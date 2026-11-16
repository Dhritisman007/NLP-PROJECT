from __future__ import annotations

from types import SimpleNamespace

import pytest

from mml.cache import DiskCache
from mml.config import load_config
from mml.llm import (
    AnthropicBackend, FakeBackend, LLMError, LLMRefused, LLMUnavailable, OpenAICompatBackend, build_llm,
)
from mml.utils import load_dotenv


def block(type_, text=""):
    return SimpleNamespace(type=type_, text=text)


def message(text="<translation>x</translation>", stop="end_turn", extra_blocks=()):
    return SimpleNamespace(content=[*extra_blocks, block("text", text)], stop_reason=stop,
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20), stop_details=None)


class FakeAnthropic:
    """Records requests; mimics client.messages.create / client.beta.messages.create."""

    def __init__(self, reply=None, error=None):
        self.requests, self.reply, self.error = [], reply or message(), error
        self.messages = SimpleNamespace(create=lambda **kw: self._call("plain", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._call("beta", kw)))

    def _call(self, kind, kw):
        self.requests.append((kind, kw))
        if self.error and kind == "beta":
            raise self.error
        return self.reply


@pytest.fixture
def cfg(tmp_path):
    return load_config(overrides=[f"paths.cache_dir={tmp_path}"])


# ------------------------------------------------------------------ request shape
def test_anthropic_request_uses_configured_model_effort_and_never_sends_sampling_params(cfg):
    fake = FakeAnthropic()
    be = AnthropicBackend(cfg, None, client=fake)
    resp = be.complete("SYS", "USER")
    kind, kw = fake.requests[0]
    assert resp.text == "<translation>x</translation>"
    assert kind == "beta" and kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "medium"}
    assert kw["system"] == "SYS" and kw["messages"] == [{"role": "user", "content": "USER"}]
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(kw)        # rejected / unnecessary on this model


def test_fallbacks_can_be_disabled_and_use_the_plain_endpoint(cfg):
    cfg.raw["llm"]["anthropic"]["use_fallbacks"] = False
    fake = FakeAnthropic()
    AnthropicBackend(cfg, None, client=fake).complete("s", "u")
    assert fake.requests[0][0] == "plain" and "fallbacks" not in fake.requests[0][1]


def test_text_comes_from_text_blocks_only_and_usage_is_counted(cfg):
    fake = FakeAnthropic(reply=message("  hello  ", extra_blocks=[block("thinking", "secret reasoning")]))
    be = AnthropicBackend(cfg, None, client=fake)
    assert be.complete("s", "u").text == "hello"
    assert (be.usage.calls, be.usage.input_tokens, be.usage.output_tokens) == (1, 100, 20)


def test_refusal_and_truncation_are_distinct_errors(cfg):
    with pytest.raises(LLMRefused):
        AnthropicBackend(cfg, None, client=FakeAnthropic(reply=message(stop="refusal"))).complete("s", "u")
    be = AnthropicBackend(cfg, None, client=FakeAnthropic(reply=message(stop="max_tokens")))
    with pytest.raises(LLMError, match="truncated"):
        be.complete("s", "u")
    assert be.usage.errors == 1


def test_missing_key_is_reported_as_unavailable_and_ambient_credentials_are_not_used(cfg, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "ambient-token-that-must-not-be-used")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://ambient.example")
    with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
        AnthropicBackend(cfg, None).complete("s", "u")


def test_client_is_built_with_explicit_key_and_explicit_base_url(cfg, monkeypatch):
    import anthropic

    captured = {}

    class Spy:
        def __init__(self, **kw):
            captured.update(kw)
            self.messages = SimpleNamespace(create=lambda **k: message())
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: message()))

    monkeypatch.setattr(anthropic, "Anthropic", Spy)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://ambient.example")
    AnthropicBackend(cfg, None).complete("s", "u")
    assert captured == {"api_key": "sk-test", "base_url": "https://api.anthropic.com"}


def test_fallback_parameter_rejection_retries_without_it(cfg):
    import anthropic
    import httpx

    resp = httpx.Response(400, request=httpx.Request("POST", "https://x"))
    err = anthropic.BadRequestError("invalid fallbacks parameter", response=resp, body=None)
    fake = FakeAnthropic(error=err)
    be = AnthropicBackend(cfg, None, client=fake)
    assert be.complete("s", "u").text
    assert [k for k, _ in fake.requests] == ["beta", "plain"] and be.use_fallbacks is False


# ------------------------------------------------------------------ cache + other backends
def test_cache_makes_repeated_prompts_free_and_reproducible(cfg, tmp_path):
    cache = DiskCache(tmp_path / "llm.sqlite")
    fake = FakeAnthropic()
    be = AnthropicBackend(cfg, cache, client=fake)
    first, second = be.complete("s", "u"), be.complete("s", "u")
    assert not first.cached and second.cached and len(fake.requests) == 1
    be.complete("s", "different")
    assert len(fake.requests) == 2 and be.usage.cache_hits == 1


def test_cache_key_depends_on_model_and_effort(cfg, tmp_path):
    cache = DiskCache(tmp_path / "llm.sqlite")
    a, fake = AnthropicBackend(cfg, cache, client=(f := FakeAnthropic())), f
    a.complete("s", "u")
    cfg.raw["llm"]["anthropic"]["effort"] = "high"
    AnthropicBackend(cfg, cache, client=fake).complete("s", "u")
    assert len(fake.requests) == 2


def test_fake_backend_and_factory(cfg, tmp_path):
    fb = FakeBackend(lambda s, u: u.upper())
    assert fb.complete("s", "abc").text == "ABC" and fb.calls == [("s", "abc")]
    assert isinstance(build_llm(cfg, "anthropic"), AnthropicBackend)
    with pytest.raises(ValueError):
        build_llm(cfg, "nope")


def test_openai_compatible_backend_with_fake_session(cfg):
    sent = {}

    class Sess:
        def post(self, url, **kw):
            sent.update(url=url, **kw)
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
                "choices": [{"message": {"content": " ok "}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    be = OpenAICompatBackend(cfg, None, session=Sess())
    assert be.complete("S", "U").text == "ok"
    assert sent["url"].endswith("/chat/completions") and sent["json"]["temperature"] == 0.0
    assert be.usage.input_tokens == 5


# ------------------------------------------------------------------ .env loader
def test_dotenv_loads_names_only_never_overrides_and_handles_quotes(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nFOO_TEST_A=value-a\nexport FOO_TEST_B='quoted b'\nFOO_TEST_C=\"dq\"\nEXISTING_TEST=new\n\nbadline\n")
    monkeypatch.delenv("FOO_TEST_A", raising=False)
    monkeypatch.delenv("FOO_TEST_B", raising=False)
    monkeypatch.delenv("FOO_TEST_C", raising=False)
    monkeypatch.setenv("EXISTING_TEST", "old")
    loaded = load_dotenv(env)
    import os

    assert sorted(loaded) == ["FOO_TEST_A", "FOO_TEST_B", "FOO_TEST_C"]
    assert os.environ["FOO_TEST_B"] == "quoted b" and os.environ["FOO_TEST_C"] == "dq"
    assert os.environ["EXISTING_TEST"] == "old"                       # existing variables win
    assert load_dotenv(tmp_path / "missing.env") == []
