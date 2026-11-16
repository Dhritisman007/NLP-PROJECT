"""Pluggable LLM backends for correction: Anthropic (default), any OpenAI-compatible server, or a local HF model.

All backends share one interface, :meth:`LLMBackend.complete`, and one on-disk cache keyed by backend, model and the
exact prompt. That cache is what makes experiments reproducible: ``claude-opus-5-5`` does not accept ``temperature``,
so identical prompts are replayed from disk instead of being re-sampled.

Credentials: the Anthropic backend reads **only** the variable named in ``llm.anthropic.api_key_env`` and always sends
requests to the configured ``base_url`` (the SDK would otherwise inherit ambient ``ANTHROPIC_BASE_URL`` /
``ANTHROPIC_AUTH_TOKEN`` values that belong to other tools). Keys are never logged.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mml.cache import DiskCache, make_key
from mml.config import Config

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The backend cannot run here (missing key / package). Callers skip correction with a warning."""


class LLMError(RuntimeError):
    """A request failed or returned unusable output."""


class LLMRefused(LLMError):
    """The model declined the request (``stop_reason == "refusal"``)."""


@dataclass
class LLMResponse:
    """Text plus bookkeeping."""

    text: str
    cached: bool = False


@dataclass
class UsageStats:
    """Running token/call counters for cost reporting."""

    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    refusals: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class LLMBackend(ABC):
    """Interface: ``complete(system, user) -> LLMResponse`` with caching and usage counters."""

    name = "base"

    def __init__(self, cache: DiskCache | None = None) -> None:
        self.cache = cache
        self.usage = UsageStats()

    @abstractmethod
    def signature(self) -> dict[str, Any]:
        """Everything that changes the output (model id, effort, ...); part of the cache key."""

    @abstractmethod
    def _complete(self, system: str, user: str) -> str:
        """One uncached request; returns the model's text."""

    def check(self) -> None:
        """Raise :class:`LLMUnavailable` if the backend cannot run."""

    def complete(self, system: str, user: str) -> LLMResponse:
        key = make_key(self.name, self.signature(), system, user)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self.usage.cache_hits += 1
                return LLMResponse(hit["text"], cached=True)
        self.check()
        try:
            text = self._complete(system, user)
        except LLMRefused:
            self.usage.refusals += 1
            raise
        except LLMError:
            self.usage.errors += 1
            raise
        self.usage.calls += 1
        if self.cache is not None:
            self.cache.set(key, {"text": text})
        return LLMResponse(text)


# --------------------------------------------------------------------------- Anthropic
class AnthropicBackend(LLMBackend):
    """Claude via the official ``anthropic`` SDK (Messages API)."""

    name = "anthropic"

    def __init__(self, cfg: Config, cache: DiskCache | None = None, client: Any = None) -> None:
        super().__init__(cache)
        spec = cfg.raw["llm"]["anthropic"]
        self.model: str = spec["model"]
        self.effort: str = spec.get("effort", "medium")
        self.max_tokens: int = int(spec.get("max_tokens", 8000))
        self.use_fallbacks: bool = bool(spec.get("use_fallbacks", True))
        self.api_key_env: str = spec.get("api_key_env", "ANTHROPIC_API_KEY")
        self.base_url: str = spec.get("base_url", "https://api.anthropic.com")
        self._client = client

    def signature(self) -> dict[str, Any]:
        return {"model": self.model, "effort": self.effort, "max_tokens": self.max_tokens}

    def check(self) -> None:
        if self._client is None and not os.environ.get(self.api_key_env):
            raise LLMUnavailable(
                f"{self.api_key_env} is not set. Put it in the git-ignored .env file (see .env.example) or export it; "
                "ambient ANTHROPIC_AUTH_TOKEN/ANTHROPIC_BASE_URL are deliberately not used.")

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=os.environ[self.api_key_env], base_url=self.base_url)
        return self._client

    def _request(self, client: Any, system: str, user: str, fallbacks: bool) -> Any:
        kwargs: dict[str, Any] = dict(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}], output_config={"effort": self.effort})
        if fallbacks:
            return client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
        return client.messages.create(**kwargs)

    def _complete(self, system: str, user: str) -> str:
        import anthropic

        client = self._get_client()
        try:
            try:
                msg = self._request(client, system, user, self.use_fallbacks)
            except anthropic.BadRequestError as exc:
                if not (self.use_fallbacks and "fallback" in str(exc).lower()):
                    raise
                log.warning("Server-side fallbacks rejected (%s); retrying without them", exc)
                self.use_fallbacks = False
                msg = self._request(client, system, user, False)
        except anthropic.AuthenticationError as exc:
            raise LLMUnavailable("Anthropic rejected the API key (check ANTHROPIC_API_KEY)") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Anthropic connection error: {exc}") from exc
        usage = getattr(msg, "usage", None)
        if usage is not None:
            self.usage.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.usage.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        if msg.stop_reason == "refusal":
            raise LLMRefused(f"model refused (category={getattr(getattr(msg, 'stop_details', None), 'category', None)})")
        if msg.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens (thinking + answer exceeded the budget)")
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


# --------------------------------------------------------------------------- OpenAI-compatible
class OpenAICompatBackend(LLMBackend):
    """Any server exposing ``POST {base_url}/chat/completions`` (OpenAI, vLLM, Ollama, ...), via ``requests``."""

    name = "openai"

    def __init__(self, cfg: Config, cache: DiskCache | None = None, session: Any = None) -> None:
        super().__init__(cache)
        spec = cfg.raw["llm"]["openai"]
        self.model, self.base_url = spec["model"], spec["base_url"].rstrip("/")
        self.temperature = float(spec.get("temperature", 0.0))
        self.api_key_env = spec.get("api_key_env", "OPENAI_API_KEY")
        self._session = session

    def signature(self) -> dict[str, Any]:
        return {"model": self.model, "base_url": self.base_url, "temperature": self.temperature}

    def check(self) -> None:
        if self._session is None and not os.environ.get(self.api_key_env):
            raise LLMUnavailable(f"{self.api_key_env} is not set")

    def _complete(self, system: str, user: str) -> str:
        import requests

        http = self._session or requests
        try:
            resp = http.post(
                f"{self.base_url}/chat/completions", timeout=120,
                headers={"Authorization": f"Bearer {os.environ.get(self.api_key_env, '')}"},
                json={"model": self.model, "temperature": self.temperature,
                      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - surface any transport/HTTP problem uniformly
            raise LLMError(f"OpenAI-compatible request failed: {type(exc).__name__}: {exc}") from exc
        usage = data.get("usage", {})
        self.usage.input_tokens += int(usage.get("prompt_tokens", 0))
        self.usage.output_tokens += int(usage.get("completion_tokens", 0))
        try:
            return str(data["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError) as exc:
            raise LLMError(f"unexpected response shape: {str(data)[:200]}") from exc


# --------------------------------------------------------------------------- local HF model
class HFLocalBackend(LLMBackend):
    """Local instruct model via transformers (greedy). Qwen2.5-7B-Instruct on Colab, 1.5B on a laptop (weak for Hindi)."""

    name = "hf_local"

    def __init__(self, cfg: Config, cache: DiskCache | None = None) -> None:
        super().__init__(cache)
        spec = cfg.raw["llm"]["hf_local"]
        heavy = bool(cfg.profile.get("heavy_models", False))
        self.model_id = spec["model_id_colab"] if heavy else spec["model_id_cpu"]
        self.four_bit = bool(cfg.profile.get("llm_local_4bit", False))
        self.max_new_tokens = int(spec.get("max_new_tokens", 256))
        self._tok = self._model = None

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id, "4bit": self.four_bit, "max_new_tokens": self.max_new_tokens}

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info("Loading local LLM %s", self.model_id)
        kwargs: dict[str, Any] = {}
        if self.four_bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        if not self.four_bit and torch.cuda.is_available():
            self._model = self._model.to("cuda")

    def _complete(self, system: str, user: str) -> str:
        import torch

        if self._model is None:
            self._load()
        assert self._tok is not None and self._model is not None
        prompt = self._tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                               tokenize=False, add_generation_prompt=True)
        enc = self._tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.inference_mode():
            out = self._model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False)
        self.usage.input_tokens += int(enc["input_ids"].shape[1])
        self.usage.output_tokens += int(out.shape[1] - enc["input_ids"].shape[1])
        return self._tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# --------------------------------------------------------------------------- test double + factory
@dataclass
class FakeBackend(LLMBackend):
    """Deterministic stand-in for tests: ``responder(system, user) -> text``; records every call."""

    responder: Any = None
    calls: list[tuple[str, str]] = field(default_factory=list)
    name = "fake"

    def __init__(self, responder: Any, cache: DiskCache | None = None) -> None:
        LLMBackend.__init__(self, cache)
        self.responder = responder
        self.calls = []

    def signature(self) -> dict[str, Any]:
        return {"v": 1}

    def _complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return str(self.responder(system, user))


def build_llm(cfg: Config, name: str | None = None, use_cache: bool = True) -> LLMBackend:
    """Backend from ``llm.backend`` (or ``name``), with the shared on-disk cache."""
    name = name or cfg.raw["llm"]["backend"]
    cache = DiskCache(Path(cfg.raw["paths"]["cache_dir"]) / "llm.sqlite") if use_cache else None
    if name == "anthropic":
        return AnthropicBackend(cfg, cache)
    if name == "openai":
        return OpenAICompatBackend(cfg, cache)
    if name == "hf_local":
        return HFLocalBackend(cfg, cache)
    raise ValueError(f"Unknown llm backend {name!r}; choose anthropic | openai | hf_local")
