from __future__ import annotations

import pytest

from mml.cache import DiskCache, make_key
from mml.config import load_config
from mml.translate import (
    BackendUnavailable, IndicTrans2Backend, MTBackend, NLLBBackend, Translator,
    load_sentences, target_script_ratio,
)


class FakeBackend(MTBackend):
    """Deterministic stand-in: 'translates' by upper-casing, records calls."""

    name = "fake"

    def __init__(self, batch_size: int = 2) -> None:
        super().__init__({"model_id": "fake/model", "revision": "r0"}, "eng_Latn", "deu_Latn",
                         device="cpu", dtype="float32", batch_size=batch_size, num_beams=1, max_length=16)
        self.calls: list[list[str]] = []

    def load(self) -> None:
        pass

    def _generate_batch(self, batch, n_best):
        self.calls.append(list(batch))
        return [[f"{s.upper()}#{k}" for k in range(n_best)] for s in batch]


def test_generate_preserves_order_despite_length_sorted_batching():
    be = FakeBackend(batch_size=2)
    srcs = ["a", "bbbb", "cc", "dddddd", "e"]
    out = be.generate(srcs, n_best=1)
    assert [o[0] for o in out] == [f"{s.upper()}#0" for s in srcs]
    assert all(len(c) <= 2 for c in be.calls)


def test_n_best_returns_n_candidates():
    out = FakeBackend().generate(["x", "y"], n_best=3)
    assert all(len(c) == 3 for c in out)


def test_translator_uses_disk_cache(tmp_path):
    cache = DiskCache(tmp_path / "mt.sqlite")
    be = FakeBackend()
    tr = Translator(be, cache)
    first = tr.translate(["hello", "world"])
    n_calls = len(be.calls)
    second = tr.translate(["hello", "world", "new"])
    assert second[:2] == first
    assert sum(len(c) for c in be.calls[n_calls:]) == 1  # only "new" hit the model
    assert len(cache) == 3


def test_cache_key_depends_on_signature_and_n_best():
    sig_a = FakeBackend().signature()
    sig_b = {**sig_a, "beams": 99}
    assert make_key(sig_a, 1, "x") != make_key(sig_b, 1, "x")
    assert make_key(sig_a, 1, "x") != make_key(sig_a, 2, "x")
    assert make_key(sig_a, 1, "x") == make_key(sig_a, 1, "x")


def test_fallback_to_nllb_is_flagged_not_silent(monkeypatch, tmp_path):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])  # en-hi -> indictrans2

    def boom(self):
        raise BackendUnavailable("gated, no token")

    monkeypatch.setattr(IndicTrans2Backend, "load", boom)
    monkeypatch.setattr(NLLBBackend, "load", lambda self: None)
    tr = Translator.from_config(cfg)
    assert tr.fallback_used and tr.requested_backend == "indictrans2" and tr.backend.name == "nllb"


def test_strict_mode_raises_instead_of_falling_back(monkeypatch, tmp_path):
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    monkeypatch.setattr(IndicTrans2Backend, "load", lambda self: (_ for _ in ()).throw(BackendUnavailable("x")))
    with pytest.raises(BackendUnavailable):
        Translator.from_config(cfg, allow_fallback=False)


def test_target_script_ratio():
    assert target_script_ratio("नमस्ते दुनिया", "hin_Deva") > 0.9
    assert target_script_ratio("hello world", "hin_Deva") == 0.0
    assert target_script_ratio("Hallo Welt", "deu_Latn") == 1.0
    assert target_script_ratio("", "hin_Deva") == 0.0


def test_smoke_file_has_20_sentences():
    from pathlib import Path

    sents = load_sentences(Path(__file__).resolve().parents[1] / "data" / "samples" / "smoke_en_20.txt")
    assert len(sents) == 20 and all(s.endswith(".") for s in sents)
