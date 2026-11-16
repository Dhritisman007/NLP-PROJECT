from __future__ import annotations

import pytest

from mml.cache import DiskCache
from mml.config import load_config
from mml.detect.base import Signal, SignalResult, SignalUnavailable, SpanFlag
from mml.detect.run import run_signals


class Counting(Signal):
    name = "counting"
    chunk_size = 2

    def __init__(self, cfg, cache=None, fail_on=None):
        super().__init__(cfg, cache)
        self.calls: list[list[tuple[str, str]]] = []
        self.fail_on = fail_on

    def signature(self):
        return {"v": 1}

    def _compute(self, pairs):
        if self.fail_on and any(p == self.fail_on for p in pairs):
            raise KeyboardInterrupt("simulated crash")
        self.calls.append(list(pairs))
        return [SignalResult({"len": float(len(h))}, [SpanFlag("tgt", 0, 1, h[:1], "x")]) for _, h in pairs]


def test_results_keep_order_dedupe_and_cache(tmp_path):
    cfg = load_config()
    cache = DiskCache(tmp_path / "s.sqlite")
    sig = Counting(cfg, cache)
    pairs = [("a", "xx"), ("b", "y"), ("a", "xx"), ("c", "zzz")]
    out = sig.compute(pairs)
    assert [r.features["len"] for r in out] == [2, 1, 2, 3]
    assert sum(len(c) for c in sig.calls) == 3                     # duplicate pair computed once
    again = Counting(cfg, cache).compute(pairs)                    # a fresh object still hits the disk cache
    assert [r.to_dict() for r in again] == [r.to_dict() for r in out]
    sig2 = Counting(cfg, cache)
    sig2.compute(pairs)
    assert sig2.calls == []


def test_progress_is_checkpointed_chunk_by_chunk(tmp_path):
    cfg = load_config()
    cache = DiskCache(tmp_path / "s.sqlite")
    pairs = [(str(i), "t" * (i + 1)) for i in range(6)]
    crashing = Counting(cfg, cache, fail_on=pairs[4])              # third chunk (items 4,5) crashes
    with pytest.raises(KeyboardInterrupt):
        crashing.compute(pairs)
    assert len(cache) == 4                                         # first two chunks were saved
    resumed = Counting(cfg, cache)
    resumed.compute(pairs)
    assert sum(len(c) for c in resumed.calls) == 2                 # only the unfinished work is redone


def test_cache_key_depends_on_signature(tmp_path):
    cfg = load_config()
    cache = DiskCache(tmp_path / "s.sqlite")
    Counting(cfg, cache).compute([("a", "b")])

    class Other(Counting):
        def signature(self):
            return {"v": 2}

    other = Other(cfg, cache)
    other.compute([("a", "b")])
    assert len(other.calls) == 1


def test_result_json_roundtrip():
    r = SignalResult({"a": 1.0}, [SpanFlag("src", 1, 3, "ab", "unaligned_src", 0.5)])
    assert SignalResult.from_dict(r.to_dict()) == r


def test_unavailable_signals_are_skipped_with_a_reason_not_faked(monkeypatch, tmp_path):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("mml.detect.qe.get_hf_token", lambda: None)
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    results, skipped = run_signals(cfg, [("Hello world.", "Hallo Welt.")], ["rules", "cometkiwi", "xcomet"])
    assert "rules" in results and "cometkiwi" not in results and "xcomet" not in results
    assert set(skipped) == {"cometkiwi", "xcomet"} and all(skipped.values())


def test_unknown_signal_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_signals(load_config(overrides=[f"paths.cache_dir={tmp_path}"]), [("a", "b")], ["nope"])
