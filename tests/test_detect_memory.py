from __future__ import annotations

import gc
import weakref

from mml.config import load_config
from mml.detect import run as run_mod
from mml.detect.base import Signal, SignalResult

_instances: list[weakref.ref] = []


class FakeModelSignal(Signal):
    """Holds a 'model' that must be released as soon as the signal has run."""

    name = "fake"

    def __init__(self, cfg, cache=None):
        super().__init__(cfg, cache)
        self.model = bytearray(1024)
        _instances.append(weakref.ref(self))

    def signature(self):
        return {"v": 1}

    def _compute(self, pairs):
        return [SignalResult({"x": 1.0}) for _ in pairs]


def test_signal_objects_and_their_models_are_freed_after_each_run(monkeypatch, tmp_path):
    monkeypatch.setitem(run_mod.SIGNAL_CLASSES, "fake", FakeModelSignal)
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    results, _ = run_mod.run_signals(cfg, [("a", "b")], ["fake", "fake"])
    gc.collect()
    assert "fake" in results and _instances and all(r() is None for r in _instances)


def test_free_memory_is_safe_to_call():
    from mml.utils import free_memory

    free_memory()
