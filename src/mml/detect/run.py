"""Run a set of detection signals over (source, translation) pairs; unavailable signals are skipped, never faked."""

from __future__ import annotations

import logging
from typing import Sequence

from mml.cache import DiskCache
from mml.config import Config
from mml.detect.alignment import WordAligner
from mml.detect.base import Signal, SignalResult, SignalUnavailable, open_cache
from mml.detect.qe import CometKiwi, XComet
from mml.detect.roundtrip import RoundTripNLI
from mml.detect.rules import RuleChecks
from mml.detect.similarity import LabseSimilarity
from mml.utils import free_memory

log = logging.getLogger(__name__)

SIGNAL_CLASSES: dict[str, type[Signal]] = {
    "rules": RuleChecks, "labse": LabseSimilarity, "align": WordAligner, "roundtrip": RoundTripNLI,
    "cometkiwi": CometKiwi, "xcomet": XComet,
}
DEFAULT_SIGNALS: tuple[str, ...] = tuple(SIGNAL_CLASSES)
UNCACHED = frozenset({"rules"})        # pure-python and fast: recomputing is cheaper than a cache lookup


def run_signals(cfg: Config, pairs: list[tuple[str, str]], names: Sequence[str] = DEFAULT_SIGNALS,
                cache: DiskCache | None = None) -> tuple[dict[str, list[SignalResult]], dict[str, str]]:
    """Compute each named signal for ``pairs``. Returns ``(results, skipped)``; ``skipped`` maps name -> reason.

    Signals run one after another and each one's models are freed before the next signal loads.
    """
    unknown = set(names) - set(SIGNAL_CLASSES)
    if unknown:
        raise ValueError(f"Unknown signals {sorted(unknown)}; choose from {sorted(SIGNAL_CLASSES)}")
    cache = cache if cache is not None else open_cache(cfg)
    results: dict[str, list[SignalResult]] = {}
    skipped: dict[str, str] = {}
    for name in names:
        signal = SIGNAL_CLASSES[name](cfg, None if name in UNCACHED else cache)
        try:
            results[name] = signal.compute(pairs)
        except SignalUnavailable as exc:
            skipped[name] = str(exc)
            log.warning("Signal '%s' SKIPPED: %s", name, exc)
        finally:
            # Each signal lazily loads its own model(s) (LaBSE, mBERT, NLLB+NLI...). On an 8 GB laptop they must not
            # pile up: drop the signal object (and with it the models), then release accelerator memory.
            del signal
            free_memory()
    return results, skipped
