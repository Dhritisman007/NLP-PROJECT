"""Common interface for detection signals, with disk caching and graceful skipping.

A *signal* maps ``(source, translation)`` pairs to a :class:`SignalResult`: a flat dict of numeric features
(consumed by ``features.py`` in Phase 4) plus optional flagged spans (evidence shown to the user and used for
span-level metrics). Expensive results are cached in SQLite, keyed by signal name, model/config signature and
the text pair; long runs are checkpointed chunk by chunk so an interrupted Colab session resumes where it stopped.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mml.cache import DiskCache, make_key
from mml.config import Config

log = logging.getLogger(__name__)


class SignalUnavailable(RuntimeError):
    """The signal cannot run here (missing token / package / hardware profile). Callers skip it with a warning."""


@dataclass
class SpanFlag:
    """A flagged span. ``side`` is ``"src"`` (e.g. omitted source content) or ``"tgt"`` (e.g. a suspicious insertion)."""

    side: str
    start: int
    end: int
    text: str
    kind: str          # e.g. unaligned, number_missing, number_extra, negation, xcomet_major
    score: float = 1.0


@dataclass
class SignalResult:
    """Features (always numeric) and flagged spans for one (source, translation) pair."""

    features: dict[str, float]
    spans: list[SpanFlag] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SignalResult":
        return cls(dict(row["features"]), [SpanFlag(**s) for s in row.get("spans", [])])


class Signal(ABC):
    """Base class: subclasses define ``name``, ``signature()``, ``check()`` and ``_compute()``."""

    name: str = "signal"
    chunk_size: int = 32          # pairs computed (and checkpointed) per chunk

    def __init__(self, cfg: Config, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache
        self.src_lang: str = cfg.pair["src_lang"]
        self.tgt_lang: str = cfg.pair["tgt_lang"]

    @abstractmethod
    def signature(self) -> dict[str, Any]:
        """Everything that changes the output (model ids/revisions, languages, thresholds)."""

    def check(self) -> None:
        """Raise :class:`SignalUnavailable` if the signal cannot run in this environment."""

    @abstractmethod
    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        """Compute results for uncached pairs (one chunk)."""

    def compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        """Cached, chunked computation; results are returned in the order of ``pairs``."""
        self.check()
        sig = self.signature()
        keys = [make_key(self.name, sig, s, h) for s, h in pairs]
        out: dict[str, SignalResult] = {}
        todo: list[tuple[str, tuple[str, str]]] = []
        seen: set[str] = set()
        for key, pair in zip(keys, pairs):
            if key in seen:
                continue
            seen.add(key)
            hit = self.cache.get(key) if self.cache is not None else None
            if hit is not None:
                out[key] = SignalResult.from_dict(hit)
            else:
                todo.append((key, pair))
        log.info("[%s] %d pairs: %d unique, %d cached, %d to compute", self.name, len(pairs), len(seen),
                 len(seen) - len(todo), len(todo))
        for i in range(0, len(todo), self.chunk_size):
            chunk = todo[i:i + self.chunk_size]
            results = self._compute([p for _, p in chunk])
            if len(results) != len(chunk):
                raise RuntimeError(f"{self.name}: got {len(results)} results for {len(chunk)} pairs")
            for (key, _), res in zip(chunk, results):
                out[key] = res
                if self.cache is not None:
                    self.cache.set(key, res.to_dict())
            log.info("[%s] %d/%d computed", self.name, min(i + self.chunk_size, len(todo)), len(todo))
        return [out[k] for k in keys]


def open_cache(cfg: Config, name: str = "signals") -> DiskCache:
    """The shared SQLite cache for detection signals (``<cache_dir>/<name>.sqlite``)."""
    return DiskCache(Path(cfg.raw["paths"]["cache_dir"]) / f"{name}.sqlite")


def model_device(cfg: Config) -> str:
    """Torch device string for the active hardware profile."""
    from mml.utils import resolve_device

    return resolve_device(cfg.profile["device"])
