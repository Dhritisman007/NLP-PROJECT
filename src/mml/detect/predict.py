"""Apply a trained detector to arbitrary (source, translation) pairs: predicted error types + flagged spans.

This is the interface the correction code (and later the pipeline/demo) uses. It runs the cached signals, assembles
the same feature columns the model was trained on, and thresholds the per-type probabilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np

from mml.config import Config
from mml.detect.base import SignalResult, SpanFlag
from mml.detect.classifier import TARGETS, TYPE_NAMES, MeaningLossClassifier
from mml.detect.features import build_table
from mml.detect.run import DEFAULT_SIGNALS, run_signals
from mml.detect.spans import kind_to_type

log = logging.getLogger(__name__)

SignalRunner = Callable[[Config, list[tuple[str, str]], Sequence[str]], "tuple[dict[str, list[SignalResult]], dict[str, str]]"]


@dataclass
class Detection:
    """Detector output for one pair: probabilities per target, predicted error types, and all flagged spans."""

    probs: dict[str, float]
    types: set[str]
    spans: list[SpanFlag] = field(default_factory=list)

    @property
    def any_error(self) -> bool:
        return bool(self.types)

    def spans_for(self, types: set[str] | None = None) -> list[SpanFlag]:
        """Spans whose kind supports one of ``types`` (default: the predicted types)."""
        wanted = self.types if types is None else types
        return [s for s in self.spans if kind_to_type(s.kind) in wanted]


class Detector:
    """Loads ``models/<pair>/detector_<kind>.joblib`` (default: the kind that won model selection)."""

    def __init__(self, cfg: Config, kind: str | None = None, model_dir: str | Path | None = None,
                 signal_runner: SignalRunner = run_signals, model: MeaningLossClassifier | None = None,
                 columns: list[str] | None = None) -> None:
        self.cfg = cfg
        self._runner = signal_runner
        root = Path(cfg.raw["paths"]["data_dir"]).parent
        if model is None:
            kind = kind or self._best_kind(root, cfg.pair_key)
            blob = joblib.load(Path(model_dir or root / "models" / cfg.pair_key) / f"detector_{kind}.joblib")
            model, columns = blob["model"], blob["columns"]
        self.kind = kind or model.kind
        self.model: MeaningLossClassifier = model
        self.columns: list[str] = columns or model.columns
        self.skipped: dict[str, str] = {}

    @staticmethod
    def _best_kind(root: Path, pair: str) -> str:
        res = root / "results" / "detector" / pair / "results.json"
        return json.loads(res.read_text(encoding="utf-8"))["best_kind"] if res.exists() else "logreg"

    def detect(self, pairs: list[tuple[str, str]], signals: Sequence[str] = DEFAULT_SIGNALS) -> list[Detection]:
        """Predict error types and spans for each (source, translation) pair."""
        if not pairs:
            return []
        results, self.skipped = self._runner(self.cfg, pairs, signals)
        table = build_table([str(i) for i in range(len(pairs))], ["OK"] * len(pairs), pairs, results)
        missing = [c for c in self.columns if c not in table.columns]
        if missing:
            raise ValueError(f"Signals for features {missing[:5]}... are unavailable (skipped: {self.skipped}); the "
                             "detector was trained with them")
        X = table.X[:, [table.columns.index(c) for c in self.columns]]
        P = self.model.predict_proba(X)
        out = []
        for i, p in enumerate(P):
            types = {t for j, t in enumerate(TYPE_NAMES) if p[j + 1] >= self.model.thresholds[t]}
            out.append(Detection({t: float(p[j]) for j, t in enumerate(TARGETS)}, types, table.spans[i]))
        return out

    def error_probability(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """P(any error) per pair (used by n-best re-ranking)."""
        return np.array([d.probs["ANY"] for d in self.detect(pairs)])
