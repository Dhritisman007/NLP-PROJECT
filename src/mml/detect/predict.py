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
from mml.detect.calibrate import UNCERTAIN_MIN, Calibrator, load_calibrator
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
    calibrated: dict[str, float] | None = None    # observed real-error rate for this score (None when no calibration file exists)
    uncertain: set[str] = field(default_factory=set)   # not flagged, but >= UNCERTAIN_MIN of comparable training cases were real errors

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
                 columns: list[str] | None = None, calibrate: bool = True) -> None:
        self.cfg = cfg
        self._runner = signal_runner
        root = Path(cfg.raw["paths"]["data_dir"]).parent
        from_disk = model is None                       # an explicitly passed model (tests) must not pick up a calibration file by accident
        if model is None:
            kind = kind or self._best_kind(root, cfg.pair_key)
            blob = joblib.load(Path(model_dir or root / "models" / cfg.pair_key) / f"detector_{kind}.joblib")
            model, columns = blob["model"], blob["columns"]
        self.kind = kind or model.kind
        self.calibrator: Calibrator | None = (load_calibrator(Path(model_dir or root / "models" / cfg.pair_key))
                                              if calibrate and (from_disk or model_dir is not None) else None)
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
            raw = {t: float(p[j]) for j, t in enumerate(TARGETS)}
            cal = self.calibrator(raw) if self.calibrator is not None else None
            uncertain = {t for t in TYPE_NAMES if cal and t not in types and cal[t] >= UNCERTAIN_MIN}
            out.append(Detection(raw, types, table.spans[i], cal, uncertain))
        return out

    def error_probability(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """P(any error) per pair (used by n-best re-ranking)."""
        return np.array([d.probs["ANY"] for d in self.detect(pairs)])
