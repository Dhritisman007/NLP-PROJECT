"""Calibrate the detector's raw scores into honest probabilities (and an "uncertain" verdict).

The classifier's per-type score is *not* the chance that an error is real: per-type decision thresholds are high (0.57-0.85, tuned
for F1), so a score of 0.71 for OMISSION can sit below the threshold and still mean "about 1 in 3 similar cases was a real omission"
(measured on out-of-fold predictions). Showing 71% next to "not flagged" is confusing and hides the useful middle ground.

``fit_calibration`` learns, per target, a monotone map from raw score to the **observed rate of real errors** (isotonic regression)
on the *out-of-fold* predictions of train+val (the same sentence-grouped folds used for model selection). The test split is never
used. The flagging decision itself is unchanged (it still uses the tuned thresholds), so no reported metric moves; calibration only
changes what is *displayed* and adds a third verdict, ``uncertain``: not flagged, but at least ``UNCERTAIN_MIN`` of similar training
cases were real errors.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from mml.config import Config
from mml.detect.classifier import TARGETS, cross_val_proba

log = logging.getLogger(__name__)

UNCERTAIN_MIN = 0.20          # not flagged, but >= 20% of comparable out-of-fold cases were real errors -> "uncertain, please check"
CALIBRATION_FILE = "calibration.joblib"


@dataclass
class Calibrator:
    """Per-target isotonic maps from raw score to observed error rate."""

    maps: dict[str, IsotonicRegression]
    n_fit: int = 0

    def __call__(self, probs: Mapping[str, float]) -> dict[str, float]:
        return {t: float(self.maps[t].predict([p])[0]) if t in self.maps else float(p) for t, p in probs.items()}


def _targets_truth(labels: np.ndarray) -> dict[str, np.ndarray]:
    return {t: ((labels != "OK") if t == "ANY" else (labels == t)) for t in TARGETS}


def fit_calibrator(P: np.ndarray, labels: list[str] | np.ndarray) -> Calibrator:
    """Fit one isotonic map per target on out-of-fold probabilities ``P`` ``[n, 5]`` (columns in ``TARGETS`` order)."""
    labels = np.asarray(labels)
    truth = _targets_truth(labels)
    maps = {}
    for j, t in enumerate(TARGETS):
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(P[:, j], truth[t].astype(float))
        maps[t] = iso
    return Calibrator(maps, len(labels))


def reliability_table(P: np.ndarray, labels: list[str] | np.ndarray, thresholds: Mapping[str, float],
                      bins: int = 5) -> dict[str, list[dict[str, float]]]:
    """Observed real-error rate per score bin (quantile bins), so the claim "scores below the threshold are not zero-risk" is auditable."""
    labels = np.asarray(labels)
    truth = _targets_truth(labels)
    out: dict[str, list[dict[str, float]]] = {}
    for j, t in enumerate(TARGETS):
        p, y = P[:, j], truth[t]
        thr = thresholds[t]
        rows = []
        below, above = p < thr, p >= thr
        for name, sel in (("below threshold", below), ("at/above threshold", above)):
            if sel.sum():
                rows.append({"zone": name, "n": int(sel.sum()), "real_error_rate": float(y[sel].mean())})
        mid = (p >= max(0.0, thr - 0.3)) & below
        if mid.sum():
            rows.append({"zone": "just below threshold (within 0.3)", "n": int(mid.sum()), "real_error_rate": float(y[mid].mean())})
        out[t] = rows
    return out


def fit_calibration(cfg: Config) -> Path:
    """Compute out-of-fold predictions of the pair's selected detector, fit the calibrator and save it next to the model."""
    from mml.detect.features import FeatureTable
    from mml.detect.train import base_id, feature_path

    root = Path(cfg.raw["paths"]["data_dir"]).parent
    res = json.loads((root / "results" / "detector" / cfg.pair_key / "results.json").read_text(encoding="utf-8"))
    kind = res["best_kind"]
    spec = res["models"][kind]
    full = FeatureTable.concat(FeatureTable.load(feature_path(cfg, "train")), FeatureTable.load(feature_path(cfg, "val")))
    groups = [base_id(i) for i in full.ids]
    P = cross_val_proba(kind, spec["params"], full.X, full.labels, groups, 5, cfg.seed)
    cal = fit_calibrator(P, full.labels)
    model_dir = root / "models" / cfg.pair_key
    joblib.dump({"calibrator": cal, "kind": kind, "n": len(full.labels), "source": "out-of-fold predictions on train+val"},
                model_dir / CALIBRATION_FILE)
    table = reliability_table(P, full.labels, spec["thresholds"])
    out = root / "results" / "detector" / cfg.pair_key / "calibration.json"
    out.write_text(json.dumps({"pair": cfg.pair_key, "kind": kind, "n": len(full.labels), "uncertain_min": UNCERTAIN_MIN,
                               "reliability": table}, indent=2), encoding="utf-8")
    log.info("Calibration for %s saved (%d out-of-fold examples)", cfg.pair_key, len(full.labels))
    return model_dir / CALIBRATION_FILE


def load_calibrator(model_dir: Path) -> Calibrator | None:
    path = Path(model_dir) / CALIBRATION_FILE
    if not path.exists():
        return None
    return joblib.load(path)["calibrator"]
