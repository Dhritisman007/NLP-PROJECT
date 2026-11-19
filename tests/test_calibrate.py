from __future__ import annotations

import joblib
import numpy as np

from mml.detect.calibrate import UNCERTAIN_MIN, Calibrator, fit_calibrator, load_calibrator, reliability_table
from mml.detect.classifier import TARGETS


def _oof(n=2000, seed=0):
    """Scores for an under-confident detector: a real error is only ~35% likely at score 0.6, ~95% at 0.9, ~2% at 0.1."""
    rng = np.random.default_rng(seed)
    labels = np.array(["OK"] * (n // 2) + ["OMISSION"] * (n // 2))
    P = np.zeros((n, len(TARGETS)))
    for i, lab in enumerate(labels):
        score = rng.choice([0.1, 0.6, 0.9], p=[0.8, 0.15, 0.05]) if lab == "OK" else rng.choice([0.1, 0.6, 0.9], p=[0.02, 0.08, 0.9])
        P[i, TARGETS.index("OMISSION")] = score
        P[i, TARGETS.index("ANY")] = score
    return P, labels


def test_calibrator_maps_scores_to_observed_error_rates_and_is_monotone():
    P, labels = _oof()
    cal = fit_calibrator(P, labels)
    out = {s: cal({"OMISSION": s})["OMISSION"] for s in (0.1, 0.6, 0.9)}
    assert out[0.1] < out[0.6] < out[0.9]
    assert abs(out[0.6] - 0.35) < 0.08 and abs(out[0.9] - 0.95) < 0.05               # empirical rates from the generating process
    xs = np.linspace(0, 1, 50)
    ys = [cal({"OMISSION": float(x)})["OMISSION"] for x in xs]
    assert all(b >= a - 1e-9 for a, b in zip(ys, ys[1:])) and 0.0 <= min(ys) and max(ys) <= 1.0


def test_unknown_targets_pass_through_and_scores_outside_the_range_are_clipped():
    cal = fit_calibrator(*_oof())
    assert cal({"SOMETHING": 0.42})["SOMETHING"] == 0.42
    assert cal({"OMISSION": 5.0})["OMISSION"] == cal({"OMISSION": 0.9})["OMISSION"]


def test_uncertain_band_is_a_real_zone_just_below_the_threshold():
    P, labels = _oof()
    cal = fit_calibrator(P, labels)
    assert cal({"OMISSION": 0.6})["OMISSION"] >= UNCERTAIN_MIN > cal({"OMISSION": 0.1})["OMISSION"]


def test_reliability_table_reports_zones():
    P, labels = _oof()
    rows = {r["zone"]: r for r in reliability_table(P, labels, {t: 0.8 for t in TARGETS})["OMISSION"]}
    assert rows["at/above threshold"]["real_error_rate"] > 0.7 and 0.15 < rows["just below threshold (within 0.3)"]["real_error_rate"] < 0.45
    assert rows["below threshold"]["n"] + rows["at/above threshold"]["n"] == len(labels)


def test_load_calibrator_round_trip_and_missing_file(tmp_path):
    assert load_calibrator(tmp_path) is None
    cal = fit_calibrator(*_oof())
    joblib.dump({"calibrator": cal}, tmp_path / "calibration.joblib")
    loaded = load_calibrator(tmp_path)
    assert isinstance(loaded, Calibrator) and loaded({"OMISSION": 0.6}) == cal({"OMISSION": 0.6})
