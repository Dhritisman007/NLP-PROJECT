"""Meaning-loss classifier: one binary model per error type (+ an "any error" model), probabilities -> labels.

Why one-vs-rest instead of a single softmax: real translations can contain several errors at once; the synthetic data
has exactly one, so single-label metrics are derived from the same probabilities (see :func:`predict_labels`).

Model selection protocol (no test data is ever used): grouped K-fold cross-validation over train+val, where both
versions of a source sentence (clean/corrupted) form one group; hyper-parameters and per-type decision thresholds are
chosen on the out-of-fold probabilities; the final model is refit on train+val.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mml.synth.types import ERROR_TYPES

log = logging.getLogger(__name__)

TARGETS: tuple[str, ...] = ("ANY", *(t.value for t in ERROR_TYPES))   # ANY = any error vs meaning preserved
TYPE_NAMES: tuple[str, ...] = TARGETS[1:]


def targets_matrix(labels: Sequence[str]) -> np.ndarray:
    """Binary target matrix ``[n, 5]`` (ANY + four types) from single labels; ``OK`` is all zeros."""
    Y = np.zeros((len(labels), len(TARGETS)), dtype=int)
    for i, lab in enumerate(labels):
        if lab != "OK":
            Y[i, 0] = 1
            Y[i, TARGETS.index(lab)] = 1
    return Y


def make_estimator(kind: str, params: dict[str, Any], seed: int) -> Any:
    """``logreg`` (standardised, class-balanced) or ``gbm`` (histogram gradient boosting, class-balanced)."""
    if kind == "logreg":
        return Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(C=params.get("C", 1.0), max_iter=5000, class_weight="balanced",
                                                   random_state=seed))])
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=params.get("max_depth", 4), learning_rate=params.get("learning_rate", 0.1),
            max_iter=params.get("max_iter", 200), l2_regularization=params.get("l2", 1.0),
            class_weight="balanced", random_state=seed)
    raise ValueError(f"Unknown classifier kind {kind!r}")


PARAM_GRIDS: dict[str, list[dict[str, Any]]] = {
    "logreg": [{"C": c} for c in (0.03, 0.3, 3.0)],
    "gbm": [{"max_depth": d, "max_iter": it} for d in (3, 5) for it in (100, 250)],
}


@dataclass
class MeaningLossClassifier:
    """Fitted detector: per-target estimators, decision thresholds and the feature columns it expects."""

    kind: str
    params: dict[str, Any]
    columns: list[str]
    seed: int = 42
    thresholds: dict[str, float] = field(default_factory=lambda: {t: 0.5 for t in TARGETS})
    models: dict[str, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, labels: Sequence[str]) -> "MeaningLossClassifier":
        Y = targets_matrix(labels)
        for j, t in enumerate(TARGETS):
            est = make_estimator(self.kind, self.params, self.seed)
            self.models[t] = est.fit(X, Y[:, j])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """``[n, 5]`` probabilities in :data:`TARGETS` order."""
        return np.column_stack([self.models[t].predict_proba(X)[:, 1] for t in TARGETS])

    def predict_labels(self, X: np.ndarray) -> list[str]:
        return predict_labels(self.predict_proba(X), self.thresholds)

    def predict_sets(self, X: np.ndarray) -> list[set[str]]:
        """Multi-label view: every type whose probability exceeds its threshold."""
        P = self.predict_proba(X)
        return [{t for j, t in enumerate(TYPE_NAMES) if p[j + 1] >= self.thresholds[t]} for p in P]


def predict_labels(P: np.ndarray, thresholds: dict[str, float]) -> list[str]:
    """Single label per example: the type with the highest ``p / threshold`` among those above threshold, else ``OK``."""
    thr = np.array([thresholds[t] for t in TYPE_NAMES])
    out = []
    for p in P:
        score = p[1:] / thr
        out.append(TYPE_NAMES[int(np.argmax(score))] if (score >= 1.0).any() else "OK")
    return out


def tune_thresholds(P: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    """Per-target threshold maximising binary F1 on (out-of-fold) probabilities."""
    Y = targets_matrix(labels)
    grid = np.linspace(0.05, 0.95, 37)
    thr = {}
    for j, t in enumerate(TARGETS):
        scores = [f1_score(Y[:, j], P[:, j] >= g, zero_division=0) for g in grid]
        thr[t] = float(grid[int(np.argmax(scores))])
    return thr


def cross_val_proba(kind: str, params: dict[str, Any], X: np.ndarray, labels: Sequence[str],
                    groups: Sequence[str], n_splits: int = 5, seed: int = 42) -> np.ndarray:
    """Out-of-fold probabilities ``[n, 5]`` with sentence-grouped folds (twins never straddle a fold)."""
    P = np.zeros((len(labels), len(TARGETS)))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, groups=groups):
        clf = MeaningLossClassifier(kind, params, [], seed).fit(X[tr], [labels[i] for i in tr])
        P[te] = clf.predict_proba(X[te])
    return P


def macro_f1_5way(P: np.ndarray, labels: Sequence[str], thresholds: dict[str, float]) -> float:
    """Macro-F1 over OK + four types of the single-label decision rule."""
    return float(f1_score(list(labels), predict_labels(P, thresholds), average="macro",
                          labels=["OK", *TYPE_NAMES], zero_division=0))


def select_model(kind: str, X: np.ndarray, labels: Sequence[str], groups: Sequence[str], columns: list[str],
                 n_splits: int = 5, seed: int = 42) -> tuple[MeaningLossClassifier, np.ndarray, list[dict[str, Any]]]:
    """Grid-search hyper-parameters on grouped-CV macro-F1; return (refit model with tuned thresholds, OOF probs, log)."""
    best: tuple[float, dict[str, Any], np.ndarray, dict[str, float]] | None = None
    trace = []
    for params in PARAM_GRIDS[kind]:
        P = cross_val_proba(kind, params, X, labels, groups, n_splits, seed)
        thr = tune_thresholds(P, labels)
        score = macro_f1_5way(P, labels, thr)
        trace.append({"params": params, "cv_macro_f1": round(score, 4)})
        log.info("%s %s -> CV macro-F1 %.4f", kind, params, score)
        if best is None or score > best[0]:
            best = (score, params, P, thr)
    assert best is not None
    _, params, P, thr = best
    model = MeaningLossClassifier(kind, params, columns, seed, thr).fit(X, labels)
    return model, P, trace
