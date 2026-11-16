"""Sentence-level detection metrics: per-type precision / recall / F1 / MCC, confusion matrix, group bootstrap CIs."""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, precision_recall_fscore_support

from mml.detect.classifier import TYPE_NAMES

CLASSES: tuple[str, ...] = ("OK", *TYPE_NAMES)


def binary_metrics(y_true: Sequence[bool], y_pred: Sequence[bool]) -> dict[str, float]:
    """P / R / F1 / MCC of the positive class (all zero when undefined)."""
    yt, yp = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    mcc = float(matthews_corrcoef(yt, yp)) if yt.any() and (~yt).any() and yp.any() and (~yp).any() else 0.0
    return {"precision": float(p), "recall": float(r), "f1": float(f), "mcc": mcc, "support": int(yt.sum())}


def sentence_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, Any]:
    """Per-class one-vs-rest metrics, 5-way macro-F1 / accuracy / multiclass MCC, and the binary "any error" task."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    per_class = {c: binary_metrics(yt == c, yp == c) for c in CLASSES}
    return {
        "per_class": per_class,
        "any_error": binary_metrics(yt != "OK", yp != "OK"),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=list(CLASSES), zero_division=0)),
        "macro_f1_errors_only": float(np.mean([per_class[c]["f1"] for c in TYPE_NAMES])),
        "accuracy": float((yt == yp).mean()),
        "mcc_multiclass": float(matthews_corrcoef(yt, yp)),
        "confusion": {"labels": list(CLASSES),
                      "matrix": confusion_matrix(yt, yp, labels=list(CLASSES)).tolist()},
        "n": int(len(yt)),
    }


def bootstrap_ci(metric: Callable[[np.ndarray, np.ndarray], float], y_true: Sequence[str], y_pred: Sequence[str],
                 groups: Sequence[str], n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of ``metric`` resampling whole source-sentence groups (twins stay together)."""
    yt, yp, g = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    uniq = np.unique(g)
    index = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([index[u] for u in pick])
        vals.append(metric(yt[rows], yp[rows]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def macro_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(f1_score(yt, yp, average="macro", labels=list(CLASSES), zero_division=0))


def any_error_f1(yt: np.ndarray, yp: np.ndarray) -> float:
    return float(f1_score(yt != "OK", yp != "OK", zero_division=0))


def format_markdown(m: dict[str, Any], title: str = "") -> str:
    """Per-class table in markdown."""
    lines = ([f"**{title}**", ""] if title else []) + ["| class | precision | recall | F1 | MCC | support |", "|---|---|---|---|---|---|"]
    for c in CLASSES:
        r = m["per_class"][c]
        lines.append(f"| {c} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['mcc']:.3f} | {r['support']} |")
    a = m["any_error"]
    lines.append(f"| *any error vs OK* | {a['precision']:.3f} | {a['recall']:.3f} | {a['f1']:.3f} | {a['mcc']:.3f} | {a['support']} |")
    lines.append(f"\nmacro-F1 (5 classes) {m['macro_f1']:.3f}, accuracy {m['accuracy']:.3f}, multiclass MCC {m['mcc_multiclass']:.3f}")
    return "\n".join(lines)
