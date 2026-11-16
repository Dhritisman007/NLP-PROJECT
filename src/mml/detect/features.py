"""Combine all detection signals into one feature table (rows = examples, columns = named features).

Features are grouped by the signal that produced them so that Phase 4 can drop one group at a time (ablation):

``rules``     number/date/negation/generic-entity consistency
``lex``       closed-lexicon entity and unit checks (``lex_*``; overlap with the error generator's gazetteer!)
``labse``     cross-lingual cosine
``align``     simalign unaligned-token statistics
``roundtrip`` back-translation chrF + bidirectional NLI
``qe``        CometKiwi / xCOMET (only present if those signals actually ran)
``surface``   raw length statistics

Signals that were skipped contribute no columns (no placeholder values).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from mml.detect.base import SignalResult, SpanFlag

SIGNAL_TO_GROUP = {"rules": "rules", "labse": "labse", "align": "align", "roundtrip": "roundtrip",
                   "cometkiwi": "qe", "xcomet": "qe"}
GROUP_ORDER = ("rules", "lex", "labse", "align", "roundtrip", "qe", "surface")


def group_of(name: str, signal: str) -> str:
    """Feature group of a column: ``lex_*`` columns of the rules signal form their own group."""
    if signal == "rules" and name.startswith("lex_"):
        return "lex"
    return SIGNAL_TO_GROUP[signal]


def surface_features(src: str, hyp: str) -> dict[str, float]:
    """Length statistics that need no model."""
    sw, hw = max(len(src.split()), 1), len(hyp.split())
    return {"len_src_words": float(sw), "len_hyp_words": float(hw), "len_word_ratio": hw / sw,
            "len_char_ratio": len(hyp) / max(len(src), 1)}


@dataclass
class FeatureTable:
    """Feature matrix with provenance: column names/groups, example ids/labels and each example's flagged spans."""

    ids: list[str]
    labels: list[str]
    columns: list[str]
    col_groups: list[str]
    X: np.ndarray
    spans: list[list[SpanFlag]]

    def groups(self) -> list[str]:
        """Feature groups present, in canonical order."""
        return [g for g in GROUP_ORDER if g in self.col_groups]

    def select(self, drop: Sequence[str] = (), keep: Sequence[str] | None = None) -> tuple[np.ndarray, list[str]]:
        """Columns after dropping/keeping whole groups. Returns ``(X_sub, column_names)``."""
        idx = [i for i, g in enumerate(self.col_groups)
               if g not in drop and (keep is None or g in keep)]
        return self.X[:, idx], [self.columns[i] for i in idx]

    def subset(self, rows: Sequence[int]) -> "FeatureTable":
        r = list(rows)
        return FeatureTable([self.ids[i] for i in r], [self.labels[i] for i in r], self.columns, self.col_groups,
                            self.X[r], [self.spans[i] for i in r])

    @classmethod
    def concat(cls, a: "FeatureTable", b: "FeatureTable") -> "FeatureTable":
        """Stack two tables with identical columns."""
        if a.columns != b.columns:
            raise ValueError("Cannot concatenate feature tables with different columns")
        return cls(a.ids + b.ids, a.labels + b.labels, a.columns, a.col_groups, np.vstack([a.X, b.X]), a.spans + b.spans)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.with_suffix(".npz"), X=self.X)
        meta = {"ids": self.ids, "labels": self.labels, "columns": self.columns, "col_groups": self.col_groups,
                "spans": [[s.__dict__ for s in sp] for sp in self.spans]}
        path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureTable":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        X = np.load(path.with_suffix(".npz"))["X"]
        return cls(meta["ids"], meta["labels"], meta["columns"], meta["col_groups"], X,
                   [[SpanFlag(**s) for s in sp] for sp in meta["spans"]])


def build_table(ids: Sequence[str], labels: Sequence[str], pairs: Sequence[tuple[str, str]],
                results: dict[str, list[SignalResult]]) -> FeatureTable:
    """Assemble the table from per-signal results (``results[name][i]`` belongs to ``pairs[i]``)."""
    n = len(pairs)
    for name, res in results.items():
        if len(res) != n:
            raise ValueError(f"signal {name!r}: {len(res)} results for {n} pairs")
    columns: list[str] = []
    groups: list[str] = []
    for sig in (s for s in SIGNAL_TO_GROUP if s in results):
        for feat in results[sig][0].features if n else {}:
            columns.append(feat)
            groups.append(group_of(feat, sig))
    surf = [surface_features(s, h) for s, h in pairs]
    for feat in (surf[0] if n else {}):
        columns.append(feat)
        groups.append("surface")
    X = np.zeros((n, len(columns)), dtype=np.float64)
    spans: list[list[SpanFlag]] = [[] for _ in range(n)]
    for i in range(n):
        merged: dict[str, float] = {}
        for sig in (s for s in SIGNAL_TO_GROUP if s in results):
            merged.update(results[sig][i].features)
            spans[i] += results[sig][i].spans
        merged.update(surf[i])
        X[i] = [merged[c] for c in columns]
    return FeatureTable(list(ids), list(labels), columns, groups, X, spans)
