"""Span-level evaluation of the flagged spans against the synthetic gold spans.

Two sides, because meaning loss shows up in different places:

* ``tgt`` - changed/inserted text in the translation (hallucination, number/entity/unit/date change, negation insertion,
  role swap). Gold = the exact injected spans (**gold**).
* ``src`` - content that was *deleted* from the translation (omission, deleted numbers/entities, removed negation) lives
  in the source. Its location is derived by aligning source <-> reference and mapping the deleted reference words back
  (**silver**: depends on the aligner; examples whose deleted text has no aligned source word are skipped and counted).

Examples of type *t* without gold on a side are skipped for that side (reported as ``skipped_no_gold``), never scored as errors.

Metric: character-level micro precision/recall/F1 over all examples of a split (clean examples contribute false
positives), computed per error type and side. Predicted spans are the signal spans mapped to a type by ``kind``;
optionally only for examples where the classifier predicted that type (``gate``).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np

from mml.cache import DiskCache, make_key
from mml.detect.alignment import AlignFn, word_tokens
from mml.detect.base import SpanFlag
from mml.detect.classifier import TYPE_NAMES
from mml.synth.types import SynthExample

log = logging.getLogger(__name__)


def kind_to_type(kind: str) -> str | None:
    """Error type a signal span kind provides evidence for (``None`` = type-agnostic, e.g. xCOMET)."""
    if kind == "unaligned_src":
        return "OMISSION"
    if kind == "unaligned_tgt":
        return "HALLUCINATION"
    if kind.startswith("negation"):
        return "MISTRANSLATION"
    if kind.startswith(("number_", "date_", "entity_", "lex_")):
        return "ENTITY_NUMBER"
    return None


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge touching/overlapping ``(start, end)`` ranges."""
    out: list[tuple[int, int]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def silver_source_spans(ex: SynthExample, align: AlignFn) -> list[tuple[int, int]] | None:
    """Source character spans corresponding to the text deleted from ``ex.ref`` (None if there is no deletion/mapping)."""
    deletions = [s for s in ex.errors if s.tgt_start == s.tgt_end and s.original_text.strip()]
    if not deletions:
        return None
    s_toks, r_toks = word_tokens(ex.src), word_tokens(ex.ref)
    if not s_toks or not r_toks:
        return None
    links = align([t for *_, t in s_toks], [t for *_, t in r_toks])
    ranges = []
    for d in deletions:
        a, b = d.edit_start, d.edit_start + len(d.original_text)       # reference coordinates (single deletion edit)
        ref_idx = {j for j, (rs, re_, _) in enumerate(r_toks) if rs < b and a < re_}
        ranges += [(s_toks[i][0], s_toks[i][1]) for i, j in links if j in ref_idx]
    return merge_ranges(ranges) or None


def cached_silver(examples: Sequence[SynthExample], align: AlignFn, cache: DiskCache | None,
                  model_sig: str = "mbert-itermax") -> dict[str, list[tuple[int, int]] | None]:
    """Silver source spans for all examples with a deletion (disk-cached)."""
    out: dict[str, list[tuple[int, int]] | None] = {}
    for ex in examples:
        if not any(s.tgt_start == s.tgt_end and s.original_text.strip() for s in ex.errors):
            continue
        key = make_key("silver", model_sig, ex.src, ex.ref, [(s.edit_start, s.original_text) for s in ex.errors])
        hit = cache.get(key) if cache is not None else None
        if hit is None:
            val = silver_source_spans(ex, align)
            hit = {"spans": val}
            if cache is not None:
                cache.set(key, hit)
        out[ex.id] = [tuple(x) for x in hit["spans"]] if hit["spans"] else None
    return out


def _mask(n: int, ranges: Sequence[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in ranges:
        m[max(0, s):min(n, e)] = True
    return m


def span_f1(examples: Sequence[SynthExample], spans: Sequence[Sequence[SpanFlag]],
            silver: dict[str, list[tuple[int, int]] | None],
            gate: Sequence[set[str]] | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Char-level P/R/F1 per ``type -> side``. ``gate[i]`` = predicted types of example ``i`` (None = no gating)."""
    acc: dict[tuple[str, str], list[int]] = {(t, side): [0, 0, 0, 0] for t in TYPE_NAMES for side in ("tgt", "src")}
    for i, ex in enumerate(examples):
        for t in TYPE_NAMES:
            for side in ("tgt", "src"):
                text = ex.hyp if side == "tgt" else ex.src
                if side == "tgt":
                    gold = [(s.tgt_start, s.tgt_end) for s in ex.errors if s.type == t and s.tgt_end > s.tgt_start]
                else:
                    gold = (silver.get(ex.id) or []) if ex.label == t else []
                if ex.label == t and not gold:
                    # a corrupted example of this type with no gold on this side (e.g. a deletion has no target span,
                    # a substitution has no silver source span): nothing to score here, not a false positive
                    acc[(t, side)][3] += 1
                    continue
                if gate is not None and t not in gate[i]:
                    pred: list[tuple[int, int]] = []
                else:
                    pred = [(s.start, s.end) for s in spans[i] if s.side == side and kind_to_type(s.kind) == t]
                g, p = _mask(len(text), gold), _mask(len(text), pred)
                a = acc[(t, side)]
                a[0] += int((g & p).sum())
                a[1] += int((p & ~g).sum())
                a[2] += int((g & ~p).sum())
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (t, side), (tp, fp, fn, skipped) in acc.items():
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out.setdefault(t, {})[side] = {"precision": prec, "recall": rec, "f1": f1, "gold_chars": tp + fn,
                                       "skipped_no_gold": skipped}
    return out
