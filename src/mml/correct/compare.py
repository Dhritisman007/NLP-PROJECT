"""Compare correction strategies on examples with known references (development comparison, Phase 5).

Strategies: ``raw`` (no correction), ``baseline_llm`` (LLM, no spans), ``span_llm`` (LLM guided by detector spans),
``constrained`` (rule-based number/date/unit/name forcing), ``full`` (constrained first, then span-guided LLM for what
the rules could not resolve). Metrics against the reference: exact restoration, mean chrF change, and over-correction
on clean examples (any change to a correct translation counts as damage).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from mml.correct.constrained import constrained_fix
from mml.correct.span_postedit import CorrectionResult, baseline_postedit, run_many, span_postedit
from mml.detect.predict import Detection
from mml.llm import LLMBackend, LLMUnavailable
from mml.synth.types import SynthExample

log = logging.getLogger(__name__)

STRATEGIES = ("raw", "constrained", "baseline_llm", "span_llm", "full")


def chrf(hyp: str, ref: str) -> float:
    from sacrebleu.metrics import CHRF

    return float(CHRF().sentence_score(hyp, [ref]).score) / 100.0


def residual_detection(det: Detection, still_flagged_numbers: bool) -> Detection:
    """Detection for the LLM step of ``full``: drop ENTITY_NUMBER when the rule-based fixer resolved everything."""
    types = set(det.types)
    if not still_flagged_numbers:
        types.discard("ENTITY_NUMBER")
    return Detection(det.probs, types, det.spans)


def correct_pairs(strategy: str, pairs: Sequence[tuple[str, str]], detections: Sequence[Detection], src_lang: str,
                  tgt_lang: str, llm: LLMBackend | None, max_edit_ratio: float = 0.6, workers: int = 4) -> list[CorrectionResult]:
    """Apply one strategy to ``(source, translation)`` pairs using the detector's findings."""
    n = len(pairs)
    if strategy == "raw":
        return [CorrectionResult(h, "not_flagged") for _, h in pairs]
    if strategy == "constrained":
        out = []
        for (src, hyp), d in zip(pairs, detections):
            if "ENTITY_NUMBER" not in d.types:
                out.append(CorrectionResult(hyp, "not_flagged"))
                continue
            r = constrained_fix(src, hyp, src_lang, tgt_lang)
            out.append(CorrectionResult(r.text, "corrected" if r.changed else "unchanged_by_llm"))
        return out
    if llm is None:
        raise LLMUnavailable("this strategy needs an LLM backend")
    if strategy == "baseline_llm":
        return run_many(lambda i: baseline_postedit(llm, pairs[i][0], pairs[i][1], src_lang, tgt_lang, max_edit_ratio),
                        range(n), workers)
    if strategy == "span_llm":
        return run_many(lambda i: span_postedit(llm, pairs[i][0], pairs[i][1], detections[i], src_lang, tgt_lang,
                                                max_edit_ratio), range(n), workers)
    if strategy == "full":
        def one(i: int) -> CorrectionResult:
            (src, hyp), d = pairs[i], detections[i]
            text = hyp
            unresolved = bool(d.types & {"ENTITY_NUMBER"})
            if "ENTITY_NUMBER" in d.types:
                r = constrained_fix(src, hyp, src_lang, tgt_lang)
                text, unresolved = r.text, bool(r.unresolved)
            det = residual_detection(d, unresolved)
            if not det.types:
                return CorrectionResult(text, "corrected" if text != hyp else "not_flagged")
            res = span_postedit(llm, src, text, det, src_lang, tgt_lang, max_edit_ratio)
            if res.status != "corrected" and text != hyp:
                return CorrectionResult(text, "corrected", problems=res.problems)
            return res
        return run_many(one, range(n), workers)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


def correct_all(strategy: str, examples: Sequence[SynthExample], detections: Sequence[Detection],
                src_lang: str, tgt_lang: str, llm: LLMBackend | None, max_edit_ratio: float = 0.6,
                workers: int = 4) -> list[CorrectionResult]:
    """Apply one strategy to labelled examples (``hyp`` is the translation to correct)."""
    return correct_pairs(strategy, [(e.src, e.hyp) for e in examples], detections, src_lang, tgt_lang, llm,
                         max_edit_ratio, workers)


def score(examples: Sequence[SynthExample], results: Sequence[CorrectionResult]) -> dict[str, Any]:
    """Metrics for one strategy (see module docstring)."""
    by_label: dict[str, list[tuple[SynthExample, CorrectionResult]]] = defaultdict(list)
    for e, r in zip(examples, results):
        by_label[e.label].append((e, r))
    clean = by_label.get("OK", [])
    bad = [(e, r) for lab, rows in by_label.items() if lab != "OK" for e, r in rows]

    def summarize(rows: list[tuple[SynthExample, CorrectionResult]]) -> dict[str, float]:
        if not rows:
            return {"n": 0}
        before = [chrf(e.hyp, e.ref) for e, _ in rows]
        after = [chrf(r.text, e.ref) for e, r in rows]
        return {"n": len(rows),
                "exact_restored": float(np.mean([r.text == e.ref for e, r in rows])),
                "changed": float(np.mean([r.text != e.hyp for e, r in rows])),
                "chrf_before": float(np.mean(before)), "chrf_after": float(np.mean(after)),
                "chrf_gain": float(np.mean(after) - np.mean(before)),
                "improved": float(np.mean([a > b + 1e-9 for a, b in zip(after, before)])),
                "worsened": float(np.mean([a < b - 1e-9 for a, b in zip(after, before)]))}

    out: dict[str, Any] = {"errors": summarize(bad), "per_type": {lab: summarize(rows) for lab, rows in by_label.items() if lab != "OK"}}
    # over-correction: correct translations that were modified (every change to a correct sentence makes it worse)
    out["clean"] = {"n": len(clean), "over_correction_rate": float(np.mean([r.text != e.hyp for e, r in clean])) if clean else 0.0}
    status: dict[str, int] = defaultdict(int)
    for r in results:
        status[r.status] += 1
    out["status"] = dict(status)
    return out
