"""Phase 5 development experiments (validation data only; the held-out test split is never touched here)."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mml.config import Config
from mml.correct.compare import STRATEGIES, chrf, correct_all, score
from mml.correct.rerank import pick, rerank_batch
from mml.correct.span_postedit import build_baseline_prompt, build_problems, build_span_prompt
from mml.detect.predict import Detection, Detector
from mml.llm import LLMBackend, LLMUnavailable, build_llm
from mml.synth.build import load_split
from mml.synth.types import ERROR_TYPES, SynthExample

log = logging.getLogger(__name__)

NO_LLM = ("raw", "constrained")


def results_dir(cfg: Config, name: str) -> Path:
    d = Path(cfg.raw["run"]["output_dir"]).parent / name / cfg.pair_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def sample_examples(cfg: Config, n: int, seed: int) -> list[SynthExample]:
    """``n`` validation examples: n/2 corrupted (balanced over the four types) plus their clean twins."""
    rows = load_split(Path(cfg.raw["paths"]["data_dir"]) / "synthetic" / cfg.pair_key, "val")
    by_id = {r.id: r for r in rows}
    rng = random.Random(f"{seed}|compare")
    per_type = max(1, n // 8)
    chosen: list[SynthExample] = []
    for t in ERROR_TYPES:
        pool = sorted((r for r in rows if r.label == t.value), key=lambda r: r.id)
        chosen += rng.sample(pool, min(per_type, len(pool)))
    twins = [by_id[c.id.rsplit("-", 1)[0] + "-clean"] for c in chosen]
    return sorted(chosen + twins, key=lambda r: r.id)


def show_prompts(cfg: Config, k: int, seed: int) -> str:
    """Render the exact prompts for ``k`` flagged examples (no LLM call) so they can be inspected."""
    examples = [e for e in sample_examples(cfg, 40, seed) if e.label != "OK"][:k]
    detector = Detector(cfg)
    dets = detector.detect([(e.src, e.hyp) for e in examples])
    out = []
    for e, d in zip(examples, dets):
        problems = build_problems(e.src, e.hyp, d)
        system, user = build_span_prompt(e.src, e.hyp, problems, cfg.pair["src_lang"], cfg.pair["tgt_lang"]) if problems else ("", "(no findings)")
        out.append(f"=== gold label {e.label}/{e.subtype}; detector types {sorted(d.types)}\n--- SYSTEM\n{system}\n--- USER\n{user}\n")
    return "\n".join(out)


def run_compare(cfg: Config, n: int, strategies: Sequence[str], workers: int, llm: LLMBackend | None = None) -> dict[str, Any]:
    """Run each strategy on the sample and score against the references. LLM strategies are skipped if no backend works."""
    examples = sample_examples(cfg, n, cfg.seed)
    detector = Detector(cfg)
    dets: list[Detection] = detector.detect([(e.src, e.hyp) for e in examples])
    src_lang, tgt_lang = cfg.pair["src_lang"], cfg.pair["tgt_lang"]
    max_ratio = float(cfg.get("correct.max_edit_ratio", 0.6))
    if llm is None:
        llm = build_llm(cfg)
    llm_ok = True
    try:
        llm.check()
    except LLMUnavailable as exc:
        llm_ok = False
        log.warning("LLM backend unavailable: %s", exc)
    report: dict[str, Any] = {"pair": cfg.pair_key, "n": len(examples), "split": "val",
                              "detector": detector.kind, "llm": llm.signature() if llm_ok else None, "strategies": {},
                              "skipped": {}}
    # detection quality on this sample (in-sample for the detector: it was fit on train+val)
    flagged = [bool(d.types) for d in dets]
    report["detector_on_sample"] = {
        "recall_on_errors": float(np.mean([f for f, e in zip(flagged, examples) if e.label != "OK"])),
        "false_alarm_rate_on_clean": float(np.mean([f for f, e in zip(flagged, examples) if e.label == "OK"])),
        "note": "optimistic: the detector was trained on train+val"}
    for strat in strategies:
        if strat not in NO_LLM and not llm_ok:
            report["skipped"][strat] = "LLM backend unavailable (set ANTHROPIC_API_KEY in .env)"
            continue
        results = correct_all(strat, examples, dets, src_lang, tgt_lang, llm if llm_ok else None, max_ratio, workers)
        report["strategies"][strat] = score(examples, results)
        report["strategies"][strat]["samples"] = [
            {"id": e.id, "label": e.label, "ref": e.ref, "before": e.hyp, "after": r.text, "status": r.status}
            for e, r in zip(examples, results) if r.text != e.hyp][:12]
    if llm_ok:
        report["llm_usage"] = llm.usage.as_dict()
    out = results_dir(cfg, "correction")
    (out / f"compare_n{len(examples)}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return report


def format_compare(report: dict[str, Any]) -> str:
    """Markdown table of the strategy comparison."""
    lines = [f"Pair {report['pair']}, {report['n']} validation examples (half corrupted, half clean twins).",
             f"Detector on this sample: recall {report['detector_on_sample']['recall_on_errors']:.2f}, "
             f"false alarms on clean {report['detector_on_sample']['false_alarm_rate_on_clean']:.2f} (in-sample, optimistic).", "",
             "| strategy | exact restored | chrF before | chrF after | gain | worsened | over-correction (clean changed) |",
             "|---|---|---|---|---|---|---|"]
    for name, m in report["strategies"].items():
        e = m["errors"]
        lines.append(f"| {name} | {e['exact_restored']:.2f} | {e['chrf_before']:.3f} | {e['chrf_after']:.3f} | "
                     f"{e['chrf_gain']:+.3f} | {e['worsened']:.2f} | {m['clean']['over_correction_rate']:.2f} |")
    for name, why in report["skipped"].items():
        lines.append(f"| {name} | skipped: {why} | | | | | |")
    return "\n".join(lines)


def run_rerank_eval(cfg: Config, n: int, n_best: int, backend: str = "nllb_ct2") -> dict[str, Any]:
    """Re-rank real MT n-best lists (validation sources) with the detector and compare chrF to the reference."""
    from mml.translate import Translator

    rows = [r for r in load_split(Path(cfg.raw["paths"]["data_dir"]) / "synthetic" / cfg.pair_key, "val") if r.label == "OK"]
    rows = sorted(rows, key=lambda r: r.id)[:n]
    translator = Translator.from_name(cfg, backend)
    cands = translator.translate([r.src for r in rows], n_best)
    cands = [list(dict.fromkeys(c)) or [""] for c in cands]            # drop duplicate hypotheses, keep MT order
    detector = Detector(cfg)
    res = rerank_batch(detector, [r.src for r in rows], cands)
    top1 = [c[0] for c in cands]
    picked = [c[r.best_idx] for c, r in zip(cands, res)]
    oracle = [max(c, key=lambda h: chrf(h, row.ref)) for c, row in zip(cands, rows)]
    mean = lambda xs: float(np.mean(xs))  # noqa: E731
    report = {
        "pair": cfg.pair_key, "n": len(rows), "n_best": n_best, "mt_backend": backend,
        "mean_distinct_candidates": mean([len(c) for c in cands]),
        "chrf_top1": mean([chrf(h, r.ref) for h, r in zip(top1, rows)]),
        "chrf_reranked": mean([chrf(h, r.ref) for h, r in zip(picked, rows)]),
        "chrf_oracle": mean([chrf(h, r.ref) for h, r in zip(oracle, rows)]),
        "p_error_top1": mean([r.scores[0] for r in res]), "p_error_reranked": mean([r.best_score for r in res]),
        # exploratory (60 sentences, one seed - NOT used to pick a default): only switch when clearly more confident
        "margin_sweep": {str(m): {"chrf": mean([chrf(c[pick(r.scores, 0.0, m)], row.ref) for c, r, row in zip(cands, res, rows)]),
                                  "changed_top1": mean([pick(r.scores, 0.0, m) != 0 for r in res])}
                         for m in (0.0, 0.1, 0.2, 0.3, 0.5)},
        "changed_top1": mean([r.best_idx != 0 for r in res]),
        "note": "sources are validation sentences the detector saw (as synthetic pairs), not MT n-best lists",
    }
    (results_dir(cfg, "correction") / f"rerank_n{len(rows)}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
