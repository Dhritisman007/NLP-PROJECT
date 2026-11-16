"""Phase 6 evaluation of detection + correction on the held-out data.

Two experiments per language pair:

1. **Synthetic test split** (labelled errors + clean twins). The detector and correction settings are frozen (chosen on
   train/val in Phases 4-5); nothing here tunes anything. Reported per error type: how often the detector flags the
   example, how often the correction restores the reference exactly, chrF before/after, and flags cleared on re-detection;
   on clean sentences: false-alarm rate and over-correction rate (a changed correct sentence is by definition worse).
2. **Real MT output** (FLORES ``devtest`` sources translated by NLLB). There are no gold error labels, so this
   reports behaviour only: flag rate, edit rate, chrF to the reference before/after, and a sample of edits for manual
   review. chrF is a weak proxy for meaning, so these numbers are descriptive, not a quality claim.

Baselines: (a) raw MT; (b) a fixed threshold on one similarity score (LaBSE - COMET-QE was unavailable) that flags
sentences; (c) LLM post-editing without spans and (d) the full pipeline need an LLM and are reported as ``not run``
when no LLM backend is configured; ``d_rules`` (detector + rule-based fixer) is what can be run without one.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from mml.config import Config
from mml.correct.compare import chrf, correct_pairs, score
from mml.data import flores_source, read_flores
from mml.detect.classifier import TYPE_NAMES
from mml.detect.features import FeatureTable
from mml.detect.predict import Detection, Detector
from mml.detect.train import feature_path
from mml.llm import LLMBackend, LLMUnavailable, build_llm
from mml.pipeline import CorrectionPipeline
from mml.synth.build import load_split

log = logging.getLogger(__name__)

SYNTH_STRATEGIES = ("raw", "constrained")           # + "full" when an LLM is available


def eval_dir(cfg: Config) -> Path:
    d = Path(cfg.raw["run"]["output_dir"]).parent / "evaluation" / cfg.pair_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _usable_llm(cfg: Config) -> LLMBackend | None:
    llm = build_llm(cfg)
    try:
        llm.check()
        return llm
    except LLMUnavailable as exc:
        log.warning("LLM strategies will be reported as 'not run': %s", exc)
        return None


def _rate(flags: Sequence[bool]) -> float:
    return float(np.mean(flags)) if len(flags) else 0.0


# --------------------------------------------------------------------------- 1. synthetic test split
def evaluate_synthetic(cfg: Config, llm: LLMBackend | None = None, strategies: Sequence[str] | None = None,
                       split: str = "test", limit: int | None = None) -> dict[str, Any]:
    """Frozen detector + correction strategies on a labelled split (the held-out ``test`` split by default).

    ``split``/``limit`` exist so the code path can be tested on validation data; the reported results use ``test``.
    """
    examples = load_split(Path(cfg.raw["paths"]["data_dir"]) / "synthetic" / cfg.pair_key, split,
                          for_final_eval=(split == "test"))[:limit]
    pairs = [(e.src, e.hyp) for e in examples]
    detector = Detector(cfg)
    dets = detector.detect(pairs)
    src_lang, tgt_lang = cfg.pair["src_lang"], cfg.pair["tgt_lang"]
    errors = [e.label != "OK" for e in examples]
    out: dict[str, Any] = {"pair": cfg.pair_key, "split": split, "n": len(examples), "detector": detector.kind,
                           "tuned_on_test": False, "skipped": {}, "strategies": {}}

    # detector as a flagger (any error) and per-type flag rates
    flagged = [bool(d.types) for d in dets]
    out["detector_flagging"] = {
        "recall_on_errors": _rate([f for f, e in zip(flagged, errors) if e]),
        "false_alarm_on_clean": _rate([f for f, e in zip(flagged, errors) if not e]),
        "per_type_flag_rate": {t: _rate([t in d.types for d, e in zip(dets, examples) if e.label == t]) for t in TYPE_NAMES}}

    # baseline (b): flag by a fixed threshold on LaBSE cosine (threshold tuned on train+val in Phase 4)
    base = json.loads((Path(cfg.raw["run"]["output_dir"]).parent / "detector" / cfg.pair_key / "results.json").read_text())
    b = base.get("baseline_labse_threshold")
    if b:
        table = FeatureTable.load(feature_path(cfg, split))
        assert table.ids[:len(examples)] == [e.id for e in examples], "feature table rows do not match the examples"
        flagged_b = table.X[:len(examples), table.columns.index(b["feature"])] <= b["threshold"]
        tp = int((flagged_b & np.array(errors)).sum())
        fp = int((flagged_b & ~np.array(errors)).sum())
        fn = int((~flagged_b & np.array(errors)).sum())
        out["baseline_b_labse_threshold"] = {
            "threshold": b["threshold"], "recall_on_errors": tp / max(tp + fn, 1),
            "false_alarm_on_clean": fp / max(int((~np.array(errors)).sum()), 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1),
            "note": "flagging only (it has no way to correct); COMET-QE not available, LaBSE cosine used instead"}

    # correction strategies
    if llm is None:
        llm = _usable_llm(cfg)
    names = list(strategies or SYNTH_STRATEGIES)
    if llm is not None and "full" not in names:
        names += ["baseline_llm", "span_llm", "full"]
    for name in ("baseline_llm", "span_llm", "full"):
        if name not in names:
            out["skipped"][name] = "no LLM backend configured (ANTHROPIC_API_KEY)"
    max_ratio = float(cfg.get("correct.max_edit_ratio", 0.6))
    for strat in names:
        results = correct_pairs(strat, pairs, dets, src_lang, tgt_lang, llm, max_ratio)
        changed = [i for i, r in enumerate(results) if r.text != pairs[i][1]]
        after = list(dets)
        for i, d in zip(changed, detector.detect([(pairs[i][0], results[i].text) for i in changed]) if changed else []):
            after[i] = d
        m = score(examples, results)
        for t in TYPE_NAMES:
            idx = [i for i, e in enumerate(examples) if e.label == t]
            fl = [i for i in idx if dets[i].types]
            m["per_type"][t].update({
                "flagged_any": _rate([bool(dets[i].types) for i in idx]),
                "flagged_correct_type": _rate([t in dets[i].types for i in idx]),
                "restored_given_flagged": _rate([results[i].text == examples[i].ref for i in fl]),
                "flags_cleared": _rate([not after[i].types for i in fl])})
        clean = [i for i, e in enumerate(examples) if e.label == "OK"]
        m["clean"].update({"false_alarm_rate": _rate([bool(dets[i].types) for i in clean]),
                           "flagged_after": _rate([bool(after[i].types) for i in clean])})
        fl_all = [i for i, e in enumerate(examples) if e.label != "OK" and dets[i].types]
        m["errors"].update({"flagged_any": _rate([bool(dets[i].types) for i, e in enumerate(examples) if e.label != "OK"]),
                            "restored_given_flagged": _rate([results[i].text == examples[i].ref for i in fl_all]),
                            "flags_cleared": _rate([not after[i].types for i in fl_all])})
        out["strategies"][strat] = m
    if llm is not None:
        out["llm_usage"] = llm.usage.as_dict()
    return out


# --------------------------------------------------------------------------- 2. real MT output
def evaluate_real_mt(cfg: Config, n: int = 300, strategy: str = "constrained", llm: LLMBackend | None = None,
                     backend: str = "nllb_ct2") -> dict[str, Any]:
    """Translate FLORES devtest sources, run the pipeline, and report behaviour + chrF to the reference."""
    from mml.translate import Translator

    data_dir = Path(cfg.raw["paths"]["data_dir"])
    source = flores_source(cfg)
    srcs = read_flores(data_dir, cfg.pair["src_lang"], "devtest", source)
    refs = read_flores(data_dir, cfg.pair["tgt_lang"], "devtest", source)
    idx = sorted(random.Random(f"{cfg.seed}|realmt").sample(range(len(srcs)), min(n, len(srcs))))
    srcs, refs = [srcs[i] for i in idx], [refs[i] for i in idx]
    mt = Translator.from_name(cfg, backend).translate_best(srcs)
    pipe = CorrectionPipeline(cfg, Detector(cfg), strategy, llm)
    res = pipe.run_pairs(list(zip(srcs, mt)))
    before = [chrf(r.mt, ref) for r, ref in zip(res, refs)]
    after = [chrf(r.corrected, ref) for r, ref in zip(res, refs)]
    changed = [i for i, r in enumerate(res) if r.changed]
    flagged = [i for i, r in enumerate(res) if r.before.types]
    type_counts = {t: sum(t in r.before.types for r in res) for t in TYPE_NAMES}
    out = {
        "pair": cfg.pair_key, "split": "devtest", "n": len(res), "mt_backend": backend, "strategy": strategy,
        "note": "FLORES devtest sources translated by NLLB-600M int8; no gold error labels, so chrF to the reference is a weak proxy",
        "flag_rate": len(flagged) / len(res), "flag_rate_by_type": {t: c / len(res) for t, c in type_counts.items()},
        "edit_rate": len(changed) / len(res),
        "chrf_mt": float(np.mean(before)), "chrf_corrected": float(np.mean(after)),
        "chrf_gain_all": float(np.mean(after) - np.mean(before)),
        "edited_improved": _rate([after[i] > before[i] + 1e-9 for i in changed]) if changed else None,
        "edited_worsened": _rate([after[i] < before[i] - 1e-9 for i in changed]) if changed else None,
        "edited_unchanged_chrf": _rate([abs(after[i] - before[i]) <= 1e-9 for i in changed]) if changed else None,
        "flags_cleared_among_edited": _rate([res[i].flags_cleared for i in changed]) if changed else None,
        "mean_chrf_flagged": float(np.mean([before[i] for i in flagged])) if flagged else None,
        "mean_chrf_unflagged": float(np.mean([before[i] for i in range(len(res)) if i not in set(flagged)])) if len(flagged) < len(res) else None,
        "edit_samples": [{"src": res[i].src, "ref": refs[i], "mt": res[i].mt, "corrected": res[i].corrected,
                          "chrf_before": before[i], "chrf_after": after[i],
                          "fixes": res[i].correction.problems or [], "types": sorted(res[i].before.types)} for i in changed[:30]],
    }
    return out


def run_pair(cfg: Config, real_n: int = 300, llm: LLMBackend | None = None) -> dict[str, Any]:
    """Both experiments for one pair; writes ``results/evaluation/<pair>/results.json``."""
    llm = llm or _usable_llm(cfg)
    res = {"synthetic_test": evaluate_synthetic(cfg, llm),
           "real_mt": evaluate_real_mt(cfg, real_n, "full" if llm else "constrained", llm)}
    (eval_dir(cfg) / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    return res
