"""Phase 4 orchestration: feature tables, model selection (grouped CV), ablation, baselines, spans, final test run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

from mml.config import Config
from mml.detect.base import open_cache
from mml.detect.classifier import (
    MeaningLossClassifier, TYPE_NAMES, cross_val_proba, macro_f1_5way, predict_labels, select_model, tune_thresholds,
)
from mml.detect.features import FeatureTable, build_table
from mml.detect.metrics import CLASSES, any_error_f1, bootstrap_ci, format_markdown, macro_f1, sentence_metrics
from mml.detect.run import DEFAULT_SIGNALS, run_signals
from mml.detect.spans import cached_silver, span_f1
from mml.synth.build import load_split
from mml.synth.types import SynthExample

log = logging.getLogger(__name__)


def pair_dir(cfg: Config, *parts: str) -> Path:
    return Path(cfg.raw["paths"]["data_dir"]).joinpath(*parts)


def feature_path(cfg: Config, split: str) -> Path:
    return pair_dir(cfg, "processed", "features", cfg.pair_key, split)


def base_id(example_id: str) -> str:
    """Source-sentence id shared by an example and its clean/corrupted twin."""
    return example_id.rsplit("-", 1)[0]


def build_features(cfg: Config, split: str, signals: Sequence[str] = DEFAULT_SIGNALS) -> tuple[FeatureTable, dict[str, str]]:
    """Compute (cached) signals for a split and save its feature table. The test split is only *featurised* here."""
    rows = load_split(pair_dir(cfg, "synthetic", cfg.pair_key), split, for_final_eval=(split == "test"))
    pairs = [(r.src, r.hyp) for r in rows]
    results, skipped = run_signals(cfg, pairs, signals)
    table = build_table([r.id for r in rows], [r.label for r in rows], pairs, results)
    table.save(feature_path(cfg, split))
    return table, skipped


def _flat(m: dict[str, Any]) -> dict[str, float]:
    return {"macro_f1": m["macro_f1"], "any_f1": m["any_error"]["f1"],
            **{f"f1_{c}": m["per_class"][c]["f1"] for c in TYPE_NAMES}}


def single_feature_baseline(table: FeatureTable, feature: str, labels: Sequence[str]) -> dict[str, Any]:
    """Detect "any error" by thresholding ONE feature (lowest values flagged); threshold picked for best F1 on the data
    given (a 1-parameter fit). Stand-in for the "QE score with a fixed threshold" baseline until CometKiwi is available."""
    x = table.X[:, table.columns.index(feature)]
    y = np.asarray(labels) != "OK"
    best = (-1.0, 0.0)
    for thr in np.unique(np.quantile(x, np.linspace(0.02, 0.98, 97))):
        pred = x <= thr
        tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best[0]:
            best = (f1, float(thr))
    return {"feature": feature, "threshold": best[1], "any_f1": best[0]}


def ablation(kind: str, params: dict[str, Any], table: FeatureTable, groups: Sequence[str], seed: int) -> list[dict[str, Any]]:
    """Grouped-CV results with the full set, each group dropped, and each group alone."""
    runs: list[tuple[str, dict[str, Any]]] = [("all", {})]
    runs += [(f"drop:{g}", {"drop": [g]}) for g in table.groups()]
    runs += [(f"only:{g}", {"keep": [g]}) for g in table.groups() if g not in ("surface",)]
    rows = []
    for name, sel in runs:
        X, cols = table.select(**sel)
        if X.shape[1] == 0:
            continue
        P = cross_val_proba(kind, params, X, table.labels, groups, seed=seed)
        thr = tune_thresholds(P, table.labels)
        m = sentence_metrics(table.labels, predict_labels(P, thr))
        rows.append({"run": name, "n_features": len(cols), **_flat(m)})
        log.info("ablation %-14s macro-F1 %.3f any-F1 %.3f", name, m["macro_f1"], m["any_error"]["f1"])
    return rows


def span_report(examples: Sequence[SynthExample], table: FeatureTable, pred_sets: Sequence[set[str]],
                align: Any, cache: Any) -> dict[str, Any]:
    """Span-level F1 with and without gating by the classifier's predicted types."""
    silver = cached_silver(examples, align, cache)
    return {"gated": span_f1(examples, table.spans, silver, pred_sets),
            "ungated": span_f1(examples, table.spans, silver, None),
            "silver_examples": len(silver), "silver_unmappable": sum(1 for v in silver.values() if not v)}


def _span_markdown(rep: dict[str, Any], mode: str) -> str:
    lines = [f"| type | side | precision | recall | F1 | gold chars |", "|---|---|---|---|---|---|"]
    for t in TYPE_NAMES:
        for side in ("tgt", "src"):
            r = rep[mode][t][side]
            if r["gold_chars"]:
                lines.append(f"| {t} | {side}{' (silver)' if side == 'src' else ''} | {r['precision']:.3f} | {r['recall']:.3f} "
                             f"| {r['f1']:.3f} | {r['gold_chars']} |")
    return "\n".join(lines)


def train_detector(cfg: Config, kinds: Sequence[str] = ("logreg", "gbm"), evaluate_test: bool = False,
                   n_splits: int = 5) -> dict[str, Any]:
    """Select, ablate and (optionally, once) test the detector for the active pair. Returns the full result dict."""
    seed = cfg.seed
    tr, va = FeatureTable.load(feature_path(cfg, "train")), FeatureTable.load(feature_path(cfg, "val"))
    full = FeatureTable.concat(tr, va)
    groups = [base_id(i) for i in full.ids]
    X, labels = full.X, full.labels
    out_dir = Path(cfg.raw["run"]["output_dir"]).parent / "detector" / cfg.pair_key   # <root>/results/detector/<pair>
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(cfg.raw["paths"]["data_dir"]).parent / "models" / cfg.pair_key
    model_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {"pair": cfg.pair_key, "seed": seed, "n_train_val": len(labels), "columns": full.columns,
                              "groups": full.groups(), "models": {}}
    fitted: dict[str, tuple[MeaningLossClassifier, np.ndarray]] = {}
    for kind in kinds:
        model, P, trace = select_model(kind, X, labels, groups, full.columns, n_splits, seed)
        m = sentence_metrics(labels, predict_labels(P, model.thresholds))
        lo, hi = bootstrap_ci(macro_f1, labels, predict_labels(P, model.thresholds), groups, seed=seed)
        result["models"][kind] = {"params": model.params, "thresholds": model.thresholds, "search": trace,
                                  "cv_metrics": m, "cv_macro_f1_ci95": [lo, hi]}
        joblib.dump({"model": model, "columns": full.columns}, model_dir / f"detector_{kind}.joblib")
        fitted[kind] = (model, P)
    best_kind = max(fitted, key=lambda k: result["models"][k]["cv_metrics"]["macro_f1"])
    best, P_oof = fitted[best_kind]
    result["best_kind"] = best_kind

    result["baseline_labse_threshold"] = (
        single_feature_baseline(full, "labse_cos", labels) if "labse_cos" in full.columns else None)
    result["ablation"] = ablation(best_kind, best.params, full, groups, seed)

    # span-level evaluation on the out-of-fold predictions of train+val (still no test data)
    from mml.detect.alignment import WordAligner

    examples = load_split(pair_dir(cfg, "synthetic", cfg.pair_key), "train") + load_split(pair_dir(cfg, "synthetic", cfg.pair_key), "val")
    pred_sets = [{t for j, t in enumerate(TYPE_NAMES) if p[j + 1] >= best.thresholds[t]} for p in P_oof]
    align = WordAligner(cfg)._get_align()
    result["spans_cv"] = span_report(examples, full, pred_sets, align, open_cache(cfg, "silver"))

    if evaluate_test:
        te = FeatureTable.load(feature_path(cfg, "test"))
        if te.columns != full.columns:
            raise ValueError("test feature columns differ from training columns")
        test_examples = load_split(pair_dir(cfg, "synthetic", cfg.pair_key), "test", for_final_eval=True)
        pred = best.predict_labels(te.X)
        m = sentence_metrics(te.labels, pred)
        g_te = [base_id(i) for i in te.ids]
        result["test"] = {"model": best_kind, "metrics": m,
                          "macro_f1_ci95": bootstrap_ci(macro_f1, te.labels, pred, g_te, seed=seed),
                          "any_f1_ci95": bootstrap_ci(any_error_f1, te.labels, pred, g_te, seed=seed),
                          "spans": span_report(test_examples, te, best.predict_sets(te.X), align, open_cache(cfg, "silver"))}
        if result["baseline_labse_threshold"]:
            b = result["baseline_labse_threshold"]
            x = te.X[:, te.columns.index(b["feature"])]
            yt = np.asarray(te.labels) != "OK"
            result["test"]["baseline_labse_any_f1"] = float(
                _binary_f1(yt, x <= b["threshold"]))
    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out_dir / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


def _binary_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def render_report(r: dict[str, Any]) -> str:
    """Markdown report for the README / project report."""
    L = [f"# Detector report - {r['pair']}", "",
         f"Features: {len(r['columns'])} in groups {r['groups']}. Selection by 5-fold sentence-grouped CV over "
         f"train+val ({r['n_train_val']} examples); the test split was not used for any choice.", ""]
    for kind, info in r["models"].items():
        m = info["cv_metrics"]
        lo, hi = info["cv_macro_f1_ci95"]
        L += [f"## {kind} (cross-validated, params {info['params']}; macro-F1 95% CI [{lo:.3f}, {hi:.3f}])", "",
              format_markdown(m), ""]
    L += [f"Best by CV macro-F1: **{r['best_kind']}**", ""]
    if r.get("baseline_labse_threshold"):
        b = r["baseline_labse_threshold"]
        L += [f"Baseline - threshold on `{b['feature']}` only (any-error F1, tuned on train+val): {b['any_f1']:.3f}", ""]
    L += ["## Ablation (grouped CV, best model)", "", "| run | #features | macro-F1 | any-F1 | " +
          " | ".join(f"F1 {t}" for t in TYPE_NAMES) + " |", "|---|---|---|---|" + "---|" * len(TYPE_NAMES)]
    for a in r["ablation"]:
        L.append(f"| {a['run']} | {a['n_features']} | {a['macro_f1']:.3f} | {a['any_f1']:.3f} | " +
                 " | ".join(f"{a['f1_' + t]:.3f}" for t in TYPE_NAMES) + " |")
    L += ["", "## Span-level F1 (cross-validated predictions, train+val)", "", "Gated by predicted type:", "",
          _span_markdown(r["spans_cv"], "gated"), "", "Ungated (raw signal evidence):", "", _span_markdown(r["spans_cv"], "ungated"),
          f"\n(src side = silver gold from alignment; {r['spans_cv']['silver_unmappable']} of "
          f"{r['spans_cv']['silver_examples']} deletion examples could not be mapped)"]
    if "test" in r:
        t = r["test"]
        L += ["", f"## FINAL held-out test ({t['model']}, evaluated once)", "", format_markdown(t["metrics"]),
              f"\nmacro-F1 95% CI {t['macro_f1_ci95'][0]:.3f}-{t['macro_f1_ci95'][1]:.3f}; any-error F1 95% CI "
              f"{t['any_f1_ci95'][0]:.3f}-{t['any_f1_ci95'][1]:.3f}"]
        if "baseline_labse_any_f1" in t:
            L.append(f"Baseline (LaBSE threshold only) any-error F1 on test: {t['baseline_labse_any_f1']:.3f}")
        L += ["", "Span-level F1 on test (gated):", "", _span_markdown(t["spans"], "gated")]
    return "\n".join(L)
