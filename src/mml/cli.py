"""Command-line interface (typer):  ``mml show-config | translate | smoke-test | download-data``."""

from __future__ import annotations

import csv
import logging
import os
from pathlib import Path
from typing import Annotated, Optional

import typer

from mml.config import Config, load_config
from mml.data import download_all
from mml.translate import Translator, load_sentences, target_script_ratio, timed
from mml.utils import RunLogger, get_hf_token, set_seed, setup_logging

app = typer.Typer(add_completion=False, help="Detection and correction of meaning loss in MT.")
log = logging.getLogger("mml.cli")

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="YAML config (default: configs/default.yaml)")]
PairOpt = Annotated[Optional[str], typer.Option(help="Language pair key from the config, e.g. en-hi, en-de")]
ProfileOpt = Annotated[Optional[str], typer.Option(help="Hardware profile: mac_dev | colab_t4")]
SetOpt = Annotated[Optional[list[str]], typer.Option("--set", help="Override, e.g. --set mt.num_beams=4 (repeatable)")]
DEFAULT_SMOKE = Path(__file__).resolve().parents[2] / "data" / "samples" / "smoke_en_20.txt"


def _load(config: Path | None, pair: str | None, profile: str | None, overrides: list[str] | None) -> Config:
    setup_logging()
    from mml.utils import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")      # secrets (HF_TOKEN, ANTHROPIC_API_KEY); never logged
    cfg = load_config(config, pair, profile, overrides)
    set_seed(cfg.seed)
    # transformers otherwise downloads a second (safetensors) copy of every .bin model in the background
    os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
    if cfg.raw["paths"].get("hf_cache"):
        os.environ.setdefault("HF_HOME", str(cfg.raw["paths"]["hf_cache"]))
    return cfg


@app.command("show-config")
def show_config(config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None) -> None:
    """Print the fully resolved configuration."""
    typer.echo(_load(config, pair, profile, set_).to_yaml())


@app.command()
def translate(
    text: Annotated[list[str], typer.Argument(help="Source sentence(s)")],
    n_best: Annotated[int, typer.Option(help="Candidates per sentence")] = 1,
    strict: Annotated[bool, typer.Option(help="Fail instead of falling back to NLLB")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Translate sentences given on the command line."""
    cfg = _load(config, pair, profile, set_)
    tr = Translator.from_config(cfg, allow_fallback=not strict)
    for src, cands in zip(text, tr.translate(text, n_best)):
        typer.echo(f"SRC: {src}")
        for rank, cand in enumerate(cands, 1):
            typer.echo(f"  [{rank}] {cand}")


@app.command("smoke-test")
def smoke_test(
    input_file: Annotated[Path, typer.Option("--input", help="Source sentences, one per line")] = DEFAULT_SMOKE,
    strict: Annotated[bool, typer.Option(help="Fail instead of falling back to NLLB")] = False,
    no_cache: Annotated[bool, typer.Option(help="Ignore/skip the MT disk cache")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Translate the 20 smoke-test sentences end to end and run basic sanity checks."""
    cfg = _load(config, pair, profile, set_)
    sources = load_sentences(input_file)
    run = RunLogger(cfg, f"smoke_{cfg.pair_key}")
    tr = Translator.from_config(cfg, allow_fallback=not strict, use_cache=not no_cache)
    if tr.fallback_used:
        typer.secho(f"WARNING: requested backend '{tr.requested_backend}' unavailable; used '{tr.backend.name}' instead.",
                    fg=typer.colors.YELLOW, err=True)
    outputs, secs = timed(tr.translate_best, sources)

    tgt = cfg.pair["tgt_lang"]
    rows, failures = [], []
    for i, (src, hyp) in enumerate(zip(sources, outputs), 1):
        ratio = target_script_ratio(hyp, tgt)
        # Latin-script targets share a script with the source, so also require hyp != src.
        script_ok = ratio >= 0.5 and (hyp.strip() != src.strip() or not tgt.endswith("Latn"))
        ok = bool(hyp.strip()) and script_ok
        if not ok:
            failures.append(i)
        rows.append({"id": i, "src": src, "hyp": hyp, "script_ratio": round(ratio, 3),
                     "len_ratio": round(len(hyp) / max(len(src), 1), 2), "ok": ok})
        typer.echo(f"{i:>2}. {src}\n    {hyp}")

    with open(run.dir / "translations.tsv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    metrics = {
        "pair": cfg.pair_key, "backend_used": tr.backend.name, "backend_requested": tr.requested_backend,
        "fallback_used": tr.fallback_used, "model": tr.backend.model_id, "device": tr.backend.device,
        "n": len(sources), "n_ok": len(rows) - len(failures), "failed_ids": failures,
        "mean_script_ratio": round(sum(r["script_ratio"] for r in rows) / len(rows), 3),
        "seconds": round(secs, 1), "hf_token_present": get_hf_token() is not None,
    }
    run.log_metrics(metrics)
    typer.echo(f"\nRun dir: {run.dir}")
    if failures:
        typer.secho(f"SMOKE TEST FAILED for sentence ids {failures}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)
    typer.secho(f"SMOKE TEST PASSED ({metrics['n_ok']}/{metrics['n']}, backend={tr.backend.name}, {secs:.1f}s)",
                fg=typer.colors.GREEN)


@app.command()
def synth(
    out: Annotated[Optional[Path], typer.Option(help="Output dir (default: data/synthetic/<pair>)")] = None,
    limit: Annotated[Optional[int], typer.Option(help="Use only the first N FLORES sentences (debug)")] = None,
    balance: Annotated[bool, typer.Option(help="Downsample all error types to the rarest one")] = False,
    ner: Annotated[bool, typer.Option(help="Also use the configured NER model to find entities")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Generate the synthetic error dataset (train/val/test JSONL + manifest) from FLORES references."""
    from mml.synth.build import build_examples, load_flores_items, write_dataset
    from mml.synth.entities import CompositeFinder, GazetteerFinder, NERFinder
    from mml.synth.lang import get_resources

    cfg = _load(config, pair, profile, set_)
    lang = get_resources(cfg.pair["tgt_lang"])
    finder = GazetteerFinder(lang.gazetteer)
    finder_desc = "gazetteer"
    if ner:
        model_id = cfg.get(f"models.ner.{lang.code}")
        if not model_id:
            raise typer.BadParameter(f"No NER model configured for '{lang.code}' (models.ner.{lang.code})")
        finder, finder_desc = CompositeFinder(finder, NERFinder(model_id, token=get_hf_token())), f"gazetteer+{model_id}"
    items = load_flores_items(cfg, limit)
    examples, stats = build_examples(items, cfg.pair_key, cfg.pair["tgt_lang"], cfg.seed, finder, balance)
    out_dir = out or Path(cfg.raw["paths"]["data_dir"]) / "synthetic" / cfg.pair_key
    run = RunLogger(cfg, f"synth_{cfg.pair_key}")
    manifest = write_dataset(examples, out_dir, {
        "pair": cfg.pair_key, "seed": cfg.seed, "source": cfg.get("data.flores_source"), "entity_finder": finder_desc,
        "balanced": balance, "n_source_sentences": len(items), **stats})
    run.log_metrics({"synthetic_summary": manifest["summary"], **stats})
    for split, info in manifest["summary"].items():
        typer.echo(f"{split:>5}: n={info['n']:>5}  {info['labels']}")
    typer.echo(f"skipped (no applicable error): {stats['skipped_no_applicable_error']}")
    typer.echo(f"Wrote {out_dir}")


@app.command()
def signals(
    split: Annotated[str, typer.Option(help="train | val (the test split is never used here)")] = "val",
    limit: Annotated[Optional[int], typer.Option(help="Use only the first N examples")] = 40,
    names: Annotated[str, typer.Option("--signals", help="Comma-separated: rules,labse,align,roundtrip,cometkiwi,xcomet")] = "rules,labse,align,roundtrip,cometkiwi,xcomet",
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Compute detection signals on synthetic examples (cached on disk) and print mean features per gold label."""
    from collections import defaultdict

    from mml.detect.run import run_signals
    from mml.synth.build import load_split

    if split == "test":
        raise typer.BadParameter("The test split is held out; use train or val.")
    cfg = _load(config, pair, profile, set_)
    rows = load_split(Path(cfg.raw["paths"]["data_dir"]) / "synthetic" / cfg.pair_key, split)[: limit or None]
    run = RunLogger(cfg, f"signals_{cfg.pair_key}_{split}")
    results, skipped = run_signals(cfg, [(r.src, r.hyp) for r in rows], [n.strip() for n in names.split(",")])
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, res in results.items():
        for row, r in zip(rows, res):
            for k, v in r.features.items():
                by_label[row.label][k].append(v)
    summary = {lab: {k: round(sum(v) / len(v), 3) for k, v in feats.items()} for lab, feats in by_label.items()}
    keys = sorted({k for feats in summary.values() for k in feats})
    labels = sorted(summary)
    typer.echo(f"{'feature':28s} " + " ".join(f"{lab[:11]:>11s}" for lab in labels))
    for k in keys:
        typer.echo(f"{k:28s} " + " ".join(f"{summary[lab].get(k, float('nan')):11.3f}" for lab in labels))
    for name, why in skipped.items():
        typer.secho(f"SKIPPED {name}: {why}", fg=typer.colors.YELLOW, err=True)
    run.log_metrics({"n": len(rows), "signals_run": sorted(results), "signals_skipped": skipped})
    run.write_json("feature_means_by_label.json", summary)


@app.command()
def features(
    splits: Annotated[str, typer.Option(help="Comma-separated: train,val,test (test is only featurised, never trained on)")] = "train,val",
    names: Annotated[str, typer.Option("--signals", help="Signals to use")] = "rules,labse,align,roundtrip,cometkiwi,xcomet",
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Compute all detection signals for whole splits (cached) and save feature tables to data/processed/features/."""
    from mml.detect.train import build_features

    cfg = _load(config, pair, profile, set_)
    run = RunLogger(cfg, f"features_{cfg.pair_key}")
    for split in [s.strip() for s in splits.split(",")]:
        table, skipped = build_features(cfg, split, [n.strip() for n in names.split(",")])
        typer.echo(f"{split}: {table.X.shape[0]} examples x {table.X.shape[1]} features, groups {table.groups()}")
        for name, why in skipped.items():
            typer.secho(f"  SKIPPED {name}: {why}", fg=typer.colors.YELLOW, err=True)
        run.log_metrics({"split": split, "n": int(table.X.shape[0]), "n_features": int(table.X.shape[1]),
                         "groups": table.groups(), "signals_skipped": skipped})


@app.command("train-detector")
def train_detector_cmd(
    kinds: Annotated[str, typer.Option(help="logreg,gbm")] = "logreg,gbm",
    evaluate_test: Annotated[bool, typer.Option("--evaluate-test", help="Also score the held-out test split. Do this ONCE, after all choices are frozen.")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Train + select the detector with grouped CV on train+val, run the ablation, write results/detector/<pair>/."""
    from mml.detect.train import train_detector

    cfg = _load(config, pair, profile, set_)
    run = RunLogger(cfg, f"detector_{cfg.pair_key}")
    res = train_detector(cfg, [k.strip() for k in kinds.split(",")], evaluate_test)
    for kind, info in res["models"].items():
        m = info["cv_metrics"]
        typer.echo(f"[CV] {kind:7s} macro-F1 {m['macro_f1']:.3f}  any-error F1 {m['any_error']['f1']:.3f}  params {info['params']}")
    if "test" in res:
        t = res["test"]["metrics"]
        typer.echo(f"[TEST] {res['best_kind']} macro-F1 {t['macro_f1']:.3f}  any-error F1 {t['any_error']['f1']:.3f}")
    run.log_metrics({"best_kind": res["best_kind"], "evaluated_test": evaluate_test})
    typer.echo(f"Report: results/detector/{cfg.pair_key}/report.md")


@app.command("correct-compare")
def correct_compare(
    n: Annotated[int, typer.Option(help="Validation examples (half corrupted, half clean twins)")] = 80,
    strategies: Annotated[str, typer.Option(help="raw,constrained,baseline_llm,span_llm,full")] = "raw,constrained,baseline_llm,span_llm,full",
    workers: Annotated[int, typer.Option(help="Concurrent LLM calls")] = 4,
    show_prompts: Annotated[int, typer.Option(help="Only print the exact prompts for K flagged examples; no LLM call")] = 0,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Compare correction strategies (raw / constrained / LLM no-spans / span-guided / full) on validation examples."""
    from mml.correct.experiments import format_compare, run_compare
    from mml.correct.experiments import show_prompts as render_prompts

    cfg = _load(config, pair, profile, set_)
    if show_prompts:
        typer.echo(render_prompts(cfg, show_prompts, cfg.seed))
        return
    report = run_compare(cfg, n, [s.strip() for s in strategies.split(",")], workers)
    typer.echo(format_compare(report))
    if "llm_usage" in report:
        typer.echo(f"\nLLM usage: {report['llm_usage']}")
    for name, why in report["skipped"].items():
        typer.secho(f"SKIPPED {name}: {why}", fg=typer.colors.YELLOW, err=True)


@app.command("rerank-eval")
def rerank_eval(
    n: Annotated[int, typer.Option(help="Validation sources")] = 60,
    n_best: Annotated[int, typer.Option(help="Candidates per source")] = 4,
    backend: Annotated[str, typer.Option(help="Forward MT backend for the n-best list")] = "nllb_ct2",
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Re-rank real MT n-best lists with the detector and compare chrF to the reference."""
    from mml.correct.experiments import run_rerank_eval

    cfg = _load(config, pair, profile, set_)
    rep = run_rerank_eval(cfg, n, n_best, backend)
    for k, v in rep.items():
        typer.echo(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")


@app.command()
def evaluate(
    real_n: Annotated[int, typer.Option(help="FLORES devtest sentences for the real-MT experiment")] = 300,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Phase 6: evaluate the frozen detector + correction on the held-out test split and on real MT output.

    Uses the TEST split. Nothing is tuned here; run it after Phases 4-5 are frozen. LLM strategies are reported
    as 'not run' unless an LLM backend is configured (ANTHROPIC_API_KEY in .env).
    """
    from mml.evaluate import run_pair

    cfg = _load(config, pair, profile, set_)
    run = RunLogger(cfg, f"evaluate_{cfg.pair_key}")
    res = run_pair(cfg, real_n)
    syn, real = res["synthetic_test"], res["real_mt"]
    c = syn["strategies"]["constrained"]
    typer.echo(f"[{cfg.pair_key}] test n={syn['n']}: detector flags {syn['detector_flagging']['recall_on_errors']:.2f} of errors, "
               f"{syn['detector_flagging']['false_alarm_on_clean']:.3f} of clean | rules-only fix restores {c['errors']['exact_restored']:.2f} of all errors, "
               f"over-correction {c['clean']['over_correction_rate']:.3f}")
    typer.echo(f"[{cfg.pair_key}] real MT n={real['n']}: flagged {real['flag_rate']:.2f}, edited {real['edit_rate']:.3f}, "
               f"chrF {real['chrf_mt']:.3f} -> {real['chrf_corrected']:.3f}")
    for name, why in syn["skipped"].items():
        typer.secho(f"  not run: {name} ({why})", fg=typer.colors.YELLOW, err=True)
    run.log_metrics({"evaluated": cfg.pair_key, "skipped": list(syn["skipped"])})


@app.command()
def report(
    pairs: Annotated[str, typer.Option(help="Comma-separated pairs with finished evaluations")] = "en-hi,en-de",
    config: ConfigOpt = None, profile: ProfileOpt = None,
) -> None:
    """Build the results tables (CSV + markdown) and plots in results/evaluation/ from saved evaluations."""
    from mml.report import build_report

    cfg = _load(config, None, profile, None)
    rep = build_report(Path(cfg.raw["paths"]["data_dir"]).parent, [p.strip() for p in pairs.split(",")])
    for k in ("tables", "plots"):
        for f in rep[k]:
            typer.echo(f)


@app.command()
def demo(
    port: Annotated[int, typer.Option(help="Local port")] = 7860,
    share: Annotated[bool, typer.Option(help="Create a public gradio.live link (off by default; the app runs on localhost only)")] = False,
    strategy: Annotated[str, typer.Option(help="auto | constrained | full | span_llm | none")] = "auto",
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Launch the Gradio demo (models load on the first request and then stay in memory, ~4-5 GB per language pair)."""
    from mml.demo import launch

    cfg = _load(config, pair, profile, set_)
    launch(cfg, list(cfg.raw["pairs"]), cfg.pair_key, port, share, strategy)


@app.command("human-eval-kit")
def human_eval_kit(
    n: Annotated[int, typer.Option(help="Items per annotator (incl. attention checks)")] = 100,
    checks: Annotated[int, typer.Option(help="Known-answer attention checks among the items")] = 10,
    pool_n: Annotated[int, typer.Option(help="Real MT sentences to translate and screen before sampling")] = 600,
    annotators: Annotated[str, typer.Option(help="Comma-separated annotator names")] = "A,B",
    reuse_pool: Annotated[bool, typer.Option(help="Reuse a saved pool instead of translating again")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Build the human-evaluation kit: stratified sample of real MT output, blind CSV/HTML forms, hidden key."""
    from mml.human_eval.kit import build_pool, load_pool, pool_path, sample_kit, write_kit
    from mml.synth.build import load_split

    cfg = _load(config, pair, profile, set_)
    pool = load_pool(cfg) if reuse_pool and pool_path(cfg).exists() else build_pool(cfg, pool_n)
    val = load_split(Path(cfg.raw["paths"]["data_dir"]) / "synthetic" / cfg.pair_key, "val")
    items, meta = sample_kit(pool, val, n, checks, cfg.seed)
    out = write_kit(cfg, items, meta, [a.strip() for a in annotators.split(",")])
    typer.echo(f"pool: {meta['pool_n']} sentences, {meta['pool_flagged']} flagged by the detector "
               f"({meta['pool_flag_rate']:.1%}); sampled {meta['sampled_flagged']} flagged + {meta['sampled_unflagged']} unflagged "
               f"+ {meta['n_checks']} attention checks = {meta['n']} items")
    typer.echo(f"kit written to {out}")


@app.command("human-eval-iaa")
def human_eval_iaa(
    a: Annotated[Path, typer.Option("--a", help="Annotator A's returned CSV")],
    b: Annotated[Path, typer.Option("--b", help="Annotator B's returned CSV")],
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Cohen's kappa, attention-check accuracy and detector-vs-human comparison from the two returned annotation files."""
    from mml.human_eval.iaa import run_iaa

    cfg = _load(config, pair, profile, set_)
    out = Path(cfg.raw["run"]["output_dir"]).parent / "human_eval" / cfg.pair_key
    rep = run_iaa(a, b, out / "key_DO_NOT_SHARE_WITH_ANNOTATORS.csv", out / "kit_meta.json", out)
    for issue in rep["issues"]:
        typer.secho(f"WARNING: {issue}", fg=typer.colors.YELLOW, err=True)
    for name, m in rep["agreement_real_items"]["labels"].items():
        typer.echo(f"{name:20s} kappa {m['kappa']:.3f}  [{m['ci95'][0]:.3f}, {m['ci95'][1]:.3f}]  agreement {m['observed_agreement']:.3f}  ({m['interpretation']})")
    typer.echo(f"Report: {out / 'iaa_report.md'}")


@app.command("convert-ct2")
def convert_ct2_cmd(
    force: Annotated[bool, typer.Option(help="Overwrite an existing converted model")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """One-time: convert NLLB to int8 CTranslate2 (used for fast, low-memory back-translation on the laptop)."""
    from mml.translate import convert_ct2

    cfg = _load(config, pair, profile, set_)
    out = convert_ct2(cfg, force)
    typer.echo(f"CTranslate2 model: {out}")


@app.command("download-data")
def download_data(
    what: Annotated[list[str], typer.Option(help="flores | halomi | mqm (repeatable)")] = ["flores", "halomi", "mqm"],  # noqa: B006
    dry_run: Annotated[bool, typer.Option(help="Only check reachability/requirements; download nothing")] = False,
    config: ConfigOpt = None, pair: PairOpt = None, profile: ProfileOpt = None, set_: SetOpt = None,
) -> None:
    """Download FLORES+, HalOmi and WMT MQM data for the active pair (each source degrades gracefully)."""
    cfg = _load(config, pair, profile, set_)
    for res in download_all(cfg, what, dry_run):
        color = {"ok": "green", "dry-run": "cyan", "skipped": "yellow", "failed": "red"}[res.status]
        typer.secho(f"[{res.status:>7}] {res.source}: {res.detail}", fg=color)


if __name__ == "__main__":
    app()
