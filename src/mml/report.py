"""Phase 6 reporting: results tables (CSV + markdown) and plots from the saved evaluation/detector JSON files.

Plot conventions (from the data-viz guidance): one y-axis per chart, categorical hues assigned in fixed order
(en-hi = slot 1 blue, en-de = slot 2 orange; the pair passes the palette validator: CVD dE 24.7, normal dE 33.6), neutral
gray for baselines and "before", a legend for every chart with two or more series, selective value labels, text in
neutral ink (never the series colour), recessive grid, thin marks with a gap between fills.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mml.detect.classifier import TYPE_NAMES  # noqa: E402

SURFACE, INK, INK2, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#a8a7a0"
PAIR_COLOR = {"en-hi": "#2a78d6", "en-de": "#eb6834"}
PAIR_LABEL = {"en-hi": "English → Hindi", "en-de": "English → German"}
TYPE_LABEL = {"OK": "Clean (OK)", "OMISSION": "Omission", "MISTRANSLATION": "Mistranslation",
              "ENTITY_NUMBER": "Entity / number", "HALLUCINATION": "Hallucination"}
GROUP_LABEL = {"rules": "rules", "lex": "closed lists (lex)", "labse": "LaBSE", "align": "alignment",
               "roundtrip": "round trip (BT+NLI)", "surface": "length"}


# --------------------------------------------------------------------------- loading
def load_all(root: Path, pairs: list[str]) -> dict[str, dict[str, Any]]:
    """Collect ``evaluation`` and ``detector`` results for each pair from ``<root>/results``."""
    out: dict[str, dict[str, Any]] = {}
    for p in pairs:
        ev = root / "results" / "evaluation" / p / "results.json"
        det = root / "results" / "detector" / p / "results.json"
        if ev.exists() and det.exists():
            out[p] = {"eval": json.loads(ev.read_text(encoding="utf-8")), "detector": json.loads(det.read_text(encoding="utf-8"))}
    return out


# --------------------------------------------------------------------------- tables
def _f(x: Any, nd: int = 3) -> Any:
    return round(float(x), nd) if isinstance(x, (int, float)) else x


def correction_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Per pair x strategy x scope (each error type, all errors, clean sentences)."""
    rows = []
    for pair, d in data.items():
        syn = d["eval"]["synthetic_test"]
        for strat, m in syn["strategies"].items():
            scopes = [(TYPE_LABEL[t], m["per_type"][t]) for t in TYPE_NAMES] + [("ALL errors", m["errors"])]
            for scope, s in scopes:
                rows.append({"pair": pair, "strategy": strat, "scope": scope, "n": s["n"],
                             "flagged": _f(s.get("flagged_any")), "restored_exactly": _f(s["exact_restored"]),
                             "restored_given_flagged": _f(s.get("restored_given_flagged")),
                             "flags_cleared": _f(s.get("flags_cleared")), "chrF_before": _f(s["chrf_before"]),
                             "chrF_after": _f(s["chrf_after"]), "chrF_gain": _f(s["chrf_gain"], 4),
                             "made_worse": _f(s["worsened"])})
            c = m["clean"]
            rows.append({"pair": pair, "strategy": strat, "scope": "Clean (over-correction)", "n": c["n"],
                         "flagged": _f(c.get("false_alarm_rate")), "restored_exactly": None, "restored_given_flagged": None,
                         "flags_cleared": None, "chrF_before": None, "chrF_after": None, "chrF_gain": None,
                         "made_worse": _f(c["over_correction_rate"])})
    return pd.DataFrame(rows)


def baselines_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """The four baselines of the project brief, per pair. LLM baselines read ``not run`` when no LLM was available."""
    rows = []
    for pair, d in data.items():
        syn = d["eval"]["synthetic_test"]
        s = syn["strategies"]
        det = syn["detector_flagging"]
        raw = s["raw"]
        rows.append({"pair": pair, "method": "(a) raw MT", "flags_errors": None, "false_alarms_on_clean": None,
                     "errors_restored_exactly": 0.0, "chrF_gain_on_errors": 0.0, "clean_sentences_changed": 0.0})
        b = syn.get("baseline_b_labse_threshold")
        if b:
            rows.append({"pair": pair, "method": "(b) fixed threshold on LaBSE similarity (flag only)",
                         "flags_errors": _f(b["recall_on_errors"]), "false_alarms_on_clean": _f(b["false_alarm_on_clean"]),
                         "errors_restored_exactly": "cannot correct", "chrF_gain_on_errors": "cannot correct",
                         "clean_sentences_changed": "cannot correct"})
        for key, label in (("baseline_llm", "(c) LLM post-editing, no spans"), ("full", "(d) full pipeline = detector + rules + span-guided LLM")):
            if key in s:
                m = s[key]
                rows.append({"pair": pair, "method": label, "flags_errors": None, "false_alarms_on_clean": _f(m["clean"].get("false_alarm_rate")),
                             "errors_restored_exactly": _f(m["errors"]["exact_restored"]), "chrF_gain_on_errors": _f(m["errors"]["chrf_gain"], 4),
                             "clean_sentences_changed": _f(m["clean"]["over_correction_rate"])})
            else:
                rows.append({"pair": pair, "method": label, "flags_errors": "not run", "false_alarms_on_clean": "not run",
                             "errors_restored_exactly": "not run", "chrF_gain_on_errors": "not run", "clean_sentences_changed": "not run"})
        c = s["constrained"]
        rows.append({"pair": pair, "method": "(d') detector + rule-based fixer (no LLM)", "flags_errors": _f(det["recall_on_errors"]),
                     "false_alarms_on_clean": _f(det["false_alarm_on_clean"]), "errors_restored_exactly": _f(c["errors"]["exact_restored"]),
                     "chrF_gain_on_errors": _f(c["errors"]["chrf_gain"], 4), "clean_sentences_changed": _f(c["clean"]["over_correction_rate"])})
        del raw
    return pd.DataFrame(rows)


def detection_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Phase 4 held-out test metrics per class (from ``results/detector``)."""
    rows = []
    for pair, d in data.items():
        t = d["detector"].get("test")
        if not t:
            continue
        for cls, r in t["metrics"]["per_class"].items():
            rows.append({"pair": pair, "model": t["model"], "class": cls, "precision": _f(r["precision"]), "recall": _f(r["recall"]),
                         "F1": _f(r["f1"]), "MCC": _f(r["mcc"]), "support": r["support"]})
        a = t["metrics"]["any_error"]
        rows.append({"pair": pair, "model": t["model"], "class": "any error vs OK", "precision": _f(a["precision"]), "recall": _f(a["recall"]),
                     "F1": _f(a["f1"]), "MCC": _f(a["mcc"]), "support": a["support"]})
    return pd.DataFrame(rows)


def real_mt_table(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for pair, d in data.items():
        r = d["eval"]["real_mt"]
        rows.append({"pair": pair, "n": r["n"], "mt": r["mt_backend"], "strategy": r["strategy"], "flagged": _f(r["flag_rate"]),
                     "edited": _f(r["edit_rate"]), "chrF_mt": _f(r["chrf_mt"]), "chrF_corrected": _f(r["chrf_corrected"]),
                     "chrF_gain_all": _f(r["chrf_gain_all"], 4), "edited_improved": _f(r["edited_improved"]),
                     "edited_worsened": _f(r["edited_worsened"]), "flags_cleared_among_edited": _f(r["flags_cleared_among_edited"])})
    return pd.DataFrame(rows)


def write_table(df: pd.DataFrame, out: Path, name: str, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{name}.csv", index=False)
    (out / f"{name}.md").write_text(f"### {title}\n\n{df.astype(object).where(df.notna(), '').to_markdown(index=False)}\n", encoding="utf-8")


# --------------------------------------------------------------------------- plots
def _style(ax: plt.Axes, ylabel: str, ylim: tuple[float, float] | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if ylim:
        ax.set_ylim(*ylim)


def _fig(w: float = 8.6, h: float = 4.6) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(w, h), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def _fmt(v: float, fmt: str) -> str:
    """Format a value label; a value that rounds to zero is shown as plain 0 (no '-0.000' noise)."""
    text = fmt.format(v)
    return fmt.replace("+", "").format(0.0) if float(text) == 0 else text


def _grouped(ax: plt.Axes, cats: list[str], series: dict[str, list[float]], fmt: str = "{:.2f}", label_all: bool = True) -> None:
    """Grouped bars with value labels placed just beyond each bar end (offset proportional to the data range)."""
    n = len(series)
    width = 0.36 if n == 2 else 0.7 / n
    off = 0.025 * max(max(abs(v) for v in vals) for vals in series.values()) or 0.01
    for k, (name, vals) in enumerate(series.items()):
        xs = [i + (k - (n - 1) / 2) * (width + 0.03) for i in range(len(cats))]
        ax.bar(xs, vals, width=width, color=PAIR_COLOR.get(name, GRAY), label=PAIR_LABEL.get(name, name), edgecolor=SURFACE, linewidth=1.5)
        if label_all:
            for x, v in zip(xs, vals):
                ax.text(x, v + (off if v >= 0 else -off), _fmt(v, fmt), ha="center", va="bottom" if v >= 0 else "top", fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)


def _legend(ax: plt.Axes) -> None:
    leg = ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    for t in leg.get_texts():
        t.set_color(INK)


def fig_detection(data: dict, out: Path) -> Path:
    """Per-class F1 on the held-out test split, both pairs, with the LaBSE-threshold baseline as a gray reference."""
    fig, ax = _fig()
    classes = ["OK", *TYPE_NAMES]
    series = {p: [d["detector"]["test"]["metrics"]["per_class"][c]["f1"] for c in classes] for p, d in data.items()}
    _grouped(ax, [TYPE_LABEL[c] for c in classes], series)
    _style(ax, "F1 on held-out test split", (0, 1.12))
    ax.set_title("Detection: F1 per class (held-out test, synthetic errors)", loc="left", color=INK, fontsize=11, pad=14)
    _legend(ax)
    path = out / "fig1_detection_f1_per_class.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_ablation(data: dict, out: Path) -> Path:
    """Change in cross-validated macro-F1 when one feature group is removed (negative = the group helps)."""
    fig, ax = _fig()
    groups = [g for g in GROUP_LABEL if all(any(a["run"] == f"drop:{g}" for a in d["detector"]["ablation"]) for d in data.values())]
    series = {}
    for p, d in data.items():
        full = next(a for a in d["detector"]["ablation"] if a["run"] == "all")["macro_f1"]
        series[p] = [next(a for a in d["detector"]["ablation"] if a["run"] == f"drop:{g}")["macro_f1"] - full for g in groups]
    _grouped(ax, [GROUP_LABEL[g] for g in groups], series, fmt="{:+.3f}")
    ax.axhline(0, color=INK2, linewidth=0.8)
    lo = min(min(v) for v in series.values())
    hi = max(0.0, max(max(v) for v in series.values()))
    _style(ax, "Δ macro-F1 when the group is removed", (lo * 1.14, hi + abs(lo) * 0.10))
    ax.set_title("Ablation: what each feature group contributes (grouped CV)", loc="left", color=INK, fontsize=11, pad=14)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    path = out / "fig2_ablation_drop_one_group.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_correction(data: dict, out: Path) -> Path:
    """Rule-based correction per error type: exact restoration (left) and chrF gain (right), as two panels (two measures, two axes)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    cats = [TYPE_LABEL[t] for t in TYPE_NAMES]
    s1 = {p: [d["eval"]["synthetic_test"]["strategies"]["constrained"]["per_type"][t]["exact_restored"] for t in TYPE_NAMES] for p, d in data.items()}
    s2 = {p: [d["eval"]["synthetic_test"]["strategies"]["constrained"]["per_type"][t]["chrf_gain"] for t in TYPE_NAMES] for p, d in data.items()}
    _grouped(a1, cats, s1)
    _style(a1, "reference restored exactly", (0, 1.05))
    a1.set_title("Share of errors fully fixed", loc="left", color=INK, fontsize=10.5)
    _grouped(a2, cats, s2, fmt="{:+.3f}")
    top = max(max(v) for v in s2.values())
    _style(a2, "chrF gain vs reference", (-0.08 * top, top * 1.18))
    a2.set_title("chrF change", loc="left", color=INK, fontsize=10.5)
    for ax in (a1, a2):
        ax.tick_params(axis="x", labelsize=8, rotation=12)
    a1.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Rule-based correction (detector + constrained fixer, no LLM) on the held-out test split", x=0.01, ha="left", color=INK, fontsize=11)
    path = out / "fig3_correction_per_type.png"
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_overcorrection(data: dict, out: Path) -> Path:
    """Harm to correct sentences: detector false alarms, the LaBSE-threshold baseline's false alarms, and sentences actually changed."""
    fig, ax = _fig()
    cats = ["LaBSE threshold:\nflagged (baseline b)", "Detector:\nflagged", "Detector + rule fixer:\nchanged (over-correction)"]
    series = {}
    for p, d in data.items():
        syn = d["eval"]["synthetic_test"]
        series[p] = [syn["baseline_b_labse_threshold"]["false_alarm_on_clean"], syn["detector_flagging"]["false_alarm_on_clean"],
                     syn["strategies"]["constrained"]["clean"]["over_correction_rate"]]
    _grouped(ax, cats, series, fmt="{:.3f}")
    _style(ax, "share of CLEAN test sentences", (0, max(0.4, max(max(v) for v in series.values()) * 1.25)))
    ax.set_title("Cost on correct translations: false alarms and over-correction", loc="left", color=INK, fontsize=11, pad=14)
    _legend(ax)
    path = out / "fig4_false_alarms_and_overcorrection.png"
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def build_report(root: Path, pairs: list[str]) -> dict[str, Any]:
    """Write all tables and plots to ``<root>/results/evaluation`` and return their paths."""
    data = load_all(root, pairs)
    if not data:
        raise FileNotFoundError("no evaluation + detector results found; run `mml evaluate` for each pair first")
    out = root / "results" / "evaluation"
    tables = {
        "results_table": (correction_table(data), "Correction results on the held-out test split (synthetic errors + clean twins)"),
        "baselines": (baselines_table(data), "Baselines (a)-(d) of the project brief"),
        "detection_test": (detection_table(data), "Detector, held-out test split (evaluated once)"),
        "real_mt": (real_mt_table(data), "Behaviour on real NLLB translations of FLORES devtest sources (no gold error labels)"),
    }
    for name, (df, title) in tables.items():
        write_table(df, out, name, title)
    plots = [fig_detection(data, out), fig_ablation(data, out), fig_correction(data, out), fig_overcorrection(data, out)]
    return {"tables": [str(out / f"{n}.csv") for n in tables], "plots": [str(p) for p in plots], "pairs": list(data)}
