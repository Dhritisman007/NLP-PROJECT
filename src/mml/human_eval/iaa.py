"""Inter-annotator agreement (Cohen's kappa) and detector-vs-human evaluation for the human-evaluation kit.

Inputs: the two annotation CSVs returned by the annotators and the hidden ``key`` file. Outputs: validation warnings, kappa and
observed agreement per label (with bootstrap 95% CIs), attention-check accuracy, a list of disagreements to adjudicate, and the
detector's precision / recall / F1 against the humans' consensus, **re-weighted for the stratified sampling** (the sample is
about half detector-flagged, so unweighted precision/recall would be biased).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from mml.human_eval.kit import LABEL_COLUMNS, TYPE_COLUMNS

TRUE_WORDS = {"1", "yes", "y", "true", "t", "x", "ja", "haan", "हाँ", "हां"}
FALSE_WORDS = {"0", "no", "n", "false", "f", "nein", "nahi", "नहीं"}


def kappa(a: Sequence[int], b: Sequence[int]) -> float:
    """Cohen's kappa for two raters over the same items (NaN if chance agreement is 1, i.e. no label variation)."""
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return float("nan")
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in np.unique(np.concatenate([a, b])))
    return float("nan") if pe >= 1.0 - 1e-12 else (po - pe) / (1.0 - pe)


def interpret(k: float) -> str:
    """Landis & Koch (1977) verbal bands."""
    if math.isnan(k):
        return "undefined (no variation in labels)"
    if k < 0:
        return "poor (worse than chance)"
    return "slight" if k < 0.2 else "fair" if k < 0.4 else "moderate" if k < 0.6 else "substantial" if k < 0.8 else "almost perfect"


def bootstrap_kappa(a: Sequence[int], b: Sequence[int], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of kappa resampling items (NaN resamples are skipped)."""
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    vals = [kappa(a[i], b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    vals = [v for v in vals if not math.isnan(v)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if len(vals) > 20 else (float("nan"), float("nan"))


# --------------------------------------------------------------------------- loading and validation
def _to_bin(value: Any) -> float:
    s = str(value).strip().lower()
    if s in TRUE_WORDS:
        return 1.0
    if s in FALSE_WORDS:
        return 0.0
    return float("nan")


def load_annotations(path: str | Path, who: str) -> tuple[pd.DataFrame, list[str]]:
    """Read and normalise one annotator file. Returns (frame indexed by item_id, issues)."""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    issues: list[str] = []
    missing_cols = [c for c in ("item_id", *LABEL_COLUMNS) if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{who}: file is missing required columns {missing_cols}")
    dup = df["item_id"][df["item_id"].duplicated()].tolist()
    if dup:
        issues.append(f"{who}: duplicate item ids {dup[:5]}")
        df = df.drop_duplicates("item_id", keep="last")
    out = pd.DataFrame(index=df["item_id"])
    out["mp"] = [_to_bin(v) for v in df["meaning_preserved"]]
    for col in TYPE_COLUMNS:
        out[col] = [0.0 if str(v).strip() == "" else _to_bin(v) for v in df[col]]
    bad_vals = [i for i, v in zip(df["item_id"], df["meaning_preserved"]) if str(v).strip() != "" and math.isnan(_to_bin(v))]
    if bad_vals:
        issues.append(f"{who}: unreadable meaning_preserved value for {bad_vals[:5]} (use 1/0 or yes/no)")
    unl = out.index[out["mp"].isna()].tolist()
    if unl:
        issues.append(f"{who}: {len(unl)} unlabeled items (excluded)")
    typed = out[list(TYPE_COLUMNS)].sum(axis=1)
    incons = out.index[(out["mp"] == 1) & (typed > 0)].tolist()
    if incons:
        issues.append(f"{who}: {len(incons)} items marked 'meaning preserved' but with an error type ticked: {incons[:5]}")
        out.loc[incons, list(TYPE_COLUMNS)] = 0.0
    untyped = out.index[(out["mp"] == 0) & (typed == 0)].tolist()
    if untyped:
        issues.append(f"{who}: {len(untyped)} errors without a type (counted as an error, no type): {untyped[:5]}")
    out["problem_text"] = df.set_index("item_id").get("problem_text", pd.Series(dtype=str)).reindex(out.index).fillna("")
    out["notes"] = df.set_index("item_id").get("notes", pd.Series(dtype=str)).reindex(out.index).fillna("")
    return out, issues


# --------------------------------------------------------------------------- agreement
def agreement(a: pd.DataFrame, b: pd.DataFrame, exclude: set[str] = frozenset()) -> dict[str, Any]:
    """Kappa, observed agreement and CI per label on items both annotators labelled (attention checks excluded)."""
    ids = [i for i in a.index.intersection(b.index) if i not in exclude and not math.isnan(a.at[i, "mp"]) and not math.isnan(b.at[i, "mp"])]
    out: dict[str, Any] = {"n_items": len(ids), "labels": {}}
    if not ids:
        return out
    cols = {"meaning_preserved": "mp", **{k: k for k in TYPE_COLUMNS}}
    for name, col in cols.items():
        x, y = a.loc[ids, col].astype(int).to_numpy(), b.loc[ids, col].astype(int).to_numpy()
        k = kappa(x, y)
        lo, hi = bootstrap_kappa(x, y)
        out["labels"][name] = {"kappa": k, "ci95": [lo, hi], "interpretation": interpret(k), "observed_agreement": float(np.mean(x == y)),
                               "prevalence_A": float(np.mean(1 - x if name == "meaning_preserved" else x)),
                               "prevalence_B": float(np.mean(1 - y if name == "meaning_preserved" else y)),
                               "n_positive_either": int(np.sum((x == 1) | (y == 1))) if name != "meaning_preserved" else int(np.sum((x == 0) | (y == 0)))}
    out["note_meaning_preserved_prevalence"] = "prevalence_* for meaning_preserved is the share marked as an ERROR"
    return out


def disagreements(a: pd.DataFrame, b: pd.DataFrame, ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for i in ids:
        if i in a.index and i in b.index and (a.loc[i, ["mp", *TYPE_COLUMNS]].astype(float).fillna(-1).tolist() != b.loc[i, ["mp", *TYPE_COLUMNS]].astype(float).fillna(-1).tolist()):
            rows.append({"item_id": i, "A_preserved": a.at[i, "mp"], "B_preserved": b.at[i, "mp"],
                         "A_types": ",".join(k for k in TYPE_COLUMNS if a.at[i, k] == 1), "B_types": ",".join(k for k in TYPE_COLUMNS if b.at[i, k] == 1),
                         "A_problem": a.at[i, "problem_text"], "B_problem": b.at[i, "problem_text"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- attention checks
def check_accuracy(ann: pd.DataFrame, key: pd.DataFrame) -> dict[str, Any]:
    """How an annotator did on the known-answer items."""
    chk = key[key["is_check"] == 1]
    rows = [(i, r["check_label"], ann.at[i, "mp"]) for i, r in chk.set_index("item_id").iterrows() if i in ann.index and not math.isnan(ann.at[i, "mp"])]
    correct = [(lab != "OK") == (mp == 0) for _, lab, mp in rows]
    wrong = [i for (i, _, _), ok in zip(rows, correct) if not ok]
    return {"n": len(rows), "accuracy": float(np.mean(correct)) if rows else float("nan"), "wrong_items": wrong}


# --------------------------------------------------------------------------- detector vs humans (stratified re-weighting)
def weighted_prf(h: np.ndarray, d: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Precision / recall / F1 of detector flags ``d`` against human labels ``h`` with item weights ``w`` (= stratum size / sampled)."""
    tp, dpos, hpos = float((w * h * d).sum()), float((w * d).sum()), float((w * h).sum())
    prec = tp / dpos if dpos else float("nan")
    rec = tp / hpos if hpos else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec and rec and not math.isnan(prec + rec) else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1}


def detector_vs_humans(a: pd.DataFrame, b: pd.DataFrame, key: pd.DataFrame, meta: dict[str, Any], n_boot: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Detector precision/recall against the humans' consensus on the real-MT items (attention checks excluded).

    Consensus = both annotators agree. Weights re-balance the flagged/unflagged strata to the pool's real flag rate.
    """
    k = key[key["is_check"] == 0].set_index("item_id")
    ids = [i for i in k.index if i in a.index and i in b.index and not math.isnan(a.at[i, "mp"]) and a.at[i, "mp"] == b.at[i, "mp"]]
    n_f = {"flagged": meta["pool_flagged"], "unflagged": meta["pool_unflagged"]}
    n_s = {s: int((k.loc[ids, "stratum"] == s).sum()) for s in n_f}
    if not ids or min(n_s.values()) == 0:
        return {"error": "need consensus items in both strata (flagged and unflagged)", "n_consensus": len(ids), "n_by_stratum": n_s}
    w = np.array([n_f[k.at[i, "stratum"]] / n_s[k.at[i, "stratum"]] for i in ids])
    strata = np.array([k.at[i, "stratum"] for i in ids])
    out: dict[str, Any] = {"n_consensus": len(ids), "n_total_real": int(len(k)), "n_by_stratum": n_s,
                           "pool_flag_rate": meta["pool_flag_rate"], "targets": {}}
    targets = {"any_error": (1 - a.loc[ids, "mp"].to_numpy(), k.loc[ids, "detector_flagged"].astype(int).to_numpy())}
    det_types = k.loc[ids, "detector_types"].fillna("").astype(str).tolist()
    for col, t in TYPE_COLUMNS.items():
        both = (a.loc[ids, col].to_numpy() == 1) & (b.loc[ids, col].to_numpy() == 1)
        targets[col] = (both.astype(float), np.array([int(t in s.split(",")) for s in det_types]))
    rng = np.random.default_rng(seed)
    for name, (h, d) in targets.items():
        point = weighted_prf(h, d, w)
        boots = {"precision": [], "recall": [], "f1": []}
        for _ in range(n_boot):
            idx = np.concatenate([rng.choice(np.where(strata == s)[0], size=(strata == s).sum(), replace=True) for s in n_f])
            r = weighted_prf(h[idx], d[idx], w[idx])
            for m in boots:
                if not math.isnan(r[m]):
                    boots[m].append(r[m])
        ci = {m: ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 20 else [float("nan")] * 2) for m, v in boots.items()}
        prev = float((w * h).sum() / w.sum())
        out["targets"][name] = {**point, "ci95": ci, "estimated_prevalence": prev, "human_positive_items": int(h.sum())}
    flagged = strata == "flagged"
    out["human_error_rate_among_flagged"] = float(np.mean(targets["any_error"][0][flagged]))
    out["human_error_rate_among_unflagged"] = float(np.mean(targets["any_error"][0][~flagged]))
    return out


# --------------------------------------------------------------------------- driver
def run_iaa(path_a: Path, path_b: Path, key_path: Path, meta_path: Path, out_dir: Path) -> dict[str, Any]:
    """Validate, compute agreement + detector comparison, and write ``iaa_report.json/.md`` and ``disagreements.csv``."""
    a, issues_a = load_annotations(path_a, "annotator A")
    b, issues_b = load_annotations(path_b, "annotator B")
    key = pd.read_csv(key_path, encoding="utf-8-sig", dtype={"detector_types": str, "check_label": str, "check_subtype": str}, keep_default_na=False)
    for col in ("detector_flagged", "is_check"):
        key[col] = key[col].astype(int)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checks = set(key.loc[key["is_check"] == 1, "item_id"])
    unknown = [i for i in list(a.index) + list(b.index) if i not in set(key["item_id"])]
    report: dict[str, Any] = {"issues": issues_a + issues_b + ([f"ids not in the key: {unknown[:5]}"] if unknown else []),
                              "agreement_real_items": agreement(a, b, exclude=checks),
                              "agreement_including_checks": agreement(a, b),
                              "attention_checks": {"A": check_accuracy(a, key), "B": check_accuracy(b, key)},
                              "detector_vs_humans": detector_vs_humans(a, b, key, meta)}
    dis = disagreements(a, b, sorted(set(a.index) & set(b.index)))
    out_dir.mkdir(parents=True, exist_ok=True)
    dis.to_csv(out_dir / "disagreements.csv", index=False, encoding="utf-8-sig")
    (out_dir / "iaa_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    (out_dir / "iaa_report.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _fmt(x: Any, nd: int = 3) -> str:
    return "n/a" if isinstance(x, float) and math.isnan(x) else (f"{x:.{nd}f}" if isinstance(x, (int, float)) else str(x))


def render_markdown(r: dict[str, Any]) -> str:
    L = ["# Human evaluation: agreement and detector comparison", ""]
    if r["issues"]:
        L += ["## Data issues", "", *[f"- {i}" for i in r["issues"]], ""]
    ag = r["agreement_real_items"]
    L += [f"## Inter-annotator agreement (Cohen's kappa), {ag['n_items']} real items (attention checks excluded)", "",
          "| label | kappa | 95% CI | reading | observed agreement | positives (either) |", "|---|---|---|---|---|---|"]
    for name, m in ag["labels"].items():
        L.append(f"| {name} | {_fmt(m['kappa'])} | [{_fmt(m['ci95'][0])}, {_fmt(m['ci95'][1])}] | {m['interpretation']} | {_fmt(m['observed_agreement'])} | {m['n_positive_either']} |")
    L += ["", "Kappa bands (Landis & Koch): <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate, 0.6-0.8 substantial, >0.8 almost perfect. "
          "Rare labels give unstable kappa even when raw agreement is high; read it together with observed agreement and the CI.", ""]
    L += ["## Attention checks (known answers)", "", "| annotator | checks answered | accuracy | wrong items |", "|---|---|---|---|"]
    for who, c in r["attention_checks"].items():
        L.append(f"| {who} | {c['n']} | {_fmt(c['accuracy'])} | {', '.join(c['wrong_items']) or '-'} |")
    d = r["detector_vs_humans"]
    L += ["", "## Detector vs human consensus (real MT output, re-weighted for the stratified sample)", ""]
    if "error" in d:
        L.append(f"Not computed: {d['error']} (consensus items: {d['n_consensus']}, per stratum: {d['n_by_stratum']}).")
    else:
        L += [f"{d['n_consensus']} of {d['n_total_real']} real items have a consensus label (annotators agree). Pool flag rate {d['pool_flag_rate']:.3f}; "
              f"humans mark {d['human_error_rate_among_flagged']:.2f} of detector-flagged and {d['human_error_rate_among_unflagged']:.2f} of unflagged items as errors.", "",
              "| target | precision | 95% CI | recall | 95% CI | F1 | est. prevalence | human positives |", "|---|---|---|---|---|---|---|---|"]
        for name, t in d["targets"].items():
            L.append(f"| {name} | {_fmt(t['precision'])} | [{_fmt(t['ci95']['precision'][0])}, {_fmt(t['ci95']['precision'][1])}] | {_fmt(t['recall'])} | "
                     f"[{_fmt(t['ci95']['recall'][0])}, {_fmt(t['ci95']['recall'][1])}] | {_fmt(t['f1'])} | {_fmt(t['estimated_prevalence'])} | {t['human_positive_items']} |")
    return "\n".join(L) + "\n"
