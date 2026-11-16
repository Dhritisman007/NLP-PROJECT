from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

from mml.config import load_config
from mml.human_eval.iaa import (
    agreement, bootstrap_kappa, check_accuracy, interpret, kappa, load_annotations, run_iaa, weighted_prf,
)
from mml.human_eval.kit import (
    ANNOTATION_COLUMNS, KEY_COLUMNS, LABEL_COLUMNS, PoolItem, order_for, render_html, sample_kit, write_kit,
)


def make_pool(n_flagged=80, n_unflagged=300):
    pool = []
    for i in range(n_flagged + n_unflagged):
        fl = i < n_flagged
        pool.append(PoolItem(i, f"SOURCE sentence {i}", f"REFERENCE {i}", f"TRANSLATION {i}", fl, ["OMISSION"] if fl else [],
                             0.9 if fl else 0.1, [f"[OMISSION] secret finding {i}"] if fl else []))
    return pool


def make_checks():
    labels = ["OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"] * 3 + ["OK"] * 12
    return [SimpleNamespace(id=f"c{i}", src=f"CHECKSRC {i}", hyp=f"CHECKHYP {i}", ref=f"CHECKREF {i}", label=l, subtype="x") for i, l in enumerate(labels)]


# ------------------------------------------------------------------ kit
def test_sample_is_stratified_has_checks_of_every_type_and_unique_ids():
    items, meta = sample_kit(make_pool(), make_checks(), n=100, n_checks=10, seed=1)
    assert len(items) == 100 and len({i.item_id for i in items}) == 100
    assert meta["sampled_flagged"] == 45 and meta["sampled_unflagged"] == 45 and meta["n_checks"] == 10
    assert sum(i.stratum == "flagged" for i in items) == 45 and sum(i.stratum == "unflagged" for i in items) == 45
    checks = [i for i in items if i.is_check]
    assert {c.check_label for c in checks} >= {"OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION", "OK"}
    assert meta["pool_flag_rate"] == pytest.approx(80 / 380)
    assert sample_kit(make_pool(), make_checks(), seed=1)[0][0].source == items[0].source          # deterministic
    assert [i.source for i in sample_kit(make_pool(), make_checks(), seed=2)[0]] != [i.source for i in items]


def test_sampling_degrades_gracefully_when_few_sentences_are_flagged():
    items, meta = sample_kit(make_pool(n_flagged=10, n_unflagged=300), make_checks(), n=100, n_checks=10)
    # the shortfall of flagged sentences is filled from the unflagged stratum so the kit still has 100 items;
    # the re-weighting by pool strata sizes (kit_meta) is what keeps the estimates unbiased
    assert meta["sampled_flagged"] == 10 and meta["sampled_unflagged"] == 80 and len(items) == 100


def test_annotator_files_are_blind_and_item_order_differs(tmp_path):
    cfg = load_config(pair="en-de")
    items, meta = sample_kit(make_pool(), make_checks(), seed=3)
    out = write_kit(cfg, items, meta, ["A", "B"], out_dir=tmp_path)
    secret = ["secret finding", "REFERENCE", "CHECKREF", "flagged", "unflagged", "detector", "OMISSION"]
    share = out / "for_annotators"
    assert sorted(p.name for p in share.iterdir()) == ["GUIDELINES.md", "annotator_A.csv", "annotator_A.html", "annotator_B.csv", "annotator_B.html"]
    assert not any("key" in p.name for p in share.iterdir())                 # the key can never travel with the annotator files
    for name in ("annotator_A.csv", "annotator_B.csv", "annotator_A.html", "annotator_B.html"):
        text = (share / name).read_text(encoding="utf-8")
        for s in secret:
            if name.endswith(".html") and s in ("OMISSION", "flagged"):
                continue                                           # the form legitimately explains error types / uses CSS words
            assert s not in text, (name, s)
    key = (out / "key_DO_NOT_SHARE_WITH_ANNOTATORS.csv").read_text(encoding="utf-8-sig")
    assert "secret finding" in key and key.splitlines()[0].split(",") == KEY_COLUMNS
    a = pd.read_csv(share / "annotator_A.csv", encoding="utf-8-sig", dtype=str)
    b = pd.read_csv(share / "annotator_B.csv", encoding="utf-8-sig", dtype=str)
    assert list(a.columns) == ANNOTATION_COLUMNS and set(a["item_id"]) == set(b["item_id"]) and list(a["item_id"]) != list(b["item_id"])
    assert (share / "annotator_A.csv").read_bytes().startswith(b"\xef\xbb\xbf")          # BOM so spreadsheets show Devanagari
    assert (share / "GUIDELINES.md").exists() and json.loads((out / "kit_meta.json").read_text())["n"] == 100
    assert [i.item_id for i in order_for(items, "A", 3)] == list(a["item_id"])


def test_ids_do_not_reveal_the_stratum():
    items, _ = sample_kit(make_pool(), make_checks(), seed=4)
    first_20 = [i.stratum for i in sorted(items, key=lambda i: i.item_id)[:20]]
    assert len(set(first_20)) > 1                                  # strata are interleaved, not grouped by id


def test_html_escapes_hostile_text_and_has_valid_javascript(tmp_path):
    items, _ = sample_kit(make_pool(), make_checks(), seed=5)
    items[0].source = "</script><img src=x onerror=alert(1)> \"quotes\" & हिन्दी"
    html = render_html(items, "en-hi", "A", "hin_Deva", "# guide\n<b>x</b>")
    assert html.count("</script>") == 1 and "<b>x</b>" not in html and "&lt;b&gt;x&lt;/b&gt;" in html
    assert "हिन्दी" in html and "Hindi" in html
    node = shutil.which("node")
    if node:
        js = re.search(r"<script>(.*)</script>", html, re.S).group(1)
        f = tmp_path / "form.js"
        f.write_text(js, encoding="utf-8")
        res = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


# ------------------------------------------------------------------ kappa
def test_kappa_known_values_and_agreement_with_sklearn():
    assert kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    # classic 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po = 0.7, pe = 0.5 -> kappa = 0.4
    a = [1] * 25 + [0] * 25
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert kappa(a, b) == pytest.approx(0.4)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x, y = rng.integers(0, 2, 60), rng.integers(0, 2, 60)
        assert kappa(x, y) == pytest.approx(cohen_kappa_score(x, y))
    assert math.isnan(kappa([1, 1, 1], [1, 1, 1])) and math.isnan(kappa([], []))


def test_interpretation_bands_and_bootstrap_ci():
    assert [interpret(k) for k in (-0.1, 0.1, 0.3, 0.5, 0.7, 0.9)] == [
        "poor (worse than chance)", "slight", "fair", "moderate", "substantial", "almost perfect"]
    assert interpret(float("nan")).startswith("undefined")
    rng = np.random.default_rng(1)
    x = rng.integers(0, 2, 200)
    y = np.where(rng.random(200) < 0.85, x, 1 - x)
    lo, hi = bootstrap_kappa(x, y, n_boot=300)
    assert lo <= kappa(x, y) <= hi and hi - lo < 0.4


# ------------------------------------------------------------------ loading / validation
def write_ann(path, rows, cols=("item_id", "source", "translation", *LABEL_COLUMNS, "problem_text", "notes")):
    pd.DataFrame(rows, columns=list(cols)).to_csv(path, index=False, encoding="utf-8-sig")


def row(i, mp="", om="", mi="", en="", ha="", pt="", nt=""):
    return [f"item_{i:03d}", "s", "t", mp, om, mi, en, ha, pt, nt]


def test_loader_normalises_values_and_reports_problems(tmp_path):
    p = tmp_path / "a.csv"
    write_ann(p, [row(1, "yes"), row(2, "No", om="1", ha="x"), row(3, "1", om="1"), row(4, ""), row(5, "0"), row(6, "maybe"), row(2, "no", om="1")])
    df, issues = load_annotations(p, "annotator A")
    text = " ".join(issues)
    assert df.at["item_001", "mp"] == 1 and df.at["item_002", "mp"] == 0 and df.at["item_002", "omission"] == 1
    assert "duplicate item ids" in text and "unlabeled" in text and "marked 'meaning preserved' but with an error type" in text
    assert "without a type" in text and "unreadable" in text
    assert df.at["item_003", "omission"] == 0                       # inconsistent row: types cleared, flagged in the report
    assert math.isnan(df.at["item_004", "mp"]) and math.isnan(df.at["item_006", "mp"])


def test_loader_rejects_files_without_the_required_columns(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"item_id": ["x"], "source": ["s"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_annotations(p, "annotator A")


# ------------------------------------------------------------------ stratified weighting
def test_weighted_prf_recovers_population_precision_and_recall():
    # pool: 100 flagged (80% real errors), 900 unflagged (10% real errors). Sample 50 + 50 in exact proportion.
    h = np.array([1] * 40 + [0] * 10 + [1] * 5 + [0] * 45)
    d = np.array([1] * 50 + [0] * 50)
    w = np.array([100 / 50] * 50 + [900 / 50] * 50)
    r = weighted_prf(h, d, w)
    assert r["precision"] == pytest.approx(0.8)                                  # 80 of the 100 flagged are real errors
    assert r["recall"] == pytest.approx(80 / (80 + 90))                          # 170 real errors in the pool, 80 flagged
    assert r["f1"] == pytest.approx(2 * 0.8 * (80 / 170) / (0.8 + 80 / 170))
    naive_recall = h[d == 1].sum() / h.sum()                                      # ignoring the weights would give 0.89
    assert naive_recall > 0.85 and r["recall"] < 0.5
    assert math.isnan(weighted_prf(np.zeros(4), np.ones(4), np.ones(4))["recall"])


# ------------------------------------------------------------------ end to end with SIMULATED annotators (test only)
def simulate(items, rng, flip=0.0, wrong_check=0):
    """Annotators who answer from a hidden 'truth' (checks use their known label), with random label noise."""
    truth = {}
    for it in items:
        if it.is_check:
            err = it.check_label != "OK"
            types = {it.check_label} if err else set()
        else:
            err = it.stratum == "flagged" and rng.random() < 0.7 or it.stratum == "unflagged" and rng.random() < 0.1
            types = {"OMISSION"} if err else set()
        truth[it.item_id] = (err, types)
    return truth


def annotator_rows(items, truth, rng, flip):
    rows = []
    for it in sorted(items, key=lambda i: i.item_id):
        err, types = truth[it.item_id]
        if rng.random() < flip:
            err, types = (not err), ({"OMISSION"} if not err else set())
        rows.append(row(int(it.item_id.split("_")[1]), "0" if err else "1",
                        *(("1" if t in types else "0") if err else "0" for t in ("OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"))))
    return rows


def test_full_iaa_pipeline_on_simulated_annotators(tmp_path):
    cfg = load_config(pair="en-de")
    items, meta = sample_kit(make_pool(80, 300), make_checks(), seed=7)
    out = write_kit(cfg, items, meta, ["A", "B"], out_dir=tmp_path)
    rng = np.random.default_rng(7)
    truth = simulate(items, rng)
    write_ann(tmp_path / "ret_A.csv", annotator_rows(items, truth, rng, flip=0.05))
    write_ann(tmp_path / "ret_B.csv", annotator_rows(items, truth, rng, flip=0.05))
    rep = run_iaa(tmp_path / "ret_A.csv", tmp_path / "ret_B.csv", out / "key_DO_NOT_SHARE_WITH_ANNOTATORS.csv", out / "kit_meta.json", tmp_path / "rep")
    mp = rep["agreement_real_items"]["labels"]["meaning_preserved"]
    assert rep["agreement_real_items"]["n_items"] == 90 and mp["kappa"] > 0.6 and mp["observed_agreement"] > 0.85
    assert rep["attention_checks"]["A"]["n"] == 10 and rep["attention_checks"]["A"]["accuracy"] >= 0.7
    d = rep["detector_vs_humans"]
    assert d["n_by_stratum"]["flagged"] > 20 and d["n_by_stratum"]["unflagged"] > 20
    t = d["targets"]["any_error"]
    assert 0.5 < t["precision"] <= 1 and 0 < t["recall"] < 1 and t["ci95"]["precision"][0] <= t["precision"] <= t["ci95"]["precision"][1]
    assert abs(t["precision"] - 0.7) < 0.2                                         # the simulated truth had 70% precision
    for f in ("iaa_report.json", "iaa_report.md", "disagreements.csv"):
        assert (tmp_path / "rep" / f).stat().st_size > 0
    assert "Cohen's kappa" in (tmp_path / "rep" / "iaa_report.md").read_text()


def test_iaa_report_handles_missing_data_without_crashing(tmp_path):
    cfg = load_config(pair="en-de")
    items, meta = sample_kit(make_pool(), make_checks(), seed=8)
    out = write_kit(cfg, items, meta, ["A", "B"], out_dir=tmp_path)
    for name in ("A", "B"):                                                          # annotators who returned nothing
        write_ann(tmp_path / f"{name}.csv", [row(i) for i in range(1, 101)])
    rep = run_iaa(tmp_path / "A.csv", tmp_path / "B.csv", out / "key_DO_NOT_SHARE_WITH_ANNOTATORS.csv", out / "kit_meta.json", tmp_path / "r")
    assert rep["agreement_real_items"]["n_items"] == 0 and "error" in rep["detector_vs_humans"]
    assert any("unlabeled" in i for i in rep["issues"])


def test_check_accuracy_counts_only_answered_checks():
    key = pd.DataFrame({"item_id": ["a", "b", "c"], "is_check": [1, 1, 0], "check_label": ["OMISSION", "OK", ""]})
    ann = pd.DataFrame({"mp": [0.0, 0.0, 1.0]}, index=["a", "b", "c"])
    r = check_accuracy(ann, key)
    assert r["n"] == 2 and r["accuracy"] == 0.5 and r["wrong_items"] == ["b"]
