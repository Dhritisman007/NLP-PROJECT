from __future__ import annotations

import json
from pathlib import Path

import pytest

from mml.config import load_config
from mml.correct.span_postedit import CorrectionResult
from mml.detect.classifier import TARGETS
from mml.detect.predict import Detection
from mml.pipeline import CorrectionPipeline
from mml.report import (
    baselines_table, build_report, correction_table, detection_table, load_all, real_mt_table, write_table,
)

ROOT = Path(__file__).resolve().parents[1]
EN, DE = "eng_Latn", "deu_Latn"


def det(*types):
    return Detection({t: 0.9 for t in TARGETS}, set(types), [])


class FakeDetector:
    """First call flags ENTITY_NUMBER for every pair; re-detection of corrected text flags nothing."""

    def __init__(self):
        self.calls = []

    def detect(self, pairs):
        self.calls.append(list(pairs))
        return [det("ENTITY_NUMBER") if "45 Millionen" in h else det() for _, h in pairs]


# ------------------------------------------------------------------ pipeline
def test_pipeline_corrects_then_redetects_only_changed_sentences():
    cfg = load_config(pair="en-de")
    fake = FakeDetector()
    pipe = CorrectionPipeline(cfg, fake, "constrained")
    pairs = [("The firm made 4.5 million dollars.", "Die Firma verdiente 45 Millionen Dollar."),
             ("The cat sleeps.", "Die Katze schläft.")]
    res = pipe.run_pairs(pairs)
    assert res[0].corrected == "Die Firma verdiente 4,5 Millionen Dollar." and res[0].changed and res[0].flags_cleared
    assert not res[1].changed and res[1].corrected == pairs[1][1]
    assert len(fake.calls) == 2 and fake.calls[1] == [(pairs[0][0], res[0].corrected)]       # re-detect = changed ones only


def test_pipeline_none_strategy_never_edits():
    pipe = CorrectionPipeline(load_config(pair="en-de"), FakeDetector(), "none")
    [r] = pipe.run_pairs([("x", "Die Firma verdiente 45 Millionen Dollar.")])
    assert not r.changed and r.before.types == {"ENTITY_NUMBER"}


def test_auto_strategy_falls_back_to_rules_without_an_llm(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import mml.pipeline as pm

    monkeypatch.setattr(pm, "Detector", lambda cfg: FakeDetector())
    pipe = CorrectionPipeline.from_config(load_config(pair="en-de"), "auto")
    assert pipe.strategy == "constrained" and pipe.llm is None
    with pytest.raises(Exception):
        CorrectionPipeline.from_config(load_config(pair="en-de"), "full")                  # explicit LLM strategy must not silently degrade


def test_pipeline_translate_then_run(monkeypatch):
    class T:
        def translate_best(self, sources):
            return ["Die Firma verdiente 45 Millionen Dollar." for _ in sources]

    pipe = CorrectionPipeline(load_config(pair="en-de"), FakeDetector(), "constrained", translator=T())
    [r] = pipe.run(["The firm made 4.5 million dollars."])
    assert r.corrected.endswith("4,5 Millionen Dollar.")


# ------------------------------------------------------------------ integration on validation data (never the test split)
HAVE = all((ROOT / p).exists() for p in ("data/synthetic/en-de/val.jsonl", "models/en-de/detector_gbm.joblib",
                                         "results/detector/en-de/results.json", "data/processed/features/en-de/val.json"))


@pytest.mark.skipif(not HAVE, reason="needs the generated data, features and trained detector")
def test_evaluate_synthetic_and_report_on_validation_data(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from mml.evaluate import evaluate_synthetic

    cfg = load_config(pair="en-de")
    syn = evaluate_synthetic(cfg, llm=None, split="val", limit=60)
    assert syn["split"] == "val" and syn["tuned_on_test"] is False and syn["n"] == 60
    assert set(syn["strategies"]) == {"raw", "constrained"}
    assert set(syn["skipped"]) == {"baseline_llm", "span_llm", "full"}                    # LLM strategies are NOT silently faked
    raw, con = syn["strategies"]["raw"], syn["strategies"]["constrained"]
    assert raw["errors"]["exact_restored"] == 0 and raw["clean"]["over_correction_rate"] == 0
    assert 0 <= con["clean"]["over_correction_rate"] <= 0.1 and con["errors"]["chrf_gain"] >= -1e-9
    assert "flags_cleared" in con["per_type"]["ENTITY_NUMBER"] and "baseline_b_labse_threshold" in syn

    detector_res = json.loads((ROOT / "results/detector/en-de/results.json").read_text())
    real = {"n": 10, "mt_backend": "nllb_ct2", "strategy": "constrained", "flag_rate": 0.3, "edit_rate": 0.1, "chrf_mt": 0.5,
            "chrf_corrected": 0.51, "chrf_gain_all": 0.01, "edited_improved": 1.0, "edited_worsened": 0.0, "flags_cleared_among_edited": 1.0}
    for pair in ("en-de", "en-hi"):                                                          # same payload for both pairs: structure test only
        d = tmp_path / "results" / "evaluation" / pair
        d.mkdir(parents=True)
        (d / "results.json").write_text(json.dumps({"synthetic_test": syn, "real_mt": real}))
        dd = tmp_path / "results" / "detector" / pair
        dd.mkdir(parents=True)
        (dd / "results.json").write_text(json.dumps(detector_res))
    data = load_all(tmp_path, ["en-de", "en-hi"])
    corr = correction_table(data)
    assert {"ALL errors", "Clean (over-correction)", "Entity / number"} <= set(corr["scope"])
    base = baselines_table(data)
    llm_rows = base[base["method"].str.startswith(("(c)", "(d) "))]
    assert len(llm_rows) == 4 and (llm_rows["errors_restored_exactly"] == "not run").all()  # honest "not run", never a number
    assert set(detection_table(data)["class"]) >= {"OK", "OMISSION", "any error vs OK"}
    assert real_mt_table(data)["flagged"].iloc[0] == 0.3
    out = tmp_path / "t"
    write_table(corr, out, "x", "title")
    assert (out / "x.csv").stat().st_size > 0 and (out / "x.md").read_text().startswith("### title")

    rep = build_report(tmp_path, ["en-de", "en-hi"])
    assert len(rep["plots"]) == 4 and len(rep["tables"]) == 4
    for f in rep["plots"]:
        p = Path(f)
        assert p.suffix == ".png" and p.stat().st_size > 10_000


def test_report_refuses_to_run_without_evaluations(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path, ["en-de"])


def test_test_split_is_only_read_by_the_explicit_final_path():
    import inspect

    from mml.evaluate import evaluate_synthetic

    src = inspect.getsource(evaluate_synthetic)
    assert 'for_final_eval=(split == "test")' in src and 'split: str = "test"' in src
