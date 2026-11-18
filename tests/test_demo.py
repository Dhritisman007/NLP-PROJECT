from __future__ import annotations

import pytest

from mml.config import load_config
from mml.correct.span_postedit import CorrectionResult
from mml.demo import (
    EXAMPLES, MAX_SENTENCES, TYPE_COLORS, analyse, legend_html, probability_rows, retitle, validate_request, build_app, build_view, diff_segments, edit_summary, highlight_segments,
    split_sentences,
)
from mml.detect.base import Signal, SignalResult, SignalUnavailable, SpanFlag
from mml.detect.classifier import TARGETS
from mml.detect.predict import Detection
from mml.pipeline import CorrectionPipeline, PipelineResult
from mml.resident import ResidentSignals


def det(types=(), spans=(), p=0.9):
    return Detection({t: p for t in TARGETS}, set(types), list(spans))


def result(src, mt, corrected=None, before=None, after=None):
    corrected = mt if corrected is None else corrected
    before = before or det()
    return PipelineResult(src, mt, corrected, before, after or before, CorrectionResult(corrected, "corrected" if corrected != mt else "not_flagged"))


# ------------------------------------------------------------------ pure helpers
def test_split_sentences_handles_punctuation_newlines_and_the_cap():
    assert split_sentences("Hello there. How are you?  Fine!\nNew line here") == ["Hello there.", "How are you?", "Fine!", "New line here"]
    assert split_sentences("   ") == [] and split_sentences("One sentence without a stop") == ["One sentence without a stop"]
    assert len(split_sentences(". ".join(f"S{i}" for i in range(20)) + ".")) == MAX_SENTENCES


def test_highlight_segments_cover_the_text_exactly_for_both_scripts():
    for text in ("Der Zug fährt um 6:45 Uhr ab.", "मुंबई की ट्रेन सुबह 6:45 बजे निकलती है।"):
        segs = highlight_segments(text, [(4, 9, "OMISSION"), (12, 16, "ENTITY_NUMBER")])
        assert "".join(p for p, _ in segs) == text
        assert {lab for _, lab in segs} == {None, "OMISSION", "ENTITY_NUMBER"}
    assert highlight_segments("abc", []) == [("abc", None)] and highlight_segments("", []) == [("", None)]


def test_overlapping_spans_resolve_by_priority_and_out_of_range_spans_are_clipped():
    segs = highlight_segments("abcdefghij", [(0, 6, "OMISSION"), (3, 8, "MISTRANSLATION"), (8, 99, "HALLUCINATION"), (-5, 1, "ENTITY_NUMBER")])
    by_char = {}
    pos = 0
    for piece, lab in segs:
        for _ in piece:
            by_char[pos] = lab
            pos += 1
    assert by_char[0] == "ENTITY_NUMBER" and by_char[1] == "OMISSION" and by_char[4] == "MISTRANSLATION"       # MISTRANSLATION beats OMISSION
    assert by_char[9] == "HALLUCINATION" and pos == 10


def test_diff_and_edit_summary():
    old, new = "Die Firma verdiente 45 Millionen Dollar.", "Die Firma verdiente 4,5 Millionen Dollar."
    segs = diff_segments(old, new)
    assert "".join(p for p, _ in segs) == new and ("4,5", "edited") in segs and all(lab in (None, "edited") for _, lab in segs)
    assert edit_summary(old, new) == [["45", "4,5"]]
    assert edit_summary("a b c", "a c") == [["b", "(removed)"]] and edit_summary("a", "a") == []
    assert diff_segments("same", "same") == [("same", None)]


# ------------------------------------------------------------------ view building
def test_view_shifts_span_offsets_across_sentences_and_reports_findings():
    s1 = result("First sentence.", "Erster Satz.")
    span = SpanFlag("tgt", 0, 4, "Zahl", "number_extra")
    s2 = result("Second one costs 45.", "Zweiter kostet 54.", "Zweiter kostet 45.", det({"ENTITY_NUMBER"}, [span, SpanFlag("src", 17, 19, "45", "number_missing")]),
                det())
    view = build_view([s1, s2], "constrained")
    mt = "".join(p for p, _ in view.mt_segments)
    assert mt == "Erster Satz. Zweiter kostet 54."
    labelled = [(p, lab) for p, lab in view.mt_segments if lab]
    assert labelled == [("Zwei", "ENTITY_NUMBER")]                                  # offset 13 + span 0..4 in the combined text
    assert "".join(p for p, _ in view.source_segments) == "First sentence. Second one costs 45."
    assert ("45", "ENTITY_NUMBER") in [(p, lab) for p, lab in view.source_segments if lab]
    assert view.corrected_text.endswith("kostet 45.") and view.edits == [["54.", "45."]] and view.flagged
    kinds = {(f[0], f[1]) for f in view.findings}
    assert ("Entity / number", "translation") in kinds and ("Entity / number", "source") in kinds
    assert "no longer flags" in view.status and "Rule-based fixing only" in view.status
    assert view.probabilities["ENTITY_NUMBER"] == 0.9 and "any error" in view.probabilities and view.flagged_types == ["ENTITY_NUMBER"]


def test_probability_table_never_headlines_an_unflagged_type_and_legend_names_every_type():
    below = build_view([result("a b c.", "x y z.", before=Detection({**{t: 0.1 for t in TARGETS}, "OMISSION": 0.38}, set(), []))], "constrained")
    rows = probability_rows(below)
    assert rows[0] == ["Omission", "38%", "no"] and all(r[2] == "no" for r in rows)          # high-ish but below threshold: not "flagged"
    flagged = build_view([result("a b c.", "x y z.", before=det({"OMISSION"}, p=0.9))], "constrained")
    rows = {r[0]: r for r in probability_rows(flagged)}
    assert rows["Omission"][2] == "yes" and rows["Any error"][2] == "yes" and rows["Hallucination"][2] == "no"
    html = legend_html()
    assert all(name in html for name in ("Omission", "Mistranslation", "Entity / number", "Hallucination", "edited word"))


def test_own_translation_skips_mt_and_requires_a_single_sentence():
    class FakePipe:
        strategy = "constrained"
        detector = type("D", (), {"skipped": {}})()
        ran = []

        def run(self, sentences):
            self.ran.append("mt")
            return [result(s, s) for s in sentences]

        def run_pairs(self, pairs):
            self.ran.append("pairs")
            return [result(s, t, "FIXED") for s, t in pairs]

    pipe = FakePipe()
    view = analyse(pipe, "Only one sentence here.", True, "  my own translation  ")
    assert pipe.ran == ["pairs"] and view.per_sentence[0]["mt"] == "my own translation" and view.corrected_text == "FIXED"
    with pytest.raises(ValueError, match="exactly one source sentence"):
        analyse(pipe, "One. Two.", True, "x")
    analyse(pipe, "One.", True, "   ")                          # blank own translation -> normal MT path
    assert pipe.ran[-1] == "mt"


def test_invalid_requests_are_rejected_before_any_model_is_loaded():
    gr = pytest.importorskip("gradio")
    assert validate_request("One. Two.", "") == ["One.", "Two."]
    for bad in (("   ", ""), ("", "x"), ("One. Two.", "own translation")):
        with pytest.raises(ValueError):
            validate_request(*bad)
    loaded = []

    def get_pipeline(pair):
        loaded.append(pair)                                   # must never be reached for invalid input
        raise AssertionError("pipeline requested for an invalid request")

    app = build_app(get_pipeline, ["en-hi"], "en-hi")
    handler = next(f.fn for f in app.fns.values() if f.fn.__name__ == "run")
    for args in (("  ", "en-hi", True, ""), ("One. Two.", "en-hi", True, "mine")):
        out = handler(*args)
        assert len(out) == 7 and out[6].startswith("**Cannot run yet:**") and out[0] == [] and out[1] == []
    assert loaded == []


def test_highlight_labels_are_readable_and_match_the_colour_map():
    from mml.demo import TITLE_COLORS

    segs = retitle([("a", None), ("b", "ENTITY_NUMBER"), ("c", "edited")])
    assert segs == [("a", None), ("b", "Entity / number"), ("c", "edited")]
    assert set(TITLE_COLORS) == {"Omission", "Mistranslation", "Entity / number", "Hallucination"}


def test_app_overrides_gradios_mid_word_line_breaking():
    pytest.importorskip("gradio")
    app = build_app(lambda p: None, ["en-hi"], "en-hi")
    assert "word-break: normal" in app.css


def test_demo_disables_gradio_analytics_by_default():
    import os

    import mml.demo  # noqa: F401

    assert os.environ.get("GRADIO_ANALYTICS_ENABLED") == "False"


def test_status_messages_for_clean_unfixable_and_skipped_cases():
    clean = build_view([result("a b c.", "x y z.")], "constrained", {"cometkiwi": "no token"})
    assert not clean.flagged and "No meaning-loss problems detected" in clean.status and "cometkiwi" in clean.status
    unfixable = build_view([result("a b c.", "x y z.", before=det({"OMISSION"}))], "constrained")
    assert unfixable.flagged and "no automatic fix applied" in unfixable.status
    assert any(f[0] == "Omission" and f[1] == "sentence" for f in unfixable.findings)        # type flagged without a located span
    not_applied = build_view([result("a.", "b.", "c.", before=det({"ENTITY_NUMBER"}))], "full", applied=False)
    assert not_applied.corrected_text == "b."


def test_analyse_runs_the_pipeline_and_rejects_empty_input():
    class FakePipe:
        strategy = "constrained"
        detector = type("D", (), {"skipped": {}})()

        def run(self, sentences):
            return [result(s, s.upper()) for s in sentences]

    view = analyse(FakePipe(), "Hello world. Second sentence here.")
    assert [x["mt"] for x in view.per_sentence] == ["HELLO WORLD.", "SECOND SENTENCE HERE."]
    with pytest.raises(ValueError):
        analyse(FakePipe(), "   ")
    assert len(analyse(FakePipe(), "x. " * 5000).per_sentence) <= MAX_SENTENCES              # input is capped


# ------------------------------------------------------------------ app construction (no server)
def test_app_builds_and_its_handler_returns_all_outputs_in_order():
    gr = pytest.importorskip("gradio")

    class FakePipe:
        strategy = "constrained"
        detector = type("D", (), {"skipped": {}})()

        def run(self, sentences):
            return [result(s, "T:" + s) for s in sentences]

    asked = []

    def get_pipeline(pair):
        asked.append(pair)
        return FakePipe()

    app = build_app(get_pipeline, ["en-hi", "en-de"], "en-hi")
    assert isinstance(app, gr.Blocks)
    handler = next(f.fn for f in app.fns.values() if f.fn.__name__ == "run")
    out = handler("Hello.", "en-de", True)
    assert len(out) == 7 and asked == ["en-de"] and out[2][0][0] in ("Omission", "Mistranslation", "Entity / number", "Hallucination", "Any error")
    assert "".join(p for p, _ in out[1]) == "T:Hello." and out[3] == [["", "", "", "No problems detected."]]
    assert "Please enter some English text." in handler("  ", "en-de", True)[6]
    assert set(TYPE_COLORS) == {"OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"} and len(EXAMPLES) >= 3


# ------------------------------------------------------------------ resident signals
CREATED: list[str] = []


class Counting(Signal):
    name = "counting"

    def __init__(self, cfg, cache=None):
        super().__init__(cfg, cache)
        CREATED.append(type(self).__name__)
        self.model = object()                                   # stands in for a loaded model

    def signature(self):
        return {"v": 1}

    def _compute(self, pairs):
        return [SignalResult({"x": 1.0}) for _ in pairs]


class Unavailable(Counting):
    def check(self):
        raise SignalUnavailable("no token")


def test_resident_signals_keep_the_same_instance_between_calls_and_skip_unavailable(monkeypatch, tmp_path):
    from mml.detect import run as run_mod

    monkeypatch.setitem(run_mod.SIGNAL_CLASSES, "c1", Counting)
    monkeypatch.setitem(run_mod.SIGNAL_CLASSES, "c2", Unavailable)
    import mml.resident as res_mod

    monkeypatch.setattr(res_mod, "SIGNAL_CLASSES", run_mod.SIGNAL_CLASSES)
    CREATED.clear()
    cfg = load_config(overrides=[f"paths.cache_dir={tmp_path}"])
    runner = ResidentSignals(cfg)
    r1, s1 = runner(cfg, [("a", "b")], ["c1", "c2"])
    r2, _ = runner(cfg, [("c", "d")], ["c1", "c2"])
    assert sorted(CREATED) == ["Counting", "Unavailable"]        # one instance per signal, built once, not once per call
    assert set(r1) == {"c1"} and s1 == {"c2": "no token"} and r2["c1"][0].features == {"x": 1.0}
    assert runner._signals["c1"] is runner._get("c1")


def test_pipeline_object_is_usable_with_a_resident_runner_signature():
    cfg = load_config(pair="en-de")

    class D:
        skipped = {}

        def detect(self, pairs):
            return [det() for _ in pairs]

    pipe = CorrectionPipeline(cfg, D(), "constrained")
    [r] = pipe.run_pairs([("x", "y")])
    assert r.corrected == "y"
