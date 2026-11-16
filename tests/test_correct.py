from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mml.config import load_config
from mml.correct.compare import STRATEGIES, correct_all, score
from mml.correct.constrained import constrained_fix, render_number
from mml.correct.rerank import rerank_batch
from mml.correct.span_postedit import (
    MAX_PROBLEMS, CorrectionResult, baseline_postedit, build_baseline_prompt, build_problems, build_span_prompt,
    describe_span, edit_ratio, parse_translation, span_postedit,
)
from mml.detect.base import SignalResult, SpanFlag
from mml.detect.classifier import TARGETS, MeaningLossClassifier
from mml.detect.predict import Detection, Detector
from mml.llm import FakeBackend, LLMError, LLMRefused
from mml.synth.types import SynthExample

EN, HI, DE = "eng_Latn", "hin_Deva", "deu_Latn"
SRC = "The doctor said that the patient should take two tablets after each meal."
HYP = "Der Arzt sagte, dass der Patient zwei Tabletten einnehmen sollte."


def det(types, spans=(), p=0.9):
    return Detection({t: p for t in TARGETS}, set(types), list(spans))


# ------------------------------------------------------------------ prompts and parsing
def test_span_prompt_contains_source_translation_types_and_exact_flagged_text():
    start = SRC.index("after each meal")
    spans = [SpanFlag("src", start, start + len("after each meal"), "", "unaligned_src"),
             SpanFlag("tgt", 0, 3, "Der", "unaligned_tgt")]
    problems = build_problems(SRC, HYP, det({"OMISSION", "HALLUCINATION"}, spans))
    system, user = build_span_prompt(SRC, HYP, problems, EN, DE)
    assert "English into German" in system and "Fix only the listed problems" in system and "<translation>" in system
    assert SRC in user and HYP in user and "Detected problems:" in user
    assert "[OMISSION]" in user and "«after each meal»" in user          # text recovered from offsets, not from span.text
    assert "[HALLUCINATION]" in user and "«Der»" in user


def test_types_without_spans_get_a_generic_hint_and_unpredicted_types_are_not_mentioned():
    problems = build_problems(SRC, HYP, det({"MISTRANSLATION"}, [SpanFlag("tgt", 0, 3, "Der", "unaligned_tgt")]))
    assert len(problems) == 1 and problems[0].startswith("[MISTRANSLATION]")   # the hallucination span is not for a predicted type


def test_problem_list_is_capped_and_deduplicated():
    spans = [SpanFlag("tgt", i, i + 1, "x", "number_extra") for i in range(20)]
    assert len(build_problems(SRC, "x" * 30, det({"ENTITY_NUMBER"}, spans))) == 1          # identical wording is deduplicated
    spans = [SpanFlag("tgt", i, i + 2, f"n{i}", "number_extra") for i in range(20)]
    assert len(build_problems(SRC, "x" * 30, det({"ENTITY_NUMBER"}, spans))) == MAX_PROBLEMS


def test_only_the_longest_unaligned_runs_are_reported_but_all_rule_findings_are_kept():
    hyp = "a b c d e f g h i j k l m n o p"
    spans = [SpanFlag("tgt", 0, 3, "", "unaligned_tgt"), SpanFlag("tgt", 4, 11, "", "unaligned_tgt"),
             SpanFlag("tgt", 12, 15, "", "unaligned_tgt"), SpanFlag("tgt", 20, 31, "", "unaligned_tgt"),
             SpanFlag("tgt", 0, 1, "a", "number_extra"), SpanFlag("tgt", 2, 3, "b", "number_extra")]
    problems = build_problems(SRC, hyp, det({"HALLUCINATION", "ENTITY_NUMBER"}, spans))
    unaligned = [p for p in problems if p.startswith("[HALLUCINATION]")]
    assert len(unaligned) == 2 and "«d e f g h i j»".replace("d e f g h i j", hyp[4:11]) in unaligned[0] + unaligned[1]
    assert len([p for p in problems if p.startswith("[ENTITY_NUMBER]")]) == 2       # precise findings are never dropped


def test_every_signal_span_kind_has_wording_or_is_deliberately_skipped():
    kinds = ["unaligned_src", "unaligned_tgt", "number_missing", "number_extra", "date_missing", "date_extra",
             "entity_missing", "lex_entity_missing", "lex_entity_extra", "lex_unit_missing", "lex_unit_extra",
             "negation_missing", "negation_extra"]
    for k in kinds:
        assert describe_span(SpanFlag("tgt", 0, 2, "ab", k), "ab", "ab")
    assert describe_span(SpanFlag("tgt", 0, 2, "ab", "xcomet_major"), "ab", "ab") is None      # type-agnostic: no wording
    assert describe_span(SpanFlag("tgt", 0, 0, "", "number_extra"), "", "") is None               # empty text


def test_baseline_prompt_has_no_detector_information():
    system, user = build_baseline_prompt(SRC, HYP, EN, DE)
    assert "PROBLEMS" not in system and "Detected" not in user and SRC in user and HYP in user


def test_parse_and_edit_ratio():
    assert parse_translation("blah <translation> Hallo Welt </translation>") == "Hallo Welt"
    assert parse_translation("<TRANSLATION>a</TRANSLATION> then <translation>b</translation>") == "b"
    assert parse_translation("no tags") is None and parse_translation("<translation>  </translation>") is None
    assert edit_ratio("abc", "abc") == 0.0 and edit_ratio("abc", "xyz") == 1.0 and 0 < edit_ratio("hello world", "hello word") < 0.2


# ------------------------------------------------------------------ span-guided and baseline post-editing
def test_unflagged_sentences_never_reach_the_llm():
    llm = FakeBackend(lambda s, u: "<translation>CHANGED</translation>")
    r = span_postedit(llm, SRC, HYP, det(set()), EN, DE)
    assert r.status == "not_flagged" and r.text == HYP and llm.calls == []


def test_flagged_sentence_is_corrected_and_prompt_carries_the_findings():
    fixed = HYP.replace("einnehmen", "nach jeder Mahlzeit einnehmen")
    llm = FakeBackend(lambda s, u: f"<translation>{fixed}</translation>")
    r = span_postedit(llm, SRC, HYP, det({"OMISSION"}), EN, DE)
    assert r.status == "corrected" and r.text == fixed and 0 < r.edit_ratio < 0.6
    assert "[OMISSION]" in llm.calls[0][1] and r.problems


def test_safeguards_reject_rewrites_unparsable_output_and_llm_failures():
    rewrite = FakeBackend(lambda s, u: "<translation>Ein völlig anderer Satz über das Wetter heute.</translation>")
    r = span_postedit(rewrite, SRC, HYP, det({"MISTRANSLATION"}), EN, DE)
    assert r.status == "rejected_edit_ratio" and r.text == HYP and r.edit_ratio > 0.6
    assert span_postedit(FakeBackend(lambda s, u: "I think it is fine"), SRC, HYP, det({"OMISSION"}), EN, DE).status == "unparsable"
    same = FakeBackend(lambda s, u: f"<translation>{HYP}</translation>")
    assert span_postedit(same, SRC, HYP, det({"OMISSION"}), EN, DE).status == "unchanged_by_llm"

    def refuse(s, u):
        raise LLMRefused("no")

    def fail(s, u):
        raise LLMError("boom")

    assert span_postedit(FakeBackend(refuse), SRC, HYP, det({"OMISSION"}), EN, DE).status == "llm_refused"
    r = span_postedit(FakeBackend(fail), SRC, HYP, det({"OMISSION"}), EN, DE)
    assert r.status == "llm_error" and r.text == HYP                           # a failure never loses the translation


def test_baseline_sends_every_sentence_to_the_llm_even_if_it_looks_fine():
    llm = FakeBackend(lambda s, u: f"<translation>{HYP}</translation>")
    r1 = baseline_postedit(llm, SRC, HYP, EN, DE)
    baseline_postedit(llm, SRC, "Ein anderer Satz.", EN, DE)
    assert len(llm.calls) == 2 and r1.status == "unchanged_by_llm" and r1.text == HYP


# ------------------------------------------------------------------ constrained fixing
def test_render_number_follows_target_conventions():
    from decimal import Decimal

    assert render_number("4.5", frozenset({Decimal("4.5")}), DE) == "4,5"
    assert render_number("2,500", frozenset(), DE) == "2.500" and render_number("2,500", frozenset(), HI) == "2,500"
    assert render_number("9:30", frozenset(), DE) == "9:30"
    assert render_number("twenty", frozenset({Decimal(20)}), HI) == "20"


@pytest.mark.parametrize("src,hyp,tgt,expected", [
    ("The firm made 4.5 million dollars.", "Die Firma verdiente 45 Millionen Dollar.", DE, "Die Firma verdiente 4,5 Millionen Dollar."),
    ("The bridge is 1.2 kilometres long.", "नया पुल 12 किलोमीटर लंबा है।", HI, "नया पुल 1.2 किलोमीटर लंबा है।"),
    ("The company reported a profit of 45 million dollars.", "कंपनी ने मिलियन डॉलर का मुनाफा दर्ज किया।", HI,
     "कंपनी ने 45 मिलियन डॉलर का मुनाफा दर्ज किया।"),
    ("On Monday he left.", "बुधवार को वह चला गया।", HI, "सोमवार को वह चला गया।"),
    ("Narendra Modi visited Germany.", "नरेंद्र मोदी ने फ्रांस का दौरा किया।", HI, "नरेंद्र मोदी ने जर्मनी का दौरा किया।"),
    ("The road is 45 kilometres long.", "Die Straße ist 45 Meter lang.", DE, "Die Straße ist 45 Kilometer lang."),
    ("They visited Munich and Vienna.", "Sie besuchten Hamburg und Wien.", DE, "Sie besuchten München und Wien."),
])
def test_constrained_fixes(src, hyp, tgt, expected):
    r = constrained_fix(src, hyp, EN, tgt)
    assert r.text == expected and r.changed and r.fixes


def test_repeated_numbers_are_matched_in_order_not_greedily():
    # source "1. ... 3." but translation "3. ... 3.": the FIRST number is wrong (greedy matching would swap them)
    r = constrained_fix("The 1st and the 3rd regiment.", "Das 3. und das 3. Regiment.", EN, DE)
    assert r.text == "Das 1. und das 3. Regiment."
    r = constrained_fix("Winds of 100-200 miles per hour.", "Winde von 200-200 Meilen pro Stunde.", EN, DE)
    assert r.text == "Winde von 100-200 Meilen pro Stunde."
    r = constrained_fix("Winds of 100-200 miles per hour.", "Winde von 100-100 Meilen pro Stunde.", EN, DE)
    assert r.text == "Winde von 100-200 Meilen pro Stunde."


@pytest.mark.parametrize("src,hyp", [
    ("The 802.11n standard operates on both the 2.4Ghz and 5.0Ghz frequencies.",
     "Der 802.11n-Standard funktioniert sowohl auf den 2,4 GHz als auch auf den 5,0 GHz-Frequenzen."),
    ("This allows backwards compatibility with 802.11a, 802.11b and 802.11g.",
     "Dies ermöglicht Kompatibilität mit 802.11a, 802.11b und 802.11g."),
    ("The group presented its report at 12.00 GMT today.", "Die Gruppe hat heute um 12.00 Uhr GMT ihren Bericht vorgestellt."),
])
def test_numbers_already_written_exactly_as_in_the_source_are_never_rewritten(src, hyp):
    """Regression: found on real German MT output - a '.' in version numbers/times was 'fixed' into a decimal comma."""
    r = constrained_fix(src, hyp, EN, DE)
    assert r.text == hyp and not r.changed


def test_ordered_match_basic():
    from mml.correct.constrained import ordered_match
    from mml.detect.rules import find_numbers

    s, t = find_numbers("5 and 7 and 9", EN), find_numbers("5 und 9 und 8", "deu_Latn")
    missing, extra = ordered_match(s, t)
    assert [m.text for m in missing] == ["7"] and [e.text for e in extra] == ["8"]
    assert ordered_match([], []) == ([], [])


def test_constrained_leaves_correct_and_unfixable_sentences_alone_and_is_idempotent():
    ok = constrained_fix("The firm made 4.5 million dollars.", "Die Firma verdiente 4,5 Millionen Dollar.", EN, DE)
    assert not ok.changed and ok.text == "Die Firma verdiente 4,5 Millionen Dollar."
    # an extra number with no source counterpart is reported, not "fixed"
    extra = constrained_fix("They came.", "Sie kamen 7 Mal.", EN, DE)
    assert not extra.changed and extra.unresolved
    once = constrained_fix("The firm made 4.5 million dollars.", "Die Firma verdiente 45 Millionen Dollar.", EN, DE)
    twice = constrained_fix("The firm made 4.5 million dollars.", once.text, EN, DE)
    assert not twice.changed and twice.text == once.text


def test_constrained_fixes_are_verified_by_the_rules():
    r = constrained_fix("The firm made 4.5 million dollars.", "Die Firma verdiente 45 Millionen Dollar.", EN, DE)
    from mml.detect.rules import check_pair

    after = check_pair("The firm made 4.5 million dollars.", r.text, EN, DE)
    assert after.features["num_missing"] == 0 and after.features["num_extra"] == 0


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"


@pytest.mark.skipif(not (DATA / "en-hi" / "val.jsonl").exists(), reason="synthetic data not generated")
@pytest.mark.parametrize("pair,tgt", [("en-hi", HI), ("en-de", DE)])
def test_constrained_restores_most_rule_detectable_entity_number_errors(pair, tgt):
    from mml.synth.build import load_split

    rows = [e for e in load_split(DATA / pair, "train") if e.label == "ENTITY_NUMBER"
            and e.subtype in ("number_change", "date_change", "unit_change", "entity_replace")]
    restored = sum(constrained_fix(e.src, e.hyp, EN, tgt).text == e.ref for e in rows)
    assert len(rows) > 60 and restored / len(rows) > 0.7, (restored, len(rows))
    clean = [e for e in load_split(DATA / pair, "train") if e.label == "OK"][:300]
    assert sum(constrained_fix(e.src, e.hyp, EN, tgt).changed for e in clean) / len(clean) < 0.03   # little collateral damage


# ------------------------------------------------------------------ re-ranking
class StubDetector:
    def __init__(self, table):
        self.table, self.calls = table, 0

    def error_probability(self, pairs):
        self.calls += 1
        return np.array([self.table[c] for _, c in pairs])


def test_rerank_picks_lowest_error_probability_and_scores_unique_pairs_once():
    det_ = StubDetector({"a": 0.9, "b": 0.2, "c": 0.5, "d": 0.1})
    res = rerank_batch(det_, ["s1", "s2"], [["a", "b", "b", "c"], ["d", "a"]])
    assert [r.best_idx for r in res] == [1, 0] and res[0].scores == [0.9, 0.2, 0.2, 0.5]
    assert det_.calls == 1


def test_margin_keeps_the_mt_first_choice_unless_the_detector_is_clearly_better():
    from mml.correct.rerank import pick

    assert pick([0.50, 0.20]) == 1 and pick([0.50, 0.45], margin=0.1) == 0 and pick([0.50, 0.20], margin=0.1) == 1
    assert pick([0.10, 0.20, 0.05], margin=0.2) == 0                   # best alternative only 0.05 better than top-1


def test_rank_penalty_prefers_mt_order_among_near_ties():
    det_ = StubDetector({"a": 0.30, "b": 0.29})
    assert rerank_batch(det_, ["s"], [["a", "b"]])[0].best_idx == 1
    assert rerank_batch(det_, ["s"], [["a", "b"]], rank_penalty=0.05)[0].best_idx == 0
    with pytest.raises(ValueError):
        rerank_batch(det_, ["s"], [[]])


# ------------------------------------------------------------------ comparison harness
def ex(label, hyp, ref, id_="x"):
    return SynthExample(id_, "en-de", "val", "dev", SRC, ref, hyp, label)


def test_score_arithmetic_and_over_correction():
    ref = "Der Arzt sagte es."
    examples = [ex("OMISSION", "Der Arzt sagte.", ref, "1"), ex("OK", ref, ref, "2"), ex("OK", ref, ref, "3")]
    results = [CorrectionResult(ref, "corrected"), CorrectionResult(ref, "not_flagged"), CorrectionResult("Der Arzt meinte es.", "corrected")]
    m = score(examples, results)
    assert m["errors"]["exact_restored"] == 1.0 and m["errors"]["chrf_gain"] > 0
    assert m["clean"]["over_correction_rate"] == 0.5 and m["status"] == {"corrected": 2, "not_flagged": 1}


def test_strategies_run_and_full_skips_the_llm_when_rules_resolve_everything():
    cfg_ = load_config(pair="en-de")
    src = "The firm made 4.5 million dollars."
    bad = SynthExample("a", "en-de", "val", "dev", src, "Die Firma verdiente 4,5 Millionen Dollar.",
                       "Die Firma verdiente 45 Millionen Dollar.", "ENTITY_NUMBER")
    flagged = det({"ENTITY_NUMBER"})
    llm = FakeBackend(lambda s, u: "<translation>SHOULD NOT BE USED</translation>")
    [res] = correct_all("full", [bad], [flagged], EN, DE, llm)
    assert res.text == bad.ref and llm.calls == []                           # fixed by rules alone: no LLM call
    [raw] = correct_all("raw", [bad], [flagged], EN, DE, None)
    assert raw.text == bad.hyp
    [con] = correct_all("constrained", [bad], [flagged], EN, DE, None)
    assert con.text == bad.ref
    with pytest.raises(Exception):
        correct_all("span_llm", [bad], [flagged], EN, DE, None)               # LLM strategies need a backend
    assert set(STRATEGIES) == {"raw", "constrained", "baseline_llm", "span_llm", "full"} and cfg_.pair_key == "en-de"


def test_span_and_baseline_strategies_differ_in_which_sentences_cost_a_call():
    exs = [ex("OK", HYP, HYP, "1"), ex("OK", HYP, HYP, "2")]
    dets = [det(set()), det({"OMISSION"})]
    llm_s = FakeBackend(lambda s, u: f"<translation>{HYP}</translation>")
    correct_all("span_llm", exs, dets, EN, DE, llm_s, workers=1)
    llm_b = FakeBackend(lambda s, u: f"<translation>{HYP}</translation>")
    correct_all("baseline_llm", exs, dets, EN, DE, llm_b, workers=1)
    assert (len(llm_s.calls), len(llm_b.calls)) == (1, 2)


# ------------------------------------------------------------------ Detector wrapper
def test_detector_thresholds_probabilities_and_orders_columns():
    rng = np.random.default_rng(0)
    types = ["OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"]
    labels = ["OK"] * 40 + [t for t in types for _ in range(10)]
    X = rng.normal(size=(len(labels), 2)) * 0.1
    X[40:, 0] += 3.0                                   # f_a separates errors from OK; f_b is noise
    model = MeaningLossClassifier("logreg", {"C": 1.0}, ["f_a", "f_b"]).fit(X, labels)

    def runner(cfg, pairs, signals):                   # stub signal: f_a = word count of the translation
        return {"labse": [SignalResult({"f_b": 0.0, "f_a": 3.0 * float(len(h.split()) > 0)}, [SpanFlag("tgt", 0, 1, "x", "unaligned_tgt")])
                          for _, h in pairs]}, {"cometkiwi": "skipped"}

    d = Detector(load_config(), model=model, columns=["f_a", "f_b"], signal_runner=runner)   # model columns ordered f_a, f_b
    out = d.detect([("s", ""), ("s", "some words")])
    assert out[0].probs["ANY"] < 0.5 < out[1].probs["ANY"] and out[1].any_error and not out[0].any_error
    assert d.skipped == {"cometkiwi": "skipped"} and out[1].spans[0].kind == "unaligned_tgt"
    assert out[1].spans_for({"HALLUCINATION"}) and not out[1].spans_for({"OMISSION"})
    assert d.detect([]) == []


def test_detector_refuses_when_a_required_signal_was_skipped():
    model = MeaningLossClassifier("logreg", {}, ["kiwi_score"])
    d = Detector(load_config(), model=model, columns=["kiwi_score"], signal_runner=lambda c, p, s: ({}, {"cometkiwi": "no token"}))
    with pytest.raises(ValueError, match="unavailable"):
        d.detect([("a", "b")])
