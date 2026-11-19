from __future__ import annotations

import pytest

from mml.langmix import classify, output_issues, script_of, to_devanagari
from mml.normalize import Prepared, describe, prepare_sentences


@pytest.mark.parametrize("text,label", [
    ("Mera naam Rahul hai aur main kal Delhi jaa raha hoon.", "hinglish"),
    ("Please mujhe kal tak report bhej do, it is urgent.", "hinglish"),
    ("Kya haal hai?", "hinglish"),
    ("She did not attend the meeting because she was ill.", "english"),
    ("Prime Minister Narendra Modi visited Germany on 12 May 2022.", "english"),
    ("The log file is in the tab.", "english"),            # English words that are also in the lexicon (log, tab) are only weak support
    ("Ko Samui is a nice island.", "english"),
    ("वह बीमार है।", "hindi_script"),
    ("Meeting कल है, please confirm.", "mixed_script"),
    ("12345 !!!", "english"),
    ("这是一个测试", "other"),
])
def test_classify(text, label):
    assert classify(text).label == label


def test_single_strong_word_needs_enough_support():
    assert classify("The bhai next door is nice and tall and kind.").label == "english"      # 1 strong word out of 10
    assert classify("Hey bhai, nice").label == "hinglish"


def test_to_devanagari_converts_only_known_words_and_keeps_names_and_english():
    out = to_devanagari("Mera naam Rahul hai, report kal bhejna.")
    assert out == "मेरा नाम Rahul है, report कल भेजना."


def test_output_issues_devanagari_target():
    src = "My name is Rahul and the bridge is 5 km long."
    assert output_issues(src, "मेरा नाम Rahul है और पुल 5 km लंबा है।", "devanagari") == []          # name, number and unit are allowed
    bad = output_issues(src, "मेरा नाम है और kal bridge", "devanagari")
    assert {i.text for i in bad} == {"kal", "bridge"}
    assert any(i.kind == "copied_source" for i in output_issues("Mera naam Rahul hai", "Mera naam Rahul hai", "devanagari"))


def test_output_issues_latin_target():
    assert output_issues("x", "Der Zug fährt um 6 Uhr ab.", "latin") == []
    kinds = {i.kind for i in output_issues("x y z w", "Das ist कल gut aur", "latin")}
    assert {"wrong_script", "untranslated"} <= kinds


def test_script_of():
    assert script_of("hin_Deva") == "devanagari" and script_of("deu_Latn") == "latin"


# ------------------------------------------------------------------ normalisation
def test_english_passes_through_and_nothing_is_called():
    calls = []
    out = prepare_sentences(["Hello there."], pivot=lambda xs: calls.append(xs) or xs, llm=lambda t: calls.append(t) or t)
    assert out[0].method == "as-is" and not out[0].converted and calls == [] and describe(out[0]) == ""


def test_hinglish_goes_through_the_pivot_after_lexicon_transliteration():
    seen = []

    def pivot(xs):
        seen.extend(xs)
        return ["I will send the report tomorrow."]

    p = prepare_sentences(["Main kal report bhejunga aur kya hai."], pivot=pivot)[0]
    assert seen == ["मैं कल report bhejunga और क्या है."] and p.method == "mt-pivot" and p.english == "I will send the report tomorrow."
    assert "Approximate" in p.note and "Hinglish" in describe(p)


def test_llm_is_preferred_and_failures_fall_back_to_the_pivot():
    good = prepare_sentences(["Mera naam Rahul hai."], pivot=lambda xs: ["PIVOT"], llm=lambda t: "My name is Rahul.")[0]
    assert good.method == "llm" and good.english == "My name is Rahul." and good.note == ""

    def boom(_):
        raise RuntimeError("server down")

    assert prepare_sentences(["Mera naam Rahul hai."], pivot=lambda xs: ["My name is Rahul."], llm=boom)[0].method == "mt-pivot"
    still_hinglish = prepare_sentences(["Mera naam Rahul hai."], pivot=lambda xs: ["x"], llm=lambda t: "Mera naam hai")[0]
    assert still_hinglish.method == "mt-pivot"                                         # the LLM's non-English answer is rejected


def test_hindi_script_uses_the_pivot_as_is_and_unconvertible_input_is_reported():
    p = prepare_sentences(["वह बीमार है।"], pivot=lambda xs: [f"EN({xs[0]})"])[0]
    assert p.method == "mt-pivot" and p.english == "EN(वह बीमार है।)"
    assert "not clearly English" in p.note                                              # "EN(...)" still contains Devanagari
    none = prepare_sentences(["Kya haal hai?"])[0]
    assert none.method == "unconverted" and none.english == "Kya haal hai?" and "unreliable" in describe(none)
    other = prepare_sentences(["这是一个测试"], pivot=lambda xs: xs)[0]
    assert other.method == "unconverted" and "not supported" in other.note


def test_llm_rewriter_reads_tags_and_strips_quotes():
    from mml.llm import FakeBackend
    from mml.normalize import llm_rewriter

    rewrite = llm_rewriter(FakeBackend(lambda system, user: f'<english>"{user} (en)"</english> extra'))
    assert rewrite("Kya haal hai") == "Kya haal hai (en)"
    assert llm_rewriter(FakeBackend(lambda s, u: "plain answer"))("x") == "plain answer"
