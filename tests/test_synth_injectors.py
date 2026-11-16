"""Injector tests: hand-written fixtures per error type/language, plus a property test over real FLORES data.

The central invariant for every injected example (checked on all of FLORES dev): the labelled spans
reproduce the corrupted text exactly and ``invert_spans(hyp, spans) == ref``.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import pytest

from mml.synth.entities import GazetteerFinder
from mml.synth.inject_errors import ErrorInjector, mutate_number
from mml.synth.lang import DE, HI, get_resources
from mml.synth.text import chunks
from mml.synth.types import ERROR_TYPES, ErrorType, apply_edits, invert_spans

FLORES = Path(__file__).resolve().parents[1] / "data" / "raw" / "flores200" / "dev"


def rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def check_span_invariants(ref: str, hyp: str, spans) -> None:
    assert hyp != ref
    assert invert_spans(hyp, spans) == ref
    for s in spans:
        assert hyp[s.tgt_start:s.tgt_end] == s.tgt_text
        assert 0 <= s.edit_start <= s.tgt_start <= s.tgt_end <= s.edit_end <= len(hyp)
        assert ref.count(s.original_text) >= 1 or s.original_text == ""


# ----------------------------------------------------------------------------- OMISSION
HI_OMIT = "सरकार ने कहा कि नई योजना अगले साल शुरू होगी, और इससे लाखों किसानों को सीधा फायदा मिलेगा।"
DE_OMIT = "Die Regierung kündigte an, dass der neue Plan im nächsten Jahr beginnt, und viele Bauern profitieren davon."


@pytest.mark.parametrize("lang,ref", [(HI, HI_OMIT), (DE, DE_OMIT)])
def test_omission_deletes_text_and_records_it(lang, ref):
    for seed in range(20):
        hyp, spans = ErrorInjector(lang).inject(ref, ErrorType.OMISSION, rng(seed))
        (s,) = spans
        check_span_invariants(ref, hyp, spans)
        assert s.tgt_start == s.tgt_end and s.tgt_text == ""         # zero-width at the deletion point
        assert len(hyp) < len(ref) and len(s.original_text.split()) >= 2
        assert s.type == "OMISSION" and s.subtype.startswith("omit_")
        assert ref[s.edit_start:s.edit_start + len(s.original_text)] == s.original_text


def test_omission_never_deletes_numbers_entities_or_negation():
    ref = "Am 12. Mai besuchte Angela Merkel nicht die Stadt Berlin, und sie blieb dort 3 Tage lang."
    inj = ErrorInjector(DE)
    for seed in range(60):
        res = inj.inject(ref, ErrorType.OMISSION, rng(seed))
        if res is None:
            continue
        _, (s,) = res
        assert not re.search(r"\d|nicht|Merkel|Berlin|Mai", s.original_text), s.original_text


def test_omission_not_applicable_to_very_short_sentences():
    assert ErrorInjector(HI).inject("वह घर गया।", ErrorType.OMISSION, rng()) is None


# ----------------------------------------------------------------------------- MISTRANSLATION
def test_hindi_negation_removed_exactly_once():
    ref = "वह बैठक में नहीं गई क्योंकि वह बीमार थी।"
    hyp, (s,) = ErrorInjector(HI).inject(ref, ErrorType.MISTRANSLATION, rng())
    check_span_invariants(ref, hyp, [s])
    assert s.subtype == "negation_remove" and s.original_text.strip() == "नहीं"
    assert "नहीं" not in hyp.split() and hyp == "वह बैठक में गई क्योंकि वह बीमार थी।"


def test_hindi_negation_inserted_in_verb_group():
    ref = "आज की बैठक में कई महत्वपूर्ण फैसले लिए जा सकते हैं।"
    hyp, (s,) = ErrorInjector(HI).inject(ref, ErrorType.MISTRANSLATION, rng())
    assert s.subtype == "negation_insert" and s.tgt_text == "नहीं"
    assert hyp.split().count("नहीं") == 1
    assert hyp.endswith("सकते हैं।") and "लिए जा नहीं सकते हैं।" in hyp.replace("जा नहीं", "जा नहीं")
    check_span_invariants(ref, hyp, [s])


def test_hindi_negation_not_removed_in_nahin_to():
    ref = "तुम समय पर आओ, नहीं तो हम चले जाएंगे और वह बहुत नाराज़ होगा।"
    inj = ErrorInjector(HI)
    for seed in range(10):
        hyp, (s,) = inj.inject(ref, ErrorType.MISTRANSLATION, rng(seed))
        assert s.subtype == "negation_insert"          # "नहीं तो" ("otherwise") is not a negation to delete
        check_span_invariants(ref, hyp, [s])


def test_german_nicht_removed_and_kein_replaced():
    ref1 = "Sie nahm nicht an der Sitzung teil, weil sie krank war."
    hyp, (s,) = ErrorInjector(DE).inject(ref1, ErrorType.MISTRANSLATION, rng())
    assert s.subtype == "negation_remove" and "nicht" not in hyp
    check_span_invariants(ref1, hyp, [s])
    ref2 = "Er hat keine Zeit für diese Aufgabe, aber er versucht es trotzdem."
    hyp, (s,) = ErrorInjector(DE).inject(ref2, ErrorType.MISTRANSLATION, rng())
    assert s.subtype == "negation_swap" and (s.original_text, s.tgt_text) == ("keine", "eine")
    assert hyp.startswith("Er hat eine Zeit")
    check_span_invariants(ref2, hyp, [s])


def test_german_nicht_inserted_after_finite_verb():
    ref = "Der Zug ist pünktlich in Berlin angekommen."
    hyp, (s,) = ErrorInjector(DE).inject(ref, ErrorType.MISTRANSLATION, rng())
    assert s.subtype == "negation_insert" and hyp == "Der Zug ist nicht pünktlich in Berlin angekommen."
    check_span_invariants(ref, hyp, [s])


def test_german_sondern_sentences_are_left_alone():
    assert ErrorInjector(DE).inject("Er kam nicht heute, sondern gestern am späten Abend an.",
                                    ErrorType.MISTRANSLATION, rng()) is None


def test_role_swap_exchanges_two_same_type_entities():
    ref = "बराक ओबामा ने पेरिस में नरेंद्र मोदी से मुलाकात की।"
    inj = ErrorInjector(HI)
    edits = inj._role_swap(ref, rng())
    assert edits and {e.subtype for e in edits} == {"role_swap"} and len({e.group for e in edits}) == 1
    hyp, spans = apply_edits(ref, edits, ErrorType.MISTRANSLATION)
    assert hyp == "नरेंद्र मोदी ने पेरिस में बराक ओबामा से मुलाकात की।"
    assert [s.tgt_text for s in spans] == ["नरेंद्र मोदी", "बराक ओबामा"]
    assert sorted(hyp.split()) == sorted(ref.split())            # same words, different roles
    check_span_invariants(ref, hyp, spans)


def test_role_swap_skips_symmetric_coordination():
    # "Apple und Google": swapping would not change the meaning, so it must not be used as an error
    assert ErrorInjector(DE)._role_swap("Apple und Google kündigten eine Partnerschaft an.", rng()) is None


# ----------------------------------------------------------------------------- ENTITY / NUMBER
def test_number_change_preserves_format_and_changes_value():
    for seed in range(40):
        for num in ("45", "4,5", "2,500", "9:30", "1947", "0230", "7"):
            new = mutate_number(num, rng(seed))
            assert new and new != num
            assert re.sub(r"\d", "0", new) == re.sub(r"\d", "0", num) or len(new) == len(num) + 1 or len(new) == len(num) - 1
            if ":" in new:
                h, m = new.split(":")
                assert int(h) < 24 and int(m) < 60
            if num == "1947":
                assert new[:2] == "19"                           # years change only in the last two digits
            assert not (len(new) > 1 and new[0] == "0" and num[0] != "0")


def test_german_decimal_comma_survives_number_change():
    ref = "Das Unternehmen erzielte im dritten Quartal einen Gewinn von 4,5 Millionen Dollar."
    inj = ErrorInjector(DE)
    seen = set()
    for seed in range(60):
        res = inj.inject(ref, ErrorType.ENTITY_NUMBER, rng(seed))
        hyp, (s,) = res
        check_span_invariants(ref, hyp, [s])
        seen.add(s.subtype)
        if s.subtype == "number_change":
            assert s.original_text == "4,5" and re.fullmatch(r"\d,\d", s.tgt_text) or len(s.tgt_text) in (3, 4)
        if s.subtype == "unit_change":
            assert (s.original_text, s.tgt_text) in {("Millionen", "Milliarden"), ("Dollar", "Euro")}
    assert {"number_change", "unit_change"} <= seen


def test_entity_replace_uses_same_type_entity_and_date_change_uses_other_day():
    ref = "Am Montag eröffnete das Museum in München seine neue Ausstellung."
    inj = ErrorInjector(DE)
    seen = {}
    for seed in range(80):
        hyp, (s,) = inj.inject(ref, ErrorType.ENTITY_NUMBER, rng(seed))
        check_span_invariants(ref, hyp, [s])
        seen[s.subtype] = s
    assert s.type == "ENTITY_NUMBER"
    ent = seen["entity_replace"]
    assert ent.original_text == "München" and ent.tgt_text in DE.gazetteer["LOC"] and ent.tgt_text != "München"
    day = seen["date_change"]
    assert day.original_text == "Montag" and day.tgt_text in DE.weekdays and day.tgt_text != "Montag"


def test_hindi_entity_number_covers_numbers_units_dates_entities():
    ref = "प्रधानमंत्री नरेंद्र मोदी ने 12 मई को जर्मनी में 45 किलोमीटर लंबी सड़क का उद्घाटन किया।"
    inj = ErrorInjector(HI)
    subtypes = set()
    for seed in range(200):
        hyp, (s,) = inj.inject(ref, ErrorType.ENTITY_NUMBER, rng(seed))
        check_span_invariants(ref, hyp, [s])
        subtypes.add(s.subtype)
    assert subtypes == {"number_change", "number_deleted", "date_change", "unit_change", "entity_replace",
                        "entity_deleted"}


def test_deleted_number_is_labelled_entity_number_not_omission():
    ref = "कंपनी ने तीसरी तिमाही में 45 मिलियन डॉलर का मुनाफा दर्ज किया।"
    inj = ErrorInjector(HI)
    for seed in range(100):
        hyp, (s,) = inj.inject(ref, ErrorType.ENTITY_NUMBER, rng(seed))
        if s.subtype == "number_deleted":
            assert s.type == "ENTITY_NUMBER" and s.original_text.strip() == "45" and "45" not in hyp
            assert s.tgt_start == s.tgt_end
            return
    pytest.fail("number_deleted never produced")


def test_entity_number_not_applicable_without_numbers_or_entities():
    assert ErrorInjector(HI).inject("यह एक बहुत सुंदर और शांत जगह है जहाँ सब खुश रहते हैं।",
                                    ErrorType.ENTITY_NUMBER, rng()) is None


# ----------------------------------------------------------------------------- HALLUCINATION
@pytest.mark.parametrize("lang,ref", [(HI, HI_OMIT), (DE, DE_OMIT)])
def test_hallucination_inserts_an_unrelated_filler_and_nothing_else(lang, ref):
    for seed in range(20):
        hyp, (s,) = ErrorInjector(lang).inject(ref, ErrorType.HALLUCINATION, rng(seed))
        check_span_invariants(ref, hyp, [s])
        assert s.original_text == "" and s.subtype in {"hallucination_append", "hallucination_prepend"}
        assert s.tgt_text.rstrip(".।|") in {f.rstrip(".।|") for f in lang.fillers}
        assert hyp.replace(s.tgt_text, "").split() == ref.split()    # the rest is the untouched reference


def test_hallucination_filler_mirrors_final_punctuation():
    ref = "वह बैठक में गई क्योंकि वह बीमार नहीं थी."
    hyp, (s,) = ErrorInjector(HI).inject(ref, ErrorType.HALLUCINATION, rng())
    assert s.tgt_text.endswith(".") and "।" not in s.tgt_text


def test_fillers_contain_no_numbers_or_negation():
    for lang in (HI, DE):
        for f in lang.fillers:
            assert not re.search(r"\d", f)
            assert not set(f.lower().split()) & ({"nicht", "kein", "keine", "nie"} | set(lang.neg_delete))


# ----------------------------------------------------------------------------- determinism / errors
def test_same_rng_seed_gives_same_output():
    inj = ErrorInjector(HI)
    for et in ERROR_TYPES:
        a = inj.inject(HI_OMIT + " 45 किलोमीटर", et, rng(5))
        b = inj.inject(HI_OMIT + " 45 किलोमीटर", et, rng(5))
        assert (a is None and b is None) or (a[0] == b[0] and a[1] == b[1])


def test_unsupported_language_has_helpful_error():
    with pytest.raises(KeyError, match="LangResources"):
        get_resources("tam_Taml")


def test_cannot_inject_ok_type():
    with pytest.raises(ValueError):
        ErrorInjector(HI).inject(HI_OMIT, ErrorType.OK, rng())


# ----------------------------------------------------------------------------- property test on real data
@pytest.mark.skipif(not FLORES.exists(), reason="FLORES-200 not downloaded (mml download-data --what flores)")
@pytest.mark.parametrize("flores_lang", ["hin_Deva", "deu_Latn"])
def test_span_invariants_hold_on_all_of_flores_dev(flores_lang):
    refs = (FLORES / f"{flores_lang}.dev").read_text(encoding="utf-8").splitlines()
    lang = get_resources(flores_lang)
    inj = ErrorInjector(lang)
    gaz = GazetteerFinder(lang.gazetteer)
    neg_words = set(lang.neg_delete) | set(lang.neg_replace)
    applicable = {et: 0 for et in ERROR_TYPES}
    for i, ref in enumerate(refs):
        for et in ERROR_TYPES:
            res = inj.inject(ref, et, random.Random(f"test|{i}|{et.value}"))
            if res is None:
                continue
            applicable[et] += 1
            hyp, spans = res
            check_span_invariants(ref, hyp, spans)                     # round trip + exact slices
            assert all(s.type == et.value for s in spans)
            if et is ErrorType.OMISSION:
                (s,) = spans
                assert len(hyp) < len(ref) and s.tgt_start == s.tgt_end
                assert not re.search(r"\d", s.original_text)
                assert not gaz.find(s.original_text)
                assert not set(w.strip(".,;:()").lower() for w in s.original_text.split()) & neg_words
            elif et is ErrorType.HALLUCINATION:
                (s,) = spans
                assert len(hyp) > len(ref) and s.original_text == ""
            elif et is ErrorType.MISTRANSLATION and spans[0].subtype != "role_swap":
                (s,) = spans
                before = sum(c.core.lower() in neg_words for c in chunks(ref))
                after = sum(c.core.lower() in neg_words for c in chunks(hyp))
                assert after == before + (1 if s.subtype == "negation_insert" else -1)
            elif et is ErrorType.ENTITY_NUMBER and spans[0].subtype == "number_change":
                (s,) = spans
                assert s.tgt_text != s.original_text and re.search(r"\d", s.tgt_text)
    n = len(refs)
    assert applicable[ErrorType.OMISSION] > 0.9 * n
    assert applicable[ErrorType.HALLUCINATION] > 0.9 * n
    assert applicable[ErrorType.MISTRANSLATION] > 0.6 * n
    assert applicable[ErrorType.ENTITY_NUMBER] > 0.2 * n
