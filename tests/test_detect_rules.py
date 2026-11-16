from __future__ import annotations

from decimal import Decimal

from mml.detect.rules import (
    check_pair, find_dates, find_numbers, match_numbers, negation_cues, number_values, source_entities,
)

EN, HI, DE = "eng_Latn", "hin_Deva", "deu_Latn"


def feats(src, hyp, tgt):
    return check_pair(src, hyp, EN, tgt)


def kinds(res):
    return {(s.side, s.kind, s.text) for s in res.spans}


# ------------------------------------------------------------------ numbers
def test_number_values_follow_language_conventions():
    assert Decimal("2500") in number_values("2,500", ".")           # English / Indian grouping
    assert Decimal("2500") in number_values("2.500", ",")           # German thousands dot
    assert Decimal("4.5") in number_values("4,5", ",")              # German decimal comma
    assert Decimal("4.5") in number_values("4.5", ".")
    assert Decimal("45") not in number_values("4,5", ",")
    assert Decimal("100000") in number_values("1,00,000", ".")      # Indian lakh grouping
    assert Decimal("45") in number_values("४५", ".")                  # Devanagari digits
    assert ("t", 9, 30) in number_values("9:30", ".") and Decimal("930") in number_values("9:30", ".")
    assert ("t", 2, 30) in number_values("0230", ".")               # "0230 UTC" == "02:30"


def test_matching_ignores_formatting_but_catches_value_changes():
    ok = feats("The firm made 4.5 million dollars on 12 May.", "Die Firma verdiente 4,5 Millionen Dollar am 12. Mai.", DE)
    assert ok.features["num_missing"] == 0 and ok.features["num_extra"] == 0
    bad = feats("The firm made 4.5 million dollars.", "Die Firma verdiente 45 Millionen Dollar.", DE)
    assert bad.features["num_missing"] == 1 and bad.features["num_extra"] == 1
    assert kinds(bad) >= {("src", "number_missing", "4.5"), ("tgt", "number_extra", "45")}


def test_number_spans_point_at_the_right_characters():
    src, hyp = "He paid 45 dollars.", "Er zahlte 450 Dollar."
    res = feats(src, hyp, DE)
    by_kind = {s.kind: s for s in res.spans}
    assert src[by_kind["number_missing"].start:by_kind["number_missing"].end] == "45"
    assert hyp[by_kind["number_extra"].start:by_kind["number_extra"].end] == "450"


def test_number_words_match_digits_and_words_in_the_target():
    res = feats("The doctor prescribed two tablets and twenty-five drops.",
                "डॉक्टर ने दो गोलियाँ और 25 बूँदें बताईं।", HI)
    assert res.features["num_missing"] == 0 and res.features["num_extra"] == 0
    res = feats("He bought three apples.", "Er kaufte drei Äpfel.", DE)
    assert res.features["num_missing"] == 0
    res = feats("He bought three apples.", "Er kaufte Äpfel.", DE)
    assert res.features["num_missing"] == 1


def test_missing_number_deleted_from_translation():
    res = feats("The bridge is 1.2 kilometres long.", "नया पुल किलोमीटर लंबा है।", HI)
    assert res.features["num_missing"] == 1 and res.features["num_extra"] == 0


def test_target_words_alone_are_never_reported_as_extra():
    res = feats("They came.", "वे दो बार आए।", HI)
    assert res.features["num_extra"] == 0


# ------------------------------------------------------------------ dates
def test_months_and_weekdays_are_compared_across_languages():
    assert feats("On Monday, 12 May he left.", "Am Montag, dem 12. Mai, reiste er ab.", DE).features["date_missing"] == 0
    r = feats("On Monday he left.", "Am Mittwoch reiste er ab.", DE)
    assert r.features["date_missing"] == 1 and r.features["date_extra"] == 1
    r = feats("He left in March.", "वह सितंबर में गया।", HI)
    assert r.features["date_missing"] == 1 and r.features["date_extra"] == 1


def test_may_the_verb_is_not_a_month():
    assert find_dates("It may rain tomorrow, they said.", EN) == []
    assert [d[2] for d in find_dates("He left on 3 May.", EN)] == ["May"]


# ------------------------------------------------------------------ negation
def test_negation_counts_and_parity():
    ok = feats("She did not attend the meeting.", "Sie nahm nicht an der Sitzung teil.", DE)
    assert ok.features["neg_src"] == 1 and ok.features["neg_tgt"] == 1 and ok.features["neg_parity_mismatch"] == 0
    flipped = feats("She did not attend the meeting.", "Sie nahm an der Sitzung teil.", DE)
    assert flipped.features["neg_parity_mismatch"] == 1 and flipped.features["neg_diff"] == -1
    assert ("src", "negation_missing", "not") in kinds(flipped)
    added = feats("She attended the meeting.", "वह बैठक में नहीं गई।", HI)
    assert added.features["neg_parity_mismatch"] == 1 and ("tgt", "negation_extra", "नहीं") in kinds(added)


def test_english_contractions_and_hindi_danda():
    assert len(negation_cues("He isn't here and can’t stay.", EN)) == 2
    assert [c[2] for c in negation_cues("वह नहीं आया।", HI)] == ["नहीं"]


def test_double_negation_keeps_parity():
    r = feats("Nobody did not know.", "Niemand wusste es nicht.", DE)
    assert r.features["neg_parity_mismatch"] == 0


# ------------------------------------------------------------------ entities
def test_source_entities_heuristics():
    ents = [e[2] for e in source_entities("Prime Minister Narendra Modi met Angela Merkel in Berlin on Monday.")]
    assert "Narendra Modi" in " ".join(ents) or "Prime Minister Narendra Modi" in ents
    assert "Berlin" in ents and "Monday" not in ents
    texts = [e[2] for e in source_entities("The Greek team and the Olympic flag, said Mr Reid, arrived in 1970s Earth.")]
    assert "Mr Reid" in texts and not {"Greek", "Olympic", "Earth", "1970s"} & set(texts)
    assert [e[2] for e in source_entities("It was flown on the JAS 39C Gripen.")] == ["JAS 39C Gripen"]


def test_latin_target_missing_entity_with_exonym_tolerance():
    ok = feats("They visited Munich and Vienna.", "Sie besuchten München und Wien.", DE)
    assert ok.features["ent_missing"] == 0
    bad = feats("They visited Munich and Vienna.", "Sie besuchten Hamburg und Wien.", DE)
    assert bad.features["ent_missing"] == 1 and ("src", "entity_missing", "Munich") in kinds(bad)


def test_non_latin_target_only_checks_preservable_entities():
    r = feats("The JAS 39C Gripen crashed near Stockholm.", "JAS 39C Gripen स्टॉकहोम के पास दुर्घटनाग्रस्त हो गया।", HI)
    assert r.features["ent_src"] == 1 and r.features["ent_missing"] == 0
    r = feats("The JAS 39C Gripen crashed near Stockholm.", "विमान स्टॉकहोम के पास दुर्घटनाग्रस्त हो गया।", HI)
    assert r.features["ent_missing"] == 1


def test_lexicon_entity_replacement_and_deletion():
    ok = feats("Narendra Modi visited Germany.", "नरेंद्र मोदी ने जर्मनी का दौरा किया।", HI)
    assert ok.features["lex_ent_missing"] == 0 and ok.features["lex_ent_extra"] == 0
    swapped = feats("Narendra Modi visited Germany.", "नरेंद्र मोदी ने फ्रांस का दौरा किया।", HI)
    assert swapped.features["lex_ent_missing"] == 1 and swapped.features["lex_ent_extra"] == 1
    assert ("tgt", "lex_entity_extra", "फ्रांस") in kinds(swapped)
    surname = feats("Mr Modi visited Germany.", "नरेंद्र मोदी ने जर्मनी का दौरा किया।", HI)
    assert surname.features["lex_ent_extra"] == 0                   # "Modi" accepts "Narendra Modi"


def test_unit_lexicon():
    ok = feats("The road is 45 km long.", "Die Straße ist 45 Kilometer lang.", DE)
    assert ok.features["lex_unit_missing"] == 0 and ok.features["lex_unit_extra"] == 0
    bad = feats("The road is 45 kilometres long.", "सड़क 45 मीटर लंबी है।", HI)
    assert bad.features["lex_unit_missing"] == 1 and bad.features["lex_unit_extra"] == 1


def test_clean_pair_has_no_flags_and_features_are_numeric():
    r = feats("The museum is open every day except Monday.", "Das Museum ist jeden Tag außer Montag geöffnet.", DE)
    assert r.spans == [] or all(s.kind.startswith("entity") for s in r.spans)
    assert all(isinstance(v, float) for v in r.features.values())
    assert match_numbers([], []) == ([], []) and find_numbers("", EN) == []


def test_german_units_do_not_match_inside_longer_units_or_words():
    # "Kilometer" must not also count as "Meter"; "Kilogramm" not as "Gramm"; "Minister" not as "min"
    r = feats("The minister walked 3 kilograms of rice.", "Der Minister trug 3 Kilogramm Reis.", DE)
    assert r.features["lex_unit_missing"] == 0 and r.features["lex_unit_extra"] == 0
