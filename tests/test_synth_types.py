import pytest

from mml.synth.types import Edit, ErrorType, SynthExample, apply_edits, invert_spans


def test_replacement_span_is_exact():
    ref = "He paid 45 dollars today"
    hyp, spans = apply_edits(ref, [Edit(8, 10, "450", "number_change")], ErrorType.ENTITY_NUMBER)
    (s,) = spans
    assert hyp == "He paid 450 dollars today"
    assert hyp[s.tgt_start:s.tgt_end] == s.tgt_text == "450" and s.original_text == "45"
    assert invert_spans(hyp, spans) == ref


def test_insertion_span_excludes_separator_space_but_inversion_restores_it():
    ref = "Hello world."
    hyp, (s,) = apply_edits(ref, [Edit(len(ref), len(ref), " Extra sentence.", "hallucination_append")],
                            ErrorType.HALLUCINATION)
    assert hyp == "Hello world. Extra sentence."
    assert s.tgt_text == "Extra sentence." and hyp[s.tgt_start:s.tgt_end] == "Extra sentence."
    assert (s.edit_start, s.edit_end) == (len(ref), len(hyp)) and s.original_text == ""
    assert invert_spans(hyp, [s]) == ref


def test_deletion_is_zero_width_and_records_deleted_text():
    ref = "A b, c d e, f g."
    hyp, (s,) = apply_edits(ref, [Edit(3, 9, "", "omit_clause")], ErrorType.OMISSION)
    assert hyp == "A be, f g." or hyp == ref[:3] + ref[9:]
    assert s.tgt_start == s.tgt_end == 3 and s.tgt_text == "" and s.original_text == ref[3:9]
    assert invert_spans(hyp, [s]) == ref


def test_multiple_edits_offsets_account_for_earlier_length_changes():
    ref = "aa bb cc dd"
    edits = [Edit(0, 2, "XXXX", "e", group=1), Edit(6, 8, "Y", "e", group=1)]
    hyp, spans = apply_edits(ref, edits, ErrorType.MISTRANSLATION)
    assert hyp == "XXXX bb Y dd"
    assert [hyp[s.tgt_start:s.tgt_end] for s in spans] == ["XXXX", "Y"]
    assert invert_spans(hyp, spans) == ref
    assert {s.group for s in spans} == {1}


def test_edit_order_does_not_matter():
    ref = "aa bb cc dd"
    e1, e2 = Edit(0, 2, "X", "e"), Edit(6, 8, "Y", "e")
    assert apply_edits(ref, [e1, e2], ErrorType.OMISSION)[0] == apply_edits(ref, [e2, e1], ErrorType.OMISSION)[0]


def test_overlapping_or_out_of_range_edits_rejected():
    with pytest.raises(ValueError):
        apply_edits("abcdef", [Edit(0, 3, "x", "e"), Edit(2, 4, "y", "e")], ErrorType.OMISSION)
    with pytest.raises(ValueError):
        apply_edits("abc", [Edit(2, 9, "x", "e")], ErrorType.OMISSION)


def test_devanagari_offsets_are_codepoint_offsets():
    ref = "वह नहीं गया।"
    hyp, (s,) = apply_edits(ref, [Edit(3, 8, "", "negation_remove")], ErrorType.MISTRANSLATION)
    assert invert_spans(hyp, [s]) == ref and s.original_text == "नहीं "


def test_example_json_roundtrip():
    ref = "one two three"
    hyp, spans = apply_edits(ref, [Edit(4, 7, "2", "x")], ErrorType.ENTITY_NUMBER)
    ex = SynthExample("id", "en-hi", "train", "dev", "src", ref, hyp, "ENTITY_NUMBER", "x", spans)
    assert SynthExample.from_dict(ex.to_dict()) == ex
