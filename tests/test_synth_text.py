from mml.synth.text import (
    chunks, clause_commas, content_end, deletion_range, is_word_boundary, number_spans, overlaps,
)


def test_chunk_cores_strip_edge_punctuation_in_both_scripts():
    text = 'वह, घर गया।'
    cores = [c.core for c in chunks(text)]
    assert cores == ["वह", "घर", "गया"]
    c = chunks('He said "no".')
    assert [x.core for x in c] == ["He", "said", '"no'.lstrip('"')]


def test_chunk_offsets_slice_back_to_core():
    text = "(0230 UTC) के मुताबिक, करीब 9:30 बजे."
    for c in chunks(text):
        assert text[c.cs:c.ce] == c.core and text[c.start:c.end] == c.text


def test_content_end_ignores_trailing_punctuation_and_quotes():
    assert content_end("वह घर गया।") == len("वह घर गया")
    assert content_end("वह घर गया.") == len("वह घर गया")
    assert content_end('Er sagte "ja".') == len('Er sagte "ja')
    assert content_end("no punct") == len("no punct")


def test_number_spans_find_times_and_grouped_numbers_but_not_names_with_digits():
    text = "Um 9:30 Uhr, 2,500 km, 4,5 Mio., JAS 39C und 1947."
    found = [text[a:b] for a, b in number_spans(text)]
    assert found == ["9:30", "2,500", "4,5", "1947"]
    assert "39C" not in found


def test_clause_commas_skip_decimal_and_thousands_commas():
    text = "Er zahlte 4,5 Euro, und sie 2,500, aber nicht mehr."
    positions = clause_commas(text)
    assert [text[:p].split()[-1] for p in positions] == ["Euro", "2,500"]


def test_deletion_range_swallows_exactly_one_space():
    t = "a bb cc dd."
    s, e = deletion_range(t, 2, 4)           # "bb" followed by space
    assert t[:s] + t[e:] == "a cc dd."
    s, e = deletion_range(t, 8, 10)           # "dd" followed by '.', preceded by space
    assert t[:s] + t[e:] == "a bb cc."


def test_overlaps_semantics():
    assert overlaps(2, 5, [(4, 8)]) and not overlaps(2, 4, [(4, 8)])
    assert overlaps(3, 3, [(0, 10)])              # a point inside a protected span counts as overlapping
    assert not overlaps(10, 10, [(0, 10)])        # ...but not one at its edge


def test_word_boundary_treats_devanagari_marks_as_word_characters():
    t = "दिल्लीवासी दिल्ली में"
    assert not is_word_boundary(t, len("दिल्ली"))          # inside "दिल्लीवासी"
    assert is_word_boundary(t, len("दिल्लीवासी दिल्ली"))   # before the space
