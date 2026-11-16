import random

from mml.synth.entities import CompositeFinder, EntityPool, GazetteerFinder, NERFinder
from mml.synth.lang import DE, HI


def test_gazetteer_finds_multiword_entity_and_reports_exact_offsets():
    text = "प्रधानमंत्री नरेंद्र मोदी ने जर्मनी का दौरा किया।"
    ents = GazetteerFinder(HI.gazetteer).find(text)
    assert [(e.text, e.etype) for e in ents] == [("नरेंद्र मोदी", "PER"), ("जर्मनी", "LOC")]
    assert all(text[e.start:e.end] == e.text for e in ents)


def test_gazetteer_respects_word_boundaries():
    assert GazetteerFinder(DE.gazetteer).find("Die Berliner Mauer fiel.") == []
    assert [e.text for e in GazetteerFinder(DE.gazetteer).find("Die Mauer in Berlin fiel.")] == ["Berlin"]
    # a Hindi word that merely starts with a known city must not match
    assert GazetteerFinder(HI.gazetteer).find("दिल्लीवासी खुश थे।") == []


def test_longest_match_wins_over_contained_entity():
    gaz = {"ORG": ("Universität Oxford",), "LOC": ("Oxford",)}
    (e,) = GazetteerFinder(gaz).find("Die Universität Oxford ist alt.")
    assert (e.text, e.etype) == ("Universität Oxford", "ORG")


def test_ner_finder_maps_labels_and_drops_unknown_types():
    text = "Angela Merkel besuchte Paris."
    fake = lambda t: [  # noqa: E731 - mimics a HF aggregated pipeline output
        {"entity_group": "PER", "start": 0, "end": 13}, {"entity_group": "LOC", "start": 23, "end": 28},
        {"entity_group": "MISC", "start": 14, "end": 22}]
    ents = NERFinder("stub", pipeline_fn=fake).find(text)
    assert [(e.text, e.etype) for e in ents] == [("Angela Merkel", "PER"), ("Paris", "LOC")]


def test_ner_finder_handles_bio_prefixed_labels():
    fake = lambda t: [{"entity_group": "B-ORG", "start": 0, "end": 6}]  # noqa: E731
    assert NERFinder("stub", pipeline_fn=fake).find("Google ist groß.")[0].etype == "ORG"


def test_composite_finder_merges_and_dedups_overlaps():
    text = "Angela Merkel besuchte Paris."
    ner = NERFinder("stub", pipeline_fn=lambda t: [{"entity_group": "PER", "start": 0, "end": 6}])
    ents = CompositeFinder(GazetteerFinder(DE.gazetteer), ner).find(text)
    assert [e.text for e in ents] == ["Angela Merkel", "Paris"]   # longer gazetteer span beats NER "Angela"


def test_pool_sample_is_same_type_different_and_deterministic():
    pool = EntityPool(DE.gazetteer)
    a = pool.sample("LOC", "Berlin", random.Random(1))
    assert a and a != "Berlin" and a in DE.gazetteer["LOC"]
    assert pool.sample("LOC", "Berlin", random.Random(1)) == a
    assert pool.sample("MISC", "x", random.Random(1)) is None


def test_pool_harvests_entities_from_corpus():
    pool = EntityPool({})
    pool.extend_from_corpus(GazetteerFinder(DE.gazetteer), ["Wir fahren nach Köln.", "Hamburg ist schön."])
    assert pool.sample("LOC", "Köln", random.Random(0)) == "Hamburg"
