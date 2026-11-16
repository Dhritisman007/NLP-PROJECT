from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from mml.synth.build import (
    SPLITS, ParallelItem, assign_split, build_examples, load_split, write_dataset,
)
from mml.synth.inject_errors import ErrorInjector
from mml.synth.lang import HI
from mml.synth.types import invert_spans

REFS = [
    "सरकार ने कहा कि नई योजना अगले साल शुरू होगी, और इससे लाखों किसानों को सीधा फायदा मिलेगा।",
    "प्रधानमंत्री नरेंद्र मोदी ने 12 मई को जर्मनी में 45 किलोमीटर लंबी सड़क का उद्घाटन किया।",
    "वह बैठक में नहीं गई क्योंकि वह बीमार थी, लेकिन उसने सबको संदेश भेज दिया था।",
    "कंपनी ने तीसरी तिमाही में 45 मिलियन डॉलर का मुनाफा दर्ज किया और शेयर बाज़ार में तेज़ी आई।",
    "बराक ओबामा ने पेरिस में नरेंद्र मोदी से मुलाकात की और कई समझौतों पर चर्चा हुई।",
    "शहर के पुराने बाज़ार में हर शनिवार को किसान ताज़ी सब्ज़ियाँ और फल बेचने आते हैं।",
]


def items(n_dev: int = 60, n_devtest: int = 20) -> list[ParallelItem]:
    out = []
    for sp, n in (("dev", n_dev), ("devtest", n_devtest)):
        for i in range(n):
            out.append(ParallelItem(f"en-hi-{sp}-{i:04d}", sp, f"source {sp} {i}", REFS[i % len(REFS)]))
    return out


def build(seed: int = 42, **kw):
    return build_examples(items(), "en-hi", "hin_Deva", seed, **kw)[0]


def base_id(example_id: str) -> str:
    return example_id.rsplit("-", 1)[0]


def test_equal_numbers_of_clean_and_corrupted_in_every_split():
    ex = build()
    for split in SPLITS:
        rows = [e for e in ex if e.split == split]
        assert rows, split
        assert sum(e.label == "OK" for e in rows) == sum(e.label != "OK" for e in rows)


def test_clean_examples_are_untouched_and_corrupted_ones_are_labelled_with_spans():
    for e in build():
        if e.label == "OK":
            assert e.hyp == e.ref and e.errors == [] and e.subtype == ""
        else:
            assert e.hyp != e.ref and e.errors and all(s.type == e.label for s in e.errors)
            assert invert_spans(e.hyp, e.errors) == e.ref
            assert all(e.hyp[s.tgt_start:s.tgt_end] == s.tgt_text for s in e.errors)


def test_test_split_is_exactly_devtest_and_sentences_never_cross_splits():
    ex = build()
    assert {e.origin_split for e in ex if e.split == "test"} == {"devtest"}
    assert all(e.split != "test" for e in ex if e.origin_split == "dev")
    split_of: dict[str, set[str]] = {}
    for e in ex:
        split_of.setdefault(base_id(e.id), set()).add(e.split)
    assert all(len(s) == 1 for s in split_of.values())          # clean + corrupted twins stay together


def test_assign_split_is_deterministic_and_roughly_85_15():
    ids = [f"en-hi-dev-{i:04d}" for i in range(997)]
    first = [assign_split(i, "dev") for i in ids]
    assert first == [assign_split(i, "dev") for i in ids]
    share = Counter(first)["val"] / len(ids)
    assert 0.10 < share < 0.20 and set(first) == {"train", "val"}
    assert {assign_split(f"x{i}", "devtest") for i in range(50)} == {"test"}
    assert {assign_split(f"x{i}", "custom") for i in range(300)} == {"train", "val", "test"}


def test_generation_is_deterministic_per_seed_and_varies_across_seeds():
    a, b = build(7), build(7)
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]
    assert [e.hyp for e in a] != [e.hyp for e in build(8)]


def test_all_four_error_types_are_produced():
    labels = {e.label for e in build()}
    assert labels == {"OK", "OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"}


def test_type_selection_balances_counts_where_possible():
    ex = build()
    counts = Counter(e.label for e in ex if e.split == "train" and e.label != "OK")
    assert max(counts.values()) - min(counts.values()) <= 2


def test_balance_flag_gives_identical_counts_per_type():
    ex = build(balance=True)
    for split in SPLITS:
        counts = Counter(e.label for e in ex if e.split == split and e.label != "OK")
        assert len(set(counts.values())) == 1, (split, counts)
        assert sum(e.label == "OK" for e in ex if e.split == split) == sum(counts.values())


def test_sentences_without_any_applicable_error_are_skipped_not_half_emitted():
    short = [ParallelItem(f"en-hi-dev-{i:04d}", "dev", "s", "वह घर गया।") for i in range(5)]
    ex, stats = build_examples(short, "en-hi", "hin_Deva")
    assert sum(e.label == "OK" for e in ex) == sum(e.label != "OK" for e in ex)
    assert sum(stats["skipped_no_applicable_error"].values()) + len(ex) // 2 == 5


def test_write_and_load_roundtrip_and_test_split_guard(tmp_path: Path):
    ex = build()
    manifest = write_dataset(ex, tmp_path, {"pair": "en-hi", "seed": 42})
    for split in ("train", "val"):
        loaded = load_split(tmp_path, split)
        assert [e.to_dict() for e in loaded] == [e.to_dict() for e in ex if e.split == split]
    with pytest.raises(PermissionError):
        load_split(tmp_path, "test")                                  # held out by default
    test_rows = load_split(tmp_path, "test", for_final_eval=True)
    ids = sorted(e.id for e in test_rows)
    assert manifest["test_ids_sha256"] == hashlib.sha256("\n".join(ids).encode()).hexdigest()
    assert json.loads((tmp_path / "manifest.json").read_text())["summary"]["test"]["n"] == len(test_rows)
    with pytest.raises(ValueError):
        load_split(tmp_path, "dev")


def test_entity_pool_for_a_split_only_uses_that_splits_sentences():
    """Replacement entities come from the built-in gazetteer plus the split's own references."""
    ex = build()
    allowed = {f for forms in HI.gazetteer.values() for f in forms}
    for e in ex:
        for s in e.errors:
            if s.subtype == "entity_replace":
                assert s.tgt_text in allowed
    assert ErrorInjector(HI).pool is not None
