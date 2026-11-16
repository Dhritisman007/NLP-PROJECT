from __future__ import annotations

import numpy as np
import pytest

from mml.detect.base import SignalResult, SpanFlag
from mml.detect.classifier import (
    TARGETS, MeaningLossClassifier, cross_val_proba, make_estimator, predict_labels, select_model, targets_matrix,
    tune_thresholds,
)
from mml.detect.features import FeatureTable, build_table, group_of, surface_features
from mml.detect.metrics import bootstrap_ci, macro_f1, sentence_metrics
from mml.detect.spans import kind_to_type, merge_ranges, silver_source_spans, span_f1
from mml.detect.train import base_id, single_feature_baseline
from mml.synth.types import ErrorSpan, SynthExample


# ------------------------------------------------------------------ features
def _results(n=3):
    return {
        "rules": [SignalResult({"num_missing": float(i), "lex_ent_extra": 0.0}, [SpanFlag("src", 0, 1, "a", "number_missing")]) for i in range(n)],
        "labse": [SignalResult({"labse_cos": 0.9 - 0.1 * i}) for i in range(n)],
    }


def test_table_columns_groups_and_missing_signals():
    pairs = [("a b", "x y")] * 3
    t = build_table(["i0", "i1", "i2"], ["OK", "OMISSION", "OK"], pairs, _results())
    assert t.groups() == ["rules", "lex", "labse", "surface"]               # no align/roundtrip/qe columns at all
    assert group_of("lex_ent_extra", "rules") == "lex" and group_of("num_missing", "rules") == "rules"
    assert t.X.shape == (3, len(t.columns)) and t.X[2, t.columns.index("num_missing")] == 2
    assert [len(s) for s in t.spans] == [1, 1, 1]


def test_select_drop_keep_and_roundtrip_and_concat(tmp_path):
    t = build_table(["a", "b", "c"], ["OK", "OK", "OMISSION"], [("a b", "x y")] * 3, _results())
    X, cols = t.select(drop=["lex", "surface"])
    assert set(cols) == {"num_missing", "labse_cos"}
    X, cols = t.select(keep=["labse"])
    assert cols == ["labse_cos"] and X.shape == (3, 1)
    t.save(tmp_path / "f")
    back = FeatureTable.load(tmp_path / "f")
    assert back.columns == t.columns and np.allclose(back.X, t.X) and back.spans[0][0].kind == "number_missing"
    both = FeatureTable.concat(t, back)
    assert both.X.shape[0] == 6
    other = t.subset([0])
    other.columns = ["z"]
    with pytest.raises(ValueError):
        FeatureTable.concat(t, other)


def test_surface_features():
    f = surface_features("one two three four", "eins zwei")
    assert f["len_word_ratio"] == 0.5 and f["len_hyp_words"] == 2


# ------------------------------------------------------------------ classifier
def test_targets_matrix():
    Y = targets_matrix(["OK", "OMISSION", "HALLUCINATION"])
    assert Y.tolist() == [[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 1]]


def test_predict_labels_ok_when_nothing_exceeds_threshold_else_best_ratio():
    thr = {t: 0.5 for t in TARGETS}
    thr["OMISSION"] = 0.2
    P = np.array([[0.1, 0.1, 0.1, 0.1, 0.1],        # nothing above -> OK
                  [0.9, 0.3, 0.6, 0.1, 0.1],        # OMISSION (0.3/0.2=1.5) beats MISTRANSLATION (0.6/0.5=1.2)
                  [0.9, 0.1, 0.4, 0.9, 0.1]])
    assert predict_labels(P, thr) == ["OK", "OMISSION", "ENTITY_NUMBER"]


def _toy(n=300, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.choice(["OK", "OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"], n)
    X = rng.normal(size=(n, 6))
    for i, lab in enumerate(labels):
        if lab != "OK":
            X[i, ["OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"].index(lab)] += 3.0
    return X, list(labels)


@pytest.mark.parametrize("kind", ["logreg", "gbm"])
def test_both_models_learn_a_separable_problem(kind):
    X, labels = _toy()
    clf = MeaningLossClassifier(kind, {}, [f"f{i}" for i in range(6)]).fit(X, labels)
    P = clf.predict_proba(X)
    clf.thresholds = tune_thresholds(P, labels)
    acc = np.mean(np.array(clf.predict_labels(X)) == np.array(labels))
    assert P.shape == (300, 5) and acc > 0.9
    assert clf.predict_sets(X)[0] <= {"OMISSION", "MISTRANSLATION", "ENTITY_NUMBER", "HALLUCINATION"}


def test_grouped_cv_has_no_leakage_between_twins():
    """A feature that is unique per sentence group must NOT let the model memorise labels across folds."""
    rng = np.random.default_rng(1)
    n_groups = 120
    labels, groups, ident = [], [], []
    for g in range(n_groups):
        for lab in ("OK", rng.choice(["OMISSION", "HALLUCINATION"])):
            labels.append(lab)
            groups.append(f"g{g}")
            ident.append(g)
    X = np.array(ident, dtype=float)[:, None]                  # perfectly identifies the group, says nothing about label
    P = cross_val_proba("gbm", {"max_depth": 5, "max_iter": 100}, X, labels, groups, n_splits=5)
    pred = np.array(predict_labels(P, {t: 0.5 for t in TARGETS})) != "OK"
    acc = np.mean(pred == (np.array(labels) != "OK"))
    assert acc < 0.7                                           # chance-level; leakage would give ~1.0


def test_select_model_returns_tuned_thresholds_and_oof_probs():
    X, labels = _toy(250)
    groups = [f"g{i // 2}" for i in range(250)]
    model, P, trace = select_model("logreg", X, labels, groups, [f"f{i}" for i in range(6)], n_splits=3)
    assert P.shape == (250, 5) and len(trace) == 3 and set(model.thresholds) == set(TARGETS)
    assert model.params in [t["params"] for t in trace]


def test_unknown_estimator_kind():
    with pytest.raises(ValueError):
        make_estimator("svm", {}, 0)


# ------------------------------------------------------------------ metrics
def test_sentence_metrics_perfect_and_imperfect():
    y = ["OK", "OMISSION", "HALLUCINATION", "OK"]
    perfect = sentence_metrics(y, y)
    assert perfect["macro_f1_errors_only"] == pytest.approx(0.5) or perfect["per_class"]["OMISSION"]["f1"] == 1.0
    assert perfect["per_class"]["OMISSION"]["mcc"] == pytest.approx(1.0) and perfect["any_error"]["f1"] == 1.0
    bad = sentence_metrics(y, ["OK", "OK", "OMISSION", "OMISSION"])
    assert bad["per_class"]["OMISSION"]["precision"] == 0.0 and bad["per_class"]["OMISSION"]["recall"] == 0.0   # 0 of 2 predictions right
    assert bad["per_class"]["HALLUCINATION"]["f1"] == 0.0 and bad["accuracy"] == 0.25
    assert bad["confusion"]["matrix"][0][0] == 1


def test_bootstrap_ci_contains_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    yt = rng.choice(["OK", "OMISSION"], 200)
    yp = np.where(rng.random(200) < 0.85, yt, "OK")
    groups = [f"g{i // 2}" for i in range(200)]
    lo, hi = bootstrap_ci(macro_f1, yt, yp, groups, n_boot=200, seed=1)
    assert lo <= macro_f1(yt, yp) <= hi and (lo, hi) == bootstrap_ci(macro_f1, yt, yp, groups, n_boot=200, seed=1)


def test_single_feature_baseline_finds_a_good_threshold():
    t = build_table([f"i{i}" for i in range(100)], ["OK"] * 50 + ["OMISSION"] * 50, [("a", "b")] * 100,
                    {"labse": [SignalResult({"labse_cos": 0.9 if i < 50 else 0.5}) for i in range(100)]})
    b = single_feature_baseline(t, "labse_cos", t.labels)
    assert b["any_f1"] == pytest.approx(1.0) and 0.5 <= b["threshold"] < 0.9


def test_base_id_groups_twins():
    assert base_id("en-hi-dev-0007-clean") == base_id("en-hi-dev-0007-OMISSION") == "en-hi-dev-0007"


# ------------------------------------------------------------------ spans
def test_kind_mapping_and_merge():
    assert kind_to_type("unaligned_src") == "OMISSION" and kind_to_type("unaligned_tgt") == "HALLUCINATION"
    assert kind_to_type("negation_extra") == "MISTRANSLATION" and kind_to_type("lex_entity_extra") == "ENTITY_NUMBER"
    assert kind_to_type("number_missing") == "ENTITY_NUMBER" and kind_to_type("xcomet_major") is None
    # ranges separated by at most one character (a space between words) are merged into one span
    assert merge_ranges([(5, 8), (0, 3), (3, 4), (20, 25)]) == [(0, 8), (20, 25)]
    assert merge_ranges([(0, 3), (5, 8)]) == [(0, 3), (5, 8)]


def _ex(label, hyp, spans, src="s" * 30, ref=None, id="x-1"):
    return SynthExample(id, "en-hi", "val", "dev", src, ref or hyp, hyp, label, "", spans)


def test_span_f1_char_level_arithmetic_and_gating():
    hyp = "0123456789" * 3
    gold = ErrorSpan("HALLUCINATION", "h", 10, 20, hyp[10:20], "", 10, 20)
    ex = _ex("HALLUCINATION", hyp, [gold])
    pred = [SpanFlag("tgt", 15, 25, "", "unaligned_tgt")]               # overlaps 5 of 10 gold chars, 5 chars wrong
    r = span_f1([ex], [pred], {})
    f = r["HALLUCINATION"]["tgt"]
    assert (f["precision"], f["recall"]) == (0.5, 0.5) and f["gold_chars"] == 10
    gated_off = span_f1([ex], [pred], {}, gate=[set()])
    assert gated_off["HALLUCINATION"]["tgt"]["recall"] == 0.0
    gated_on = span_f1([ex], [pred], {}, gate=[{"HALLUCINATION"}])
    assert gated_on["HALLUCINATION"]["tgt"]["f1"] == pytest.approx(0.5)


def test_clean_examples_count_as_false_positives_and_wrong_type_gets_no_credit():
    clean = _ex("OK", "a" * 30, [], id="c")
    pred = [SpanFlag("tgt", 0, 5, "", "unaligned_tgt"), SpanFlag("tgt", 6, 9, "", "number_extra")]
    r = span_f1([clean], [pred], {})
    assert r["HALLUCINATION"]["tgt"]["precision"] == 0.0 and r["ENTITY_NUMBER"]["tgt"]["recall"] == 0.0
    assert r["HALLUCINATION"]["tgt"]["gold_chars"] == 0


def test_examples_without_gold_on_a_side_are_skipped_not_penalised():
    delete = ErrorSpan("OMISSION", "omit_phrase", 5, 5, "", "gone words", 5, 5)
    ex = _ex("OMISSION", "0123456789", [delete])
    r = span_f1([ex], [[SpanFlag("tgt", 0, 3, "", "unaligned_tgt")]], {ex.id: None})
    assert r["OMISSION"]["src"]["skipped_no_gold"] == 1 and r["OMISSION"]["tgt"]["skipped_no_gold"] == 1
    assert r["OMISSION"]["src"]["gold_chars"] == 0


def test_silver_source_spans_map_deleted_reference_words_to_source_words():
    src = "The doctor said the patient should take two tablets after each meal ."
    ref = "Arzt sagte Patient solle zwei Tabletten nach jeder Mahlzeit nehmen"
    deleted = "nach jeder Mahlzeit "
    pos = ref.index(deleted)
    hyp = ref.replace(deleted, "")
    err = ErrorSpan("OMISSION", "omit_phrase", pos, pos, "", deleted, pos, pos)
    ex = SynthExample("e", "en-de", "val", "dev", src, ref, hyp, "OMISSION", "omit_phrase", [err])
    s_words = src.replace(" .", "").split()
    r_words = ref.split()
    links = []
    for j, w in enumerate(r_words):                                      # fake aligner: hand-made lexical links
        mapping = {"nach": "after", "jeder": "each", "Mahlzeit": "meal", "Arzt": "doctor", "zwei": "two"}
        if w in mapping:
            links.append((s_words.index(mapping[w]), j))
    spans = silver_source_spans(ex, lambda a, b: links)
    assert spans is not None and src[spans[0][0]:spans[0][1]] == "after each meal"
    assert silver_source_spans(_ex("OK", "abc", []), lambda a, b: []) is None
