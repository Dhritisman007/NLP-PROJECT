"""Fast tests for LaBSE / alignment / round-trip / QE signals using injected stand-ins (no model downloads)."""

from __future__ import annotations

import numpy as np
import pytest

from mml.cache import DiskCache
from mml.config import load_config
from mml.detect.alignment import WordAligner, alignment_features, unaligned_runs, word_tokens
from mml.detect.base import SignalUnavailable
from mml.detect.qe import CometKiwi, XComet
from mml.detect.roundtrip import RoundTripNLI, chrf, roundtrip_features
from mml.detect.similarity import LabseSimilarity


def cfg(tmp_path, **kw):
    return load_config(overrides=[f"paths.cache_dir={tmp_path}"], **kw)


# ------------------------------------------------------------------ LaBSE
def test_labse_cosine_from_injected_encoder(tmp_path):
    vecs = {"a": [1, 0, 0], "b": [1, 0, 0], "c": [0, 1, 0], "d": [1, 1, 0]}
    enc = lambda texts: np.array([vecs[t] for t in texts], dtype=np.float32)  # noqa: E731
    sig = LabseSimilarity(cfg(tmp_path), None, encoder=enc)
    out = [r.features["labse_cos"] for r in sig.compute([("a", "b"), ("a", "c"), ("a", "d")])]
    assert out == pytest.approx([1.0, 0.0, 2 ** -0.5], abs=1e-6)


# ------------------------------------------------------------------ alignment
def test_word_tokens_drop_punctuation_and_keep_offsets():
    text = "वह घर गया।"
    toks = word_tokens(text)
    assert [t for *_, t in toks] == ["वह", "घर", "गया"]
    assert all(text[s:e] == t for s, e, t in toks)


def test_unaligned_runs():
    assert unaligned_runs(6, {0, 3}) == [(1, 3), (4, 6)]
    assert unaligned_runs(3, {0, 1, 2}) == [] and unaligned_runs(0, set()) == []


def test_alignment_features_flag_omitted_source_and_added_target_runs():
    src = "He said that the new plan starts next year"
    hyp_omit = "He said that the new plan starts"          # last two source words have no counterpart
    s_toks, t_toks = word_tokens(src), word_tokens(hyp_omit)
    links = [(i, i) for i in range(len(t_toks))]
    r = alignment_features(s_toks, t_toks, links)
    assert r.features["align_src_unaligned_ratio"] == pytest.approx(2 / 9) and r.features["align_src_max_run"] == 2
    (span,) = r.spans
    assert span.side == "src" and src[span.start:span.end] == "next year"

    hyp_add = "He said that the new plan starts next year and tomorrow it rains"
    s_toks, t_toks = word_tokens(src), word_tokens(hyp_add)
    r = alignment_features(s_toks, t_toks, [(i, i) for i in range(len(s_toks))])
    assert r.features["align_tgt_unaligned_ratio"] == pytest.approx(4 / 13)
    (span,) = [s for s in r.spans if s.side == "tgt"]
    assert hyp_add[span.start:span.end] == "and tomorrow it rains"


def test_word_aligner_uses_injected_aligner_and_caches(tmp_path):
    calls = []

    def fake(src, tgt):
        calls.append((src, tgt))
        return [(i, i) for i in range(min(len(src), len(tgt)))]

    c = cfg(tmp_path)
    cache = DiskCache(tmp_path / "a.sqlite")
    sig = WordAligner(c, cache, align_fn=fake)
    pairs = [("a b c d", "x y z"), ("one two", "")]
    out = sig.compute(pairs)
    assert out[0].features["align_src_unaligned_ratio"] == pytest.approx(0.25)
    assert out[1].features["align_n_tgt"] == 0                       # empty translation does not crash
    sig.compute(pairs)
    assert len(calls) == 1                                           # second call fully served from the cache


# ------------------------------------------------------------------ round trip
def test_roundtrip_features_and_chrf():
    assert chrf("the cat sat", "the cat sat") == pytest.approx(1.0)
    assert chrf("completely different words", "the cat sat") < 0.3
    f = roundtrip_features("a b c", "a b", np.array([0.9, 0.05, 0.05]), np.array([0.1, 0.2, 0.7]))
    assert f["nli_fwd_ent"] == pytest.approx(0.9) and f["nli_bwd_con"] == pytest.approx(0.7)
    assert f["bt_len_ratio"] == pytest.approx(2 / 3)


def test_roundtrip_uses_both_nli_directions_in_order(tmp_path):
    seen = []

    def nli(pairs):
        seen.append(list(pairs))
        return np.array([[0.8, 0.1, 0.1]] * len(pairs))

    sig = RoundTripNLI(cfg(tmp_path), None, back_translate=lambda hyps: [h.upper() for h in hyps], nli=nli)
    (res,) = sig.compute([("src text", "hyp text")])
    assert seen[0] == [("src text", "HYP TEXT")] and seen[1] == [("HYP TEXT", "src text")]   # fwd then bwd
    assert res.features["nli_fwd_ent"] == pytest.approx(0.8)
    assert sig.last_back_translations == ["HYP TEXT"]


# ------------------------------------------------------------------ QE
def test_cometkiwi_with_injected_predictor(tmp_path):
    sig = CometKiwi(cfg(tmp_path), None, predict_fn=lambda data: ([0.5 + 0.1 * i for i in range(len(data))], [[]] * len(data)))
    out = sig.compute([("a", "b"), ("c", "d")])
    assert [r.features["kiwi_score"] for r in out] == pytest.approx([0.5, 0.6])


def test_xcomet_parses_error_spans_by_severity(tmp_path):
    hyp = "Das ist ein Test."
    raw = [{"text": "ein", "start": 8, "end": 11, "severity": "MAJOR", "confidence": 0.8},
           {"text": "Test", "start": 12, "end": 16, "severity": "minor", "confidence": 0.3}]
    sig = XComet(cfg(tmp_path), None, predict_fn=lambda data: ([0.42], [raw]))
    (res,) = sig.compute([("This is a test.", hyp)])
    f = res.features
    assert f["xcomet_score"] == pytest.approx(0.42) and f["xcomet_n_major"] == 1 and f["xcomet_n_minor"] == 1
    assert f["xcomet_err_frac"] == pytest.approx(7 / len(hyp)) and f["xcomet_max_conf"] == pytest.approx(0.8)
    assert [(s.kind, hyp[s.start:s.end]) for s in res.spans] == [("xcomet_major", "ein"), ("xcomet_minor", "Test")]


def test_xcomet_is_gated_by_hardware_profile_then_package_then_token(tmp_path, monkeypatch):
    mac = XComet(cfg(tmp_path, profile="mac_dev"))
    with pytest.raises(SignalUnavailable, match="GPU profile"):
        mac.check()
    colab = XComet(cfg(tmp_path, profile="colab_t4"))
    monkeypatch.setattr("mml.detect.qe.importlib.util.find_spec", lambda name: None)
    with pytest.raises(SignalUnavailable, match="unbabel-comet"):
        colab.check()
    monkeypatch.setattr("mml.detect.qe.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("mml.detect.qe.get_hf_token", lambda: None)
    with pytest.raises(SignalUnavailable, match="HF_TOKEN"):
        colab.check()
    monkeypatch.setattr("mml.detect.qe.get_hf_token", lambda: "x")
    colab.check()                                                    # all gates passed
