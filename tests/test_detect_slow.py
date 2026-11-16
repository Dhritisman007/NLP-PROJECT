"""Real-model checks (downloads weights, slow). Run with:  pytest -m slow -q"""

from __future__ import annotations

import pytest

from mml.config import load_config
from mml.detect.alignment import WordAligner
from mml.detect.roundtrip import RoundTripNLI, chrf
from mml.detect.similarity import LabseSimilarity

pytestmark = pytest.mark.slow

SRC = "The doctor said that the patient should take two tablets after each meal."
DE_OK = "Der Arzt sagte, dass der Patient nach jeder Mahlzeit zwei Tabletten einnehmen sollte."
DE_OMIT = "Der Arzt sagte, dass der Patient zwei Tabletten einnehmen sollte."
DE_HALL = DE_OK + " Am selben Abend wurde außerdem ein alter Film im Kino gezeigt."
DE_UNRELATED = "Die Regierung kündigte an, dass der neue Plan im nächsten Jahr beginnt."


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return load_config(pair="en-de", overrides=[f"paths.cache_dir={tmp_path_factory.mktemp('cache')}"])


def test_labse_prefers_the_correct_translation(cfg):
    out = LabseSimilarity(cfg).compute([(SRC, DE_OK), (SRC, DE_UNRELATED)])
    assert out[0].features["labse_cos"] > out[1].features["labse_cos"] + 0.15
    assert out[0].features["labse_cos"] > 0.7


def test_simalign_unaligned_evidence_for_omission_and_hallucination(cfg):
    ok, omit, hall = WordAligner(cfg).compute([(SRC, DE_OK), (SRC, DE_OMIT), (SRC, DE_HALL)])
    assert omit.features["align_src_unaligned_ratio"] > ok.features["align_src_unaligned_ratio"]
    assert hall.features["align_tgt_unaligned_ratio"] > ok.features["align_tgt_unaligned_ratio"]
    assert hall.features["align_tgt_max_run"] >= 5
    assert any(s.side == "tgt" and "Film" in DE_HALL[s.start:s.end] for s in hall.spans)


def test_nli_direction_semantics(cfg):
    sig = RoundTripNLI(cfg, None, back_translate=lambda hyps: hyps)         # identity "back-translation"
    nli = sig._nli_probs
    ent = nli([("A man is playing a guitar on stage.", "A man is playing an instrument.")])[0]
    con = nli([("The cat is sleeping on the sofa.", "The cat is not sleeping on the sofa.")])[0]
    assert ent[0] > 0.5 and con[2] > 0.5


def test_back_translation_roundtrip_end_to_end(cfg):
    sig = RoundTripNLI(cfg, None)
    ok, flipped = sig.compute([(SRC, DE_OK), (SRC, DE_OK.replace("zwei", "keine"))])
    assert chrf(sig.last_back_translations[0], SRC) > 0.45
    assert ok.features["bt_chrf"] > 0.45
    assert ok.features["nli_fwd_ent"] > 0.4 and ok.features["nli_bwd_ent"] > 0.4
