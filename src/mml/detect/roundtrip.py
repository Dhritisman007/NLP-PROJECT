"""Round-trip check: back-translate the translation to the source language, then compare with bidirectional NLI.

Intuition (src = original English, bt = back-translated English):
* omission        -> bt says less: ``src => bt`` still holds, ``bt => src`` fails (low backward entailment)
* hallucination   -> bt says more: ``src => bt`` fails (low forward entailment)
* mistranslation  -> contradiction in either direction (negation flip, role swap)
Back-translation uses NLLB (target->source) and is cached in the MT cache; NLI probabilities are cached per pair.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from mml.cache import DiskCache
from mml.config import Config
from mml.detect.base import Signal, SignalResult, model_device

log = logging.getLogger(__name__)

BackTranslateFn = Callable[[list[str]], list[str]]
NLIFn = Callable[[list[tuple[str, str]]], np.ndarray]   # (premise, hypothesis) pairs -> [n, 3] probs (ent, neu, con)


def chrf(hyp: str, ref: str) -> float:
    """chrF of ``hyp`` against ``ref`` in [0, 1]."""
    from sacrebleu.metrics import CHRF

    return float(CHRF().sentence_score(hyp, [ref]).score) / 100.0


def roundtrip_features(src: str, bt: str, fwd: np.ndarray, bwd: np.ndarray) -> dict[str, float]:
    """Feature dict from NLI probabilities (ent, neu, con) in both directions and surface similarity."""
    return {
        "nli_fwd_ent": float(fwd[0]), "nli_fwd_con": float(fwd[2]),
        "nli_bwd_ent": float(bwd[0]), "nli_bwd_con": float(bwd[2]),
        "bt_chrf": chrf(bt, src),
        "bt_len_ratio": len(bt.split()) / max(len(src.split()), 1),
    }


class RoundTripNLI(Signal):
    """Back-translation + bidirectional NLI. Both models can be injected for tests."""

    name = "roundtrip"
    chunk_size = 64

    def __init__(self, cfg: Config, cache: DiskCache | None = None, back_translate: BackTranslateFn | None = None,
                 nli: NLIFn | None = None) -> None:
        super().__init__(cfg, cache)
        self.nli_id: str = cfg.raw["models"]["nli"]
        from mml.translate import reverse_backend_name

        self.bt_name = reverse_backend_name(cfg)
        bt = cfg.raw["mt"]["backends"][self.bt_name]
        self.bt_id, self.bt_rev = bt["model_id"], bt.get("revision")
        self._bt, self._nli = back_translate, nli
        self.last_back_translations: list[str] = []

    def signature(self) -> dict[str, Any]:
        return {"bt": [self.bt_name, self.bt_id, self.bt_rev, self.cfg.raw["mt"]["num_beams"]], "nli": self.nli_id,
                "dir": [self.tgt_lang, self.src_lang], "v": 1}

    def _back_translate(self, hyps: list[str]) -> list[str]:
        if self._bt is None:
            from mml.translate import Translator

            translator = Translator.reverse_from_config(self.cfg)
            self._bt = translator.translate_best
        return self._bt(hyps)

    def _nli_probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if self._nli is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            log.info("Loading NLI model %s", self.nli_id)
            device = model_device(self.cfg)
            tok = AutoTokenizer.from_pretrained(self.nli_id)
            model = AutoModelForSequenceClassification.from_pretrained(self.nli_id).to(device).eval()
            labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
            order = [labels["entailment"], labels["neutral"], labels["contradiction"]]

            def run(batch: list[tuple[str, str]]) -> np.ndarray:
                enc = tok([p for p, _ in batch], [h for _, h in batch], return_tensors="pt", padding=True,
                          truncation=True, max_length=256).to(device)
                with torch.inference_mode():
                    probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().numpy()
                return probs[:, order]

            self._nli = run
        return np.asarray(self._nli(pairs))

    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        srcs, hyps = [s for s, _ in pairs], [h for _, h in pairs]
        bts = self._back_translate(hyps)
        self.last_back_translations = bts
        fwd = self._nli_probs(list(zip(srcs, bts)))
        bwd = self._nli_probs(list(zip(bts, srcs)))
        return [SignalResult(roundtrip_features(s, b, f, w)) for s, b, f, w in zip(srcs, bts, fwd, bwd)]
