"""Reference-free quality estimation with COMET models: CometKiwi (sentence score) and xCOMET (score + error spans).

Both models are **gated** on Hugging Face (accept the license and set ``HF_TOKEN``) and need the ``unbabel-comet``
package, which is installed only in the Colab environment (``requirements-colab.txt``). xCOMET-XL (3.5B) additionally
needs the ``colab_t4`` hardware profile (``heavy_models: true``). If anything is missing the signal raises
:class:`~mml.detect.base.SignalUnavailable` and the runner skips it with a warning (no placeholder scores).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Callable

from mml.cache import DiskCache
from mml.config import Config
from mml.detect.base import Signal, SignalResult, SignalUnavailable, SpanFlag
from mml.utils import get_hf_token

log = logging.getLogger(__name__)

# (list of {"src","mt"}) -> (scores, error spans per item as dicts with text/start/end/severity/confidence)
PredictFn = Callable[[list[dict[str, str]]], "tuple[list[float], list[list[dict[str, Any]]]]"]


class _CometSignal(Signal):
    """Shared loading/gating logic for COMET-family models."""

    chunk_size = 16
    needs_heavy = False
    model_key = ""

    def __init__(self, cfg: Config, cache: DiskCache | None = None, predict_fn: PredictFn | None = None) -> None:
        super().__init__(cfg, cache)
        self.model_id: str = cfg.raw["models"][self.model_key]
        self._predict = predict_fn

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id, "v": 1}

    def check(self) -> None:
        if self._predict is not None:
            return                                   # injected predictor (tests): nothing to gate
        if self.needs_heavy and not self.cfg.profile.get("heavy_models", False):
            raise SignalUnavailable(f"{self.model_id} needs a GPU profile (heavy_models); active profile is "
                                    f"'{self.cfg.raw['run']['profile']}'. Run on Colab with --profile colab_t4.")
        if importlib.util.find_spec("comet") is None:
            raise SignalUnavailable("unbabel-comet is not installed (pip install -r requirements-colab.txt)")
        if not get_hf_token():
            raise SignalUnavailable(f"{self.model_id} is gated: set HF_TOKEN and accept its license on Hugging Face")

    def _get_predict(self) -> PredictFn:
        if self._predict is None:
            from comet import download_model, load_from_checkpoint

            os.environ.setdefault("HF_TOKEN", get_hf_token() or "")
            log.info("Loading %s", self.model_id)
            model = load_from_checkpoint(download_model(self.model_id))
            gpus = 1 if self.cfg.profile["device"] in ("cuda", "auto") and _cuda() else 0

            def run(data: list[dict[str, str]]) -> tuple[list[float], list[list[dict[str, Any]]]]:
                out = model.predict(data, batch_size=8, gpus=gpus, progress_bar=False)
                meta = getattr(out, "metadata", None)
                spans = getattr(meta, "error_spans", None) or [[] for _ in data]
                return list(map(float, out.scores)), [list(s) for s in spans]

            self._predict = run
        return self._predict


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


class CometKiwi(_CometSignal):
    """Feature ``kiwi_score`` (higher = better, roughly 0..1)."""

    name = "cometkiwi"
    model_key = "qe_sentence"

    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        scores, _ = self._get_predict()([{"src": s, "mt": h} for s, h in pairs])
        return [SignalResult({"kiwi_score": float(x)}) for x in scores]


class XComet(_CometSignal):
    """xCOMET in QE mode: sentence score, error-span counts by severity, and the error spans themselves."""

    name = "xcomet"
    model_key = "qe_spans"
    needs_heavy = True

    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        scores, all_spans = self._get_predict()([{"src": s, "mt": h} for s, h in pairs])
        return [self._to_result(h, float(sc), spans) for (_, h), sc, spans in zip(pairs, scores, all_spans)]

    @staticmethod
    def _to_result(hyp: str, score: float, raw_spans: list[dict[str, Any]]) -> SignalResult:
        spans, counts, covered = [], {"MINOR": 0, "MAJOR": 0, "CRITICAL": 0}, set()
        for sp in raw_spans:
            sev = str(sp.get("severity", "MINOR")).upper()
            start, end = int(sp["start"]), int(sp["end"])
            counts[sev] = counts.get(sev, 0) + 1
            covered.update(range(start, end))
            spans.append(SpanFlag("tgt", start, end, str(sp.get("text", hyp[start:end])), f"xcomet_{sev.lower()}",
                                  float(sp.get("confidence", 1.0))))
        feats = {
            "xcomet_score": score, "xcomet_n_minor": float(counts["MINOR"]), "xcomet_n_major": float(counts["MAJOR"]),
            "xcomet_n_critical": float(counts["CRITICAL"]),
            "xcomet_err_frac": len(covered) / max(len(hyp), 1),
            "xcomet_max_conf": max((s.score for s in spans), default=0.0),
        }
        return SignalResult(feats, spans)
