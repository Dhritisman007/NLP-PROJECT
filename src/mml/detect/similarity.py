"""Cross-lingual semantic similarity of source and translation with LaBSE (cosine of sentence embeddings)."""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np

from mml.cache import DiskCache
from mml.config import Config
from mml.detect.base import Signal, SignalResult, model_device

log = logging.getLogger(__name__)

Encoder = Callable[[list[str]], np.ndarray]


class LabseSimilarity(Signal):
    """Feature ``labse_cos``: cosine(LaBSE(src), LaBSE(hyp)). Low values suggest omission or hallucination.

    ``encoder`` can be injected (tests); by default ``sentence-transformers`` loads ``models.similarity`` lazily.
    """

    name = "labse"
    chunk_size = 64

    def __init__(self, cfg: Config, cache: DiskCache | None = None, encoder: Encoder | None = None) -> None:
        super().__init__(cfg, cache)
        self.model_id: str = cfg.raw["models"]["similarity"]
        self._encoder = encoder

    def signature(self) -> dict[str, Any]:
        return {"model": self.model_id}

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            log.info("Loading %s", self.model_id)
            model = SentenceTransformer(self.model_id, device=model_device(self.cfg))
            self._encoder = lambda t: model.encode(t, batch_size=32, normalize_embeddings=True,
                                                   convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(self._encoder(texts), dtype=np.float32)

    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        src = self._encode([s for s, _ in pairs])
        hyp = self._encode([h for _, h in pairs])
        src = src / np.clip(np.linalg.norm(src, axis=1, keepdims=True), 1e-9, None)
        hyp = hyp / np.clip(np.linalg.norm(hyp, axis=1, keepdims=True), 1e-9, None)
        cos = (src * hyp).sum(axis=1)
        return [SignalResult({"labse_cos": float(c)}) for c in cos]
