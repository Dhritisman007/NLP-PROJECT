"""N-best re-ranking: among N MT candidates for one source, pick the one the detector considers least likely to be wrong.

Score = detector P(any error) (lower is better); ties and near-ties keep the MT model's own order (a small
``rank_penalty`` per position). Duplicate candidates are scored once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mml.detect.predict import Detector


@dataclass
class RerankResult:
    best_idx: int              # index into the ORIGINAL candidate list
    scores: list[float]        # P(any error) per original candidate (duplicates share the score of their first copy)

    @property
    def best_score(self) -> float:
        return self.scores[self.best_idx]


def pick(scores: Sequence[float], rank_penalty: float = 0.0, margin: float = 0.0) -> int:
    """Index of the chosen candidate: lowest P(error) (+ rank penalty), but keep the MT model's first choice unless the
    best alternative is better by more than ``margin`` (a conservative switch rule)."""
    adjusted = [s + rank_penalty * i for i, s in enumerate(scores)]
    best = int(np.argmin(adjusted))
    return best if adjusted[0] - adjusted[best] > margin else 0


def rerank_batch(detector: Detector, sources: Sequence[str], candidates: Sequence[Sequence[str]],
                 rank_penalty: float = 0.0, margin: float = 0.0) -> list[RerankResult]:
    """Re-rank every source's candidate list with one detector pass over all unique (source, candidate) pairs."""
    pairs: list[tuple[str, str]] = []
    index: dict[tuple[str, str], int] = {}
    for src, cands in zip(sources, candidates):
        for c in cands:
            if (src, c) not in index:
                index[(src, c)] = len(pairs)
                pairs.append((src, c))
    probs = detector.error_probability(pairs) if pairs else np.array([])
    out = []
    for src, cands in zip(sources, candidates):
        if not cands:
            raise ValueError("every source needs at least one candidate")
        scores = [float(probs[index[(src, c)]]) for c in cands]
        out.append(RerankResult(pick(scores, rank_penalty, margin), scores))
    return out
