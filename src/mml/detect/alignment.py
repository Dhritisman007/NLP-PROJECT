"""Word-alignment evidence for omission / hallucination (simalign, mBERT embeddings, ``itermax`` matching).

Source tokens with no aligned target token suggest content that was **omitted**; unaligned target tokens suggest
**added (hallucinated)** content. Runs of unaligned tokens are returned as flagged spans with character offsets, so
they also serve as span-level predictions. Function words (articles, postpositions) are often legitimately unaligned,
so these are features for the classifier, not decisions.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from mml.cache import DiskCache
from mml.config import Config
from mml.detect.base import Signal, SignalResult, SpanFlag
from mml.synth.text import chunks

log = logging.getLogger(__name__)

AlignFn = Callable[[list[str], list[str]], list[tuple[int, int]]]
MIN_SPAN_RUN = 2   # unaligned runs of at least this many tokens are reported as spans


def word_tokens(text: str) -> list[tuple[int, int, str]]:
    """Tokens with offsets: whitespace chunks stripped of edge punctuation; punctuation-only chunks are dropped."""
    return [(c.cs, c.ce, c.core) for c in chunks(text) if c.core]


def unaligned_runs(n: int, aligned: set[int]) -> list[tuple[int, int]]:
    """Maximal runs ``[i, j)`` of token indices in ``range(n)`` that are not in ``aligned``."""
    runs, i = [], 0
    while i < n:
        if i in aligned:
            i += 1
            continue
        j = i
        while j < n and j not in aligned:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def alignment_features(src_toks: list[tuple[int, int, str]], tgt_toks: list[tuple[int, int, str]],
                       pairs: list[tuple[int, int]]) -> SignalResult:
    """Turn an alignment into features and unaligned-run spans (pure function; unit-testable without a model)."""
    n_s, n_t = len(src_toks), len(tgt_toks)
    a_s, a_t = {i for i, _ in pairs}, {j for _, j in pairs}
    runs_s, runs_t = unaligned_runs(n_s, a_s), unaligned_runs(n_t, a_t)
    max_s = max((j - i for i, j in runs_s), default=0)
    max_t = max((j - i for i, j in runs_t), default=0)
    feats = {
        "align_n_src": float(n_s), "align_n_tgt": float(n_t),
        "align_len_ratio": n_t / n_s if n_s else 0.0,
        "align_src_unaligned_ratio": (n_s - len(a_s)) / n_s if n_s else 0.0,
        "align_tgt_unaligned_ratio": (n_t - len(a_t)) / n_t if n_t else 0.0,
        "align_src_max_run": float(max_s), "align_tgt_max_run": float(max_t),
        "align_src_max_run_frac": max_s / n_s if n_s else 0.0,
        "align_tgt_max_run_frac": max_t / n_t if n_t else 0.0,
    }
    spans: list[SpanFlag] = []
    for i, j in runs_s:
        if j - i >= MIN_SPAN_RUN:
            spans.append(SpanFlag("src", src_toks[i][0], src_toks[j - 1][1], "", "unaligned_src", (j - i) / n_s))
    for i, j in runs_t:
        if j - i >= MIN_SPAN_RUN:
            spans.append(SpanFlag("tgt", tgt_toks[i][0], tgt_toks[j - 1][1], "", "unaligned_tgt", (j - i) / n_t))
    return SignalResult(feats, spans)


class WordAligner(Signal):
    """simalign wrapper. ``align_fn`` can be injected (tests); by default simalign is loaded lazily on CPU."""

    name = "align"
    chunk_size = 16

    def __init__(self, cfg: Config, cache: DiskCache | None = None, align_fn: AlignFn | None = None,
                 model: str = "bert-base-multilingual-cased", method: str = "itermax") -> None:
        super().__init__(cfg, cache)
        self.model, self.method = model, method
        self._align = align_fn

    def signature(self) -> dict[str, Any]:
        return {"model": self.model, "method": self.method, "min_span_run": MIN_SPAN_RUN, "v": 1}

    def _get_align(self) -> AlignFn:
        if self._align is None:
            from simalign import SentenceAligner

            log.info("Loading simalign (%s, %s)", self.model, self.method)
            aligner = SentenceAligner(model=self.model, token_type="bpe", matching_methods="i", device="cpu")

            def run(src: list[str], tgt: list[str]) -> list[tuple[int, int]]:
                return [tuple(p) for p in aligner.get_word_aligns(src, tgt)[self.method]]  # type: ignore[misc]

            self._align = run
        return self._align

    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        align = self._get_align()
        out = []
        for src, hyp in pairs:
            s_toks, t_toks = word_tokens(src), word_tokens(hyp)
            links = align([t for *_, t in s_toks], [t for *_, t in t_toks]) if s_toks and t_toks else []
            out.append(alignment_features(s_toks, t_toks, links))
        return out
