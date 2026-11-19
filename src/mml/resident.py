"""Keep models loaded between requests (for the interactive demo).

The batch tools (``run_signals``) free every model right after its signal ran, because an 8 GB laptop cannot hold them all
while processing thousands of sentences. An interactive app does the opposite: it handles one short request at a time and must
not reload ~3 GB of models on every click. :class:`ResidentSignals` builds each signal once and keeps it (and its model) alive.
Memory footprint with all signals resident (int8 NLLB shared by both translation directions): roughly 4-5 GB.
"""

from __future__ import annotations

import logging
from typing import Sequence

from mml.cache import DiskCache
from mml.config import Config
from mml.detect.base import Signal, SignalResult, SignalUnavailable, open_cache
from mml.detect.predict import Detector
from mml.detect.roundtrip import RoundTripNLI
from mml.detect.run import DEFAULT_SIGNALS, SIGNAL_CLASSES, UNCACHED
from mml.pipeline import CorrectionPipeline
from mml.translate import Translator, build_backend
from mml.utils import get_hf_token

log = logging.getLogger(__name__)


class ResidentSignals:
    """Callable with the ``SignalRunner`` signature that reuses signal instances (and their loaded models) across calls."""

    def __init__(self, cfg: Config, shared_reverse: Translator | None = None, cache: DiskCache | None = None) -> None:
        self.cfg = cfg
        self.cache = cache if cache is not None else open_cache(cfg)
        self.shared_reverse = shared_reverse
        self._signals: dict[str, Signal] = {}

    def _get(self, name: str) -> Signal:
        if name not in self._signals:
            sig = SIGNAL_CLASSES[name](self.cfg, None if name in UNCACHED else self.cache)
            if isinstance(sig, RoundTripNLI) and self.shared_reverse is not None:
                sig._bt = self.shared_reverse.translate_best        # reuse the already loaded translation model
            self._signals[name] = sig
        return self._signals[name]

    def __call__(self, cfg: Config, pairs: list[tuple[str, str]], names: Sequence[str] = DEFAULT_SIGNALS
                 ) -> tuple[dict[str, list[SignalResult]], dict[str, str]]:
        results: dict[str, list[SignalResult]] = {}
        skipped: dict[str, str] = {}
        for name in names:
            try:
                results[name] = self._get(name).compute(pairs)
            except SignalUnavailable as exc:
                skipped[name] = str(exc)
        return results, skipped


def share_ct2(forward: Translator, cfg: Config) -> Translator | None:
    """Back-translation translator that reuses the forward CTranslate2 model (NLLB translates both directions)."""
    backend = forward.backend
    if backend.name != "nllb_ct2" or not hasattr(backend, "translator"):
        return None
    from transformers import AutoTokenizer

    rev = build_backend(cfg, reverse=True)
    if rev.name != "nllb_ct2":
        return None
    rev.tokenizer = AutoTokenizer.from_pretrained(rev.model_id, src_lang=rev.src_lang, revision=rev.revision, token=get_hf_token())
    rev.translator = backend.translator
    rev._loaded = True
    cache = DiskCache(cfg.raw["paths"]["cache_dir"] + "/mt.sqlite")
    return Translator(rev, cache, requested_backend=rev.name)


def build_pivot(forward: Translator, cfg: Config) -> Translator | None:
    """Hindi (Devanagari, possibly code-mixed) -> English translator for normalising non-English input, whatever the chosen pair.

    On the laptop profile it reuses the already loaded int8 NLLB model (no extra memory); otherwise it loads NLLB for ``hin_Deva -> eng_Latn``.
    """
    from transformers import AutoTokenizer

    rev = build_backend(cfg, reverse=True)
    rev.src_lang, rev.tgt_lang = "hin_Deva", "eng_Latn"
    backend = forward.backend
    if rev.name == "nllb_ct2":
        if backend.name != "nllb_ct2" or not hasattr(backend, "translator"):
            return None
        rev.tokenizer = AutoTokenizer.from_pretrained(rev.model_id, src_lang=rev.src_lang, revision=rev.revision, token=get_hf_token())
        rev.translator = backend.translator
    else:
        rev.load()
    rev._loaded = True
    cache = DiskCache(cfg.raw["paths"]["cache_dir"] + "/mt.sqlite")
    return Translator(rev, cache, requested_backend=rev.name)


def build_resident_pipeline(cfg: Config, strategy: str = "auto") -> CorrectionPipeline:
    """Detector + corrector + forward translator with every model kept in memory.

    On the laptop profile the forward MT is int8 NLLB (as in the Phase 6 evaluation; for en-hi this is NLLB because IndicTrans2 is
    gated); on the GPU profile it is the pair's configured backend.
    """
    if cfg.profile.get("bt_backend", "hf") == "ct2":
        forward = Translator.from_name(cfg, "nllb_ct2")
    else:
        forward = Translator.from_config(cfg)
    runner = ResidentSignals(cfg, share_ct2(forward, cfg))
    pipe = CorrectionPipeline.from_config(cfg, strategy)
    pipe.detector = Detector(cfg, signal_runner=runner)
    pipe.translator = forward
    pipe.pivot = build_pivot(forward, cfg)                    # Hindi -> English, used only for non-English input (see mml.normalize)
    return pipe
