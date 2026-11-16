"""End-to-end pipeline: source -> MT -> detect -> correct -> re-detect.

Strategies: ``none`` (report only), ``constrained`` (rule-based number/date/unit/name forcing; no LLM needed),
``span_llm`` and ``full`` (constrained fixing first, then span-guided LLM post-editing; need an LLM backend).
After correction the changed sentences are run through the detector again, so every result says whether the flags
were actually cleared.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

from mml.config import Config
from mml.correct.compare import correct_pairs
from mml.correct.span_postedit import CorrectionResult
from mml.detect.predict import Detection, Detector
from mml.llm import LLMBackend, LLMUnavailable, build_llm
from mml.translate import Translator

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Everything the pipeline knows about one sentence."""

    src: str
    mt: str
    corrected: str
    before: Detection
    after: Detection
    correction: CorrectionResult

    @property
    def changed(self) -> bool:
        return self.corrected != self.mt

    @property
    def flags_cleared(self) -> bool:
        """The detector flagged the MT output and no longer flags the corrected text."""
        return bool(self.before.types) and not self.after.types


@dataclass
class CorrectionPipeline:
    """Detect -> correct -> re-detect on (source, translation) pairs; optionally starts from source sentences."""

    cfg: Config
    detector: Detector
    strategy: str = "constrained"
    llm: LLMBackend | None = None
    translator: Translator | None = None
    workers: int = 4
    _used: str = field(default="", init=False)

    @classmethod
    def from_config(cls, cfg: Config, strategy: str = "auto", with_translator: bool = False) -> "CorrectionPipeline":
        """``auto`` = ``full`` when an LLM backend is usable, otherwise ``constrained`` (with a logged warning)."""
        llm: LLMBackend | None = None
        if strategy in ("auto", "full", "span_llm"):
            candidate = build_llm(cfg)
            try:
                candidate.check()
                llm = candidate
            except LLMUnavailable as exc:
                if strategy != "auto":
                    raise
                log.warning("No LLM available (%s): falling back to the rule-based 'constrained' strategy", exc)
        if strategy == "auto":
            strategy = "full" if llm is not None else "constrained"
        translator = Translator.from_config(cfg) if with_translator else None
        return cls(cfg, Detector(cfg), strategy, llm, translator)

    def translate(self, sources: Sequence[str]) -> list[str]:
        if self.translator is None:
            self.translator = Translator.from_config(self.cfg)
        return self.translator.translate_best(list(sources))

    def run_pairs(self, pairs: list[tuple[str, str]], before: list[Detection] | None = None) -> list[PipelineResult]:
        """Correct existing translations. ``before`` may carry pre-computed detections."""
        before = before or self.detector.detect(pairs)
        src_lang, tgt_lang = self.cfg.pair["src_lang"], self.cfg.pair["tgt_lang"]
        if self.strategy == "none":
            corrections = [CorrectionResult(h, "not_flagged") for _, h in pairs]
        else:
            corrections = correct_pairs(self.strategy, pairs, before, src_lang, tgt_lang, self.llm,
                                        float(self.cfg.get("correct.max_edit_ratio", 0.6)), self.workers)
        changed = [i for i, c in enumerate(corrections) if c.text != pairs[i][1]]
        after = list(before)
        if changed:
            for i, d in zip(changed, self.detector.detect([(pairs[i][0], corrections[i].text) for i in changed])):
                after[i] = d
        return [PipelineResult(s, h, c.text, b, a, c) for (s, h), c, b, a in zip(pairs, corrections, before, after)]

    def run(self, sources: Sequence[str]) -> list[PipelineResult]:
        """Full pipeline from source sentences."""
        mt = self.translate(sources)
        return self.run_pairs(list(zip(sources, mt)))
