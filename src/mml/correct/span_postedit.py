"""LLM post-editing: span-guided (uses the detector's flagged spans) and the no-spans baseline.

Span-guided: the prompt contains the source, the translation and the detector's findings (type + the exact flagged text
on the source or translation side) and instructs the model to change **only** what those findings concern.
Baseline: the same model sees only source and translation and is asked to fix meaning errors if there are any.

Both go through the same safeguards: output must be wrapped in ``<translation>`` tags, and edits that rewrite more
than ``correct.max_edit_ratio`` of the sentence are rejected (the original is kept). A sentence the detector does not
flag is never sent to the LLM in the span-guided mode, which is what keeps over-correction low.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Sequence

from mml.detect.base import SpanFlag
from mml.detect.predict import Detection
from mml.detect.spans import kind_to_type
from mml.llm import LLMBackend, LLMError, LLMRefused, LLMUnavailable

log = logging.getLogger(__name__)

LANG_NAMES = {"eng_Latn": "English", "hin_Deva": "Hindi", "deu_Latn": "German"}
TAG_RE = re.compile(r"<translation>(.*?)</translation>", re.DOTALL | re.IGNORECASE)
MAX_PROBLEMS = 8
MAX_UNALIGNED = 2      # unaligned runs are the noisiest evidence (function words): keep only the longest per side

SPAN_SYSTEM = """You are a meticulous post-editor of machine translation from {src} into {tgt}.

You receive a source sentence, its current {tgt} translation, and a list of PROBLEMS found by an automatic detector. The detector can be wrong, so check each problem against the source.

Rules:
- Fix only the listed problems, with the smallest possible edit. Keep every other word of the translation exactly as it is: same wording, word order, punctuation and script. Do not paraphrase, polish or re-translate.
- OMISSION: add back the missing meaning (translate the quoted source words) at the natural position.
- HALLUCINATION: delete or correct the quoted translation words that have no basis in the source.
- ENTITY_NUMBER: make names, numbers, dates and units match the source exactly (use the target language's usual spelling and number format).
- MISTRANSLATION: restore the source meaning (e.g. negation, who did what to whom) without touching unrelated words.
- If a reported problem is a false alarm, leave that part unchanged.
Reply with the complete corrected translation inside <translation></translation> tags and nothing else."""

BASELINE_SYSTEM = """You are a meticulous post-editor of machine translation from {src} into {tgt}.

You receive a source sentence and its current {tgt} translation. Check whether the translation loses or changes the source meaning: omitted content, mistranslations (e.g. negation or roles), wrong names, numbers, dates or units, or content that is not in the source.
If there are such errors, fix them with the smallest possible edit and keep everything else exactly as it is. If the translation is already correct, return it unchanged. Do not paraphrase or polish.
Reply with the complete translation inside <translation></translation> tags and nothing else."""


@dataclass
class CorrectionResult:
    """Outcome for one sentence. ``status`` says what happened; ``text`` is always a usable translation."""

    text: str
    status: str            # see STATUSES
    edit_ratio: float = 0.0
    problems: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.status == "corrected"


STATUSES = ("not_flagged", "corrected", "unchanged_by_llm", "rejected_edit_ratio", "unparsable", "llm_refused",
            "llm_error", "no_findings")


def edit_ratio(old: str, new: str) -> float:
    """Fraction of the sentence changed (1 - character-level similarity)."""
    return 1.0 - SequenceMatcher(None, old, new, autojunk=False).ratio()


def parse_translation(raw: str) -> str | None:
    """Text inside ``<translation>`` tags (last match wins); ``None`` if absent or empty."""
    matches = TAG_RE.findall(raw)
    if not matches:
        return None
    out = matches[-1].strip()
    return out or None


# --------------------------------------------------------------------------- problem descriptions
def _quote(text: str) -> str:
    return "«" + text.strip() + "»"


def describe_span(span: SpanFlag, src: str, hyp: str) -> str | None:
    """One human-readable finding for a flagged span (None for kinds with no useful wording)."""
    text = span.text or (src if span.side == "src" else hyp)[span.start:span.end]
    t = kind_to_type(span.kind)
    if t is None or not text.strip():
        return None
    k = span.kind
    if k == "unaligned_src":
        return f"[OMISSION] The source words {_quote(text)} seem to be missing from the translation."
    if k == "unaligned_tgt":
        return f"[HALLUCINATION] The translation words {_quote(text)} have no counterpart in the source."
    if k == "number_missing":
        return f"[ENTITY_NUMBER] The source number {_quote(text)} is not reflected in the translation."
    if k == "number_extra":
        return f"[ENTITY_NUMBER] The translation contains the number {_quote(text)}, which does not match any number in the source."
    if k == "date_missing":
        return f"[ENTITY_NUMBER] The source date word {_quote(text)} is not reflected in the translation."
    if k == "date_extra":
        return f"[ENTITY_NUMBER] The translation date word {_quote(text)} does not match the source."
    if k in ("entity_missing", "lex_entity_missing"):
        return f"[ENTITY_NUMBER] The source name {_quote(text)} seems to be missing or wrong in the translation."
    if k == "lex_entity_extra":
        return f"[ENTITY_NUMBER] The translation name {_quote(text)} does not correspond to any name in the source."
    if k == "lex_unit_missing":
        return f"[ENTITY_NUMBER] The source unit/currency {_quote(text)} is not reflected in the translation."
    if k == "lex_unit_extra":
        return f"[ENTITY_NUMBER] The translation unit/currency {_quote(text)} does not match the source."
    if k == "negation_missing":
        return f"[MISTRANSLATION] The source negation {_quote(text)} is not expressed in the translation."
    if k == "negation_extra":
        return f"[MISTRANSLATION] The translation negation {_quote(text)} has no counterpart in the source."
    return None


TYPE_HINTS = {
    "OMISSION": "[OMISSION] Some source content seems to be missing from the translation.",
    "HALLUCINATION": "[HALLUCINATION] The translation seems to contain content that is not in the source.",
    "ENTITY_NUMBER": "[ENTITY_NUMBER] A name, number, date or unit seems to differ from the source.",
    "MISTRANSLATION": "[MISTRANSLATION] The meaning seems to differ from the source (e.g. negation or roles).",
}


def _select_spans(spans: list[SpanFlag], src: str, hyp: str) -> list[SpanFlag]:
    """Keep all precise rule findings, but only the longest ``MAX_UNALIGNED`` unaligned runs per side."""
    def words(sp: SpanFlag) -> int:
        return len((sp.text or (src if sp.side == "src" else hyp)[sp.start:sp.end]).split())

    keep = [s for s in spans if not s.kind.startswith("unaligned")]
    for kind in ("unaligned_src", "unaligned_tgt"):
        runs = sorted((s for s in spans if s.kind == kind), key=lambda s: (-words(s), s.start))
        keep += runs[:MAX_UNALIGNED]
    return sorted(keep, key=lambda s: (s.side, s.start))


select_spans = _select_spans        # public alias (used by the demo to show the same evidence the LLM gets)


def build_problems(src: str, hyp: str, det: Detection) -> list[str]:
    """Findings for the predicted types: flagged spans first, then a generic hint for types without any span."""
    problems: list[str] = []
    covered: set[str] = set()
    for span in _select_spans(det.spans_for(), src, hyp):
        desc = describe_span(span, src, hyp)
        if desc and desc not in problems:
            problems.append(desc)
            covered.add(kind_to_type(span.kind) or "")
    problems += [TYPE_HINTS[t] for t in sorted(det.types) if t not in covered]
    return problems[:MAX_PROBLEMS]


def build_span_prompt(src: str, hyp: str, problems: Sequence[str], src_lang: str, tgt_lang: str) -> tuple[str, str]:
    """(system, user) prompt for span-guided post-editing."""
    sn, tn = LANG_NAMES.get(src_lang, src_lang), LANG_NAMES.get(tgt_lang, tgt_lang)
    listing = "\n".join(f"{i}. {p}" for i, p in enumerate(problems, 1))
    user = (f"Source ({sn}):\n{src}\n\nCurrent translation ({tn}):\n{hyp}\n\nDetected problems:\n{listing}")
    return SPAN_SYSTEM.format(src=sn, tgt=tn), user


def build_baseline_prompt(src: str, hyp: str, src_lang: str, tgt_lang: str) -> tuple[str, str]:
    """(system, user) prompt for the no-spans baseline."""
    sn, tn = LANG_NAMES.get(src_lang, src_lang), LANG_NAMES.get(tgt_lang, tgt_lang)
    return (BASELINE_SYSTEM.format(src=sn, tgt=tn),
            f"Source ({sn}):\n{src}\n\nCurrent translation ({tn}):\n{hyp}")


# --------------------------------------------------------------------------- running a prompt safely
def _run(llm: LLMBackend, system: str, user: str, hyp: str, max_edit_ratio: float,
         problems: list[str]) -> CorrectionResult:
    try:
        raw = llm.complete(system, user).text
    except LLMUnavailable:
        raise
    except LLMRefused as exc:
        log.warning("LLM refused: %s", exc)
        return CorrectionResult(hyp, "llm_refused", problems=problems)
    except LLMError as exc:
        log.warning("LLM error: %s", exc)
        return CorrectionResult(hyp, "llm_error", problems=problems)
    new = parse_translation(raw)
    if new is None:
        return CorrectionResult(hyp, "unparsable", problems=problems)
    ratio = edit_ratio(hyp, new)
    if new == hyp:
        return CorrectionResult(hyp, "unchanged_by_llm", 0.0, problems)
    if ratio > max_edit_ratio:
        return CorrectionResult(hyp, "rejected_edit_ratio", ratio, problems)
    return CorrectionResult(new, "corrected", ratio, problems)


def span_postedit(llm: LLMBackend, src: str, hyp: str, det: Detection, src_lang: str, tgt_lang: str,
                  max_edit_ratio: float = 0.6) -> CorrectionResult:
    """Span-guided correction. Sentences the detector did not flag are returned untouched without an LLM call."""
    if not det.types:
        return CorrectionResult(hyp, "not_flagged")
    problems = build_problems(src, hyp, det)
    if not problems:
        return CorrectionResult(hyp, "no_findings")
    system, user = build_span_prompt(src, hyp, problems, src_lang, tgt_lang)
    return _run(llm, system, user, hyp, max_edit_ratio, problems)


def baseline_postedit(llm: LLMBackend, src: str, hyp: str, src_lang: str, tgt_lang: str,
                      max_edit_ratio: float = 0.6) -> CorrectionResult:
    """Baseline (c): post-edit with no detector information; every sentence is sent to the LLM."""
    system, user = build_baseline_prompt(src, hyp, src_lang, tgt_lang)
    return _run(llm, system, user, hyp, max_edit_ratio, [])


def run_many(fn, items: Sequence, workers: int = 4) -> list[CorrectionResult]:
    """Apply ``fn`` to every item with a small thread pool (LLM calls are I/O bound); order preserved."""
    if workers <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))
