"""Turn non-English input (Hinglish, Hindi script, a mix) into English **before** the English->target pipeline runs.

Why: the detector, the aligner and the translator all assume an English source. Given Romanised Hindi, NLLB copies it through and the
output mixes scripts (observed: "Mera naam Rahul hai ... कृपया ..."). So each sentence is classified (:mod:`mml.langmix`) and, if it is
not English, converted first. Two routes, best first:

1. an **LLM** (if one is configured and reachable): "rewrite this as plain English";
2. the **MT model itself** as a pivot: known Romanised-Hindi words are rewritten in Devanagari, then NLLB translates Hindi -> English.
   Cheap and offline, but approximate for heavy Hinglish (unknown words stay as typed).

The user always sees what was understood ("Interpreted as: ..."), because a wrong interpretation would otherwise silently produce a
fluent translation of the wrong sentence. Nothing here is hidden or guessed silently; when no conversion is possible the sentence is
passed on unchanged with an explicit warning.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from mml.langmix import LangInfo, classify, to_devanagari

log = logging.getLogger(__name__)

Pivot = Callable[[list[str]], list[str]]        # Hindi (possibly code-mixed) -> English, one output per input
LLMRewrite = Callable[[str], str]                # any text -> plain English

LABEL_TEXT = {"hinglish": "Hinglish (Hindi in Roman letters)", "hindi_script": "Hindi (Devanagari)",
              "mixed_script": "mixed Hindi and English", "other": "a script this app does not support"}

SYSTEM_PROMPT = (
    "You convert text to plain English. The input may be Hinglish (Hindi written in Roman letters, often mixed with English), Hindi, or a "
    "mix. Output ONLY the English version of the text inside <english></english> tags. Keep the meaning exactly; keep names, numbers, dates "
    "and units unchanged. Do not add or remove information, do not explain, do not answer questions in the text.")


@dataclass
class Prepared:
    """One input sentence after normalisation."""

    original: str
    english: str
    info: LangInfo
    method: str          # "as-is" | "llm" | "mt-pivot" | "unconverted"
    note: str = ""

    @property
    def converted(self) -> bool:
        return self.english != self.original


def llm_rewriter(llm) -> LLMRewrite:                         # noqa: ANN001 - any mml.llm.LLMBackend
    """Wrap an :class:`mml.llm.LLMBackend` as a ``text -> English`` function (reads the ``<english>`` tags)."""
    def rewrite(text: str) -> str:
        reply = llm.complete(SYSTEM_PROMPT, text).text
        m = re.search(r"<english>(.*?)</english>", reply, flags=re.S | re.I)
        return (m.group(1) if m else reply).strip().strip('"“”')
    return rewrite


def _convert(text: str, info: LangInfo, pivot: Pivot | None, llm: LLMRewrite | None) -> tuple[str, str, str]:
    """``(english, method, note)`` for one non-English sentence."""
    if llm is not None:
        try:
            out = llm(text)
            if out and classify(out).is_english:
                return out, "llm", ""
            log.warning("LLM normalisation returned non-English or empty text for %r; trying the MT pivot", text)
        except Exception as exc:  # noqa: BLE001 - an unreachable / failing LLM must not break the request
            log.warning("LLM normalisation failed (%s: %s); trying the MT pivot", type(exc).__name__, exc)
    if pivot is not None:
        source = to_devanagari(text) if info.label == "hinglish" else text
        out = pivot([source])[0].strip()
        note = ""
        if not classify(out).is_english:
            note = "The conversion is still not clearly English, so the translation below may be unreliable. Please rewrite it in English."
        elif info.label == "hinglish":
            note = "Approximate: Romanised Hindi was converted with a built-in word list, so unusual words may be misread."
        return out, "mt-pivot", note
    return text, "unconverted", ("No way to convert this to English is available (no LLM, no Hindi translation model), so it is used as typed "
                                 "and the result is unreliable. Please enter English.")


def prepare_sentences(sentences: list[str], pivot: Pivot | None = None, llm: LLMRewrite | None = None) -> list[Prepared]:
    """Classify every sentence and convert the non-English ones to English. English sentences are returned untouched."""
    out: list[Prepared] = []
    for s in sentences:
        info = classify(s)
        if info.is_english:
            out.append(Prepared(s, s, info, "as-is"))
            continue
        if info.label == "other":
            out.append(Prepared(s, s, info, "unconverted", "This script is not supported (English, Hinglish and Hindi are). Used as typed; "
                                                           "the result is unreliable."))
            continue
        english, method, note = _convert(s, info, pivot, llm)
        out.append(Prepared(s, english, info, method, note))
    return out


def describe(p: Prepared) -> str:
    """One readable line for the UI."""
    if p.method == "as-is":
        return ""
    kind = LABEL_TEXT.get(p.info.label, p.info.label)
    if p.method == "unconverted":
        return f"«{p.original}» looks like {kind}. {p.note}"
    via = "an LLM" if p.method == "llm" else "the translation model"
    line = f"«{p.original}» looks like {kind}; understood as «{p.english}» (via {via})."
    return f"{line} {p.note}".strip()
