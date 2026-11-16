"""Data model for synthetic errors: error types, edits, labelled spans, and exact span bookkeeping.

All injectors express a corruption as a list of :class:`Edit` (positions in the *reference*).
:func:`apply_edits` is the only place offsets in the corrupted text are computed, and :func:`invert_spans`
undoes them, so every labelled span can be checked mechanically (``invert_spans(hyp, spans) == ref``).
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any


class ErrorType(str, enum.Enum):
    """Meaning-loss classes. ``OK`` marks an untouched, correct translation."""

    OK = "OK"
    OMISSION = "OMISSION"
    MISTRANSLATION = "MISTRANSLATION"
    ENTITY_NUMBER = "ENTITY_NUMBER"
    HALLUCINATION = "HALLUCINATION"


ERROR_TYPES: tuple[ErrorType, ...] = (
    ErrorType.OMISSION, ErrorType.MISTRANSLATION, ErrorType.ENTITY_NUMBER, ErrorType.HALLUCINATION)


@dataclass(frozen=True)
class Edit:
    """Replace ``ref[start:end]`` by ``new`` (``start == end``: insertion, ``new == ""``: deletion).

    ``group`` ties together the edits of one logical error (e.g. the two halves of a role swap).
    """

    start: int
    end: int
    new: str
    subtype: str
    group: int = 0


@dataclass
class ErrorSpan:
    """One labelled span in the corrupted translation ``hyp``.

    * ``tgt_start:tgt_end`` is the span of the *changed* text in ``hyp`` (surrounding separator spaces
      excluded). For pure deletions (omissions) it is zero-width at the deletion point.
    * ``original_text`` is the reference text that was replaced/deleted ("" for insertions).
    * ``edit_start:edit_end`` is the full region of ``hyp`` produced by the edit (includes separator spaces);
      replacing it by ``original_text`` restores the reference.
    * ``src_start/src_end`` can hold the aligned *source* span (filled by an aligner in later phases).
    """

    type: str
    subtype: str
    tgt_start: int
    tgt_end: int
    tgt_text: str
    original_text: str
    edit_start: int
    edit_end: int
    group: int = 0
    src_start: int | None = None
    src_end: int | None = None


@dataclass
class SynthExample:
    """One row of the synthetic dataset."""

    id: str
    pair: str
    split: str                 # train | val | test
    origin_split: str          # FLORES split the sentence came from (dev | devtest)
    src: str
    ref: str
    hyp: str                   # translation under test (== ref for clean examples)
    label: str                 # ErrorType value
    subtype: str = ""
    errors: list[ErrorSpan] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "SynthExample":
        errors = [ErrorSpan(**e) for e in row.get("errors", [])]
        return cls(**{**row, "errors": errors})


def apply_edits(ref: str, edits: list[Edit], etype: ErrorType) -> tuple[str, list[ErrorSpan]]:
    """Apply non-overlapping edits to ``ref``; return the corrupted text and exact spans in it."""
    ordered = sorted(edits, key=lambda e: (e.start, e.end))
    for a, b in zip(ordered, ordered[1:]):
        if a.end > b.start:
            raise ValueError(f"Overlapping edits: {a} / {b}")
    for e in ordered:
        if not (0 <= e.start <= e.end <= len(ref)):
            raise ValueError(f"Edit out of range for ref of length {len(ref)}: {e}")
    parts: list[str] = []
    spans: list[ErrorSpan] = []
    cursor, delta = 0, 0
    for e in ordered:
        parts.append(ref[cursor:e.start])
        parts.append(e.new)
        edit_start = e.start + delta
        edit_end = edit_start + len(e.new)
        lead = len(e.new) - len(e.new.lstrip())
        trail = len(e.new) - len(e.new.rstrip())
        t0, t1 = (edit_start + lead, edit_end - trail) if e.new.strip() else (edit_start, edit_start)
        spans.append(ErrorSpan(
            type=etype.value, subtype=e.subtype, tgt_start=t0, tgt_end=t1, tgt_text="",
            original_text=ref[e.start:e.end], edit_start=edit_start, edit_end=edit_end, group=e.group))
        delta += len(e.new) - (e.end - e.start)
        cursor = e.end
    parts.append(ref[cursor:])
    hyp = "".join(parts)
    for s in spans:
        s.tgt_text = hyp[s.tgt_start:s.tgt_end]
    return hyp, spans


def invert_spans(hyp: str, spans: list[ErrorSpan]) -> str:
    """Undo all edits using only the recorded spans (right to left); returns the original reference."""
    out = hyp
    for s in sorted(spans, key=lambda s: s.edit_start, reverse=True):
        out = out[:s.edit_start] + s.original_text + out[s.edit_end:]
    return out
