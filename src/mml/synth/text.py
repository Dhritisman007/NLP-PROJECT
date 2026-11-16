"""Script-agnostic text helpers: whitespace chunks with punctuation-free cores, number finding, deletion ranges.

Python's ``\\w`` splits Devanagari words at vowel signs, so tokenisation here is whitespace based and trims
punctuation from the chunk edges instead.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

TRAIL = frozenset(".,;:!?।|)]}”’\"'»…")
LEAD = frozenset("([{“‘\"'«„‚¿¡")
BRACKETS_QUOTES = frozenset("\"“”„‘’'«»()[]{}")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*(?!\w)")


@dataclass(frozen=True)
class Chunk:
    """A whitespace-delimited chunk: ``start:end`` incl. punctuation, ``cs:ce`` the core without edge punctuation."""

    start: int
    end: int
    cs: int
    ce: int
    text: str

    @property
    def core(self) -> str:
        return self.text[self.cs - self.start:self.ce - self.start]


def chunks(text: str) -> list[Chunk]:
    """Split ``text`` into :class:`Chunk` objects (chunks that are pure punctuation get an empty core)."""
    out: list[Chunk] = []
    for m in re.finditer(r"\S+", text):
        s, e = m.start(), m.end()
        cs, ce = s, e
        while cs < ce and text[cs] in LEAD:
            cs += 1
        while ce > cs and text[ce - 1] in TRAIL:
            ce -= 1
        out.append(Chunk(s, e, cs, ce, m.group()))
    return out


def content_end(text: str) -> int:
    """Index just after the last content character (before trailing sentence punctuation / quotes / spaces)."""
    i = len(text)
    while i > 0 and (text[i - 1] in TRAIL or text[i - 1].isspace()):
        i -= 1
    return i


def clause_commas(text: str) -> list[int]:
    """Positions of commas that separate clauses (decimal / thousands commas like ``4,5`` are skipped)."""
    return [i for i, c in enumerate(text)
            if c == "," and not (0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit())]


def number_spans(text: str) -> list[tuple[int, int]]:
    """Spans of numbers (digits with optional ``. , :`` groups) that are not glued to letters."""
    return [m.span() for m in NUMBER_RE.finditer(text)]


def deletion_range(text: str, s: int, e: int) -> tuple[int, int]:
    """Widen ``s:e`` by exactly one adjacent space so that deleting it leaves single spacing."""
    if e < len(text) and text[e] == " ":
        return s, e + 1
    if s > 0 and text[s - 1] == " ":
        return s - 1, e
    return s, e


def overlaps(s: int, e: int, spans: list[tuple[int, int]]) -> bool:
    """True if ``s:e`` intersects any span (a zero-width point strictly inside a span counts; one on its edge doesn't)."""
    return any(s < b and a < e for a, b in spans)


def is_word_boundary(text: str, i: int) -> bool:
    """True if position ``i`` is not inside a word (letters, digits and combining marks are word characters)."""
    if i <= 0 or i >= len(text):
        return True
    a, b = text[i - 1], text[i]
    wa = unicodedata.category(a)[0] in "LMN"
    wb = unicodedata.category(b)[0] in "LMN"
    return not (wa and wb)
