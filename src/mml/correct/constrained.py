"""Constrained fixing of numbers, dates, units and known names - deterministic, no LLM.

"Force entities/numbers into the output where feasible": the consistency rules (``detect/rules.py``) tell us which
source numbers/dates/units/names are missing from the translation and which translation items have no source
counterpart. Where a missing item and a wrong item pair up one-to-one, the wrong text is replaced by the correct form
in the target language's conventions (German decimal comma, month/weekday names, unit and name lexicons). A missing
number can also be inserted when it is followed in the source by a unit word that is present in the translation.
Anything else is reported as ``unresolved`` (left to the LLM).

Every fix is verified: the rules are re-run on the edited sentence and the fix is reverted if it leaves *more* flags
than before. Caveat (documented in the README): the date/unit/name lexicons are closed lists that overlap with the
synthetic-error generator, so on synthetic data this component looks better than it will on real MT output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from mml.detect import lexicon as lx
from mml.detect.rules import (
    Mention, _find_unit, check_pair, find_dates, find_numbers,
)
from mml.synth.lang import get_resources
from mml.synth.text import chunks

FIXABLE_KINDS = ("number_missing", "number_extra", "date_missing", "date_extra", "lex_unit_missing", "lex_unit_extra",
                 "lex_entity_missing", "lex_entity_extra")


@dataclass
class AppliedFix:
    """One replacement/insertion applied to the translation (offsets refer to the ORIGINAL translation)."""

    kind: str            # number | number_insert | date | unit | entity
    start: int
    end: int
    old: str
    new: str


@dataclass
class ConstrainedResult:
    text: str
    fixes: list[AppliedFix] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.fixes)


def render_number(src_text: str, src_value: frozenset, tgt_lang: str) -> str:
    """Write a source number the way the target language writes it (German swaps decimal/thousands separators)."""
    if src_text and src_text[0].isdigit():
        if tgt_lang == "deu_Latn" and ":" not in src_text:
            return src_text.translate(str.maketrans({",": ".", ".": ","}))
        return src_text
    value = next((v for v in src_value if isinstance(v, Decimal)), None)       # number word -> digits
    return str(int(value)) if value is not None else src_text


def _match_case(old: str, new: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() and new else new


def _flag_count(src: str, hyp: str, src_lang: str, tgt_lang: str) -> int:
    return sum(s.kind in FIXABLE_KINDS for s in check_pair(src, hyp, src_lang, tgt_lang).spans)


def ordered_match(src: list[Mention], tgt: list[Mention]) -> tuple[list[Mention], list[Mention]]:
    """Order-preserving matching of numbers by value (DP). Returns (source mentions missing, unmatched target digit mentions).

    Unlike the greedy matcher used for the detector's features, this keeps the sequence order, and among equally good
    matchings prefers pairs at similar relative positions - so in "3 ... 3" vs source "1 ... 3" the *first* 3 is the
    wrong one, not the second.
    """
    n, m = len(src), len(tgt)
    best = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]          # (number of matches, -position cost)
    take = [[False] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cand = best[i - 1][j] if best[i - 1][j] >= best[i][j - 1] else best[i][j - 1]
            if src[i - 1].values & tgt[j - 1].values:
                cost = abs((i - 1) / max(n, 1) - (j - 1) / max(m, 1))
                diag = (best[i - 1][j - 1][0] + 1, best[i - 1][j - 1][1] - cost)
                if diag > cand:
                    cand, take[i][j] = diag, True
            best[i][j] = cand
    matched_s, matched_t = set(), set()
    i, j = n, m
    while i > 0 and j > 0:
        if take[i][j]:
            matched_s.add(i - 1)
            matched_t.add(j - 1)
            i, j = i - 1, j - 1
        elif best[i - 1][j] >= best[i][j - 1]:
            i -= 1
        else:
            j -= 1
    missing = [s for k, s in enumerate(src) if k not in matched_s]
    extra = [t for k, t in enumerate(tgt) if k not in matched_t and not t.is_word]
    return missing, extra


def _number_fixes(src: str, hyp: str, src_lang: str, tgt_lang: str) -> tuple[list[AppliedFix], list[str]]:
    s_num, t_num = find_numbers(src, src_lang), find_numbers(hyp, tgt_lang)
    missing, extra = ordered_match(s_num, t_num)
    fixes: list[AppliedFix] = []
    unresolved: list[str] = []
    n_pair = min(len(missing), len(extra))
    for m, e in zip(missing[:n_pair], extra[:n_pair]):
        if m.text == e.text:
            # The translation already writes the number exactly as the source does ("802.11n", "12.00"): the rules only
            # disagree because German reads "." as a thousands separator. That is not an error - never rewrite it.
            continue
        fixes.append(AppliedFix("number", e.start, e.end, e.text, render_number(m.text, m.values, tgt_lang)))
    for m in missing[n_pair:]:
        fix = _insert_missing_number(src, hyp, m, tgt_lang)
        if fix:
            fixes.append(fix)
        else:
            unresolved.append(f"source number {m.text!r} missing and no safe insertion point")
    unresolved += [f"translation number {e.text!r} has no source counterpart" for e in extra[n_pair:]]
    return fixes, unresolved


def _insert_missing_number(src: str, hyp: str, m: Mention, tgt_lang: str) -> AppliedFix | None:
    """Insert a missing number in front of its unit when the source word after the number is a known unit."""
    following = re.match(r"\s*([A-Za-z$€]+)", src[m.end:])
    if not following:
        return None
    word = following.group(1).lower()
    for key, forms in lx.UNITS.get(tgt_lang, {}).items():
        if word in (v.lower() for v in key.split("|")):
            for form in forms:
                pos = _find_unit(hyp, form, tgt_lang)
                if pos != -1:
                    return AppliedFix("number_insert", pos, pos, "", render_number(m.text, m.values, tgt_lang) + " ")
    return None


def _date_fixes(src: str, hyp: str, src_lang: str, tgt_lang: str) -> list[AppliedFix]:
    s_dates, t_dates = find_dates(src, src_lang), find_dates(hyp, tgt_lang)
    pool = list(t_dates)
    missing = []
    for d in s_dates:
        match = next((t for t in pool if t[3] == d[3]), None)
        if match:
            pool.remove(match)
        else:
            missing.append(d)
    res = get_resources(tgt_lang)
    fixes = []
    for d in missing:
        kind, idx = d[3]
        partner = next((t for t in pool if t[3][0] == kind), None)
        names = res.months if kind == "month" else res.weekdays
        if partner is not None and idx < len(names):
            pool.remove(partner)
            fixes.append(AppliedFix("date", partner[0], partner[1], partner[2], names[idx]))
    return fixes


def _lexicon_fixes(src: str, hyp: str, src_lang: str, tgt_lang: str) -> list[AppliedFix]:
    spans = check_pair(src, hyp, src_lang, tgt_lang).spans
    fixes: list[AppliedFix] = []
    for missing_kind, extra_kind, table, label in (
            ("lex_unit_missing", "lex_unit_extra", lx.UNITS.get(tgt_lang, {}), "unit"),
            ("lex_entity_missing", "lex_entity_extra", lx.BILINGUAL_ENTITIES.get(tgt_lang, {}), "entity")):
        miss = [s for s in spans if s.kind == missing_kind]
        extra = [s for s in spans if s.kind == extra_kind]
        for m, e in zip(miss, extra):
            for key, forms in table.items():
                variants = [v.lower() for v in key.split("|")] + [p.lower() for v in key.split("|") for p in v.split()[-1:]]
                if m.text.lower() in variants and forms:
                    surface = hyp[e.start:e.end]          # the flag's text may be a lower-cased lexicon form
                    fixes.append(AppliedFix(label, e.start, e.end, surface, _match_case(surface, forms[0])))
                    break
    return fixes


def _apply(text: str, fixes: list[AppliedFix]) -> tuple[str, list[AppliedFix]]:
    """Apply non-overlapping fixes right to left; overlapping ones (later in the list) are dropped."""
    kept: list[AppliedFix] = []
    for f in sorted(fixes, key=lambda f: (f.start, f.end)):
        if kept and f.start < kept[-1].end:
            continue
        kept.append(f)
    for f in sorted(kept, key=lambda f: f.start, reverse=True):
        text = text[:f.start] + f.new + text[f.end:]
    return text, kept


def constrained_fix(src: str, hyp: str, src_lang: str, tgt_lang: str) -> ConstrainedResult:
    """Force source numbers/dates/units/names into ``hyp`` where feasible; verified and reverted if it makes things worse."""
    num_fixes, unresolved = _number_fixes(src, hyp, src_lang, tgt_lang)
    fixes = num_fixes + _date_fixes(src, hyp, src_lang, tgt_lang) + _lexicon_fixes(src, hyp, src_lang, tgt_lang)
    if not fixes:
        return ConstrainedResult(hyp, [], unresolved)
    new_text, kept = _apply(hyp, fixes)
    if new_text == hyp:
        return ConstrainedResult(hyp, [], unresolved)
    before = _flag_count(src, hyp, src_lang, tgt_lang)
    after = _flag_count(src, new_text, src_lang, tgt_lang)
    if after > before or (after == before and not _improves(src, hyp, new_text, src_lang, tgt_lang)):
        return ConstrainedResult(hyp, [], unresolved + ["fixes reverted: they did not reduce the number of consistency flags"])
    return ConstrainedResult(new_text, kept, unresolved if after else [])


def _improves(src: str, old: str, new: str, src_lang: str, tgt_lang: str) -> bool:
    """True if the edited text has strictly fewer fixable flags of at least one kind (guards the equal-count case)."""
    from collections import Counter

    a = Counter(s.kind for s in check_pair(src, old, src_lang, tgt_lang).spans if s.kind in FIXABLE_KINDS)
    b = Counter(s.kind for s in check_pair(src, new, src_lang, tgt_lang).spans if s.kind in FIXABLE_KINDS)
    return any(b[k] < a[k] for k in a) and sum(b.values()) <= sum(a.values())
