"""Rule-based consistency checks between source and translation: numbers, dates, negation, entities.

No model is needed. Each check compares what the English source contains with what the translation contains and
emits numeric features plus flagged spans (``side="src"`` for source content missing from the translation,
``side="tgt"`` for suspicious content in the translation). Checks are language aware (German decimal commas,
Indian digit grouping, number words, month names in the target language).

Known noise sources (documented, not hidden): number words other than 2-20/tens, "twice"/"half", exonyms outside
the lexicon, and negation cues like "neither...nor" make the parity features approximate.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from typing import Any

from mml.detect import lexicon as lx
from mml.detect.base import Signal, SignalResult, SpanFlag
from mml.synth.lang import get_resources
from mml.synth.text import chunks, is_word_boundary

NUM_RE = re.compile(r"(?<!\w)\d+(?:[.,:]\d+)*")
EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'’][A-Za-z0-9]+)*")
STARTERS = frozenset({"The", "A", "An", "In", "On", "At", "He", "She", "It", "They", "We", "I", "This", "That",
                      "These", "Those", "If", "When", "While", "After", "Before", "As", "But", "And", "Or", "However",
                      "According", "Our", "My", "His", "Her", "Their", "There", "What", "Who", "How", "For", "With"})
NON_PERSON_KEYS = frozenset({"New York", "Sri Lanka", "United Nations", "United States", "United Kingdom",
                             "Oxford University", "University of Oxford"})


@dataclass(frozen=True)
class Mention:
    """A number / date / entity mention with comparable ``values`` (sets so several readings can match)."""

    start: int
    end: int
    text: str
    values: frozenset[Any]
    is_word: bool = False


def _ascii_digits(s: str) -> str:
    return "".join(str(unicodedata.decimal(c)) if c.isdigit() else c for c in s)


def number_values(raw: str, decimal: str = ".") -> frozenset[Any]:
    """Comparable readings of a numeric string: a ``Decimal``, and for clock-like strings a ``("t", h, m)`` tuple."""
    t = _ascii_digits(raw)
    if ":" in t:
        parts = t.split(":")
        try:
            h, m = int(parts[0]), int(parts[1])
        except ValueError:
            return frozenset()
        vals: set[Any] = {("t", h, m)}
        if h < 24 and m < 60:
            vals.add(Decimal(h * 100 + m))
        return frozenset(vals)
    t = t.replace(".", "").replace(",", ".") if decimal == "," else t.replace(",", "")
    try:
        value = Decimal(t)
    except InvalidOperation:
        digits = re.sub(r"\D", "", t)
        value = Decimal(digits) if digits else Decimal(0)
    vals = {value}
    if t.isdigit() and len(t) == 4 and int(t[:2]) < 24 and int(t[2:]) < 60:
        vals.add(("t", int(t[:2]), int(t[2:])))
    return frozenset(vals)


def find_numbers(text: str, lang: str) -> list[Mention]:
    """Digit numbers plus (2..20, tens) number words of ``lang`` (an FLORES code), with character spans."""
    decimal = lx.DECIMAL_SEPARATOR.get(lang, ".")
    out = [Mention(m.start(), m.end(), m.group(), number_values(m.group(), decimal)) for m in NUM_RE.finditer(text)]
    if lang == "eng_Latn":
        taken: list[tuple[int, int]] = []
        for m in re.finditer(r"\b(%s)[- ](%s)\b" % ("|".join(lx.EN_TENS), "|".join(lx.EN_UNITS)), text, re.I):
            taken.append(m.span())
            val = lx.EN_TENS[m.group(1).lower()] + lx.EN_UNITS[m.group(2).lower()]
            out.append(Mention(m.start(), m.end(), m.group(), frozenset({Decimal(val)}), True))
        for m in re.finditer(r"\b[A-Za-z]+\b", text):
            w = m.group().lower()
            if w in lx.EN_NUM_WORDS and w not in lx.AMBIGUOUS_NUMBER_WORDS and not any(a <= m.start() < b for a, b in taken):
                out.append(Mention(m.start(), m.end(), m.group(), frozenset({Decimal(lx.EN_NUM_WORDS[w])}), True))
    else:
        words = lx.TARGET_NUM_WORDS.get(lang, {})
        for c in chunks(text):
            if c.core in words:
                out.append(Mention(c.cs, c.ce, c.core, frozenset({Decimal(words[c.core])}), True))
    return sorted(out, key=lambda m: m.start)


def match_numbers(src: list[Mention], tgt: list[Mention]) -> tuple[list[Mention], list[Mention]]:
    """Greedy multiset matching by value. Returns (source mentions missing in tgt, unmatched tgt digit mentions)."""
    used: set[int] = set()
    missing: list[Mention] = []
    for sm in src:
        for j, tm in enumerate(tgt):
            if j not in used and sm.values & tm.values:
                used.add(j)
                break
        else:
            missing.append(sm)
    extra = [tm for j, tm in enumerate(tgt) if j not in used and not tm.is_word]
    return missing, extra


# ------------------------------------------------------------------ dates
def _date_table(lang: str) -> dict[str, tuple[str, int]]:
    res = get_resources(lang)
    table = {m: ("month", i) for i, m in enumerate(res.months)}
    table.update({d: ("weekday", i) for i, d in enumerate(res.weekdays)})
    if lang == "hin_Deva":
        table.update({k: ("month", v) for k, v in lx.HI_MONTH_VARIANTS.items()})
        table.update({k: ("weekday", v) for k, v in lx.HI_WEEKDAY_VARIANTS.items()})
    return table


def find_dates(text: str, lang: str) -> list[tuple[int, int, str, tuple[str, int]]]:
    """Month / weekday mentions as ``(start, end, text, (kind, index))``."""
    found = []
    if lang == "eng_Latn":
        for m in re.finditer(r"\b[A-Za-z]+\b", text):
            w = m.group().lower()
            if w in lx.EN_MONTHS:
                near_digit = bool(re.search(r"\d", text[max(0, m.start() - 6):m.start()] + text[m.end():m.end() + 6]))
                if w == "may" and not near_digit:
                    continue                                  # modal verb, not the month
                found.append((m.start(), m.end(), m.group(), ("month", lx.EN_MONTHS.index(w))))
            elif w in lx.EN_WEEKDAYS:
                found.append((m.start(), m.end(), m.group(), ("weekday", lx.EN_WEEKDAYS.index(w))))
    else:
        table = _date_table(lang)
        for c in chunks(text):
            if c.core in table:
                found.append((c.cs, c.ce, c.core, table[c.core]))
    return found


# ------------------------------------------------------------------ negation
def negation_cues(text: str, lang: str) -> list[tuple[int, int, str]]:
    """Negation cue tokens with spans (English cues, ``n't`` contractions, or the target-language cue list)."""
    cues = []
    if lang == "eng_Latn":
        for m in re.finditer(r"[A-Za-z]+(?:[’']t)?", text):
            w = m.group().lower().replace("’", "'")
            if (w in lx.EN_NEG_WORDS and w != "nor") or w.endswith("n't"):
                cues.append((m.start(), m.end(), m.group()))
    else:
        words = lx.NEG_TARGET.get(lang, frozenset())
        cues = [(c.cs, c.ce, c.core) for c in chunks(text) if c.core.lower() in words]
    return cues


# ------------------------------------------------------------------ entities
def source_entities(src: str) -> list[tuple[int, int, str]]:
    """Heuristic English named-entity candidates: capitalised runs (not sentence-initial singletons), acronyms, IDs."""
    toks = [(m.start(), m.end(), m.group()) for m in EN_WORD_RE.finditer(src)]
    month_day = set(lx.EN_MONTHS) | set(lx.EN_WEEKDAYS)

    def is_cap(i: int) -> bool:
        s, _, w = toks[i]
        if w.lower() in month_day or (w in STARTERS and w != "I") or w == "I":
            return False
        return (w[0].isupper() and not w.isdigit()) or (any(c.isdigit() for c in w) and any(c.isalpha() for c in w))

    groups: list[list[int]] = []
    i = 0
    while i < len(toks):
        if is_cap(i):
            g = [i]
            j = i + 1
            while j < len(toks) and src[toks[j - 1][1]:toks[j][0]] == " ":
                if is_cap(j) or (toks[j][2] in ("of", "de", "van") and j + 1 < len(toks) and is_cap(j + 1)):
                    g.append(j)
                    j += 1
                else:
                    break
            groups.append(g)
            i = j
        else:
            i += 1
    out = []
    for g in groups:
        s, e = toks[g[0]][0], toks[g[-1]][1]
        sentence_initial = s == 0 or src[:s].rstrip().endswith((".", "?", "!", ":", '"', "“"))
        text = re.sub(r"['’]s$", "", src[s:e])
        low = text.lower()
        if (low in lx.NON_ENTITY_WORDS or low in lx.DEMONYMS or text[0].isdigit()
                or (len(g) == 1 and not text.isupper() and low.endswith(lx.DEMONYM_SUFFIXES) and len(text) > 4
                    and low not in lx.EXONYMS_DE)):
            continue                                          # nationality adjectives, ordinals/decades, common nouns
        e = s + len(text)
        if sentence_initial and len(g) == 1 and not (text.isupper() and len(text) >= 2) and not any(c.isdigit() for c in text):
            continue
        out.append((s, e, text))
    return out


def _present_latin(entity: str, tgt: str) -> bool:
    """Entity (or its German exonym) appears in a Latin-script target, exactly or fuzzily (ratio >= 0.8)."""
    low = tgt.lower()
    variants = {entity.lower(), lx.EXONYMS_DE.get(entity.lower(), entity.lower())}
    if any(v in low for v in variants):
        return True
    words = low.split()
    n = len(entity.split())
    for v in variants:
        for i in range(max(1, len(words) - n + 1)):
            if SequenceMatcher(None, v, " ".join(words[i:i + n]).strip(".,;:!?\"()")).ratio() >= 0.8:
                return True
    return False


def _lexicon_variants(key: str) -> list[str]:
    variants = key.split("|")
    for v in list(variants):
        parts = v.split()
        if len(parts) == 2 and v not in NON_PERSON_KEYS and len(parts[1]) >= 4:
            variants.append(parts[1])                         # "Modi" for "Narendra Modi"
    return variants


def _find_word(text: str, needle: str, ci: bool = False) -> int:
    hay, nd = (text.lower(), needle.lower()) if ci else (text, needle)
    start = hay.find(nd)
    while start != -1:
        if is_word_boundary(text, start) and is_word_boundary(text, start + len(nd)):
            return start
        start = hay.find(nd, start + 1)
    return -1


def _find_unit(text: str, form: str, lang: str) -> int:
    """German: case-insensitive substring for forms longer than 3 letters (compounds such as *Quadratkilometer*);
    short forms and other languages: whole word."""
    if lang == "deu_Latn" and len(form) > 3:
        return text.lower().find(form.lower())
    return _find_word(text, form, ci=lang == "deu_Latn")


def _drop_nested(hits: dict[str, tuple[int, str]]) -> dict[str, tuple[int, str]]:
    """Discard unit hits that lie inside a longer hit of another unit ("meter" inside "Kilometer")."""
    def inside(a: tuple[int, str], b: tuple[int, str]) -> bool:
        return len(b[1]) > len(a[1]) and b[0] <= a[0] and a[0] + len(a[1]) <= b[0] + len(b[1])
    return {k: v for k, v in hits.items() if not any(inside(v, w) for kk, w in hits.items() if kk != k)}


# ------------------------------------------------------------------ the combined check
def check_pair(src: str, hyp: str, src_lang: str, tgt_lang: str) -> SignalResult:
    """Run all rule checks on one pair and return features + flagged spans."""
    spans: list[SpanFlag] = []

    # numbers
    s_num, t_num = find_numbers(src, src_lang), find_numbers(hyp, tgt_lang)
    missing, extra = match_numbers(s_num, t_num)
    spans += [SpanFlag("src", m.start, m.end, m.text, "number_missing") for m in missing]
    spans += [SpanFlag("tgt", m.start, m.end, m.text, "number_extra") for m in extra]

    # dates
    s_dates, t_dates = find_dates(src, src_lang), find_dates(hyp, tgt_lang)
    t_pool = [d for d in t_dates]
    d_missing = []
    for d in s_dates:
        match = next((t for t in t_pool if t[3] == d[3]), None)
        if match:
            t_pool.remove(match)
        else:
            d_missing.append(d)
    spans += [SpanFlag("src", d[0], d[1], d[2], "date_missing") for d in d_missing]
    spans += [SpanFlag("tgt", d[0], d[1], d[2], "date_extra") for d in t_pool]

    # negation
    s_neg, t_neg = negation_cues(src, src_lang), negation_cues(hyp, tgt_lang)
    parity = float((len(s_neg) - len(t_neg)) % 2 != 0)
    if len(t_neg) > len(s_neg):
        spans += [SpanFlag("tgt", a, b, w, "negation_extra") for a, b, w in t_neg]
    elif len(s_neg) > len(t_neg):
        spans += [SpanFlag("src", a, b, w, "negation_missing") for a, b, w in s_neg]

    # generic entities
    latin = tgt_lang.endswith("Latn")
    checkable, ent_missing = 0, 0
    for s, e, text in source_entities(src):
        preservable = any(c.isdigit() for c in text) or (text.isupper() and len(text) >= 2)
        if not (latin or preservable):
            continue
        checkable += 1
        present = _present_latin(text, hyp) if latin else text.lower() in hyp.lower()
        if not present:
            ent_missing += 1
            spans.append(SpanFlag("src", s, e, text, "entity_missing"))

    # closed-lexicon entities (see lexicon.py caveat)
    lex_src = lex_missing = lex_extra = 0
    for key, forms in lx.BILINGUAL_ENTITIES.get(tgt_lang, {}).items():
        variants = _lexicon_variants(key)
        s_pos = next((p for p in (_find_word(src, v) for v in variants) if p != -1), -1)
        t_hit = next(((p, f) for f in forms if (p := _find_word(hyp, f)) != -1), None)
        if s_pos != -1:
            lex_src += 1
            if t_hit is None:
                lex_missing += 1
                v0 = next(v for v in variants if _find_word(src, v) != -1)
                spans.append(SpanFlag("src", s_pos, s_pos + len(v0), v0, "lex_entity_missing"))
        elif t_hit is not None:
            lex_extra += 1
            spans.append(SpanFlag("tgt", t_hit[0], t_hit[0] + len(t_hit[1]), t_hit[1], "lex_entity_extra"))

    # closed-lexicon units (see lexicon.py caveat)
    unit_src = unit_missing = unit_extra = 0
    unit_table = lx.UNITS.get(tgt_lang, {})
    t_hits = _drop_nested({k: hit for k, fs in unit_table.items()
                           if (hit := next(((p, f) for f in fs if (p := _find_unit(hyp, f, tgt_lang)) != -1), None))})
    for key in unit_table:
        variants = key.split("|")
        s_hit = next(((p, v) for v in variants if (p := _find_word(src, v, ci=True)) != -1), None)
        t_hit = t_hits.get(key)
        if s_hit:
            unit_src += 1
            if t_hit is None:
                unit_missing += 1
                spans.append(SpanFlag("src", s_hit[0], s_hit[0] + len(s_hit[1]), s_hit[1], "lex_unit_missing"))
        elif t_hit is not None:
            unit_extra += 1
            spans.append(SpanFlag("tgt", t_hit[0], t_hit[0] + len(t_hit[1]), t_hit[1], "lex_unit_extra"))

    feats = {
        "num_src": float(len(s_num)), "num_tgt": float(len(t_num)), "num_missing": float(len(missing)),
        "num_extra": float(len(extra)), "num_missing_frac": len(missing) / len(s_num) if s_num else 0.0,
        "date_src": float(len(s_dates)), "date_missing": float(len(d_missing)), "date_extra": float(len(t_pool)),
        "neg_src": float(len(s_neg)), "neg_tgt": float(len(t_neg)), "neg_diff": float(len(t_neg) - len(s_neg)),
        "neg_parity_mismatch": parity,
        "ent_src": float(checkable), "ent_missing": float(ent_missing),
        "ent_missing_frac": ent_missing / checkable if checkable else 0.0,
        "lex_ent_src": float(lex_src), "lex_ent_missing": float(lex_missing), "lex_ent_extra": float(lex_extra),
        "lex_unit_src": float(unit_src), "lex_unit_missing": float(unit_missing), "lex_unit_extra": float(unit_extra),
    }
    return SignalResult(feats, sorted(spans, key=lambda s: (s.side, s.start)))


class RuleChecks(Signal):
    """Signal wrapper around :func:`check_pair` (cheap, so usually run without a cache)."""

    name = "rules"
    chunk_size = 512

    def signature(self) -> dict[str, Any]:
        return {"version": 1, "src": self.src_lang, "tgt": self.tgt_lang}

    def _compute(self, pairs: list[tuple[str, str]]) -> list[SignalResult]:
        return [check_pair(s, h, self.src_lang, self.tgt_lang) for s, h in pairs]
