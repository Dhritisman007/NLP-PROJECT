"""Synthetic meaning-loss errors injected into correct reference translations, with exact labelled spans.

Error types and mechanisms
--------------------------
* ``OMISSION``       delete a comma clause, a trailing conjunction clause, or a 2-4 word phrase that contains no
                     number, entity, date, unit or negation (those belong to other classes).
* ``MISTRANSLATION`` flip negation (remove/replace a negation word, or insert one) or swap two entity mentions
                     (role swap; an approximation of subject/object swap without a parser).
* ``ENTITY_NUMBER``  change a digit / magnitude of a number, delete a number, change a month/weekday, swap a unit,
                     replace an entity with another of the same type, or delete an entity.
* ``HALLUCINATION``  add an unrelated sentence before or after the translation.

Every injector returns :class:`mml.synth.types.Edit` objects; spans are computed centrally by ``apply_edits``.
"""

from __future__ import annotations

import logging
import random
from typing import Callable

from mml.synth.entities import EntityFinder, EntityPool, GazetteerFinder
from mml.synth.lang import LangResources
from mml.synth.text import (
    BRACKETS_QUOTES, Chunk, chunks, clause_commas, content_end, deletion_range, number_spans, overlaps,
)
from mml.synth.types import Edit, ErrorSpan, ErrorType, apply_edits, invert_spans

log = logging.getLogger(__name__)

Injector = Callable[[str, random.Random], "list[Edit] | None"]


# ------------------------------------------------------------------ number mutation
def _valid_number(s: str) -> bool:
    """Reject impossible clock times produced by digit mutation."""
    if ":" in s:
        parts = s.split(":")
        try:
            return int(parts[0]) < 24 and int(parts[1]) < 60
        except ValueError:
            return False
    return True


def mutate_number(num: str, rng: random.Random) -> str | None:
    """Return a different number string with the same formatting (separators, width), or None.

    Ops: change one digit (years: only the last two), or change magnitude (append/drop a trailing zero).
    """
    digits = [i for i, c in enumerate(num) if c.isdigit()]
    if not digits:
        return None
    is_year = num.isdigit() and len(num) == 4 and 1000 <= int(num) <= 2100
    if is_year:
        digits = digits[2:]
    for _ in range(30):
        if num.isdigit() and not is_year and not num.startswith("0") and rng.random() < 0.3:
            # magnitude change: append a zero, or drop one if there is a trailing zero
            new = num[:-1] if (num.endswith("0") and len(num) > 1 and rng.random() < 0.5) else num + "0"
        else:
            i = rng.choice(digits)
            options = [d for d in "0123456789" if d != num[i]]
            if i == digits[0] and num[0] != "0":
                options = [d for d in options if d != "0"]  # never turn the leading digit into 0
            new = num[:i] + rng.choice(options) + num[i + 1:]
        if new != num and _valid_number(new):
            return new
    return None


# ------------------------------------------------------------------ injector
class ErrorInjector:
    """Injects one labelled error into a reference translation (deterministic given the ``rng``)."""

    def __init__(self, lang: LangResources, finder: EntityFinder | None = None,
                 pool: EntityPool | None = None) -> None:
        self.lang = lang
        self.finder: EntityFinder = finder or GazetteerFinder(lang.gazetteer)
        self.pool = pool or EntityPool(lang.gazetteer)
        self._dispatch: dict[ErrorType, Injector] = {
            ErrorType.OMISSION: self._omission,
            ErrorType.MISTRANSLATION: self._mistranslation,
            ErrorType.ENTITY_NUMBER: self._entity_number,
            ErrorType.HALLUCINATION: self._hallucination,
        }

    # -- public API
    def inject(self, ref: str, etype: ErrorType, rng: random.Random) -> tuple[str, list[ErrorSpan]] | None:
        """Corrupt ``ref`` with one error of ``etype``; ``None`` if the sentence offers no suitable site.

        Raises ``AssertionError`` if the recorded spans do not restore the reference (a bug, never data noise).
        """
        if etype not in self._dispatch:
            raise ValueError(f"Cannot inject {etype}")
        edits = self._dispatch[etype](ref, rng)
        if not edits:
            return None
        hyp, spans = apply_edits(ref, edits, etype)
        if not self._acceptable(ref, hyp):
            return None
        restored = invert_spans(hyp, spans)
        assert restored == ref, f"span bookkeeping bug for {etype}: {ref!r} -> {hyp!r} -> {restored!r}"
        return hyp, spans

    @staticmethod
    def _acceptable(ref: str, hyp: str) -> bool:
        """Reject no-ops and formatting artefacts introduced by an edit (double spaces, space before punctuation)."""
        if hyp == ref or not hyp.strip():
            return False
        for artefact in ("  ", " ,", " .", " ।", " ;", " :"):
            if artefact in hyp and artefact not in ref:
                return False
        return True

    def _protected(self, ref: str) -> list[tuple[int, int]]:
        """Spans that carry ENTITY/NUMBER or polarity content and must not be touched by OMISSION."""
        lang = self.lang
        out = list(number_spans(ref))
        out += [(e.start, e.end) for e in self.finder.find(ref)]
        special = {w.lower() for w in (*lang.months, *lang.weekdays, *lang.units, *lang.neg_delete, *lang.neg_replace)}
        out += [(c.cs, c.ce) for c in chunks(ref) if c.core.lower() in special]
        # any chunk with a digit, incl. ones glued to letters ("60वाँ", "39C") that number_spans() skips
        out += [(c.start, c.end) for c in chunks(ref) if any(ch.isdigit() for ch in c.core)]
        return out

    # -- OMISSION
    def _omission(self, ref: str, rng: random.Random) -> list[Edit] | None:
        ch = chunks(ref)
        n = len(ch)
        if n < 7:
            return None
        ce = content_end(ref)
        prot = self._protected(ref)
        cands: dict[str, list[tuple[int, int]]] = {"clause": [], "conj_clause": [], "phrase": []}

        def ok(s: int, e: int) -> bool:
            region = ref[s:e]
            k = len(region.split())
            return (2 <= k <= int(0.6 * n) and n - k >= 4 and not overlaps(s, e, prot)
                    and not any(c in BRACKETS_QUOTES for c in region) and e > s)

        commas = [c for c in clause_commas(ref) if c < ce]
        bounds = [-1, *commas, ce]
        for i in range(len(bounds) - 1 if len(bounds) >= 3 else 0):  # clause deletion needs >= 2 clauses
            seg_s = bounds[i] + 1
            while seg_s < bounds[i + 1] and ref[seg_s] == " ":
                seg_s += 1
            if i == len(bounds) - 2:
                region = (bounds[i], ce)                      # ", last clause"
            elif i == 0:
                nxt = bounds[1] + 1
                while nxt < len(ref) and ref[nxt] == " ":
                    nxt += 1
                region = (seg_s, nxt)                         # "first clause, "
                if nxt < len(ref) and ref[nxt].islower():
                    continue                                   # would leave a lower-case sentence start
            else:
                region = (bounds[i], bounds[i + 1])           # ", middle clause"
            if ok(*region):
                cands["clause"].append(region)

        for k, c in enumerate(ch):
            if k >= 3 and c.core.lower() in self.lang.conj and ok(ch[k - 1].ce, ce):
                cands["conj_clause"].append((ch[k - 1].ce, ce))

        for size in (2, 3, 4):
            for i in range(1, n - size - 1):
                s, e = deletion_range(ref, ch[i].start, ch[i + size - 1].end)
                if ok(s, e):
                    cands["phrase"].append((s, e))

        avail = [(name, weight) for name, weight in (("clause", 3), ("conj_clause", 2), ("phrase", 1)) if cands[name]]
        if not avail:
            return None
        kind = rng.choices([name for name, _ in avail], weights=[weight for _, weight in avail])[0]
        s, e = rng.choice(sorted(cands[kind]))
        return [Edit(s, e, "", f"omit_{kind}")]

    # -- MISTRANSLATION
    def _mistranslation(self, ref: str, rng: random.Random) -> list[Edit] | None:
        options: list[tuple[int, Callable[[], list[Edit] | None]]] = [
            (4, lambda: self._negation(ref, rng)), (1, lambda: self._role_swap(ref, rng))]
        while options:
            idx = rng.choices(range(len(options)), weights=[w for w, _ in options])[0]
            edits = options.pop(idx)[1]()
            if edits:
                return edits
        return None

    def _negation(self, ref: str, rng: random.Random) -> list[Edit] | None:
        lang, ch, low = self.lang, chunks(ref), ref.lower()
        if any(s in low for s in lang.neg_skip_if_contains):
            return None
        removals: list[Edit] = []
        for k, c in enumerate(ch):
            w, nxt = c.core.lower(), (ch[k + 1].core.lower() if k + 1 < len(ch) else "")
            if w in lang.neg_delete and nxt not in lang.neg_skip_next and not (k == 0 and lang.code == "de"):
                s, e = deletion_range(ref, c.cs, c.ce)
                removals.append(Edit(s, e, "", "negation_remove"))
            elif w in lang.neg_replace:
                pos = lang.neg_replace[w]
                pos = pos.capitalize() if c.core[:1].isupper() else pos
                removals.append(Edit(c.cs, c.ce, pos, "negation_swap"))
        if removals:
            return [rng.choice(removals)]
        return self._negation_insert(ref, ch)

    def _negation_insert(self, ref: str, ch: list[Chunk]) -> list[Edit] | None:
        words = [c for c in ch if c.core and c.core[0].isalpha()]
        if len(words) < 4:
            return None
        lang = self.lang
        if lang.neg_insert_rule == "before_verb_tail":
            # Hindi is verb-final: put the negation before the trailing auxiliary run ("हो सकती है" ->
            # "हो नहीं सकती है"); with no auxiliary, before the last word ("पहचान करेगा" -> "पहचान नहीं करेगा").
            tail = len(words) - 1
            while tail > 1 and words[tail].core in lang.verb_aux and len(words) - tail <= 4:
                tail -= 1
            if words[tail].core in lang.verb_aux or tail == len(words) - 1:
                pos = words[tail]
            else:
                pos = words[tail + 1]
            return [Edit(pos.cs, pos.cs, lang.neg_word + " ", "negation_insert")]
        if lang.neg_insert_rule == "german":
            first_comma = next(iter(clause_commas(ref)), len(ref))
            for i, c in enumerate(ch):
                if 1 <= i < len(ch) - 1 and c.ce <= first_comma and c.core.lower() in lang.finite_verbs:
                    return [Edit(c.ce, c.ce, " " + lang.neg_word, "negation_insert")]
            last = words[-1]
            if last.core.islower() and last.core.endswith(("en", "t", "n")) and last.core.lower() not in lang.finite_verbs:
                return [Edit(last.cs, last.cs, lang.neg_word + " ", "negation_insert")]
        return None

    def _role_swap(self, ref: str, rng: random.Random) -> list[Edit] | None:
        ents = self.finder.find(ref)
        sym = {w.lower() for w in self.lang.symmetric_between}
        pairs = []
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                between = [t.strip(".,;:!?()\"'").lower() for t in ref[a.end:b.start].split()]
                content = [t for t in between if t and t not in sym]
                if a.text != b.text and content:
                    pairs.append((a, b))
        same_type = [p for p in pairs if p[0].etype == p[1].etype]
        pairs = same_type or pairs
        if not pairs:
            return None
        a, b = rng.choice(pairs)
        return [Edit(a.start, a.end, b.text, "role_swap", group=1), Edit(b.start, b.end, a.text, "role_swap", group=1)]

    # -- ENTITY / NUMBER
    def _entity_number(self, ref: str, rng: random.Random) -> list[Edit] | None:
        lang, ch = self.lang, chunks(ref)
        builders: list[tuple[int, Callable[[], Edit | None]]] = []
        for s, e in number_spans(ref):
            builders.append((3, lambda s=s, e=e: self._number_change(ref, s, e, rng)))
            builders.append((1, lambda s=s, e=e: self._delete_span(ref, s, e, "number_deleted")))
        for c in ch:
            core = c.core
            if core in lang.months or core in lang.weekdays:
                pool = lang.months if core in lang.months else lang.weekdays
                builders.append((3, lambda c=c, pool=pool: Edit(
                    c.cs, c.ce, rng.choice([m for m in pool if m != c.core]), "date_change")))
            elif core in lang.units:
                builders.append((3, lambda c=c: Edit(c.cs, c.ce, lang.units[c.core], "unit_change")))
        for ent in self.finder.find(ref):
            builders.append((3, lambda ent=ent: self._entity_replace(ent, rng)))
            builders.append((1, lambda ent=ent: self._delete_span(ref, ent.start, ent.end, "entity_deleted")))
        while builders:
            idx = rng.choices(range(len(builders)), weights=[w for w, _ in builders])[0]
            edit = builders.pop(idx)[1]()
            if edit is not None:
                return [edit]
        return None

    @staticmethod
    def _number_change(ref: str, s: int, e: int, rng: random.Random) -> Edit | None:
        new = mutate_number(ref[s:e], rng)
        return Edit(s, e, new, "number_change") if new else None

    @staticmethod
    def _delete_span(ref: str, s: int, e: int, subtype: str) -> Edit | None:
        if (s > 0 and ref[s - 1] in BRACKETS_QUOTES) or (e < len(ref) and ref[e] in BRACKETS_QUOTES):
            return None
        ds, de = deletion_range(ref, s, e)
        return Edit(ds, de, "", subtype)

    def _entity_replace(self, ent, rng: random.Random) -> Edit | None:  # type: ignore[no-untyped-def]
        new = self.pool.sample(ent.etype, ent.text, rng)
        return Edit(ent.start, ent.end, new, "entity_replace") if new else None

    # -- HALLUCINATION
    def _hallucination(self, ref: str, rng: random.Random) -> list[Edit] | None:
        if not self.lang.fillers:
            return None
        filler = rng.choice(self.lang.fillers)
        end = len(ref.rstrip())
        final = ref[end - 1] if end else ""
        if final in ".।|" and filler[-1] in ".।|":
            filler = filler[:-1] + final
        if rng.random() < 0.75:
            return [Edit(end, end, " " + filler, "hallucination_append")]
        return [Edit(0, 0, filler + " ", "hallucination_prepend")]
