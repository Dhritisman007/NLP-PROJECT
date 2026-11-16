"""Entity finding for the injectors: gazetteer (default, no model) and NER (plug-in), plus a replacement pool."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from mml.synth.text import is_word_boundary, overlaps

ENTITY_TYPES = ("PER", "LOC", "ORG")


@dataclass(frozen=True)
class EntitySpan:
    """An entity mention: ``text[start:end]`` of type PER/LOC/ORG."""

    start: int
    end: int
    text: str
    etype: str


class EntityFinder(Protocol):
    """Anything that can list entity mentions in a sentence."""

    def find(self, text: str) -> list[EntitySpan]: ...


def _resolve_overlaps(spans: Iterable[EntitySpan]) -> list[EntitySpan]:
    """Keep the longest span first, drop anything overlapping it; return sorted by position."""
    kept: list[EntitySpan] = []
    taken: list[tuple[int, int]] = []
    for sp in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if not overlaps(sp.start, sp.end, taken):
            kept.append(sp)
            taken.append((sp.start, sp.end))
    return sorted(kept, key=lambda s: s.start)


class GazetteerFinder:
    """Exact-match finder over ``{type: surface forms}`` with word-boundary checks (longest match wins)."""

    def __init__(self, gazetteer: dict[str, tuple[str, ...]]) -> None:
        self._forms = sorted(((form, t) for t, forms in gazetteer.items() for form in forms),
                             key=lambda ft: -len(ft[0]))

    def find(self, text: str) -> list[EntitySpan]:
        found: list[EntitySpan] = []
        for form, etype in self._forms:
            start = text.find(form)
            while start != -1:
                end = start + len(form)
                if is_word_boundary(text, start) and is_word_boundary(text, end):
                    found.append(EntitySpan(start, end, form, etype))
                start = text.find(form, start + 1)
        return _resolve_overlaps(found)


class NERFinder:
    """Wraps a Hugging Face token-classification pipeline (``aggregation_strategy='simple'``).

    ``pipeline_fn`` is injectable for tests; by default the model is loaded lazily from ``model_id``.
    """

    def __init__(self, model_id: str, device: int | str = -1, pipeline_fn: Callable[[str], list[dict[str, Any]]] | None = None,
                 token: str | None = None) -> None:
        self.model_id, self.device, self.token = model_id, device, token
        self._fn = pipeline_fn

    def _pipe(self) -> Callable[[str], list[dict[str, Any]]]:
        if self._fn is None:
            from transformers import pipeline

            self._fn = pipeline("token-classification", model=self.model_id, aggregation_strategy="simple",
                                device=self.device, token=self.token)
        return self._fn

    @staticmethod
    def _norm(label: str) -> str | None:
        label = label.upper().split("-")[-1]
        return label if label in ENTITY_TYPES else None

    def find(self, text: str) -> list[EntitySpan]:
        spans = []
        for ent in self._pipe()(text):
            etype = self._norm(str(ent.get("entity_group") or ent.get("entity", "")))
            s, e = int(ent["start"]), int(ent["end"])
            if etype and e > s and text[s:e].strip() == text[s:e]:
                spans.append(EntitySpan(s, e, text[s:e], etype))
        return _resolve_overlaps(spans)


class CompositeFinder:
    """Union of several finders (e.g. gazetteer + NER); overlaps resolved longest-first."""

    def __init__(self, *finders: EntityFinder) -> None:
        self.finders = finders

    def find(self, text: str) -> list[EntitySpan]:
        return _resolve_overlaps(sp for f in self.finders for sp in f.find(text))


class EntityPool:
    """Replacement candidates per entity type: built-in gazetteer forms plus entities seen in a corpus."""

    def __init__(self, builtin: dict[str, tuple[str, ...]] | None = None) -> None:
        self._pool: dict[str, list[str]] = {t: list(forms) for t, forms in (builtin or {}).items()}

    def add(self, span: EntitySpan) -> None:
        bucket = self._pool.setdefault(span.etype, [])
        if span.text not in bucket:
            bucket.append(span.text)

    def extend_from_corpus(self, finder: EntityFinder, texts: Iterable[str]) -> None:
        """Harvest entity strings from ``texts`` (use the *same split* that the examples come from)."""
        for text in texts:
            for sp in finder.find(text):
                self.add(sp)

    def sample(self, etype: str, exclude: str, rng: random.Random) -> str | None:
        """A random pool entry of ``etype`` different from ``exclude`` (None if the pool has no alternative)."""
        options = [e for e in self._pool.get(etype, []) if e != exclude and e not in exclude and exclude not in e]
        return rng.choice(sorted(options)) if options else None
