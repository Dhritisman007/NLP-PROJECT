"""Build the synthetic dataset: split assignment, type-balanced injection, paired clean examples, manifest.

Split policy (no leakage): FLORES ``devtest`` -> ``test`` (held out, never used for training/tuning);
FLORES ``dev`` -> ``train`` / ``val`` by a hash of the sentence id (85/15). The clean and corrupted versions of
one sentence always land in the same split. Entity replacement pools are built per split.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mml.config import Config
from mml.data import flores_source, read_flores
from mml.synth.entities import EntityFinder, EntityPool, GazetteerFinder
from mml.synth.inject_errors import ErrorInjector
from mml.synth.lang import get_resources
from mml.synth.types import ERROR_TYPES, ErrorType, SynthExample

log = logging.getLogger(__name__)

GENERATOR_VERSION = "1"
VAL_PERCENT = 15
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class ParallelItem:
    """One source sentence with its correct reference translation."""

    id: str
    origin_split: str   # dev | devtest | custom
    src: str
    ref: str


def assign_split(item_id: str, origin_split: str) -> str:
    """Deterministic split. ``devtest`` is always test; ``dev`` is train/val; ``custom`` is 70/15/15."""
    bucket = int(hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if origin_split == "devtest":
        return "test"
    if origin_split == "dev":
        return "val" if bucket < VAL_PERCENT else "train"
    return "test" if bucket < 15 else "val" if bucket < 30 else "train"


def load_flores_items(cfg: Config, limit: int | None = None) -> list[ParallelItem]:
    """FLORES dev + devtest as parallel (source, reference) items for the active pair."""
    data_dir = Path(cfg.raw["paths"]["data_dir"])
    source = flores_source(cfg)
    items: list[ParallelItem] = []
    for sp in ("dev", "devtest"):
        src = read_flores(data_dir, cfg.pair["src_lang"], sp, source)
        ref = read_flores(data_dir, cfg.pair["tgt_lang"], sp, source)
        if len(src) != len(ref):
            raise ValueError(f"FLORES {sp}: {len(src)} source vs {len(ref)} target lines")
        items += [ParallelItem(f"{cfg.pair_key}-{sp}-{i:04d}", sp, s.strip(), r.strip())
                  for i, (s, r) in enumerate(zip(src, ref)) if s.strip() and r.strip()]
    return items[:limit] if limit else items


def _tiebreak(seed: int, item_id: str, etype: ErrorType) -> float:
    return random.Random(f"{seed}|tie|{item_id}|{etype.value}").random()


def build_examples(items: list[ParallelItem], pair: str, tgt_lang: str, seed: int = 42,
                   finder: EntityFinder | None = None, balance: bool = False) -> tuple[list[SynthExample], dict[str, Any]]:
    """Create paired clean/corrupted examples. Returns (examples, stats).

    For each sentence the error type with the lowest running count (per split) is tried first, then the others;
    sentences where no type applies are skipped, so clean and corrupted counts are exactly equal.
    """
    lang = get_resources(tgt_lang)
    finder = finder or GazetteerFinder(lang.gazetteer)
    by_split: dict[str, list[ParallelItem]] = {s: [] for s in SPLITS}
    for it in items:
        by_split[assign_split(it.id, it.origin_split)].append(it)

    examples: list[SynthExample] = []
    skipped: Counter[str] = Counter()
    for split, split_items in by_split.items():
        pool = EntityPool(lang.gazetteer)
        pool.extend_from_corpus(finder, (i.ref for i in split_items))
        injector = ErrorInjector(lang, finder, pool)
        counts: Counter[ErrorType] = Counter()
        built: list[tuple[SynthExample, SynthExample]] = []
        for it in sorted(split_items, key=lambda i: i.id):
            for etype in sorted(ERROR_TYPES, key=lambda t: (counts[t], _tiebreak(seed, it.id, t))):
                res = injector.inject(it.ref, etype, random.Random(f"{seed}|{it.id}|{etype.value}"))
                if res is None:
                    continue
                hyp, spans = res
                counts[etype] += 1
                base = dict(pair=pair, split=split, origin_split=it.origin_split, src=it.src, ref=it.ref)
                clean = SynthExample(id=f"{it.id}-clean", hyp=it.ref, label=ErrorType.OK.value, **base)
                bad = SynthExample(id=f"{it.id}-{etype.value}", hyp=hyp, label=etype.value,
                                   subtype="+".join(sorted({s.subtype for s in spans})), errors=spans, **base)
                built.append((clean, bad))
                break
            else:
                skipped[split] += 1
        if balance and built:
            built = _balance(built, seed, split)
        for clean, bad in built:
            examples += [clean, bad]
    return examples, {"skipped_no_applicable_error": dict(skipped)}


def _balance(pairs: list[tuple[SynthExample, SynthExample]], seed: int, split: str) -> list[tuple[SynthExample, SynthExample]]:
    """Downsample every error type to the size of the rarest one (clean twins are dropped with them)."""
    by_type: dict[str, list[tuple[SynthExample, SynthExample]]] = {}
    for p in pairs:
        by_type.setdefault(p[1].label, []).append(p)
    k = min(len(v) for v in by_type.values())
    rng = random.Random(f"{seed}|balance|{split}")
    kept = [p for v in by_type.values() for p in rng.sample(sorted(v, key=lambda p: p[1].id), k)]
    return sorted(kept, key=lambda p: p[1].id)


def summarize(examples: list[SynthExample]) -> dict[str, Any]:
    """Counts per split / label / subtype (for the manifest and README)."""
    out: dict[str, Any] = {}
    for split in SPLITS:
        rows = [e for e in examples if e.split == split]
        out[split] = {
            "n": len(rows),
            "labels": dict(Counter(e.label for e in rows)),
            "subtypes": dict(Counter(e.subtype for e in rows if e.label != "OK")),
        }
    return out


def write_dataset(examples: list[SynthExample], out_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    """Write ``{train,val,test}.jsonl`` and ``manifest.json`` (with a hash of the test ids)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for e in (x for x in examples if x.split == split):
                fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")
    test_ids = sorted(e.id for e in examples if e.split == "test")
    manifest = {**meta, "generator_version": GENERATOR_VERSION, "summary": summarize(examples),
                "test_ids_sha256": hashlib.sha256("\n".join(test_ids).encode("utf-8")).hexdigest()}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_split(out_dir: str | Path, split: str, *, for_final_eval: bool = False) -> list[SynthExample]:
    """Read one split. The held-out ``test`` split is refused unless ``for_final_eval=True`` (final reporting only)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if split == "test" and not for_final_eval:
        raise PermissionError("The test split is held out: pass for_final_eval=True only in the final evaluation step.")
    path = Path(out_dir) / f"{split}.jsonl"
    return [SynthExample.from_dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line]
