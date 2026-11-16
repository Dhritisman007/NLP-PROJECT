"""Machine-translation wrapper: NLLB and IndicTrans2 backends, n-best generation, disk caching."""

from __future__ import annotations

import logging
import time
import unicodedata
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from mml.cache import DiskCache, make_key
from mml.config import Config
from mml.utils import get_hf_token, resolve_device

log = logging.getLogger(__name__)

DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


class BackendUnavailable(RuntimeError):
    """Raised when an MT backend cannot be loaded (gated model without token, missing package...)."""


# --------------------------------------------------------------------------- backends
class MTBackend(ABC):
    """One MT model. Subclasses load lazily on first use."""

    name: str = "base"

    def __init__(self, spec: dict[str, Any], src_lang: str, tgt_lang: str, *, device: str,
                 dtype: str, batch_size: int, num_beams: int, max_length: int) -> None:
        self.spec = spec
        self.model_id: str = spec["model_id"]
        self.revision: str | None = spec.get("revision")
        self.src_lang, self.tgt_lang = src_lang, tgt_lang
        self.device, self.dtype = device, dtype
        self.batch_size, self.num_beams, self.max_length = batch_size, num_beams, max_length
        self._loaded = False

    # -- identity used for cache keys
    def signature(self) -> dict[str, Any]:
        return {"backend": self.name, "model": self.model_id, "revision": self.revision,
                "src": self.src_lang, "tgt": self.tgt_lang, "beams": self.num_beams,
                "max_len": self.max_length, "dtype": self.dtype}

    @abstractmethod
    def load(self) -> None:
        """Load tokenizer and model onto ``self.device``."""

    @abstractmethod
    def _generate_batch(self, batch: list[str], n_best: int) -> list[list[str]]:
        """Translate one batch; returns ``n_best`` candidates per sentence (best first)."""

    def generate(self, sources: list[str], n_best: int = 1) -> list[list[str]]:
        """Translate ``sources`` (order preserved); length-sorted batching for speed."""
        if not self._loaded:
            self.load()
            self._loaded = True
        order = sorted(range(len(sources)), key=lambda i: -len(sources[i]))
        results: list[list[str]] = [[] for _ in sources]
        for start in range(0, len(order), self.batch_size):
            idx = order[start:start + self.batch_size]
            outs = self._generate_batch([sources[i] for i in idx], n_best)
            for i, cands in zip(idx, outs):
                results[i] = cands
        return results


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, DTYPES[name])


class NLLBBackend(MTBackend):
    """facebook/nllb-200-distilled-600M (open) via transformers."""

    name = "nllb"

    def load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        kwargs = {"revision": self.revision, "token": get_hf_token()}
        log.info("Loading %s on %s (%s)", self.model_id, self.device, self.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.src_lang, **kwargs)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id, torch_dtype=_torch_dtype(self.dtype), **kwargs).to(self.device).eval()
        self.tgt_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

    def _generate_batch(self, batch: list[str], n_best: int) -> list[list[str]]:
        import torch

        enc = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.max_length).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc, forced_bos_token_id=self.tgt_id, num_beams=max(self.num_beams, n_best),
                num_return_sequences=n_best, max_length=self.max_length)
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [texts[i * n_best:(i + 1) * n_best] for i in range(len(batch))]


class NLLBCT2Backend(MTBackend):
    """NLLB-200-distilled-600M converted to int8 with CTranslate2 (~0.6 GB, fast on CPU). Used for back-translation on
    the laptop profile; create the converted model once with ``mml convert-ct2``. Outputs differ slightly from fp32."""

    name = "nllb_ct2"

    def signature(self) -> dict[str, Any]:
        return {**super().signature(), "quant": self.spec.get("quantization", "int8")}

    def load(self) -> None:
        import os

        try:
            import ctranslate2
        except ImportError as exc:
            raise BackendUnavailable(f"ctranslate2 is not installed: {exc}") from exc
        from transformers import AutoTokenizer

        ct2_dir = Path(self.spec["ct2_dir"])
        if not (ct2_dir / "model.bin").exists():
            raise BackendUnavailable(f"No converted model at {ct2_dir}. Create it once with: mml convert-ct2")
        log.info("Loading CTranslate2 %s (%s) from %s", self.model_id, self.spec.get("quantization", "int8"), ct2_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, src_lang=self.src_lang, revision=self.revision,
                                                       token=get_hf_token())
        self.translator = ctranslate2.Translator(str(ct2_dir), device="cpu", compute_type=self.spec.get("quantization", "int8"),
                                                 inter_threads=1, intra_threads=os.cpu_count() or 4)

    def _generate_batch(self, batch: list[str], n_best: int) -> list[list[str]]:
        tok = self.tokenizer
        source = [tok.convert_ids_to_tokens(tok.encode(t, truncation=True, max_length=self.max_length)) for t in batch]
        results = self.translator.translate_batch(
            source, target_prefix=[[self.tgt_lang]] * len(batch), beam_size=max(self.num_beams, n_best),
            num_hypotheses=n_best, max_decoding_length=self.max_length)
        out = []
        for res in results:
            cands = []
            for hyp in res.hypotheses[:n_best]:
                hyp = hyp[1:] if hyp and hyp[0] == self.tgt_lang else hyp
                cands.append(tok.decode(tok.convert_tokens_to_ids(hyp), skip_special_tokens=True))
            out.append(cands)
        return out


def convert_ct2(cfg: Config, force: bool = False) -> Path:
    """One-time conversion of NLLB to an int8 CTranslate2 model (loads the weights in fp16 to limit peak memory)."""
    from ctranslate2.converters import TransformersConverter

    spec = _ct2_spec(cfg)
    out = Path(spec["ct2_dir"])
    if (out / "model.bin").exists() and not force:
        log.info("CTranslate2 model already present: %s", out)
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    TransformersConverter(spec["model_id"], revision=spec.get("revision"), load_as_float16=True,
                          low_cpu_mem_usage=True).convert(str(out), quantization=spec.get("quantization", "int8"), force=force)
    return out


def _ct2_spec(cfg: Config) -> dict[str, Any]:
    """The ``nllb_ct2`` backend spec with ``ct2_dir`` resolved against the project root."""
    spec = dict(cfg.raw["mt"]["backends"]["nllb_ct2"])
    if not Path(spec["ct2_dir"]).is_absolute():
        spec["ct2_dir"] = str(Path(cfg.raw["paths"]["data_dir"]).parent / spec["ct2_dir"])
    return spec


def reverse_backend_name(cfg: Config) -> str:
    """Backend used for target->source (back-)translation: int8 CTranslate2 on the laptop profile, else HF NLLB."""
    return "nllb_ct2" if cfg.profile.get("bt_backend", "hf") == "ct2" else "nllb"


class IndicTrans2Backend(MTBackend):
    """ai4bharat/indictrans2-en-indic-dist-200M. Gated; needs trust_remote_code and IndicTransToolkit."""

    name = "indictrans2"

    def load(self) -> None:
        try:
            try:
                from IndicTransToolkit.processor import IndicProcessor
            except ImportError:
                from IndicTransToolkit import IndicProcessor
        except ImportError as exc:
            raise BackendUnavailable(f"IndicTransToolkit is not installed/buildable: {exc}") from exc
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        token = get_hf_token()
        kwargs = {"revision": self.revision, "token": token,
                  "trust_remote_code": bool(self.spec.get("trust_remote_code", True))}
        log.info("Loading %s on %s (%s)", self.model_id, self.device, self.dtype)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id, torch_dtype=_torch_dtype(self.dtype), **kwargs).to(self.device).eval()
        except Exception as exc:  # gated repo, no token, offline, remote-code/transformers mismatch
            hint = "" if token else " (no HF token found: set HF_TOKEN and accept the model terms on its HF page)"
            raise BackendUnavailable(f"Could not load {self.model_id}{hint}: {type(exc).__name__}: {exc}") from exc
        self.processor = IndicProcessor(inference=True)

    def _generate_batch(self, batch: list[str], n_best: int) -> list[list[str]]:
        import torch

        prepared = self.processor.preprocess_batch(batch, src_lang=self.src_lang, tgt_lang=self.tgt_lang)
        enc = self.tokenizer(prepared, truncation=True, padding="longest", max_length=self.max_length,
                             return_tensors="pt", return_attention_mask=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **enc, use_cache=True, min_length=0, max_length=self.max_length,
                num_beams=max(self.num_beams, n_best), num_return_sequences=n_best)
        decoded = self.tokenizer.batch_decode(out.detach().cpu().tolist(), skip_special_tokens=True,
                                              clean_up_tokenization_spaces=True)
        final = self.processor.postprocess_batch(decoded, lang=self.tgt_lang)
        return [final[i * n_best:(i + 1) * n_best] for i in range(len(batch))]


BACKENDS: dict[str, type[MTBackend]] = {"nllb": NLLBBackend, "nllb_ct2": NLLBCT2Backend, "indictrans2": IndicTrans2Backend}


def build_backend(cfg: Config, name: str | None = None, *, reverse: bool = False) -> MTBackend:
    """Instantiate (but do not load) the named backend, or the active pair's default.

    ``reverse=True`` builds the target->source direction (used for back-translation); IndicTrans2's en->indic
    checkpoint cannot do that, so reverse always uses NLLB (HF fp32/fp16, or int8 CTranslate2 on the laptop profile).
    """
    if reverse:
        name = reverse_backend_name(cfg)
    name = name or cfg.pair["mt_backend"]
    src_lang, tgt_lang = (cfg.pair["tgt_lang"], cfg.pair["src_lang"]) if reverse else (cfg.pair["src_lang"], cfg.pair["tgt_lang"])
    if name not in BACKENDS:
        raise KeyError(f"Unknown MT backend {name!r}; available: {sorted(BACKENDS)}")
    prof = cfg.profile
    spec = _ct2_spec(cfg) if name == "nllb_ct2" else cfg.raw["mt"]["backends"][name]
    return BACKENDS[name](
        spec, src_lang, tgt_lang,
        device="cpu" if name == "nllb_ct2" else resolve_device(prof["device"]), dtype=prof["dtype"],
        batch_size=32 if name == "nllb_ct2" else int(prof["mt_batch_size"]),
        num_beams=int(cfg.raw["mt"]["num_beams"]), max_length=int(cfg.raw["mt"]["max_length"]))


# --------------------------------------------------------------------------- translator
class Translator:
    """Cache-aware translator. ``fallback_used`` records whether the configured backend was replaced."""

    def __init__(self, backend: MTBackend, cache: DiskCache | None = None, *, fallback_used: bool = False,
                 requested_backend: str | None = None) -> None:
        self.backend = backend
        self.cache = cache
        self.fallback_used = fallback_used
        self.requested_backend = requested_backend or backend.name

    @classmethod
    def reverse_from_config(cls, cfg: Config, *, use_cache: bool = True) -> "Translator":
        """Target->source translator (NLLB) for back-translation; shares the MT disk cache (keys include the languages)."""
        backend = build_backend(cfg, reverse=True)
        backend.load()
        backend._loaded = True
        cache = DiskCache(Path(cfg.raw["paths"]["cache_dir"]) / "mt.sqlite") if use_cache else None
        return cls(backend, cache, requested_backend=backend.name)

    @classmethod
    def from_name(cls, cfg: Config, name: str, *, use_cache: bool = True) -> "Translator":
        """Forward translator for an explicit backend (e.g. ``nllb_ct2`` for fast low-memory n-best on a laptop)."""
        backend = build_backend(cfg, name)
        backend.load()
        backend._loaded = True
        cache = DiskCache(Path(cfg.raw["paths"]["cache_dir"]) / "mt.sqlite") if use_cache else None
        return cls(backend, cache, requested_backend=name)

    @classmethod
    def from_config(cls, cfg: Config, *, allow_fallback: bool = True, use_cache: bool = True) -> "Translator":
        """Build the pair's backend, loading it now so failures surface here.

        If the configured backend is unavailable and ``allow_fallback`` is set, falls back to NLLB with a
        loud warning (never silently).
        """
        requested = cfg.pair["mt_backend"]
        backend = build_backend(cfg, requested)
        fallback = False
        try:
            backend.load()
            backend._loaded = True
        except BackendUnavailable as exc:
            if not allow_fallback or requested == "nllb":
                raise
            log.warning("MT backend %r unavailable -> FALLING BACK to NLLB. Reason: %s", requested, exc)
            backend, fallback = build_backend(cfg, "nllb"), True
            backend.load()
            backend._loaded = True
        cache = DiskCache(Path(cfg.raw["paths"]["cache_dir"]) / "mt.sqlite") if use_cache else None
        return cls(backend, cache, fallback_used=fallback, requested_backend=requested)

    def translate(self, sources: list[str], n_best: int = 1) -> list[list[str]]:
        """Return ``n_best`` candidates per source, using the disk cache where possible."""
        sig = self.backend.signature()
        keys = [make_key(sig, n_best, s) for s in sources]
        # NB: compare with `is not None` - DiskCache defines __len__, so an empty cache is falsy.
        results: list[list[str] | None] = [self.cache.get(k) if self.cache is not None else None for k in keys]
        todo = [i for i, r in enumerate(results) if r is None]
        log.info("MT: %d sentences, %d cache hits, %d to translate (%s)",
                 len(sources), len(sources) - len(todo), len(todo), self.backend.name)
        if todo:
            fresh = self.backend.generate([sources[i] for i in todo], n_best)
            for i, cands in zip(todo, fresh):
                results[i] = cands
                if self.cache is not None:
                    self.cache.set(keys[i], cands)
        return [r or [] for r in results]

    def translate_best(self, sources: list[str]) -> list[str]:
        """Top-1 translation per source."""
        return [c[0] for c in self.translate(sources, 1)]


# --------------------------------------------------------------------------- sanity checks
_SCRIPT_PREFIX = {"Deva": "DEVANAGARI", "Beng": "BENGALI", "Taml": "TAMIL", "Arab": "ARABIC",
                  "Cyrl": "CYRILLIC", "Latn": "LATIN", "Hans": "CJK", "Hant": "CJK"}


def target_script_ratio(text: str, tgt_lang: str) -> float:
    """Fraction of alphabetic characters in ``text`` that belong to the target language's script."""
    prefix = _SCRIPT_PREFIX.get(tgt_lang.split("_")[-1])
    letters = [c for c in text if c.isalpha()]
    if not letters or prefix is None:
        return 0.0
    return sum(unicodedata.name(c, "").startswith(prefix) for c in letters) / len(letters)


def load_sentences(path: str | Path) -> list[str]:
    """Read one sentence per line, skipping blanks and ``#`` comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def timed(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    """Run ``fn`` and return ``(result, seconds)``."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0
