"""Shared helpers: seeding, device choice, HF token lookup, run logging."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from mml.config import Config

log = logging.getLogger("mml")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if installed) for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_device(preference: str = "auto") -> str:
    """Map ``auto|cuda|mps|cpu`` to an available torch device string."""
    import torch

    if preference != "auto":
        if preference == "cuda" and not torch.cuda.is_available():
            log.warning("cuda requested but unavailable; falling back to cpu")
            return "cpu"
        if preference == "mps" and not torch.backends.mps.is_available():
            log.warning("mps requested but unavailable; falling back to cpu")
            return "cpu"
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dotenv(path: str | Path) -> list[str]:
    """Load ``KEY=VALUE`` lines from a git-ignored ``.env`` into ``os.environ`` (existing variables win).

    Returns the NAMES of the variables set; values are never logged or returned. Lets secrets such as HF_TOKEN or
    ANTHROPIC_API_KEY reach the process without going through chat or shell profiles.
    """
    path = Path(path)
    if not path.is_file():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
        if key and value and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def free_memory() -> None:
    """Run the garbage collector and release cached accelerator memory (call after dropping a model)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover
        pass


def get_hf_token() -> str | None:
    """HF token from ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN`` or ``huggingface-cli login``; never logged."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:  # pragma: no cover - token lookup must never crash a run
        return None


def package_versions(names: tuple[str, ...] = ("torch", "transformers", "datasets", "sentencepiece", "IndicTransToolkit")) -> dict[str, str]:
    """Installed versions of key packages (missing ones reported as 'not installed')."""
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


class RunLogger:
    """Creates ``<output_dir>/<timestamp>_<name>/`` holding the config, env info and metrics of one run."""

    def __init__(self, cfg: Config, name: str, root: str | Path | None = None) -> None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = Path(root or cfg.raw["run"]["output_dir"])
        self.dir = base / f"{stamp}_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
        env: dict[str, Any] = {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "seed": cfg.seed,
            "hf_token_present": get_hf_token() is not None,  # presence only, never the value
        }
        (self.dir / "env.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        self._metrics = self.dir / "metrics.jsonl"

    def log_metrics(self, metrics: dict[str, Any]) -> None:
        """Append one metrics record (JSON line) and echo it to the log."""
        with open(self._metrics, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        log.info("metrics: %s", metrics)

    def write_json(self, filename: str, obj: Any) -> Path:
        """Write an arbitrary JSON artifact into the run directory."""
        path = self.dir / filename
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        return path
