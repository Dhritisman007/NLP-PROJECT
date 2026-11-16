"""YAML configuration: loading, hardware-profile merge and ``--set a.b=c`` overrides."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` recursively merged into ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def apply_override(raw: dict[str, Any], assignment: str) -> None:
    """Apply one ``dotted.key=value`` override in place (value parsed as YAML)."""
    if "=" not in assignment:
        raise ValueError(f"Override must look like a.b=c, got: {assignment!r}")
    dotted, value = assignment.split("=", 1)
    keys = dotted.strip().split(".")
    node = raw
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot descend into non-dict at {key!r} in {dotted!r}")
    node[keys[-1]] = yaml.safe_load(value)


@dataclass
class Config:
    """Resolved configuration with convenience accessors."""

    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def pair_key(self) -> str:
        return str(self.raw["run"]["pair"])

    @property
    def pair(self) -> dict[str, Any]:
        """Settings of the active language pair."""
        pairs = self.raw["pairs"]
        if self.pair_key not in pairs:
            raise KeyError(f"Unknown pair {self.pair_key!r}; available: {sorted(pairs)}")
        return pairs[self.pair_key]

    @property
    def profile(self) -> dict[str, Any]:
        """Settings of the active hardware profile."""
        profiles = self.raw["profiles"]
        name = self.raw["run"]["profile"]
        if name not in profiles:
            raise KeyError(f"Unknown profile {name!r}; available: {sorted(profiles)}")
        return profiles[name]

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up ``a.b.c`` in the raw config."""
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def to_yaml(self) -> str:
        """Serialise the resolved config (for run logs)."""
        return yaml.safe_dump(self.raw, sort_keys=False, allow_unicode=True)


def _resolve_paths(raw: dict[str, Any], root: Path) -> None:
    """Make relative ``paths.*`` / ``run.output_dir`` absolute against the project root (not the cwd)."""
    for section, key in (("paths", "data_dir"), ("paths", "cache_dir"), ("paths", "hf_cache"), ("run", "output_dir")):
        value = raw.get(section, {}).get(key)
        if value and not Path(value).is_absolute():
            raw[section][key] = str(root / value)


def load_config(
    path: str | Path | None = None,
    pair: str | None = None,
    profile: str | None = None,
    overrides: list[str] | None = None,
) -> Config:
    """Load YAML, then apply ``--pair``/``--profile`` and ``--set`` overrides (in that order)."""
    cfg_path = Path(path or DEFAULT_CONFIG_PATH).resolve()
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if pair:
        raw["run"]["pair"] = pair
    if profile:
        raw["run"]["profile"] = profile
    for assignment in overrides or []:
        apply_override(raw, assignment)
    _resolve_paths(raw, cfg_path.parent.parent)
    cfg = Config(raw)
    _ = cfg.pair, cfg.profile  # fail fast on bad keys
    return cfg
