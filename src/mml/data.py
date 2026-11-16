"""Dataset download/loading: FLORES+ (HF, gated), HalOmi (Meta zip), WMT MQM (GitHub TSV).

Every source degrades gracefully: a missing token / unreachable URL is logged and reported, never faked.
"""

from __future__ import annotations

import json
import logging
import shutil
import ssl
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import certifi

from mml.config import Config
from mml.utils import get_hf_token

log = logging.getLogger(__name__)

HALOMI_URL = "https://dl.fbaipublicfiles.com/nllb/halomi_release_v2.zip"
MQM_URL_TMPL = "https://raw.githubusercontent.com/google/wmt-mqm-human-evaluation/main/newstest2020/{lp}/mqm_newstest2020_{lp}.tsv"
FLORES_REPO = "openlanguagedata/flores_plus"
FLORES200_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"  # public, 25.6 MB


@dataclass
class DownloadResult:
    """Outcome of one data source for the report (status: ok | skipped | failed | dry-run)."""

    source: str
    status: str
    detail: str = ""


def flores_source(cfg: Config) -> str:
    """Which FLORES release is used: ``flores200`` (public tarball, default) or ``flores_plus`` (HF, gated)."""
    src = str(cfg.get("data.flores_source", "flores200"))
    if src not in ("flores200", "flores_plus"):
        raise ValueError(f"data.flores_source must be flores200|flores_plus, got {src!r}")
    return src


def flores_path(data_dir: Path, lang: str, split: str, source: str = "flores200") -> Path:
    """Local path of a FLORES split for one language."""
    if source == "flores200":
        return data_dir / "raw" / "flores200" / split / f"{lang}.{split}"
    return data_dir / "raw" / "flores_plus" / split / f"{lang}.jsonl"


def read_flores(data_dir: Path, lang: str, split: str, source: str = "flores200") -> list[str]:
    """Read the sentences of a downloaded FLORES split (line/``id`` order; line i is parallel across languages)."""
    path = flores_path(data_dir, lang, split, source)
    if source == "flores200":
        return path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: r.get("id", 0))
    return [r["text"] for r in rows]


def download_flores(cfg: Config, data_dir: Path, dry_run: bool = False) -> DownloadResult:
    """Fetch FLORES dev+devtest for the active pair from the configured source."""
    if flores_source(cfg) == "flores200":
        return _download_flores200(cfg, data_dir, dry_run)
    return _download_flores_plus(cfg, data_dir, dry_run)


def _download_flores200(cfg: Config, data_dir: Path, dry_run: bool) -> DownloadResult:
    """FLORES-200 from Meta's public tarball; only the needed language files are extracted."""
    langs = cfg.pair["flores_plus"]
    try:
        if dry_run:
            return DownloadResult("flores200", "dry-run", f"{FLORES200_URL} reachable: {_head(FLORES200_URL)}")
        dst_root = data_dir / "raw" / "flores200"
        dst_root.mkdir(parents=True, exist_ok=True)
        tar_path = data_dir / "raw" / "flores200_dataset.tar.gz"
        if not tar_path.exists():
            _fetch(FLORES200_URL, tar_path)
        wanted = {f"{sp}/{lg}.{sp}" for lg in langs for sp in ("dev", "devtest")}
        found: set[str] = set()
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                rel = "/".join(member.name.split("/")[-2:])
                if member.isfile() and rel in wanted:
                    out = dst_root / rel  # fixed relative name: member paths from the archive are never used
                    out.parent.mkdir(parents=True, exist_ok=True)
                    fh = tf.extractfile(member)
                    assert fh is not None
                    out.write_bytes(fh.read())
                    found.add(rel)
        missing = wanted - found
        if missing:
            return DownloadResult("flores200", "failed", f"missing in archive: {sorted(missing)}")
        return DownloadResult("flores200", "ok", f"{langs} -> {dst_root}")
    except (urllib.error.URLError, OSError, tarfile.TarError) as exc:
        return DownloadResult("flores200", "failed", f"{type(exc).__name__}: {exc}")


def _download_flores_plus(cfg: Config, data_dir: Path, dry_run: bool) -> DownloadResult:
    """FLORES+ from the gated HF dataset (needs a token with the terms accepted)."""
    langs = cfg.pair["flores_plus"]
    if dry_run:
        return DownloadResult("flores_plus", "dry-run", f"would fetch dev/devtest for {langs} from {FLORES_REPO}")
    token = get_hf_token()
    if not token:
        return DownloadResult("flores_plus", "skipped",
                              "no HF token (gated dataset): set HF_TOKEN and accept terms at "
                              f"https://huggingface.co/datasets/{FLORES_REPO}")
    from huggingface_hub import hf_hub_download

    try:
        for lang in langs:
            for split in ("dev", "devtest"):
                src = hf_hub_download(FLORES_REPO, f"{split}/{lang}.jsonl", repo_type="dataset", token=token)
                dst = flores_path(data_dir, lang, split, "flores_plus")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(Path(src).read_bytes())
    except Exception as exc:
        return DownloadResult("flores_plus", "failed", f"{type(exc).__name__}: {exc}")
    return DownloadResult("flores_plus", "ok", f"{langs} -> {data_dir / 'raw' / 'flores_plus'}")


def _ssl_context() -> ssl.SSLContext:
    """TLS context using certifi's CA bundle (python.org builds on macOS ship without system CAs)."""
    return ssl.create_default_context(cafile=certifi.where())


def _fetch(url: str, dst: Path) -> None:
    """Stream ``url`` to ``dst`` over verified HTTPS."""
    with urllib.request.urlopen(url, timeout=60, context=_ssl_context()) as resp, open(dst, "wb") as fh:  # noqa: S310
        shutil.copyfileobj(resp, fh)


def _head(url: str) -> str:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=20, context=_ssl_context()) as resp:  # noqa: S310
        size = resp.headers.get("Content-Length", "?")
        return f"HTTP {resp.status}, {int(size) / 1e6:.1f} MB" if size != "?" else f"HTTP {resp.status}"


def download_halomi(cfg: Config, data_dir: Path, dry_run: bool = False) -> DownloadResult:
    """Fetch and unzip HalOmi (hallucination/omission annotations). Only useful for pairs it covers."""
    if not cfg.pair.get("halomi"):
        return DownloadResult("halomi", "skipped", f"HalOmi does not cover pair {cfg.pair_key}")
    try:
        if dry_run:
            return DownloadResult("halomi", "dry-run", f"{HALOMI_URL} reachable: {_head(HALOMI_URL)}")
        dst_dir = data_dir / "raw" / "halomi"
        dst_dir.mkdir(parents=True, exist_ok=True)
        zip_path = dst_dir / "halomi_release_v2.zip"
        _fetch(HALOMI_URL, zip_path)  # fixed https URL from the HalOmi README
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dst_dir)  # zipfile strips absolute paths and '..' components
        return DownloadResult("halomi", "ok", f"-> {dst_dir}")
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        return DownloadResult("halomi", "failed", f"{type(exc).__name__}: {exc}")


def download_mqm(cfg: Config, data_dir: Path, dry_run: bool = False) -> DownloadResult:
    """Fetch WMT20 MQM annotations (with error spans) for the pairs that have them (en-de)."""
    if not cfg.pair.get("mqm"):
        return DownloadResult("wmt_mqm", "skipped", f"no WMT MQM data for pair {cfg.pair_key}")
    lp = cfg.pair_key.replace("-", "")  # en-de -> ende
    url = MQM_URL_TMPL.format(lp=lp)
    try:
        if dry_run:
            return DownloadResult("wmt_mqm", "dry-run", f"{url} reachable: {_head(url)}")
        dst = data_dir / "raw" / "wmt_mqm" / f"mqm_newstest2020_{lp}.tsv"
        dst.parent.mkdir(parents=True, exist_ok=True)
        _fetch(url, dst)
        return DownloadResult("wmt_mqm", "ok", f"-> {dst}")
    except (urllib.error.URLError, OSError) as exc:
        return DownloadResult("wmt_mqm", "failed", f"{type(exc).__name__}: {exc}")


def download_all(cfg: Config, what: list[str], dry_run: bool = False) -> list[DownloadResult]:
    """Run the selected download steps (``flores``, ``halomi``, ``mqm``)."""
    data_dir = Path(cfg.raw["paths"]["data_dir"])
    steps = {"flores": download_flores, "halomi": download_halomi, "mqm": download_mqm}
    unknown = set(what) - set(steps)
    if unknown:
        raise ValueError(f"Unknown data sources {sorted(unknown)}; choose from {sorted(steps)}")
    return [steps[name](cfg, data_dir, dry_run) for name in what]
