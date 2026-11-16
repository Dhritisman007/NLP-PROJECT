"""Tiny SQLite key-value cache for expensive model outputs (MT, back-translation, QE)."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def make_key(*parts: Any) -> str:
    """Stable SHA-256 key from JSON-serialisable parts."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class DiskCache:
    """Persistent ``str -> JSON`` store backed by one SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._lock = threading.Lock()      # one connection shared by worker threads (LLM calls run concurrently)
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._db.commit()

    def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` on a miss."""
        with self._lock:
            row = self._db.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        """Store ``value`` (must be JSON-serialisable)."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)",
                (key, json.dumps(value, ensure_ascii=False)),
            )
            self._db.commit()

    def __len__(self) -> int:
        with self._lock:
            return int(self._db.execute("SELECT COUNT(*) FROM kv").fetchone()[0])

    def close(self) -> None:
        self._db.close()
