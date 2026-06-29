"""Per-user persistent state (chosen answer language).

A tiny JSON-backed key→value store. The bot runs on a single asyncio loop, but a
lock keeps it safe and the write-through keeps the choice across restarts.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path


class UserStore:
    def __init__(self, path: str, default_language: str = "ru") -> None:
        self._path = Path(path)
        self._default = default_language if default_language in ("ru", "en") else "ru"
        self._lock = threading.Lock()
        self._langs: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.is_file():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._langs = {
                str(k): (v if v in ("ru", "en") else self._default)
                for k, v in data.get("languages", {}).items()
            }
        except (ValueError, OSError):
            self._langs = {}

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"languages": self._langs}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def has_language(self, user_id: int | str) -> bool:
        with self._lock:
            return str(user_id) in self._langs

    def get_language(self, user_id: int | str) -> str:
        with self._lock:
            return self._langs.get(str(user_id), self._default)

    def set_language(self, user_id: int | str, lang: str) -> None:
        lang = lang if lang in ("ru", "en") else self._default
        with self._lock:
            self._langs[str(user_id)] = lang
            self._flush()
