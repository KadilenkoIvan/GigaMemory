"""Load DST_memory/.env into os.environ before reading run_config.json."""

from __future__ import annotations

import os
from pathlib import Path


def _dst_memory_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def load_dst_memory_dotenv() -> None:
    path = _dst_memory_root() / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[7:].strip()
        if "=" not in s:
            continue
        key, _, val = s.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key and not os.environ.get(key):
            os.environ[key] = val
