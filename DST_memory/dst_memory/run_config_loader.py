"""Load run configuration (JSON) for DST_memory CLI."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

_ENV_REF = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        m = _ENV_REF.match(value.strip())
        if m:
            return os.environ.get(m.group(1), "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "run_config.json"


def load_run_config(path: str | Path | None) -> dict[str, Any]:
    cfg_path = Path(path) if path else default_config_path()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    return _expand_env(raw)


def shared_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    shared = cfg.get("shared")
    if not isinstance(shared, dict):
        raise ValueError('Run config must contain a "shared" object')
    return dict(shared)


def subsection(cfg: Mapping[str, Any], name: str) -> dict[str, Any]:
    block = cfg.get(name)
    if block is None:
        return {}
    if not isinstance(block, dict):
        raise ValueError(f'Run config section "{name}" must be an object')
    return dict(block)
