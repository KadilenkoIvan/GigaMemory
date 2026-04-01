"""Resolve paths to the parent GigaMemory DST_memory package."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_dst_memory_on_path() -> Path:
    """
    This repo lives next to DST_memory, e.g. GigaMemory/slot_eval and GigaMemory/DST_memory.
    Returns the absolute path to the DST_memory directory added to sys.path.
    """
    here = Path(__file__).resolve()
    # slot_eval/package/paths.py -> slot_eval -> GigaMemory
    gigamemory_root = here.parents[2]
    dst_root = gigamemory_root / "DST_memory"
    if not dst_root.is_dir():
        raise FileNotFoundError(
            f"DST_memory not found at {dst_root}. "
            "Place the slot_eval folder next to DST_memory (same parent as in GigaMemory repo)."
        )
    s = str(dst_root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return dst_root
