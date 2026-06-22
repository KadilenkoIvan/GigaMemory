"""
Shared pytest fixtures for GigaMemory unit tests.

All tests run in stub mode — no GPU, no real LLM required.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make DST_memory importable without installing the package
_ROOT = Path(__file__).resolve().parents[1]
_DST = _ROOT / "DST_memory"
if str(_DST) not in sys.path:
    sys.path.insert(0, str(_DST))
