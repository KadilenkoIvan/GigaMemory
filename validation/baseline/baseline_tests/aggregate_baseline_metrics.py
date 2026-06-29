#!/usr/bin/env python3
"""
Build baseline_tests/metrics.json from per-model validation_results.json.

Top level: one key per baseline run folder (full_context_*). Each value is
metrics only: ``statistics`` (without memory-hit / MHE fields) plus ``timing``.
No metadata, no per-answer ``results``.

Default inputs (fixed order):
  full_context_LLama-3-8B-Instruct
  full_context_mistral-nemo
  full_context_Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keys to drop anywhere under ``statistics`` (baseline should not expose MHT/MHE).
_MEMORY_HIT_BLOCKLIST = frozenset(
    {
        "memory_hit_evaluation",
        "memory_hit",
        "memory_miss",
        "mhe_count",
        "mhe_hits",
        "mhe_misses",
        "mhe_hit_rate",
        "memory_hit_rate",
    }
)


DEFAULT_MODEL_DIRS = (
    "full_context_LLama-3-8B-Instruct",
    "full_context_mistral-nemo",
    "full_context_Qwen2.5-7B-Instruct",
)


def _strip_memory_hit_tree(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if k in _MEMORY_HIT_BLOCKLIST:
                continue
            if k == "by_type" and isinstance(v, dict):
                out[k] = {
                    qt: _strip_memory_hit_tree(st)
                    for qt, st in v.items()
                }
                continue
            out[k] = _strip_memory_hit_tree(v)
        return out
    if isinstance(obj, list):
        return [_strip_memory_hit_tree(x) for x in obj]
    return obj


def _load_metrics_leaf(results_path: Path) -> Optional[Dict[str, Any]]:
    if not results_path.is_file():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    stats = blob.get("statistics")
    timing = blob.get("timing")
    if not isinstance(stats, dict):
        return None
    stats_clean = _strip_memory_hit_tree(copy.deepcopy(stats))
    out: Dict[str, Any] = dict(stats_clean)
    if isinstance(timing, dict):
        out["timing"] = timing
    return out


def aggregate(
    base: Path,
    model_dir_names: List[str],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for name in model_dir_names:
        vr = base / name / "validation_results.json"
        leaf = _load_metrics_leaf(vr)
        if leaf is None and strict:
            raise FileNotFoundError(f"Missing or invalid: {vr}")
        tree[name] = leaf
    return tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory that contains full_context_* model subfolders",
    )
    ap.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_DIRS),
        help="Comma-separated subdir names under --base",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: <base>/metrics.json)",
    )
    ap.add_argument("--strict", action="store_true", help="Fail if any validation_results.json is missing")
    args = ap.parse_args()
    base: Path = args.base
    names = [n.strip() for n in str(args.models).split(",") if n.strip()]
    out_path = args.out if args.out is not None else base / "metrics.json"
    data = aggregate(base, names, strict=args.strict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
