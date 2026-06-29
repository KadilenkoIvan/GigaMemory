#!/usr/bin/env python3
"""
Merge judge_only validation_results.json shards into metrics.json.

Layout: model -> memory_strategy -> inactive_facts_mode -> metrics only
(no metadata, no per-answer results).

Default inputs: results_judge_bundle_Qwen_final-LLM-{LLama,Mistral-Nemo,Qwen25}
Output: metrics.json next to this script (override with --out).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORY_STRATEGIES = ("full_graph_json", "relevant_slots_full", "topk_graph_records")
INACTIVE_TAGS = ("active_only", "with_inactive")
BUNDLE_DIR_PREFIX = "results_judge_bundle_"

# Default judge bundle dirs (fixed order for stable metrics.json).
DEFAULT_BUNDLE_NAMES = (
    "results_judge_bundle_Qwen_final-LLM-LLama",
    "results_judge_bundle_Qwen_final-LLM-Mistral-Nemo",
    "results_judge_bundle_Qwen_final-LLM-Qwen25",
)


def _default_bundle_roots(base: Path) -> List[Path]:
    roots: List[Path] = []
    for name in DEFAULT_BUNDLE_NAMES:
        p = base / name
        if p.is_dir():
            roots.append(p)
    return roots


def _model_key(bundle_dir: Path) -> str:
    name = bundle_dir.name
    if name.startswith(BUNDLE_DIR_PREFIX):
        return name[len(BUNDLE_DIR_PREFIX) :]
    return name


def _load_metrics_leaf(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    stats = blob.get("statistics")
    timing = blob.get("timing")
    if not isinstance(stats, dict):
        return None
    out: Dict[str, Any] = dict(stats)
    if isinstance(timing, dict):
        out["timing"] = timing
    return out


def aggregate(
    bundle_roots: List[Path],
    *,
    strict: bool = False,
) -> Dict[str, Any]:
    tree: Dict[str, Any] = {}
    for root in bundle_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Not a directory: {root}")
        mk = _model_key(root)
        tree[mk] = {}
        for strategy in MEMORY_STRATEGIES:
            tree[mk][strategy] = {}
            for inactive in INACTIVE_TAGS:
                vr = root / strategy / inactive / "validation_results.json"
                leaf = _load_metrics_leaf(vr)
                if leaf is None and strict:
                    raise FileNotFoundError(f"Missing or invalid: {vr}")
                tree[mk][strategy][inactive] = leaf
    return tree


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing results_judge_bundle_* folders",
    )
    ap.add_argument(
        "--bundles",
        type=str,
        default="",
        help="Comma-separated bundle dir names under --base (default: three Qwen_final-LLM-* bundles)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <base>/metrics.json)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected validation_results.json is missing",
    )
    args = ap.parse_args()
    base: Path = args.base
    if args.bundles.strip():
        roots = [base / n.strip() for n in args.bundles.split(",") if n.strip()]
    else:
        roots = _default_bundle_roots(base)
    if not roots:
        raise SystemExit(f"No bundle dirs found under {base}")
    out_path = args.out if args.out is not None else base / "metrics.json"
    data = aggregate(roots, strict=args.strict)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
