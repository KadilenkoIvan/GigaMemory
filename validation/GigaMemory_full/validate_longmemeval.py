"""
LongMemEval validation script for GigaMemory DST pipeline - Version 3.

Advanced features:
- Multiple validation modes: full, memory_only, final_llm_only, judge_only
- Batch processing for final LLM (accumulate N dialogues before answering)
- Batch processing for judge (accumulate M answers before judging)
- Memory Hit Rate metric (separate LLM call to check if fact was in context)
- Model unloading before final LLM for local mode
- Full configuration via JSON config file (mirrors DST_memory structure)

Usage:
    # Full pipeline (default): memory -> final LLM -> judge
    python validate_longmemeval.py

    # Memory only: process dialogues and save memory state
    python validate_longmemeval.py --validation-mode memory_only

    # Final LLM only: load saved memory state and generate answers
    python validate_longmemeval.py --validation-mode final_llm_only \
        --input-state-dir ./results_memory_only

    # Judge only: evaluate saved answers (one intermediate_answers.json)
    python validate_longmemeval.py --validation-mode judge_only \\
        --input-answers-path ./results_final_llm/intermediate_answers.json

    # Judge only: several intermediate_answers.json in one run (e.g. all strategies × inactive modes)
    python validate_longmemeval.py --validation-mode judge_only \\
        --input-answers-paths ./out/s1.json,./out/s2.json

    # Using custom config
    python validate_longmemeval.py --config ./my_config.json

    # final_llm_only: resume from global_index N (merge into existing intermediate_answers.json)
    python validate_longmemeval.py --config ./my_config.json \\
        --final-llm-resume-from-global-index 123

    # Multiple questions per row (one memory pass, then each question scored separately):
    # use a non-empty ``questions`` list on the dataset object; see test_data/minimal_test.json.

Config file structure mirrors DST_memory/run_config.json with additional validation parameters:
    {
      "shared": { ... validation dataset/output params ... },
      "batch_processing": { ... batch sizes ... },
      "judge": { ... judge configuration ... },
      "giga_memory": { ... GigaMemory pipeline config ... },
      "validation_mode": { ... mode-specific settings ... }
    }
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
import urllib.error
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent directories to path for imports
repo_root = Path(__file__).resolve().parents[2]
dst_memory_path = repo_root / "DST_memory"
ragu_path = repo_root / "RAGU"

if str(dst_memory_path) not in sys.path:
    sys.path.insert(0, str(dst_memory_path))
if str(ragu_path) not in sys.path:
    sys.path.insert(0, str(ragu_path))

from dst_memory.utils.dotenv_loader import load_dst_memory_dotenv
from dst_memory.utils.run_config_loader import load_run_config, shared_section
from dst_memory.clients.llm_client import CHAT_API_OUTPUT_POLICY, _normalize_assistant_message_text
from dst_memory.clients.memory_context_payload import (
    MEMORY_PAYLOAD_MODES,
    finalize_memory_context_for_llm,
    normalize_memory_payload_mode,
)
from dst_memory.clients.context_limit import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    clamp_chat_messages_to_max_tokens,
)
from dst_memory.core.dataset_time import (
    fact_clock_iso_for_haystack_session,
    optional_clock_display_for_validation,
)
import random


# ============================================================================
# Timing Utilities
# ============================================================================

class TimingStats:
    """Collect and compute timing statistics."""

    def __init__(self):
        self.items: List[Dict[str, float]] = []  # one row per LongMemEval item (dialogue)
        self.user_message_seconds: List[float] = []  # one sample per write_to_memory call
        self.total_start: Optional[float] = None
        self.total_time: float = 0.0

    def start_total(self):
        self.total_start = time.time()

    def end_total(self):
        if self.total_start:
            self.total_time = time.time() - self.total_start

    def add_user_message(self, seconds: float) -> None:
        """Record wall time for a single user message (write_to_memory)."""
        if seconds >= 0:
            self.user_message_seconds.append(seconds)

    def add_item(self, num_messages: int, processing_time: float):
        """Add wall-clock timing for one full validation item (entire dialogue memory pass)."""
        self.items.append({
            "num_messages": num_messages,
            "time": processing_time,
            "time_per_message": processing_time / num_messages if num_messages > 0 else 0,
        })

    def compute_percentile(self, values: List[float], p: float) -> float:
        """Compute percentile (0-100)."""
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * p / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_vals) else f
        return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])

    def _percentile_block(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
            }
        return {
            "min": min(values),
            "max": max(values),
            "p50": self.compute_percentile(values, 50),
            "p95": self.compute_percentile(values, 95),
            "p99": self.compute_percentile(values, 99),
            "mean": sum(values) / len(values),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get computed statistics."""
        times = [item["time"] for item in self.items]
        # Legacy: one amortized ratio per dialogue (wall / session turns in haystack)
        per_dialogue_amortized = [
            item["time_per_message"] for item in self.items if item["num_messages"] > 0
        ]
        total_messages = sum(item["num_messages"] for item in self.items)
        um = self.user_message_seconds

        if not times and not um:
            return {
                "total_time": self.total_time,
                "total_items": 0,
                "total_messages": 0,
                "total_user_message_writes": 0,
            }

        out: Dict[str, Any] = {
            "total_time": self.total_time,
            "total_items": len(self.items),
            "total_messages": total_messages,
            "total_user_message_writes": len(um),
        }
        if times:
            dialogue_block = self._percentile_block(times)
            out["time_per_dialogue"] = dialogue_block
            # Backward-compatible alias (was misread as "per message" in logs)
            out["time_per_item"] = dict(dialogue_block)
        if um:
            out["time_per_user_message"] = self._percentile_block(um)
        if per_dialogue_amortized:
            out["time_per_message"] = self._percentile_block(per_dialogue_amortized)
        return out


# ============================================================================
# Retry Decorator
# ============================================================================

def retry_with_backoff(max_retries: int = 3, backoff_base: float = 1.0):
    """Decorator for retrying with exponential backoff."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except urllib.error.HTTPError as e:
                    last_exception = e
                    if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                        wait_time = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning("HTTP %d error, retrying in %.1fs (attempt %d/%d)",
                                       e.code, wait_time, attempt + 1, max_retries)
                        time.sleep(wait_time)
                    else:
                        raise
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning("Error: %s, retrying in %.1fs (attempt %d/%d)",
                                       e, wait_time, attempt + 1, max_retries)
                        time.sleep(wait_time)
                    else:
                        raise
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Question Types (for judge and per-type metrics)
# ============================================================================

QUESTION_TYPES = {
    "single-session-user": "User mentioned a fact about themselves in one session - system should remember it",
    "single-session-preference": "User previously shared their preferences - system should use them when answering a new request",
    "multi-session": "Facts about user are scattered across multiple sessions - system should collect them together",
    "knowledge-update": "User provided new fact contradicting old one - system should return the current one",
}

RELEVANT_TYPES = list(QUESTION_TYPES.keys())
MEMORY_STRATEGIES = ("full_graph_json", "relevant_slots_full", "topk_graph_records")
MEMORY_STRATEGY_IDS = frozenset(MEMORY_STRATEGIES)
_JUDGE_SHARD_INACTIVE_TAGS = frozenset({"active_only", "with_inactive"})
MEMORY_ONLY_WRITE_MODES = ("standard", "single_path_only")
INACTIVE_FACTS_MEMORY_MODES = ("active_only", "with_inactive")


def _validation_results_path_for_judge_shard(intermediate_answers_json: Path, output_root: Path) -> Path:
    """
    Map ``.../<strategy>/<inactive_tag>/intermediate_answers.json`` →
    ``output_root/<strategy>/<inactive_tag>/validation_results.json``.
    If the path does not match that layout, fall back to ``output_root/validation_results.json``.
    """
    parts = Path(intermediate_answers_json).resolve().parts
    if parts and parts[-1] == "intermediate_answers.json" and len(parts) >= 3:
        inactive_tag = parts[-2]
        strategy = parts[-3]
        if strategy in MEMORY_STRATEGY_IDS and inactive_tag in _JUDGE_SHARD_INACTIVE_TAGS:
            return Path(output_root) / strategy / inactive_tag / "validation_results.json"
    return Path(output_root) / "validation_results.json"


def coerce_openrouter_reasoning_body(raw: Any, backend_mode: str) -> Optional[Dict[str, Any]]:
    """
    Optional OpenRouter chat-completions ``reasoning`` body field.

    Only OpenAI-compatible API backends use this. Omit unless JSON explicitly sets it —
    disable-style defaults trigger HTTP 400 on some routes (mandatory reasoning).
    """
    bm = str(backend_mode or "").lower().strip()
    if bm not in ("openrouter", "puter", "api"):
        return None
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw) if raw else None
    logging.getLogger(__name__).warning("Ignoring invalid openrouter_reasoning (expected object or null): %s", raw)
    return None


def _normalize_inactive_facts_memory_modes(raw: Any) -> List[str]:
    """Modes for inactive DST facts in ``final_llm_only`` memory payloads (can combine → more LLM calls)."""
    if raw is None:
        return ["active_only"]
    items: List[str] = []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        for v in raw:
            s = str(v).strip()
            if s:
                items.append(s)
    seen = set()
    out: List[str] = []
    for s in items:
        if s in INACTIVE_FACTS_MEMORY_MODES and s not in seen:
            out.append(s)
            seen.add(s)
    return out or ["active_only"]


def _inactive_facts_use_strategy_subdir(modes: List[str]) -> bool:
    """When True, nest outputs under ``<strategy>/<inactive_mode>/``."""
    return len(modes) > 1 or (len(modes) == 1 and modes[0] == "with_inactive")


def _narrow_memory_context_active_facts(memory_context: Any) -> Any:
    """Remove records/messages with ``is_active`` false (default True keeps row)."""
    if not isinstance(memory_context, dict):
        return memory_context
    out = dict(memory_context)
    slots = out.get("slots")
    if isinstance(slots, list):
        new_slots: List[Dict[str, Any]] = []
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            msgs = slot.get("messages") or []
            if not isinstance(msgs, list):
                new_slots.append(slot)
                continue
            kept = [m for m in msgs if isinstance(m, dict) and m.get("is_active", True)]
            if kept:
                sc = dict(slot)
                sc["messages"] = kept
                new_slots.append(sc)
        out["slots"] = new_slots
    records = out.get("records")
    if isinstance(records, list):
        out["records"] = [
            r for r in records if not isinstance(r, dict) or r.get("is_active", True)
        ]
    return out


def _filter_topk_records_for_inactive_mode(records: List[Any], include_inactive: bool) -> List[Any]:
    if include_inactive:
        return list(records)
    filtered: List[Any] = []
    for r in records:
        if not isinstance(r, dict):
            filtered.append(r)
            continue
        if r.get("is_active", True):
            filtered.append(r)
    return filtered


def _normalize_memory_strategies(raw: Any) -> List[str]:
    """Normalize configured memory strategies list preserving order."""
    if raw is None:
        return []
    items: List[str] = []
    if isinstance(raw, str):
        items = [p.strip() for p in raw.split(",") if p.strip()]
    elif isinstance(raw, (list, tuple)):
        for v in raw:
            s = str(v).strip()
            if s:
                items.append(s)
    else:
        s = str(raw).strip()
        if s:
            items = [s]

    seen = set()
    out: List[str] = []
    for s in items:
        if s in MEMORY_STRATEGIES and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _normalize_memory_payload_mode(raw: Any) -> str:
    return normalize_memory_payload_mode(raw)


def _memory_context_for_payload_mode(memory_context: Any, payload_mode: str) -> Any:
    """Reduce memory JSON to the fields the final LLM is allowed to see (see memory_context_payload)."""
    return finalize_memory_context_for_llm(memory_context, payload_mode)


def _normalize_input_answers_paths(raw: Any) -> List[str]:
    """Normalize ``input_answers_paths`` from JSON (list) or CLI (comma-separated string)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for x in raw:
            s = str(x).strip()
            if s:
                out.append(s)
        return out
    return []


def _normalize_memory_only_write_mode(raw: Any) -> str:
    mode = str(raw or "").strip().lower()
    if mode in MEMORY_ONLY_WRITE_MODES:
        return mode
    return "standard"


def _coerce_int_list(raw: Any) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(x) for x in raw]
    return []


def _coerce_str_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _normalize_dialogue_id_str(item: Dict[str, Any]) -> str:
    raw = item.get("dialogue_id")
    if raw is None:
        return ""
    return str(raw).strip()


def memory_only_row_filter_sets(args: Any) -> Tuple[Optional[Set[int]], Optional[Set[str]]]:
    """Row indices are positions in the balanced dataset list (same numbering as chunk_XXXX)."""
    ri = _coerce_int_list(getattr(args, "memory_only_dialogue_row_indices", []))
    di = _coerce_str_list(getattr(args, "memory_only_dialogue_ids", []))
    idx_set: Optional[Set[int]] = set(ri) if ri else None
    id_set: Optional[Set[str]] = set(di) if di else None
    return idx_set, id_set


def memory_only_should_process_row(
    dialogue_row_index: int,
    item: Dict[str, Any],
    row_indices: Optional[Set[int]],
    dialogue_ids: Optional[Set[str]],
) -> bool:
    if row_indices is None and dialogue_ids is None:
        return True
    if row_indices is not None and dialogue_row_index in row_indices:
        return True
    did = _normalize_dialogue_id_str(item)
    if dialogue_ids is not None and did and did in dialogue_ids:
        return True
    return False


def _dialogue_row_index_from_saved_state(row: Dict[str, Any]) -> int:
    if "_validation_dialogue_row_index" in row:
        return int(row["_validation_dialogue_row_index"])
    if "dialogue_row_index" in row:
        return int(row["dialogue_row_index"])
    return -1


def _load_json_optional(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None


# ============================================================================
# Config Loading
# ============================================================================

def load_validation_config(config_path: str) -> Dict[str, Any]:
    """Load validation config from JSON file with fallback to defaults."""
    default_config = {
        "shared": {
            "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
            "output_dir": "./results",
            "num_items_per_type": 10,  # Balanced sampling: N per question type
            "question_types": list(QUESTION_TYPES.keys()),
            "log_level": "INFO",
            "log_file": True,
            "save_memory_state": True,
            "save_intermediate": True,
        },
        "batch_processing": {
            "final_llm_batch_size": 1,
            "judge_batch_size": 1,
            "calculate_memory_hit_rate": False,
        },
        "judge": {
            "mode": "openrouter",
            "model": "openai/gpt-oss-120b:free",
            "api_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "temperature": 0.0,
            "max_tokens": 1024,
            "max_context_tokens": 128 * 1024,
            "tokenizer_model": "",
            "local_model_path": "",
            "load_dtype": "float16",
            "load_quantization": "none",
            "unload_between_items": False,
        },
        "giga_memory": {},  # Will be merged with DST_memory defaults
        "validation_mode": {
            "mode": "full",  # full, memory_only, final_llm_only, judge_only
            "input_state_dir": "",  # For final_llm_only: directory with saved memory states
            "input_answers_path": "",  # For judge_only: single intermediate_answers.json (legacy)
            # For judge_only: several intermediate_answers.json in one run. Typical layout after
            # final_llm_only with ``inactive_facts_memory_modes: [active_only, with_inactive]``:
            # full_graph × 2, relevant_slots_full × 2, topk_graph_records × 1 (top-k has no separate with_inactive output).
            # When non-empty, overrides ``input_answers_path``.
            "input_answers_paths": [],
            "memory_only_output_suffix": "_memory_only",  # Suffix for memory_only output dirs
            "memory_only_write_mode": "standard",  # standard | single_path_only
            "final_llm_memory_strategies": [],  # Optional strategies list for final_llm_only
            "final_llm_memory_payload_mode": "with_metadata",  # with_metadata | triplets_only
            # memory_only: process only these rows (balanced-dataset indices → chunk_XXXX) and/or dialogue_id strings.
            # Empty lists = process all rows. Merges into existing memory_only_states.json / giga_memory_validation_logs.json.
            "memory_only_dialogue_row_indices": [],
            "memory_only_dialogue_ids": [],
            # final_llm_only: DST inactive facts (is_active=false) — combine modes → multiply LLM calls & separate JSON dirs.
            "inactive_facts_memory_modes": ["active_only"],
            # final_llm_only: if set (int >= 0), reload each intermediate_answers.json, keep rows with global_index < value,
            # re-run LLM only for states with global_index >= value, merge back into the same files (all strategies × inactive modes).
            "final_llm_resume_from_global_index": None,
        },
    }

    if not Path(config_path).exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        # Merge with defaults (validation_mode must be included — otherwise mode stays "full")
        merged = copy.deepcopy(default_config)
        for section in ["shared", "batch_processing", "judge", "giga_memory", "validation_mode"]:
            if section in user_config:
                if isinstance(merged.get(section), dict) and isinstance(user_config[section], dict):
                    merged[section].update(user_config[section])
                else:
                    merged[section] = user_config[section]

        logger.info("Loaded validation config from: %s", config_path)
        return merged
    except Exception as e:
        logger.error("Failed to load config: %s, using defaults", e)
        return default_config


def config_to_args(config: Dict[str, Any]) -> argparse.Namespace:
    """Convert config dict to argparse.Namespace for CLI compatibility."""
    shared = config.get("shared", {})
    batch = config.get("batch_processing", {})
    judge = config.get("judge", {})
    gm = config.get("giga_memory", {})
    val_mode = config.get("validation_mode", {})

    class Args:
        pass

    args = Args()

    # Validation params (shared)
    args.dataset_path = shared.get("dataset_path", "")
    args.output_dir = shared.get("output_dir", "./results")
    args.num_items_per_type = shared.get("num_items_per_type", 10)
    args.question_types = shared.get("question_types", list(QUESTION_TYPES.keys()))
    args.log_level = shared.get("log_level", "INFO")
    args.log_file = shared.get("log_file", True)
    args.save_memory_state = shared.get("save_memory_state", True)
    args.save_intermediate = shared.get("save_intermediate", True)

    # Batch processing params
    args.final_llm_batch_size = batch.get("final_llm_batch_size", 1)
    args.judge_batch_size = batch.get("judge_batch_size", 1)
    args.calculate_memory_hit_rate = batch.get("calculate_memory_hit_rate", False)

    # Judge params
    args.judge_mode = judge.get("mode", "openrouter")
    args.judge_model = judge.get("model", "openai/gpt-oss-120b:free")
    args.judge_api_url = judge.get("api_url", "https://openrouter.ai/api/v1")
    args.judge_api_key = judge.get("api_key", "")
    args.judge_temperature = judge.get("temperature", 0.0)
    args.judge_max_tokens = judge.get("max_tokens", 1024)
    args.judge_max_context_tokens = int(judge.get("max_context_tokens", 128 * 1024))
    args.judge_tokenizer_model = judge.get("tokenizer_model", "")
    args.judge_local_model_path = judge.get("local_model_path", "")
    args.judge_load_dtype = judge.get("load_dtype", "float16")
    args.judge_load_quantization = judge.get("load_quantization", "none")
    args.unload_judge_between_items = judge.get("unload_between_items", False)
    args.judge_enable_thinking = bool(judge.get("judge_enable_thinking", True))
    _jreason_raw = judge.get("openrouter_reasoning")
    if _jreason_raw is None and "reasoning" in judge:
        _jreason_raw = judge.get("reasoning")
    args.judge_openrouter_reasoning = coerce_openrouter_reasoning_body(
        _jreason_raw,
        judge.get("mode", "openrouter"),
    )

    # Validation mode params
    args.validation_mode = val_mode.get("mode", "full")
    args.input_state_dir = val_mode.get("input_state_dir", "")
    args.input_answers_path = val_mode.get("input_answers_path", "")
    args.input_answers_paths = _normalize_input_answers_paths(val_mode.get("input_answers_paths"))
    args.memory_only_output_suffix = val_mode.get("memory_only_output_suffix", "_memory_only")
    args.memory_only_write_mode = _normalize_memory_only_write_mode(
        val_mode.get("memory_only_write_mode", "standard")
    )
    args.final_llm_memory_strategies = _normalize_memory_strategies(
        val_mode.get("final_llm_memory_strategies", [])
    )
    args.final_llm_memory_payload_mode = _normalize_memory_payload_mode(
        val_mode.get("final_llm_memory_payload_mode", "with_metadata")
    )
    args.memory_only_dialogue_row_indices = _coerce_int_list(
        val_mode.get("memory_only_dialogue_row_indices")
    )
    args.memory_only_dialogue_ids = _coerce_str_list(val_mode.get("memory_only_dialogue_ids"))
    args.inactive_facts_memory_modes = _normalize_inactive_facts_memory_modes(
        val_mode.get("inactive_facts_memory_modes")
    )
    _rf_idx = val_mode.get("final_llm_resume_from_global_index")
    if _rf_idx is None or (isinstance(_rf_idx, str) and not str(_rf_idx).strip()):
        args.final_llm_resume_from_global_index = None
    else:
        args.final_llm_resume_from_global_index = int(_rf_idx)

    # GigaMemory config (for building CLI overrides)
    args.config = "DST_memory/run_config.json"  # Base GigaMemory config

    # GigaMemory CLI overrides from config file
    for key, value in gm.items():
        setattr(args, f"gm_{key}", value)

    # Ensure all gm_* attributes exist (even if None)
    gm_defaults = [
        "importance_model_path", "importance_threshold", "retrieval_top_k",
        "graph_top_k_records", "recent_history_pairs", "disable_memory_gate",
        "memory_gate_use_stub", "memory_strategy", "llm_mode", "llm_model",
        "llm_load_dtype", "llm_load_quantization", "llm_max_context_tokens", "llm_tokenizer_model",
        "llm_api_key", "llm_api_url", "llm_temperature", "llm_max_tokens",
        "openrouter_http_referer", "openrouter_x_title", "slot_use_stub",
        "slot_model_path", "slot_max_slots_per_message", "ragu_storage_path",
        "ragu_embedder_model", "ttl_mode", "ttl_semantic_dedup_enabled",
        "ttl_semantic_dedup_threshold", "slot_context_enabled",
        "slot_context_max_facts", "triplet_deletion_mode", "deletion_use_pymorphy",
        "conflict_allow_multi_relation_same_object", "conflict_rule_same_relation_updates",
        "slot_model_enable_thinking",
        "slot_llm_inject_no_think_prompt", "slot_llm_lm_format_enforcer",
        "slot_llm_load_quantization",
        "slot_fallback_on_no_slots", "triplet_fallback_on_empty", "prompt_language",
        "unload_models_before_final_llm", "use_dataset_datetime", "force_infinite_ttl",
        "llm_enable_thinking",
    ]
    for attr in gm_defaults:
        if not hasattr(args, f"gm_{attr}"):
            setattr(args, f"gm_{attr}", None)

    return args


def setup_logging(level: str, log_file: Optional[str] = None) -> None:
    """Setup logging configuration."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )


logger = logging.getLogger(__name__)

# Keys in validation JSON ``shared`` that belong only to validate_longmemeval routing/logging,
# not to DST ``PipelineConfig`` / ``DST_memory/run_config.json`` ``shared``.
_VALIDATION_SHARED_RESERVED_KEYS = frozenset(
    {
        "dataset_path",
        "output_dir",
        "num_items_per_type",
        "question_types",
        "log_level",
        "log_file",
        "save_memory_state",
        "save_intermediate",
    }
)


def _pipeline_overrides_from_validation_shared(shared: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extra pipeline keys placed alongside dataset/output params in validation ``shared``.

    They merge into ``build_pipeline_config`` overrides after ``DST_memory/run_config.json``
    but before ``giga_memory`` / ``--gm-*`` (which win on key collision).
    """
    return {k: v for k, v in shared.items() if k not in _VALIDATION_SHARED_RESERVED_KEYS}


def _effective_prompt_token_budget(max_context_tokens: int, max_completion_tokens: int) -> int:
    cap = int(max_context_tokens or 0)
    if cap <= 0:
        return 0
    reserve = max(64, int(max_completion_tokens or 0))
    return max(256, cap - reserve)


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for validation process."""
    # Dataset selection
    dataset_path: str
    output_dir: str
    start_index: int = 0
    num_items: int = 10

    # Batch processing
    final_llm_batch_size: int = 1  # Accumulate N dialogues before answering
    judge_batch_size: int = 1  # Accumulate M answers before judging

    # Memory hit rate calculation
    calculate_memory_hit_rate: bool = False

    # GigaMemory config path
    giga_memory_config: str = ""

    # Judge configuration
    judge_mode: str = "openrouter"  # "openrouter", "local", "none"
    judge_model: str = "openai/gpt-oss-120b:free"
    judge_api_url: str = "https://openrouter.ai/api/v1"
    judge_api_key: str = ""
    judge_temperature: float = 0.0
    judge_max_tokens: int = 1024
    judge_local_model_path: str = ""

    # Output options
    save_memory_state: bool = True
    save_intermediate: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: bool = True


# ============================================================================
# LongMemEval Dataset Types
# ============================================================================

RELEVANT_QUESTION_TYPES = [
    "single-session-user",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
]


# ============================================================================
# Judge Client (LLM-as-Judge)
# ============================================================================

class JudgeClient:
    """
    LLM-as-Judge client for evaluating answer correctness.
    Supports 'openrouter', 'puter', 'local', and 'none' modes.
    """

    def __init__(
        self,
        mode: str = "openrouter",
        model: str = "openai/gpt-oss-120b:free",
        api_url: str = "https://openrouter.ai/api/v1",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        local_model_path: Optional[str] = None,
        enable_thinking: bool = True,
        load_dtype: str = "float16",
        load_quantization: str = "none",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        tokenizer_model: str = "",
        openrouter_reasoning: Optional[Dict[str, Any]] = None,
    ):
        self.mode = mode
        self.model = model
        self.api_url = api_url
        if self.mode == "puter" and ((not self.api_url) or ("openrouter.ai" in self.api_url)):
            self.api_url = "https://api.puter.com/puterai/openai/v1"
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("PUTER_API_KEY", "")
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_model_path = local_model_path
        self.enable_thinking = bool(enable_thinking)
        self.load_dtype = load_dtype or "float16"
        self.load_quantization = (load_quantization or "none").strip().lower()
        self.max_context_tokens = int(max_context_tokens)
        self.tokenizer_model = (tokenizer_model or "").strip()
        self._local_serving = None
        self._tokenizer_limit: Any = None
        self.openrouter_reasoning: Optional[Dict[str, Any]] = (
            dict(openrouter_reasoning) if isinstance(openrouter_reasoning, dict) and openrouter_reasoning else None
        )

        logger.info(
            "JudgeClient initialized mode=%s model=%s tokenizer_model=%s enable_thinking=%s load_dtype=%s "
            "load_quantization=%s max_context_tokens=%s",
            mode,
            model if mode in ("openrouter", "puter") else local_model_path,
            self.tokenizer_model or "(auto)",
            self.enable_thinking,
            self.load_dtype,
            self.load_quantization,
            self.max_context_tokens,
        )

    def _tokenizer_for_prompt_limit(self):
        if self._local_serving is not None and getattr(self._local_serving, "tokenizer", None):
            return self._local_serving.tokenizer
        if self._tokenizer_limit is False:
            return None
        if self._tokenizer_limit is not None:
            return self._tokenizer_limit

        model_id = (
            (self.tokenizer_model or "").strip()
            or (self.local_model_path or "").strip()
            or (self.model or "").strip()
        )
        if not model_id:
            self._tokenizer_limit = False
            return None
        try:
            from transformers import AutoTokenizer

            self._tokenizer_limit = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            return self._tokenizer_limit
        except Exception as e:
            logger.warning("Judge tokenizer unavailable for max_context_tokens clamp: %s", e)
            self._tokenizer_limit = False
            return None

    def _build_messages(self, system_msg: str, user_msg: str) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        if self.max_context_tokens > 0:
            tok = self._tokenizer_for_prompt_limit()
            if tok is not None:
                prompt_budget = _effective_prompt_token_budget(
                    self.max_context_tokens,
                    self.max_tokens,
                )
                messages = clamp_chat_messages_to_max_tokens(
                    tok,
                    messages,
                    prompt_budget,
                    enable_thinking=self.enable_thinking,
                )
        return messages

    def _get_system_prompt_answer_correctness(self, question_type: str) -> str:
        """Get system prompt with 0-1 scoring criteria."""
        type_desc = QUESTION_TYPES.get(question_type, "General question answering")

        return CHAT_API_OUTPUT_POLICY + f"""You are an expert evaluator assessing answer quality.

    Question Type: {question_type}
    Type Description: {type_desc}

    Your task: Compare the predicted answer with the reference (gold) answer and return a score from 0.0 to 1.0 representing how well the predicted answer covers the factual content of the reference.

    Core Evaluation Principle:
    Focus ONLY on whether the required facts/entities from the reference are present in the predicted answer.
    IGNORE everything else: additional context, caveats, disclaimers, uncertainty phrases ("I think", "probably", "as of my knowledge"), verbose explanations, or any extra information beyond what was asked. Their presence or absence does NOT affect the score.

    Scoring Scale:
    1.0 - Perfect match: All key entities and facts from the reference are present. Wording may differ, but the required content is there.
    0.8 - Minor inaccuracy: All key entities present, but one is slightly distorted (wrong number, approximate date, slight name variation).
    0.6 - Partial answer: Most of the reference is covered, but one of several equally important entities is missing or replaced.
    0.4 - Weak coverage: Only one correct entity from several needed is mentioned, OR correct category but wrong specific fact.
    0.2 - Minimal match: Thematically related to the question but factually almost no overlap with the reference.
    0.0 - No match: Factually incorrect, contradicts the reference, or no relevant facts present.

    Special Rules:
    - For knowledge-update: If the system named the old/outdated fact instead of the new one → 0.0 (old fact does not count).
    - For single-session-preference: Judge whether the correct user fact was used, not phrasing. Different wording with the correct fact = 1.0.
    - For multi-session: If aggregation is needed (e.g., "how many total"), score proportionally to how many required entities were found: 2 of 4 → 0.4–0.6 depending on the importance of missing ones.

    Respond with ONLY JSON: {{"score": 0.0-1.0, "reasoning": "brief explanation"}}"""

    def _get_system_prompt_memory_hit(self) -> str:
        return CHAT_API_OUTPUT_POLICY + (
            "You are an expert evaluator assessing memory retrieval quality.\n"
            "Your task is to examine the memory context provided to an LLM and determine\n"
            "if a specific fact (needed to answer a question) is present in that context.\n\n"
            "Evaluation criteria:\n"
            "1. Check if the fact needed to answer the question appears in the memory context\n"
            "2. The fact may be worded differently but must convey the same information\n"
            "3. Partial matches count if they contain the essential information\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"fact_present": true/false, "reasoning": "brief explanation", "location": "where found or why missing"}'
        )

    def _get_user_prompt_answer_correctness(self, question: str, predicted: str, reference: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Reference Answer: {reference}\n\n"
            f"Predicted Answer: {predicted}\n\n"
            "Score the predicted answer's coverage of the reference (0.0 to 1.0)."
        )

    def _get_user_prompt_memory_hit(self, question: str, reference_answer: str, memory_context: Dict) -> str:
        memory_json = json.dumps(memory_context, ensure_ascii=False, indent=2)
        return (
            f"Question: {question}\n\n"
            f"Reference Answer (the fact that should be in memory): {reference_answer}\n\n"
            f"Memory Context provided to LLM:\n{memory_json}\n\n"
            "Is the fact needed to answer this question present in the memory context?"
        )

    def evaluate_answer(
        self, question: str, predicted_answer: str, reference_answer: str,
        question_type: str
    ) -> Tuple[float, str, Optional[str]]:
        """One dedicated LLM call: predicted vs reference (correctness score 0..1). Not combined with memory_hit."""
        if not predicted_answer or not predicted_answer.strip():
            return 0.0, "Empty predicted answer", None
        if not reference_answer or not reference_answer.strip():
            return 0.0, "Empty reference answer", None

        system_msg = self._get_system_prompt_answer_correctness(question_type)
        user_msg = self._get_user_prompt_answer_correctness(question, predicted_answer, reference_answer)

        try:
            result = self._call_judge(system_msg, user_msg, mode="correctness")
            return float(result.get("score", 0)), str(result.get("reasoning", "No reasoning")), None
        except Exception as e:
            logger.error("Judge evaluation failed: %s", e)
            return 0.0, f"Error: {e}", str(e)

    def evaluate_memory_hit(
        self, question: str, reference_answer: str, memory_context: Dict
    ) -> Dict[str, Any]:
        """One dedicated LLM call: whether reference fact appears in memory_context. Not combined with correctness."""
        if not reference_answer or not reference_answer.strip():
            return {"fact_present": False, "reasoning": "Empty reference answer"}

        system_msg = self._get_system_prompt_memory_hit()
        user_msg = self._get_user_prompt_memory_hit(question, reference_answer, memory_context)

        return self._call_judge(system_msg, user_msg, mode="memory_hit")

    def _call_judge(self, system_msg: str, user_msg: str, mode: str = "correctness") -> Dict[str, Any]:
        """Single chat completion: one metric per invocation (correctness OR memory_hit)."""
        task = "answer_correctness" if mode == "correctness" else "memory_presence_in_context"
        logger.info(
            "[Judge] Separate LLM invocation: task=%s (mode=%s backend=%s)",
            task,
            mode,
            self.mode,
        )
        if self.mode == "openrouter":
            return self._call_openrouter(system_msg, user_msg, mode)
        elif self.mode == "puter":
            return self._call_puter(system_msg, user_msg, mode)
        elif self.mode == "local":
            return self._call_local(system_msg, user_msg, mode)
        else:
            raise ValueError(f"Unknown judge mode: {self.mode}")

    @retry_with_backoff(max_retries=3)
    def _call_openrouter_api(self, body: Dict, headers: Dict, url: str) -> str:
        """Call OpenAI-compatible API with retry."""
        import urllib.request

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read().decode("utf-8")

    def _call_openrouter(self, system_msg: str, user_msg: str, mode: str) -> Dict[str, Any]:
        """Call OpenRouter API with retry."""
        messages = self._build_messages(system_msg, user_msg)

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "tool_choice": "none",
        }
        if self.openrouter_reasoning:
            body["reasoning"] = dict(self.openrouter_reasoning)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.api_url.rstrip('/')}/chat/completions"
        if not self.api_url.strip():
            url = "https://openrouter.ai/api/v1/chat/completions"

        try:
            raw = self._call_openrouter_api(body, headers, url)
        except Exception as e:
            logger.error("Judge API failed after retries: %s", e)
            return self._error_result(mode, f"API error: {e}")

        try:
            data = json.loads(raw)
            choices = data.get("choices") or []
            if not choices:
                return self._error_result(mode, "Judge response has no choices")
            msg = choices[0].get("message") or {}
            content = _normalize_assistant_message_text(msg)
            if not content:
                return self._error_result(mode, "Empty judge assistant content")
            return self._parse_judge_response(content, mode)
        except Exception as e:
            logger.error("Failed to parse judge response: %s", e)
            return self._error_result(mode, f"Parse error: {e}")

    def _call_puter(self, system_msg: str, user_msg: str, mode: str) -> Dict[str, Any]:
        """Call Puter OpenAI-compatible endpoint."""
        if not self.api_key.strip():
            return self._error_result(mode, "PUTER API key is empty (set judge.api_key or PUTER_API_KEY)")

        if not self.api_url.strip():
            self.api_url = "https://api.puter.com/puterai/openai/v1"

        return self._call_openrouter(system_msg, user_msg, mode)

    def _call_local(self, system_msg: str, user_msg: str, mode: str) -> Dict[str, Any]:
        """Call local model (HF causal LM via LocalHFServing; dtype / BitsAndBytes from config)."""
        from dst_memory.clients.llm_client import _torch_dtype_from_string
        from dst_memory.clients.serving import GenerationConfig, LocalHFServing

        if self._local_serving is None:
            resolved = (self.local_model_path or "").strip() or (self.model or "").strip()
            if not resolved:
                raise ValueError(
                    "local judge mode: set judge.local_model_path or judge.model (HF id / directory)"
                )
            td = _torch_dtype_from_string(self.load_dtype)
            logger.info(
                "Loading local judge model: %s torch_dtype=%s load_quantization=%s",
                resolved,
                td,
                self.load_quantization,
            )
            self._local_serving = LocalHFServing(
                resolved,
                torch_dtype=td,
                enable_thinking=self.enable_thinking,
                load_quantization=self.load_quantization,
            )

        messages = self._build_messages(system_msg, user_msg)

        try:
            gen_cfg = GenerationConfig(
                max_new_tokens=int(self.max_tokens),
                do_sample=float(self.temperature) > 0.0,
                temperature=float(self.temperature) if float(self.temperature) > 0.0 else 1.0,
            )
            response = self._local_serving.generate_chat(messages, generation_config=gen_cfg)
            return self._parse_judge_response(response, mode)
        except Exception as e:
            logger.error("Local judge evaluation error: %s", e)
            return self._error_result(mode, f"Local eval error: {e}")

    def _parse_judge_response(self, content: str, mode: str) -> Dict[str, Any]:
        """Parse JSON from judge response."""
        # Extract JSON from markdown code blocks if present
        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 3:
                json_str = parts[1].strip()

        result = json.loads(json_str)

        if mode == "correctness":
            return {
                "score": float(result.get("score", 0)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        else:  # memory_hit
            return {
                "fact_present": bool(result.get("fact_present", False)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
                "location": str(result.get("location", "unknown")),
            }

    def _error_result(self, mode: str, error_msg: str) -> Dict[str, Any]:
        """Return error result."""
        if mode == "correctness":
            return {"score": 0.0, "reasoning": error_msg}
        else:
            return {"fact_present": False, "reasoning": error_msg, "location": "error"}

    def unload(self):
        """Unload local model to free memory."""
        if self._local_serving is not None:
            logger.info("Unloading local judge model")
            import gc

            try:
                self._local_serving.release()
            except Exception as e:
                logger.warning("Judge LocalHFServing.release failed: %s", e)
            self._local_serving = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass


# ============================================================================
# Memory State Persistence
# ============================================================================

class MemoryStatePersistence:
    """Handles saving and loading of memory state and RAGU storage."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_chunk_state(
        self,
        chunk_id: str,
        pipeline: "DSTMemoryPipeline",
        dialogue_id: str,
    ) -> Dict[str, Path]:
        """Save memory state for a processed chunk."""
        chunk_dir = self.output_dir / f"chunk_{chunk_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # Save DST state
        state = pipeline.dst.get_state(dialogue_id)
        state_dict = {
            "dialogue_id": state.dialogue_id,
            "step": state.step,
            "next_record_id": state.next_record_id,
            "slots": {
                slot: [
                    {
                        "record_id": r.record_id,
                        "value": r.value,
                        "source_text": r.source_text,
                        "created_at_step": r.created_at_step,
                        "updated_at_step": r.updated_at_step,
                        "subject": r.subject,
                        "relation": r.relation,
                        "object": r.object,
                        "is_active": r.is_active,
                        "ttl": r.ttl,
                        "created_at_datetime": r.created_at_datetime,
                    }
                    for r in records
                ]
                for slot, records in state.slots.items()
            },
            "deleted_facts": [
                {
                    "slot": d.slot,
                    "record_id": d.record_id,
                    "subject": d.subject,
                    "relation": d.relation,
                    "object": d.object,
                    "value": d.value,
                    "source_text": d.source_text,
                    "created_at_step": d.created_at_step,
                    "created_at_datetime": d.created_at_datetime,
                    "deleted_at_step": d.deleted_at_step,
                    "deletion_reason": d.deletion_reason,
                    "deletion_source": d.deletion_source,
                    "deletion_details": d.deletion_details,
                }
                for d in state.deleted_facts
            ],
            "recent_pairs": state.recent_pairs,
        }

        state_path = chunk_dir / "dst_state.json"
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)
        saved_paths["dst_state"] = state_path

        # Save RAGU storage (path from Settings / config — not kg.storage_path)
        if pipeline.ragu_processor is not None:
            ragu_storage_src = resolve_ragu_storage_directory(pipeline)
            if ragu_storage_src is not None and ragu_storage_src.exists():
                ragu_storage_dst = chunk_dir / "ragu_storage"
                if ragu_storage_dst.exists():
                    shutil.rmtree(ragu_storage_dst)
                shutil.copytree(ragu_storage_src, ragu_storage_dst)
                saved_paths["ragu_storage"] = ragu_storage_dst
            else:
                logger.warning(
                    "RAGU storage directory not found or empty — skipping copy into %s "
                    "(set giga_memory.ragu_storage_path or rely on RAGU Settings.storage_folder)",
                    chunk_dir,
                )

        logger.info("Saved chunk state to %s", chunk_dir)
        return saved_paths


# ============================================================================
# Dataset Loading
# ============================================================================

def _validation_sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
    """Stable ordering: index in source JSON array, then global_index within the run."""
    raw = row.get("_validation_dataset_ordinal")
    ds = int(raw) if raw is not None else (1 << 30)
    gi = row.get("global_index")
    gi_int = int(gi) if gi is not None else 0
    return (ds, gi_int)


def load_dataset_balanced(
    dataset_path: str,
    question_types: List[str],
    num_per_type: int
) -> List[Dict[str, Any]]:
    """Load dataset with balanced sampling across question types.

    Selection is deterministic: a single top-to-bottom scan of the JSON array.
    For each row, if ``question_type`` is in ``question_types`` and that type's
    quota is not yet filled, the row is taken (first occurrences win). Order in
    the returned list follows **file order** of selected rows (not grouped by
    type). Each item includes ``_validation_dataset_ordinal`` = 0-based index in
    the source ``data`` array for traceability across validation modes.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    limits = {qt: num_per_type for qt in question_types}
    counts = {qt: 0 for qt in question_types}
    result: List[Dict[str, Any]] = []

    for dataset_ordinal, item in enumerate(data):
        qt = item.get("question_type", "")
        if qt not in counts:
            continue
        if counts[qt] >= limits[qt]:
            continue
        row = dict(item)
        row["_validation_dataset_ordinal"] = dataset_ordinal
        result.append(row)
        counts[qt] += 1
        if all(counts[t] >= limits[t] for t in question_types):
            break

    for qt in question_types:
        got = counts.get(qt, 0)
        logger.info("Type %s: sampled %d (requested up to %d per type)", qt, got, num_per_type)

    logger.info("Total loaded: %d items (deterministic file-order sampling)", len(result))
    return result


def extract_user_messages_from_sessions(sessions: List[List[Dict]]) -> List[str]:
    """Extract all user messages from a list of sessions."""
    user_messages = []
    for session in sessions:
        for turn in session:
            if turn.get("role", "").lower() == "user":
                content = (turn.get("content") or "").strip()
                if content:
                    user_messages.append(content)
    return user_messages


def normalize_question_specs(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Build the list of evaluation questions for one dataset row.

    If ``questions`` is a non-empty list of dicts, each dict should have
    ``question_id``, ``question``, and ``answer`` (or ``reference_answer``).
    Otherwise the legacy single fields ``question_id`` / ``question`` / ``answer``
    on the row are used as one question.

    All questions share the same ``haystack_sessions`` and one memory pass
    (``write_to_memory`` for the whole dialogue), then are scored separately.
    """
    raw = item.get("questions")
    if isinstance(raw, list) and len(raw) > 0:
        specs: List[Dict[str, str]] = []
        for i, q in enumerate(raw):
            if not isinstance(q, dict):
                continue
            qid = str(q.get("question_id", "") or "").strip() or f"q_{i}"
            qtext = str(q.get("question", "") or "")
            ref = q.get("answer", q.get("reference_answer", ""))
            specs.append({
                "question_id": qid,
                "question": qtext,
                "reference_answer": str(ref if ref is not None else ""),
            })
        if specs:
            return specs
    return [{
        "question_id": str(item.get("question_id", "") or ""),
        "question": str(item.get("question", "") or ""),
        "reference_answer": str(item.get("answer", "") or ""),
    }]


# ============================================================================
# Pipeline Building
# ============================================================================

def build_pipeline_config(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None):
    """Build PipelineConfig from DST run_config + overrides. Does not load any PyTorch models."""
    from dst_memory import PipelineConfig

    file_cfg = load_run_config(config_path)
    shared = shared_section(file_cfg)
    if cli_overrides:
        shared.update(cli_overrides)

    return PipelineConfig(
        importance_model_path=shared.get("importance_model_path", ""),
        importance_threshold=float(shared.get("importance_threshold", 0.5)),
        retrieval_top_k=int(shared.get("retrieval_top_k", 5)),
        graph_top_k_records=int(shared.get("graph_top_k_records", 20)),
        recent_history_pairs=int(shared.get("recent_history_pairs", 5)),
        use_memory_gate=not shared.get("disable_memory_gate", False),
        memory_gate_use_stub=shared.get("memory_gate_use_stub", False),
        memory_strategy=shared.get("memory_strategy", "full_graph_json"),
        llm_mode=shared.get("llm_mode", "openrouter"),
        llm_api_url=shared.get("llm_api_url", "https://openrouter.ai/api/v1"),
        llm_api_key=shared.get("llm_api_key", ""),
        llm_model=shared.get("llm_model", "openai/gpt-oss-120b:free"),
        llm_tokenizer_model=shared.get("llm_tokenizer_model", ""),
        llm_load_dtype=str(shared.get("llm_load_dtype", "float16")),
        llm_load_quantization=str(shared.get("llm_load_quantization", "none")),
        llm_max_context_tokens=int(shared.get("llm_max_context_tokens", 128 * 1024)),
        llm_max_tokens=int(shared.get("llm_max_tokens", 1024)),
        llm_temperature=float(shared.get("llm_temperature", 0.0)),
        llm_enable_thinking=bool(shared.get("llm_enable_thinking", True)),
        openrouter_http_referer=shared.get("openrouter_http_referer", ""),
        openrouter_x_title=shared.get("openrouter_x_title", ""),
        slot_use_stub=shared.get("slot_use_stub", False),
        slot_model_path=shared.get("slot_model_path", "Qwen/Qwen3-0.6B"),
        slot_max_slots_per_message=int(shared.get("slot_max_slots_per_message", 5)),
        use_ragu=True,
        ragu_embedder_model=shared.get("ragu_embedder_model", "deepvk/USER-bge-m3"),
        ragu_storage_path=shared.get("ragu_storage_path", ""),
        ttl_mode=shared.get("ttl_mode", "mode2"),
        ttl_slot_overrides=shared.get("ttl_slot_overrides", {}),
        ttl_semantic_dedup_enabled=shared.get("ttl_semantic_dedup_enabled", True),
        ttl_semantic_dedup_threshold=float(shared.get("ttl_semantic_dedup_threshold", 0.9)),
        slot_context_enabled=shared.get("slot_context_enabled", False),
        slot_context_max_facts=int(shared.get("slot_context_max_facts", 10)),
        triplet_deletion_mode=shared.get("triplet_deletion_mode", "none"),
        deletion_use_pymorphy=shared.get("deletion_use_pymorphy", False),
        conflict_allow_multi_relation_same_object=shared.get(
            "conflict_allow_multi_relation_same_object", True
        ),
        conflict_rule_same_relation_updates=bool(
            shared.get("conflict_rule_same_relation_updates", True)
        ),
        slot_model_enable_thinking=shared.get("slot_model_enable_thinking", False),
        slot_llm_inject_no_think_prompt=bool(shared.get("slot_llm_inject_no_think_prompt", True)),
        slot_llm_lm_format_enforcer=bool(shared.get("slot_llm_lm_format_enforcer", False)),
        slot_llm_load_quantization=str(shared.get("slot_llm_load_quantization", "none") or "none"),
        slot_fallback_on_no_slots=shared.get("slot_fallback_on_no_slots", True),
        triplet_fallback_on_empty=shared.get("triplet_fallback_on_empty", True),
        prompt_language=shared.get("prompt_language", "ru"),
        unload_models_before_final_llm=shared.get("unload_models_before_final_llm", True),
        use_dataset_datetime=bool(shared.get("use_dataset_datetime", False)),
        force_infinite_ttl=bool(shared.get("force_infinite_ttl", True)),
        openrouter_reasoning=coerce_openrouter_reasoning_body(
            shared.get("openrouter_reasoning"),
            shared.get("llm_mode", "stub"),
        ),
        final_llm_memory_payload_mode=normalize_memory_payload_mode(
            shared.get("final_llm_memory_payload_mode", "with_metadata")
        ),
    )


class FinalLLMOnlyPipelineFacade:
    """
    Minimal stand-in for ``DSTMemoryPipeline`` in ``final_llm_only`` validation mode.

    Avoids loading slot/triplet models, importance classifier, RAGU graph, and DST —
    those already ran in ``memory_only``. Only ``FinalLLMClient`` touches the GPU
    (on first ``generate`` for local mode).
    """

    def __init__(self, config: Any, final_llm: Any):
        self.config = config
        self.final_llm = final_llm
        self.ragu_processor = None

    def unload_local_models(self) -> None:
        """Release local final LLM weights; nothing else was loaded on this facade."""
        import gc

        logger.info(
            "[FinalLLMOnly] unload_local_models: releasing final LLM only "
            "(slot/triplet/classifier were not loaded in this process)"
        )
        if hasattr(self.final_llm, "release_local_serving"):
            self.final_llm.release_local_serving()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def reload_local_models(self) -> None:
        """No-op: memory subsystem is absent in final_llm_only facade."""
        logger.info(
            "[FinalLLMOnly] reload_local_models: skipped (no slot model to restore in this mode)"
        )


def build_final_llm_only_facade(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None):
    """Lightweight pipeline for final_llm_only — FinalLLMClient + config only."""
    from dst_memory.clients.llm_client import FinalLLMClient

    cfg = build_pipeline_config(config_path, cli_overrides)
    final_llm = FinalLLMClient(
        mode=cfg.llm_mode,
        api_url=cfg.llm_api_url,
        api_key=cfg.llm_api_key,
        model=cfg.llm_model,
        tokenizer_model=getattr(cfg, "llm_tokenizer_model", ""),
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        http_referer=cfg.openrouter_http_referer,
        x_title=cfg.openrouter_x_title,
        prompt_language=cfg.prompt_language,
        load_dtype=cfg.llm_load_dtype,
        enable_thinking=getattr(cfg, "llm_enable_thinking", True),
        load_quantization=getattr(cfg, "llm_load_quantization", "none"),
        max_context_tokens=getattr(cfg, "llm_max_context_tokens", 128 * 1024),
        openrouter_reasoning=getattr(cfg, "openrouter_reasoning", None),
    )
    logger.info(
        "final_llm_only: using FinalLLMOnlyPipelineFacade — no DST/RAGU/slot/triplet "
        "or importance models in this process (memory was built in memory_only)."
    )
    return FinalLLMOnlyPipelineFacade(cfg, final_llm)


def build_pipeline_from_config(config_path: str, cli_overrides: Optional[Dict[str, Any]] = None):
    """Build full DSTMemoryPipeline from config file with optional CLI overrides."""
    from dst_memory.core.pipeline import DSTMemoryPipeline
    from dst_memory.storage.ragu_graph_processor import build_ragu_processor

    cfg = build_pipeline_config(config_path, cli_overrides)

    logger.info("Initializing RAGU backend...")
    _kg, ragu_processor = build_ragu_processor(
        embedder_model=cfg.ragu_embedder_model,
        storage_path=cfg.ragu_storage_path or None,
    )

    return DSTMemoryPipeline(cfg, ragu_processor=ragu_processor)


# ============================================================================
# Batch Processing Classes
# ============================================================================

def _atomic_write_validation_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


def resolve_ragu_storage_directory(pipeline: Any) -> Optional[Path]:
    """
    Resolve the on-disk RAGU storage folder for the running pipeline.

    RAGU's ``KnowledgeGraph`` does not expose ``storage_path``; persistence uses
    ``Settings.storage_folder`` (set in ``build_ragu_processor``) and/or
    ``PipelineConfig.ragu_storage_path``.
    """
    if pipeline is None or getattr(pipeline, "ragu_processor", None) is None:
        return None
    kg = pipeline.ragu_processor.kg

    legacy = getattr(kg, "storage_path", None)
    if legacy:
        p = Path(str(legacy))
        if p.exists():
            return p

    cfg = getattr(pipeline, "config", None)
    if cfg is not None:
        rp = (getattr(cfg, "ragu_storage_path", None) or "").strip()
        if rp:
            p = Path(rp)
            if p.exists():
                return p

    try:
        from ragu.common.global_parameters import Settings as RaguSettings

        RaguSettings.init_storage_folder()
        sf = getattr(RaguSettings, "storage_folder", None)
        if sf:
            p = Path(str(sf))
            if p.exists():
                return p
    except Exception as exc:
        logger.debug("resolve_ragu_storage_directory: RAGU Settings fallback failed: %s", exc)

    return None


def maybe_build_ragu_graph_html(output_dir: Path, pipeline: Any, stem: str = "validation_knowledge_graph") -> None:
    """
    Build HTML graph visualization (same flow as ``DST_memory/run.py`` pipeline test).

    Writes ``{output_dir}/{stem}.html`` when ``knowledge_graph.gml`` exists under RAGU storage.
    """
    storage = resolve_ragu_storage_directory(pipeline)
    if storage is None:
        logger.warning("Graph HTML: could not resolve RAGU storage directory — skipping")
        return
    graph_path = storage / "knowledge_graph.gml"
    if not graph_path.exists():
        logger.warning(
            "Graph HTML: knowledge_graph.gml not found at %s — skipping visualization",
            graph_path,
        )
        return

    repo_root = Path(__file__).resolve().parents[2]
    viz_script = repo_root / "RAGU" / "scripts" / "visualize_knowledge_graph.py"
    if not viz_script.exists():
        logger.warning("Graph HTML: script not found at %s — skipping", viz_script)
        return

    html_out = Path(output_dir) / f"{stem}.html"
    cmd = [
        sys.executable,
        str(viz_script),
        "--graph-path",
        str(graph_path),
        "--output",
        str(html_out),
    ]
    logger.info("Graph HTML: running %s", " ".join(cmd))
    try:
        import subprocess

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info("Graph HTML saved: %s", html_out)
        else:
            logger.warning(
                "Graph HTML: visualization failed (code %d): %s",
                result.returncode,
                (result.stderr or "")[:1000],
            )
    except Exception as exc:
        logger.warning("Graph HTML: could not run visualization: %s", exc)


@dataclass
class AccumulatedDialogue:
    """Accumulated dialogue data ready for final LLM."""
    global_index: int
    dialogue_id: str
    question_id: str
    question: str
    reference_answer: str
    question_type: str
    pipeline_state: Dict[str, Any]  # Memory slots, deleted facts, etc.
    dataset_ordinal: Optional[int] = None  # index in source LongMemEval JSON array
    dialogue_row_index: Optional[int] = None  # dataset row index (one chunk_* per row)
    # LongMemEval row ``question_date`` (shared for all questions in the row); replay for final_llm_only.
    question_date: Optional[Any] = None
    # Preformatted clock for FinalLLMClient when use_dataset_datetime (same as pipeline would use).
    final_llm_clock_display: Optional[str] = None


@dataclass
class AccumulatedAnswer:
    """Accumulated answer ready for judge evaluation."""
    global_index: int
    question_id: str
    question: str
    reference_answer: str
    question_type: str
    predicted_answer: str
    memory_context: Dict[str, Any]
    memory_strategy: str = ""
    inactive_facts_mode: str = ""
    judge_intermediate_source: str = ""
    dialogue_context_chars: int = 0
    final_llm_prompt_chars_before_clamp: int = 0
    final_llm_prompt_chars_after_clamp: int = 0
    dataset_ordinal: Optional[int] = None
    dialogue_row_index: Optional[int] = None


@dataclass
class MemoryOnlyState:
    """State saved in memory_only mode for later processing."""
    global_index: int
    dialogue_id: str
    question_id: str
    question: str
    reference_answer: str
    question_type: str
    memory_state_path: str  # Path to chunk_XXXX directory
    dst_state: Dict[str, Any]  # Serialized DST state
    ragu_storage_path: Optional[str]  # Path to RAGU storage
    pipeline_state: Dict[str, Any]  # Full pipeline state including memory_context
    dataset_ordinal: Optional[int] = None  # index in source LongMemEval JSON array
    dialogue_row_index: Optional[int] = None  # one chunk dir per row; shared across questions
    question_date: Optional[Any] = None  # LongMemEval row field; for final_llm_only clock replay


@dataclass
class IntermediateAnswer:
    """Intermediate answer saved for judge_only mode."""
    global_index: int
    question_id: str
    question: str
    reference_answer: str
    question_type: str
    predicted_answer: str
    memory_context: Dict[str, Any]
    memory_state_path: str  # Path for Memory Hit Rate calculation
    memory_strategy: str = ""
    inactive_facts_mode: str = "active_only"
    dialogue_context_chars: int = 0
    final_llm_prompt_chars_before_clamp: int = 0
    final_llm_prompt_chars_after_clamp: int = 0
    dataset_ordinal: Optional[int] = None
    dialogue_row_index: Optional[int] = None
    question_date: Optional[Any] = None  # LongMemEval row; final_llm_only uses with use_dataset_datetime


def _stats_from_intermediate_answer_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recompute intermediate_answers.json ``statistics`` from the ``answers`` list."""
    n = len(rows)
    dcc_total = sum(int(r.get("dialogue_context_chars", 0) or 0) for r in rows)
    pb = sum(int(r.get("final_llm_prompt_chars_before_clamp", 0) or 0) for r in rows)
    pa = sum(int(r.get("final_llm_prompt_chars_after_clamp", 0) or 0) for r in rows)
    return {
        "answers_count": n,
        "dialogue_context_chars_total": dcc_total,
        "dialogue_context_chars_avg": float(dcc_total / n) if n else 0.0,
        "final_llm_calls_total": n,
        "final_llm_prompt_chars_before_clamp_total": pb,
        "final_llm_prompt_chars_after_clamp_total": pa,
    }


def _merge_intermediate_answer_rows_for_resume(
    prev_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
    resume_from: int,
) -> List[Dict[str, Any]]:
    """
    Rows with ``global_index`` < ``resume_from`` are kept from ``prev_rows``.
    Rows with ``global_index`` >= ``resume_from`` come from ``new_rows`` (replacing any stale tail in prev).
    """
    by_gix: Dict[int, Dict[str, Any]] = {}
    for r in prev_rows:
        try:
            gix = int(r.get("global_index", -1))
        except (TypeError, ValueError):
            continue
        if gix < resume_from:
            by_gix[gix] = r
    for r in new_rows:
        try:
            gix = int(r.get("global_index", -1))
        except (TypeError, ValueError):
            continue
        if gix >= resume_from:
            by_gix[gix] = r
    return sorted(
        by_gix.values(),
        key=lambda x: (int(x.get("global_index", 0)), str(x.get("question_id", ""))),
    )


def _intermediate_answer_to_json_row(
    ans: IntermediateAnswer,
    strategy: str,
    inactive_tag: str,
) -> Dict[str, Any]:
    """Serialize one ``IntermediateAnswer`` like ``_write_intermediate_answers``."""
    row_a: Dict[str, Any] = {
        "global_index": ans.global_index,
        "question_id": ans.question_id,
        "question": ans.question,
        "reference_answer": ans.reference_answer,
        "question_type": ans.question_type,
        "predicted_answer": ans.predicted_answer,
        "memory_context": ans.memory_context,
        "memory_strategy": strategy,
        "inactive_facts_mode": inactive_tag,
        "dialogue_context_chars": int(ans.dialogue_context_chars or 0),
        "memory_state_path": ans.memory_state_path,
        "final_llm_prompt_chars_before_clamp": ans.final_llm_prompt_chars_before_clamp,
        "final_llm_prompt_chars_after_clamp": ans.final_llm_prompt_chars_after_clamp,
    }
    if ans.dataset_ordinal is not None:
        row_a["_validation_dataset_ordinal"] = ans.dataset_ordinal
    if ans.dialogue_row_index is not None:
        row_a["_validation_dialogue_row_index"] = ans.dialogue_row_index
    if ans.question_date is not None:
        row_a["question_date"] = ans.question_date
    return row_a


def build_judge_statistics_export(
    stats: Dict[str, Any],
    calculate_memory_hit_rate: bool,
) -> Dict[str, Any]:
    """Split final JSON statistics: answer correctness vs memory-hit evaluation (MHE)."""
    total = int(stats.get("total", 0) or 0)
    total_score = float(stats.get("total_score", 0.0) or 0.0)
    by_type_raw = stats.get("by_type") or {}

    answer_by_type: Dict[str, Any] = {}
    for qt, st in by_type_raw.items():
        if not isinstance(st, dict):
            continue
        c = int(st.get("count", 0) or 0)
        ts = float(st.get("total_score", 0.0) or 0.0)
        err = int(st.get("errors", 0) or 0)
        entry: Dict[str, Any] = {
            "count": c,
            "total_score": ts,
            "errors": err,
            "average_score": (ts / c) if c > 0 else 0.0,
        }
        answer_by_type[qt] = entry

    hits = int(stats.get("memory_hit", 0) or 0)
    misses = int(stats.get("memory_miss", 0) or 0)
    mhe_total = hits + misses
    final_llm_calls = int(stats.get("final_llm_calls", 0) or 0)
    chars_before_total = int(stats.get("final_llm_prompt_chars_before_clamp_total", 0) or 0)
    chars_after_total = int(stats.get("final_llm_prompt_chars_after_clamp_total", 0) or 0)
    dialogue_chars_total = int(stats.get("dialogue_context_chars_total", 0) or 0)
    dialogue_chars_count = int(stats.get("dialogue_context_chars_count", 0) or 0)

    mhe_by_type: Dict[str, Any] = {}
    for qt, st in by_type_raw.items():
        if not isinstance(st, dict):
            continue
        mc = int(st.get("mhe_count", 0) or 0)
        mh = int(st.get("mhe_hits", 0) or 0)
        mm = int(st.get("mhe_misses", 0) or 0)
        row: Dict[str, Any] = {"count": mc, "hits": mh, "misses": mm}
        if mc > 0:
            row["hit_rate"] = mh / mc
        mhe_by_type[qt] = row

    return {
        "answer_evaluation": {
            "total": total,
            "total_score": total_score,
            "average_score": (total_score / total) if total > 0 else 0.0,
            "errors_judge": int(stats.get("errors_judge", 0) or 0),
            "by_type": answer_by_type,
        },
        "memory_hit_evaluation": {
            "enabled": bool(calculate_memory_hit_rate),
            "total": mhe_total,
            "hits": hits,
            "misses": misses,
            "hit_rate": (hits / mhe_total) if mhe_total > 0 else None,
            "by_type": mhe_by_type,
        },
        "final_llm_prompt_chars": {
            "calls": final_llm_calls,
            "before_clamp_total": chars_before_total,
            "after_clamp_total": chars_after_total,
            "before_clamp_avg": (chars_before_total / final_llm_calls) if final_llm_calls > 0 else 0.0,
            "after_clamp_avg": (chars_after_total / final_llm_calls) if final_llm_calls > 0 else 0.0,
        },
        "dialogue_context_chars": {
            "count": dialogue_chars_count,
            "total": dialogue_chars_total,
            "avg": (dialogue_chars_total / dialogue_chars_count) if dialogue_chars_count > 0 else 0.0,
        },
    }


def _is_full_dst_pipeline(pipeline: Any) -> bool:
    if pipeline is None:
        return False


def _enable_memory_only_single_path_mode(pipeline: Any) -> None:
    """
    Force memory writes to use single-path extraction only in validation memory_only.

    Implementation detail: monkey-patch slot selector to return no slots, which
    deterministically activates existing DSTManager single-pass fallback path.
    """
    if not _is_full_dst_pipeline(pipeline):
        logger.warning("[MemoryOnly] single_path_only requested, but pipeline is not DSTMemoryPipeline")
        return
    dst = getattr(pipeline, "dst", None)
    if dst is None:
        logger.warning("[MemoryOnly] single_path_only requested, but pipeline.dst is missing")
        return
    slot_selector = getattr(dst, "slot_selector", None)
    if slot_selector is None or not hasattr(slot_selector, "select_slots"):
        logger.warning("[MemoryOnly] single_path_only requested, but slot_selector is missing")
        return
    if getattr(slot_selector, "_validation_single_path_forced", False):
        return

    original = slot_selector.select_slots

    def _select_slots_single_path_only(_user_text: str) -> List[str]:
        return []

    setattr(slot_selector, "_validation_original_select_slots", original)
    slot_selector.select_slots = _select_slots_single_path_only  # type: ignore[method-assign]
    setattr(slot_selector, "_validation_single_path_forced", True)
    # Ensure fallback route is enabled.
    if hasattr(dst, "slot_fallback_on_no_slots"):
        dst.slot_fallback_on_no_slots = True
    logger.info("[MemoryOnly] single_path_only enabled: slot selection bypassed, using only single-pass extraction")
    try:
        from dst_memory.core.pipeline import DSTMemoryPipeline

        return isinstance(pipeline, DSTMemoryPipeline)
    except Exception:
        return False


class BatchProcessor:
    """
    Handles batch processing for final LLM and judge.
    Supports multiple validation modes: full, memory_only, final_llm_only, judge_only

    Flow:
    1. Process dialogues through memory pipeline (write_to_memory for all sessions)
    2. Accumulate processed dialogues (without calling final LLM)
    3. When batch is full (or at end), call final LLM for all accumulated dialogues
    4. Accumulate answers
    5. When judge batch is full (or at end), call judge for evaluation
    """

    def __init__(
        self,
        pipeline: Optional["DSTMemoryPipeline"],
        judge_client: Optional[JudgeClient],
        final_llm_batch_size: int,
        judge_batch_size: int,
        calculate_memory_hit_rate: bool,
        persistence: MemoryStatePersistence,
        results_json_path: Optional[Path] = None,
        validation_metadata: Optional[Dict[str, Any]] = None,
        timing: Optional[Any] = None,
        validation_mode: str = "full",
        input_state_dir: Optional[Path] = None,
        input_answers_path: Optional[Path] = None,
        input_answers_paths: Optional[List[str]] = None,
        final_llm_memory_strategies: Optional[List[str]] = None,
        final_llm_memory_payload_mode: str = "with_metadata",
        inactive_facts_memory_modes: Optional[List[str]] = None,
        final_llm_resume_from_global_index: Optional[int] = None,
        save_giga_memory_logs: bool = True,
    ):
        self.pipeline = pipeline
        self.judge_client = judge_client
        self.final_llm_batch_size = final_llm_batch_size
        self.judge_batch_size = judge_batch_size
        self.calculate_memory_hit_rate = calculate_memory_hit_rate
        self.persistence = persistence
        self._results_json_path = results_json_path
        self._validation_metadata = validation_metadata
        self._timing = timing
        self.validation_mode = validation_mode
        self.input_state_dir = input_state_dir
        paths_norm = _normalize_input_answers_paths(input_answers_paths)
        if paths_norm:
            self.input_answers_paths: List[Path] = [Path(p) for p in paths_norm]
            self.input_answers_path: Optional[Path] = self.input_answers_paths[0]
        elif input_answers_path is not None:
            self.input_answers_path = Path(input_answers_path)
            self.input_answers_paths = [self.input_answers_path]
        else:
            self.input_answers_path = None
            self.input_answers_paths = []
        self.save_giga_memory_logs = save_giga_memory_logs
        cfg_strategy = getattr(self.pipeline.config, "memory_strategy", "full_graph_json") if self.pipeline else "full_graph_json"
        selected = _normalize_memory_strategies(final_llm_memory_strategies or [])
        self.final_llm_memory_strategies = selected or [cfg_strategy]
        self.final_llm_memory_payload_mode = _normalize_memory_payload_mode(final_llm_memory_payload_mode)
        self.inactive_facts_memory_modes = _normalize_inactive_facts_memory_modes(inactive_facts_memory_modes)
        if final_llm_resume_from_global_index is None:
            self.final_llm_resume_from_global_index: Optional[int] = None
        else:
            self.final_llm_resume_from_global_index = int(final_llm_resume_from_global_index)
        # When local judge follows local final LLM on full DST pipeline, defer slot reload until judge finishes.
        self._pending_slot_reload_after_judge = False

        # Accumulators
        self.dialogue_buffer: List[AccumulatedDialogue] = []
        self.answer_buffer: List[AccumulatedAnswer] = []

        # Results storage
        self.results: List[Dict[str, Any]] = []

        # Verbose GigaMemory logs (same role as DST_memory pipeline test *_logs.json)
        self.dialogue_logs: List[Dict[str, Any]] = []

        # Mode-specific storage
        self.memory_only_states: List[MemoryOnlyState] = []  # For memory_only mode
        self.intermediate_answers: List[IntermediateAnswer] = []  # For final_llm_only output

        # Statistics (MHE = memory hit evaluation: second judge call when calculate_memory_hit_rate)
        self.stats = {
            "total": 0,
            "total_score": 0.0,
            "errors_final_llm": 0,
            "errors_judge": 0,
            "final_llm_calls": 0,
            "final_llm_prompt_chars_before_clamp_total": 0,
            "final_llm_prompt_chars_after_clamp_total": 0,
            "dialogue_context_chars_total": 0,
            "dialogue_context_chars_count": 0,
            "memory_hit": 0,
            "memory_miss": 0,
            "by_type": {
                qt: {
                    "count": 0,
                    "total_score": 0.0,
                    "errors": 0,
                    "mhe_count": 0,
                    "mhe_hits": 0,
                    "mhe_misses": 0,
                }
                for qt in QUESTION_TYPES.keys()
            },
        }
        self._judge_only_total_rows: int = 0
        self._judge_incremental_results_json_path: Optional[Path] = None
        self._judge_shard_metadata_extra: Optional[Dict[str, Any]] = None

    def _fresh_judge_stats(self) -> Dict[str, Any]:
        """Reset counters for a fresh judge_only shard (one intermediate_answers.json file)."""
        return {
            "total": 0,
            "total_score": 0.0,
            "errors_final_llm": 0,
            "errors_judge": 0,
            "final_llm_calls": 0,
            "final_llm_prompt_chars_before_clamp_total": 0,
            "final_llm_prompt_chars_after_clamp_total": 0,
            "dialogue_context_chars_total": 0,
            "dialogue_context_chars_count": 0,
            "memory_hit": 0,
            "memory_miss": 0,
            "by_type": {
                qt: {
                    "count": 0,
                    "total_score": 0.0,
                    "errors": 0,
                    "mhe_count": 0,
                    "mhe_hits": 0,
                    "mhe_misses": 0,
                }
                for qt in QUESTION_TYPES.keys()
            },
        }

    def _finalize_judge_by_type_averages(self) -> None:
        """Fill average_score / hit_rate on ``self.stats[\"by_type\"]`` before exporting JSON."""
        for qt in self.stats["by_type"]:
            count = self.stats["by_type"][qt]["count"]
            if count > 0:
                self.stats["by_type"][qt]["average_score"] = (
                    self.stats["by_type"][qt]["total_score"] / count
                )
            mc = self.stats["by_type"][qt].get("mhe_count", 0)
            if mc > 0:
                self.stats["by_type"][qt]["mhe_hit_rate"] = (
                    self.stats["by_type"][qt]["mhe_hits"] / mc
                )

    def _capture_memory_state_for_all_strategies(
        self,
        dialogue_id: str,
        question: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        """
        Capture answer_without_final_llm payload per memory strategy.

        This is used in memory_only mode so final_llm_only can later switch
        strategy without rebuilding memory.
        """
        details_by_strategy: Dict[str, Dict[str, Any]] = {}
        contexts: Dict[str, Dict[str, Any]] = {}
        original = getattr(self.pipeline.config, "memory_strategy", "full_graph_json")
        try:
            for strategy in MEMORY_STRATEGIES:
                self.pipeline.config.memory_strategy = strategy
                details = self.pipeline.answer_without_final_llm(dialogue_id, question)
                details_by_strategy[strategy] = details
                contexts[strategy] = details.get("memory_context_for_final_llm", {}) or {}
        finally:
            self.pipeline.config.memory_strategy = original
        return details_by_strategy, contexts

    def process_single_item(
        self,
        item: Dict[str, Any],
        dialogue_row_index: int,
        flat_index_start: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Process one dataset row through the memory pipeline.

        One row may define multiple questions (``questions`` list). All questions
        share a single ``write_to_memory`` pass over ``haystack_sessions``, then
        ``answer_without_final_llm`` / final LLM / judge run per question in order.

        ``dialogue_row_index`` indexes the dataset row (and ``chunk_*`` folder).
        ``flat_index_start`` is the running global index for the first question
        of this row; each question gets ``flat_index_start + k``.

        Behavior depends on validation_mode:
        - full/memory_only: processes through memory pipeline
        - final_llm_only/judge_only: loads from saved state (no processing)

        Returns:
            Result dict if this was the last item and batch was flushed,
            None otherwise (result will be added later in batch)
        """
        # Skip processing in modes that load from saved state
        if self.validation_mode in ("final_llm_only", "judge_only"):
            return None

        question_specs = normalize_question_specs(item)
        if not question_specs:
            raise ValueError(f"No questions in dataset row {dialogue_row_index}")

        question_type = item.get("question_type", "")
        sessions = item.get("haystack_sessions", [])
        haystack_dates = item.get("haystack_dates")
        use_dataset_dt = bool(getattr(self.pipeline.config, "use_dataset_datetime", False))

        did = str(item.get("dialogue_id", "") or "").strip()
        if did:
            dialogue_id = did
        elif len(question_specs) == 1:
            qid0 = question_specs[0]["question_id"] or "unknown"
            dialogue_id = f"longmemeval_{dialogue_row_index}_{qid0}"
        else:
            dialogue_id = f"longmemeval_{dialogue_row_index}"

        qdate = item.get("question_date")
        if hasattr(self.pipeline, "set_dialogue_dataset_clock"):
            self.pipeline.set_dialogue_dataset_clock(dialogue_id, qdate)
        clock_disp = optional_clock_display_for_validation(use_dataset_dt, qdate)

        user_messages = extract_user_messages_from_sessions(sessions)
        dialogue_context_chars = 0
        for session in sessions:
            if not isinstance(session, list):
                continue
            for turn in session:
                if not isinstance(turn, dict):
                    continue
                content = str(turn.get("content") or "")
                dialogue_context_chars += len(content)

        logger.info(
            "[Batch] Row %d: one memory pass, %d question(s), %d sessions, %d user messages, type=%s",
            dialogue_row_index,
            len(question_specs),
            len(sessions),
            len(user_messages),
            question_type,
        )

        write_logs = []
        from dst_memory.core.models import Message

        _msg_preview_len = 500
        mi = 0
        for si, session in enumerate(sessions):
            if not isinstance(session, list):
                continue
            fact_clock_iso = fact_clock_iso_for_haystack_session(
                use_dataset_dt, haystack_dates, si, qdate,
            )
            for ti, turn in enumerate(session):
                if not isinstance(turn, dict):
                    continue
                if str(turn.get("role", "")).lower() != "user":
                    continue
                msg = str(turn.get("content") or "").strip()
                if not msg:
                    continue
                mi += 1
                if len(msg) <= _msg_preview_len:
                    msg_for_log = msg
                else:
                    msg_for_log = msg[:_msg_preview_len] + "…"
                logger.info(
                    "[Batch] Row %d session %d write_to_memory message %d/%d — %s",
                    dialogue_row_index,
                    si,
                    mi,
                    len(user_messages),
                    msg_for_log.replace("\n", "\\n"),
                )
                t0 = time.time()
                log = self.pipeline.write_to_memory(
                    dialogue_id,
                    Message(role="user", content=msg),
                    fact_created_at_iso=fact_clock_iso,
                )
                if self._timing is not None:
                    self._timing.add_user_message(time.time() - t0)
                write_logs.append(log)
                # Haystack user/assistant turns → final-LLM "recent pairs" (same dialogue_id).
                if ti + 1 < len(session) and hasattr(self.pipeline, "add_recent_pair"):
                    nxt = session[ti + 1]
                    if isinstance(nxt, dict) and str(nxt.get("role", "")).lower() == "assistant":
                        assist = str(nxt.get("content") or "").strip()
                        if assist:
                            self.pipeline.add_recent_pair(dialogue_id, msg, assist)

        chunk_id = f"{dialogue_row_index:04d}"
        saved_paths = self.persistence.save_chunk_state(chunk_id, self.pipeline, dialogue_id)

        state = self.pipeline.dst.get_state(dialogue_id)
        dst_snapshot = state.to_dict() if hasattr(state, "to_dict") else {}

        ds_ord = item.get("_validation_dataset_ordinal")

        for qi, qspec in enumerate(question_specs):
            flat_gix = flat_index_start + qi
            question = qspec["question"]
            reference_answer = qspec["reference_answer"]
            qid = qspec["question_id"]

            logger.info(
                "[Batch] Row %d question %d/%d (flat index %d, question_id=%s): %s",
                dialogue_row_index,
                qi + 1,
                len(question_specs),
                flat_gix,
                qid,
                (question[:200] + "…") if len(question) > 200 else question,
            )

            strategy_details: Dict[str, Dict[str, Any]] = {}
            if self.validation_mode == "memory_only":
                strategy_details, _ = self._capture_memory_state_for_all_strategies(
                    dialogue_id, question
                )
                current_strategy = str(getattr(self.pipeline.config, "memory_strategy", "full_graph_json"))
                answer_details = strategy_details.get(current_strategy) or strategy_details.get("full_graph_json", {})
            else:
                answer_details = self.pipeline.answer_without_final_llm(dialogue_id, question)

            if self.save_giga_memory_logs:
                self.dialogue_logs.append({
                    "dialogue_id": dialogue_id,
                    "dialogue_row_index": dialogue_row_index,
                    "global_index": flat_gix,
                    "question_index_in_row": qi,
                    "question_id": qid,
                    "question": question,
                    "question_type": question_type,
                    "reference_answer": reference_answer,
                    "write_logs": write_logs,
                    "answer_without_final_llm": answer_details,
                    "final_llm_prompt": answer_details.get("final_llm_prompt"),
                    "memory_context_for_final_llm": answer_details.get("memory_context_for_final_llm"),
                    "expired_facts": answer_details.get("expired_facts", []),
                    "deleted_facts_with_reasons": answer_details.get("deleted_facts_with_reasons", []),
                })

            if self.validation_mode == "memory_only":
                strategy_state_by_strategy: Dict[str, Dict[str, Any]] = {}
                for strategy, details in strategy_details.items():
                    mc = details.get("memory_context_for_final_llm", {}) or {}
                    selected_slots = []
                    if isinstance(mc, dict):
                        for s in (mc.get("slots") or []):
                            if isinstance(s, dict):
                                sn = str(s.get("slot") or "").strip()
                                if sn:
                                    selected_slots.append(sn)
                    strategy_state_by_strategy[strategy] = {
                        "selected_slot_names": selected_slots,
                        "memory_gate": details.get("memory_gate", {}),
                        "topk_records": list((mc.get("records") if isinstance(mc, dict) else []) or []),
                        "use_memory": details.get("use_memory", False),
                    }
                memory_state = MemoryOnlyState(
                    global_index=flat_gix,
                    dialogue_id=dialogue_id,
                    question_id=qid,
                    question=question,
                    reference_answer=reference_answer,
                    question_type=question_type,
                    memory_state_path=str(saved_paths.get("dst_state", "")),
                    dst_state=copy.deepcopy(dst_snapshot),
                    ragu_storage_path=str(saved_paths.get("ragu_storage", ""))
                    if saved_paths.get("ragu_storage")
                    else None,
                    dataset_ordinal=ds_ord,
                    dialogue_row_index=dialogue_row_index,
                    question_date=qdate,
                    pipeline_state={
                        "write_logs": write_logs,
                        "dialogue_context_chars": dialogue_context_chars,
                        "memory_slots": answer_details.get("memory_slots", []),
                        "memory_gate": answer_details.get("memory_gate", {}),
                        "retrieved": answer_details.get("retrieved", []),
                        "recent_pairs": answer_details.get("recent_pairs", []),
                        "expired_facts": answer_details.get("expired_facts", []),
                        "deleted_facts_with_reasons": answer_details.get("deleted_facts_with_reasons", []),
                        "use_memory": answer_details.get("use_memory", False),
                        "memory_context": answer_details.get("memory_context_for_final_llm", {}),
                        "strategy_state_by_strategy": strategy_state_by_strategy,
                        "final_llm_prompt": answer_details.get("final_llm_prompt", []),
                    },
                )
                self.memory_only_states.append(memory_state)
                self._write_memory_only_states()

            if self.validation_mode == "full":
                acc_dialogue = AccumulatedDialogue(
                    global_index=flat_gix,
                    dialogue_id=dialogue_id,
                    question_id=qid,
                    question=question,
                    reference_answer=reference_answer,
                    question_type=question_type,
                    dataset_ordinal=ds_ord,
                    dialogue_row_index=dialogue_row_index,
                    question_date=qdate,
                    final_llm_clock_display=clock_disp,
                    pipeline_state={
                        "write_logs": write_logs,
                        "dialogue_context_chars": dialogue_context_chars,
                        "memory_slots": answer_details.get("memory_slots", []),
                        "memory_gate": answer_details.get("memory_gate", {}),
                        "retrieved": answer_details.get("retrieved", []),
                        "recent_pairs": answer_details.get("recent_pairs", []),
                        "expired_facts": answer_details.get("expired_facts", []),
                        "deleted_facts_with_reasons": answer_details.get("deleted_facts_with_reasons", []),
                        "use_memory": answer_details.get("use_memory", False),
                        "memory_context": answer_details.get("memory_context_for_final_llm", {}),
                        "final_llm_prompt": answer_details.get("final_llm_prompt", []),
                    },
                )
                self.dialogue_buffer.append(acc_dialogue)

                if len(self.dialogue_buffer) >= self.final_llm_batch_size:
                    self._flush_final_llm_batch()

        self.pipeline.clear_memory(dialogue_id)

        return None  # Result will be added by batch processing

    def _memory_only_active_row_indices(self) -> Set[int]:
        """Row indices present in this run's in-memory buffers (merge replaces these in on-disk JSON)."""
        out: Set[int] = set()
        if self.validation_mode != "memory_only":
            return out
        for s in self.memory_only_states:
            if s.dialogue_row_index is not None:
                out.add(int(s.dialogue_row_index))
        for d in self.dialogue_logs:
            ri = d.get("dialogue_row_index")
            if ri is not None:
                out.add(int(ri))
        return out

    def _memory_only_states_to_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for state in self.memory_only_states:
            row: Dict[str, Any] = {
                "global_index": state.global_index,
                "dialogue_id": state.dialogue_id,
                "question_id": state.question_id,
                "question": state.question,
                "reference_answer": state.reference_answer,
                "question_type": state.question_type,
                "memory_state_path": state.memory_state_path,
                "ragu_storage_path": state.ragu_storage_path,
                "dialogue_context_chars": int((state.pipeline_state or {}).get("dialogue_context_chars", 0) or 0),
                "pipeline_state": state.pipeline_state,
            }
            if state.dataset_ordinal is not None:
                row["_validation_dataset_ordinal"] = state.dataset_ordinal
            if state.dialogue_row_index is not None:
                row["_validation_dialogue_row_index"] = state.dialogue_row_index
            if state.question_date is not None:
                row["question_date"] = state.question_date
            rows.append(row)
        return rows

    def _write_memory_only_states(self) -> None:
        """Write memory_only states to disk for later processing."""
        output_path = self.persistence.output_dir / "memory_only_states.json"
        new_rows = self._memory_only_states_to_rows()
        replaced_rows = self._memory_only_active_row_indices()
        prev_blob = _load_json_optional(output_path)
        prev_states = list((prev_blob or {}).get("states") or []) if prev_blob else []
        kept = [
            s for s in prev_states
            if _dialogue_row_index_from_saved_state(s) not in replaced_rows
        ]
        merged = kept + new_rows
        merged.sort(
            key=lambda r: (
                _dialogue_row_index_from_saved_state(r),
                int(r.get("global_index", 0)),
                str(r.get("question_id", "")),
            ),
        )
        timing_stats = self._timing.get_stats() if self._timing is not None else {}
        _atomic_write_validation_json(output_path, {
            "metadata": self._validation_metadata,
            "timing": timing_stats,
            "statistics": {
                "states_count": len(merged),
                "total_user_message_writes": int(timing_stats.get("total_user_message_writes", 0) or 0),
                "avg_write_to_memory_seconds": float(
                    (timing_stats.get("time_per_user_message") or {}).get("mean", 0.0) or 0.0
                ),
            },
            "states": merged,
        })

    def _flush_final_llm_batch(self):
        """Process accumulated dialogues through final LLM."""
        if not self.dialogue_buffer and self.validation_mode != "final_llm_only":
            return

        # Handle final_llm_only mode - load from saved state
        if self.validation_mode == "final_llm_only":
            self._process_final_llm_only()
            return

        # Skip in memory_only mode
        if self.validation_mode == "memory_only":
            logger.info("[Batch] Skipping final LLM (memory_only mode)")
            self.dialogue_buffer.clear()
            return

        logger.info(
            "[Batch] Flushing final LLM batch: %d dialogues",
            len(self.dialogue_buffer)
        )

        # Unload models if configured and using local final LLM
        if (
            self.pipeline.config.llm_mode == "local"
            and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
        ):
            logger.info("[Batch] Unloading models before final LLM processing...")
            self.pipeline.unload_local_models()

        # Process each dialogue through final LLM
        for acc in self.dialogue_buffer:
            # Restore memory context (we need to rebuild it)
            # Actually, we saved the prompt, so we can use that
            predicted_answer = "[no_final_llm]"
            prompt_chars_before = 0
            prompt_chars_after = 0
            dialogue_context_chars = int(acc.pipeline_state.get("dialogue_context_chars", 0) or 0)

            if self.pipeline.config.llm_mode != "stub":
                # Rebuild memory context
                memory_context = _memory_context_for_payload_mode(
                    acc.pipeline_state["memory_context"],
                    self.final_llm_memory_payload_mode,
                )
                recent_pairs = list(acc.pipeline_state.get("recent_pairs") or [])

                # Call final LLM
                try:
                    predicted_answer = self.pipeline.final_llm.generate(
                        question=acc.question,
                        memory_context=memory_context,
                        recent_pairs=recent_pairs,
                        clock_display=acc.final_llm_clock_display,
                    )
                    char_stats = self.pipeline.final_llm.get_last_prompt_char_stats()
                    prompt_chars_before = int(char_stats.get("before_clamp_chars", 0))
                    prompt_chars_after = int(char_stats.get("after_clamp_chars", 0))
                    self.stats["final_llm_calls"] += 1
                    self.stats["final_llm_prompt_chars_before_clamp_total"] += prompt_chars_before
                    self.stats["final_llm_prompt_chars_after_clamp_total"] += prompt_chars_after
                    logger.info(
                        "[Batch] Final LLM answer for item %d: %s...",
                        acc.global_index, predicted_answer[:100]
                    )
                except Exception as e:
                    logger.error(
                        "[Batch] Final LLM failed for item %d: %s",
                        acc.global_index, e
                    )
                    predicted_answer = f"[error: {e}]"

            # Create accumulated answer
            acc_answer = AccumulatedAnswer(
                global_index=acc.global_index,
                question_id=acc.question_id,
                question=acc.question,
                reference_answer=acc.reference_answer,
                question_type=acc.question_type,
                predicted_answer=predicted_answer,
                memory_context=memory_context if self.pipeline.config.llm_mode != "stub" else acc.pipeline_state["memory_context"],
                dialogue_context_chars=dialogue_context_chars,
                final_llm_prompt_chars_before_clamp=prompt_chars_before,
                final_llm_prompt_chars_after_clamp=prompt_chars_after,
                dataset_ordinal=acc.dataset_ordinal,
                dialogue_row_index=acc.dialogue_row_index,
            )

            self.answer_buffer.append(acc_answer)

            # Save intermediate answer for possible later judge processing
            self.intermediate_answers.append(IntermediateAnswer(
                global_index=acc.global_index,
                question_id=acc.question_id,
                question=acc.question,
                reference_answer=acc.reference_answer,
                question_type=acc.question_type,
                predicted_answer=predicted_answer,
                memory_context=memory_context if self.pipeline.config.llm_mode != "stub" else acc.pipeline_state["memory_context"],
                memory_state_path="",  # Not available in full mode
                dialogue_context_chars=dialogue_context_chars,
                final_llm_prompt_chars_before_clamp=prompt_chars_before,
                final_llm_prompt_chars_after_clamp=prompt_chars_after,
                dataset_ordinal=acc.dataset_ordinal,
                dialogue_row_index=acc.dialogue_row_index,
                question_date=acc.question_date,
            ))

        # Write intermediate answers
        self._write_intermediate_answers()

        # Clear dialogue buffer
        self.dialogue_buffer.clear()

        judge_local = (
            self.judge_client is not None
            and getattr(self.judge_client, "mode", "") == "local"
        )
        if judge_local and self.pipeline is not None and hasattr(self.pipeline, "final_llm"):
            if hasattr(self.pipeline.final_llm, "release_local_serving"):
                logger.info("[Batch] Releasing local final LLM before local judge (save VRAM)")
                self.pipeline.final_llm.release_local_serving()

        defer_slot = (
            judge_local
            and _is_full_dst_pipeline(self.pipeline)
            and self.pipeline.config.llm_mode == "local"
            and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
        )
        self._pending_slot_reload_after_judge = bool(defer_slot)

        if (
            self.pipeline.config.llm_mode == "local"
            and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
        ):
            if defer_slot:
                logger.info(
                    "[Batch] Deferring slot model reload until after local judge "
                    "(avoid slot + final LLM + judge on VRAM at once)"
                )
            else:
                logger.info("[Batch] Reloading models after final LLM processing...")
                self.pipeline.reload_local_models()

        # Check if we should flush judge batch
        if len(self.answer_buffer) >= self.judge_batch_size:
            self._flush_judge_batch()

    def _process_final_llm_only(self):
        """Process final LLM using saved memory states (final_llm_only mode)."""
        if not self.input_state_dir:
            raise ValueError("input_state_dir required for final_llm_only mode")

        input_dir = Path(self.input_state_dir)
        states_path = input_dir / "memory_only_states.json"

        if not states_path.exists():
            raise FileNotFoundError(f"Memory states file not found: {states_path}")

        logger.info("[FinalLLMOnly] Loading memory states from %s", states_path)

        with open(states_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        states = list(data.get("states", []))
        states.sort(key=_validation_sort_key)
        logger.info("[FinalLLMOnly] Loaded %d memory states (sorted by dataset ordinal)", len(states))

        resume_from = getattr(self, "final_llm_resume_from_global_index", None)
        if resume_from is not None:
            rf = int(resume_from)
            before = len(states)
            states = [s for s in states if int(s.get("global_index", -1)) >= rf]
            logger.info(
                "[FinalLLMOnly] Resume from global_index=%d: running LLM for %d/%d states "
                "(indices <%d left unchanged on disk)",
                rf,
                len(states),
                before,
                rf,
            )

        # Group states into batches
        state_batches = [
            states[i:i + self.final_llm_batch_size]
            for i in range(0, len(states), self.final_llm_batch_size)
        ]

        for batch_idx, batch in enumerate(state_batches):
            logger.info("[FinalLLMOnly] Processing batch %d/%d", batch_idx + 1, len(state_batches))

            # Unload models if configured and using local final LLM
            if (
                self.pipeline.config.llm_mode == "local"
                and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
            ):
                logger.info("[FinalLLMOnly] Unloading models before final LLM processing...")
                self.pipeline.unload_local_models()

            for state_data in batch:
                global_index = state_data.get("global_index")
                if global_index is None:
                    logger.warning(
                        "[FinalLLMOnly] Skipping state entry without global_index (question_id=%s)",
                        state_data.get("question_id"),
                    )
                    continue
                question = state_data.get("question", "")
                reference_answer = state_data.get("reference_answer", "")
                question_type = state_data.get("question_type", "")
                question_id = state_data.get("question_id", "")
                pipeline_state = state_data.get("pipeline_state") or {}
                memory_state_path = state_data.get("memory_state_path", "")
                qd = state_data.get("question_date")

                clock = optional_clock_display_for_validation(
                    bool(getattr(self.pipeline.config, "use_dataset_datetime", False)),
                    qd,
                )
                recent_pairs = list(pipeline_state.get("recent_pairs") or [])
                strategy_state_map = (
                    pipeline_state.get("strategy_state_by_strategy")
                    or {}
                )
                legacy_context = pipeline_state.get("memory_context", {}) or {}
                dialogue_context_chars = int(pipeline_state.get("dialogue_context_chars", 0) or 0)

                # Top-k payload is identical across inactive_facts_memory_modes — cache one LLM call per question.
                topk_llm_once: Optional[Tuple[str, int, int]] = None

                for inactive_mode in self.inactive_facts_memory_modes:
                    include_inactive = inactive_mode == "with_inactive"
                    loaded_ctx = self._load_memory_context_from_state(
                        memory_state_path,
                        include_inactive=include_inactive,
                    )
                    if loaded_ctx:
                        full_context = loaded_ctx
                    else:
                        full_context = copy.deepcopy(legacy_context) if legacy_context else {}
                        if include_inactive:
                            logger.warning(
                                "[FinalLLMOnly] No dst_state.json for global_index=%s — legacy snapshot "
                                "may omit inactive facts",
                                global_index,
                            )
                        else:
                            full_context = _narrow_memory_context_active_facts(full_context)

                    for strategy in self.final_llm_memory_strategies:
                        if strategy == "full_graph_json":
                            memory_context = full_context
                        elif strategy == "relevant_slots_full":
                            slot_state = (strategy_state_map.get("relevant_slots_full") or {})
                            slots_raw = slot_state.get("selected_slot_names", [])
                            selected_slots = set()
                            for s in slots_raw:
                                if isinstance(s, dict):
                                    sn = str(s.get("slot") or "").strip()
                                else:
                                    sn = str(s or "").strip()
                                if sn:
                                    selected_slots.add(sn)
                            if selected_slots and isinstance(full_context, dict):
                                memory_context = {
                                    "slots": [
                                        x for x in (full_context.get("slots") or [])
                                        if str((x or {}).get("slot", "")).strip() in selected_slots
                                    ]
                                }
                            else:
                                memory_context = full_context
                        elif strategy == "topk_graph_records":
                            # RAGU top-k from memory_only contains graph retrieval only; DST-deactivated
                            # triplets are not re-injected here — filter inactive defensively.
                            topk_state = (strategy_state_map.get("topk_graph_records") or {})
                            retrieved_raw = list(topk_state.get("topk_records", []) or [])
                            retrieved = _filter_topk_records_for_inactive_mode(retrieved_raw, False)
                            k = int(getattr(self.pipeline.config, "graph_top_k_records", 20) or 20)
                            memory_context = {"records": retrieved[: max(0, k)]}
                        else:
                            memory_context = full_context if isinstance(full_context, dict) else legacy_context
                        memory_context = _memory_context_for_payload_mode(
                            memory_context,
                            self.final_llm_memory_payload_mode,
                        )
                        predicted_answer = "[no_final_llm]"
                        prompt_chars_before = 0
                        prompt_chars_after = 0

                        if self.pipeline.config.llm_mode != "stub":
                            try:
                                reuse_topk = (
                                    strategy == "topk_graph_records" and topk_llm_once is not None
                                )
                                if reuse_topk:
                                    predicted_answer, prompt_chars_before, prompt_chars_after = topk_llm_once
                                    logger.info(
                                        "[FinalLLMOnly] Reusing top-k LLM output item=%d inactive=%s strategy=%s",
                                        global_index,
                                        inactive_mode,
                                        strategy,
                                    )
                                else:
                                    predicted_answer = self.pipeline.final_llm.generate(
                                        question=question,
                                        memory_context=memory_context,
                                        recent_pairs=recent_pairs,
                                        clock_display=clock,
                                    )
                                    char_stats = self.pipeline.final_llm.get_last_prompt_char_stats()
                                    prompt_chars_before = int(char_stats.get("before_clamp_chars", 0))
                                    prompt_chars_after = int(char_stats.get("after_clamp_chars", 0))
                                    self.stats["final_llm_calls"] += 1
                                    self.stats["final_llm_prompt_chars_before_clamp_total"] += prompt_chars_before
                                    self.stats["final_llm_prompt_chars_after_clamp_total"] += prompt_chars_after
                                    if strategy == "topk_graph_records":
                                        topk_llm_once = (
                                            predicted_answer,
                                            prompt_chars_before,
                                            prompt_chars_after,
                                        )
                                logger.info(
                                    "[FinalLLMOnly] Final LLM answer for item %d inactive=%s strategy=%s: %s...",
                                    global_index,
                                    inactive_mode,
                                    strategy,
                                    predicted_answer[:100],
                                )
                            except Exception as e:
                                logger.error(
                                    "[FinalLLMOnly] Final LLM failed for item %d inactive=%s strategy=%s: %s",
                                    global_index,
                                    inactive_mode,
                                    strategy,
                                    e,
                                )
                                predicted_answer = f"[error: {e}]"
                                self.stats["errors_final_llm"] += 1

                        intermediate = IntermediateAnswer(
                            global_index=global_index,
                            question_id=question_id,
                            question=question,
                            reference_answer=reference_answer,
                            question_type=question_type,
                            predicted_answer=predicted_answer,
                            memory_context=memory_context,
                            memory_strategy=strategy,
                            inactive_facts_mode=inactive_mode,
                            memory_state_path=memory_state_path,
                            dialogue_context_chars=dialogue_context_chars,
                            final_llm_prompt_chars_before_clamp=prompt_chars_before,
                            final_llm_prompt_chars_after_clamp=prompt_chars_after,
                            dataset_ordinal=state_data.get("_validation_dataset_ordinal"),
                            dialogue_row_index=state_data.get("_validation_dialogue_row_index"),
                            question_date=qd,
                        )
                        self.intermediate_answers.append(intermediate)

                        acc_answer = AccumulatedAnswer(
                            global_index=global_index,
                            question_id=question_id,
                            question=question,
                            reference_answer=reference_answer,
                            question_type=question_type,
                            predicted_answer=predicted_answer,
                            memory_context=memory_context,
                            memory_strategy=strategy,
                            dialogue_context_chars=dialogue_context_chars,
                            final_llm_prompt_chars_before_clamp=prompt_chars_before,
                            final_llm_prompt_chars_after_clamp=prompt_chars_after,
                            dataset_ordinal=state_data.get("_validation_dataset_ordinal"),
                            dialogue_row_index=state_data.get("_validation_dialogue_row_index"),
                        )
                        self.answer_buffer.append(acc_answer)

            # Write intermediate answers after each batch
            self._write_intermediate_answers()

            judge_local = (
                self.judge_client is not None
                and getattr(self.judge_client, "mode", "") == "local"
            )
            if judge_local and hasattr(self.pipeline, "final_llm"):
                if hasattr(self.pipeline.final_llm, "release_local_serving"):
                    logger.info("[FinalLLMOnly] Releasing local final LLM before judge")
                    self.pipeline.final_llm.release_local_serving()

            if (
                self.pipeline.config.llm_mode == "local"
                and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
                and not judge_local
            ):
                logger.info("[FinalLLMOnly] Reloading models after final LLM processing...")
                self.pipeline.reload_local_models()

            # Check if we should flush judge batch
            if len(self.answer_buffer) >= self.judge_batch_size:
                self._flush_judge_batch()

    def _write_intermediate_answers(self) -> None:
        """Write intermediate answers to disk for later judge processing.

        With ``final_llm_resume_from_global_index`` set (final_llm_only), merges each existing
        ``intermediate_answers.json`` (same strategy × inactive subdirectory) with this run:
        rows with ``global_index`` below the threshold are kept from disk; rows ``>=`` threshold
        are replaced/appended from ``self.intermediate_answers``. Per-file ``statistics`` are
        recomputed from the merged ``answers`` list.
        """
        grouped: Dict[Tuple[str, str], List[IntermediateAnswer]] = {}
        for ans in self.intermediate_answers:
            strategy = ans.memory_strategy or "full_graph_json"
            inactive_tag = getattr(ans, "inactive_facts_mode", "") or "active_only"
            grouped.setdefault((strategy, inactive_tag), []).append(ans)

        layout_subdirs = _inactive_facts_use_strategy_subdir(self.inactive_facts_memory_modes)
        resume_raw = getattr(self, "final_llm_resume_from_global_index", None)
        resume_from_int: Optional[int] = int(resume_raw) if resume_raw is not None else None

        expected_groups: List[Tuple[str, str]] = [
            (strategy, inactive_tag)
            for strategy in self.final_llm_memory_strategies
            for inactive_tag in self.inactive_facts_memory_modes
        ]
        keys_to_write: List[Tuple[str, str]] = (
            expected_groups if resume_from_int is not None else list(grouped.keys())
        )

        agg_calls = 0
        agg_pb = 0
        agg_pa = 0
        agg_dcc = 0

        for strategy, inactive_tag in keys_to_write:
            answers = grouped.get((strategy, inactive_tag), [])
            strat_root = self.persistence.output_dir / strategy
            out_dir = strat_root / inactive_tag if layout_subdirs else strat_root
            output_path = out_dir / "intermediate_answers.json"

            new_rows = [
                _intermediate_answer_to_json_row(ans, strategy, inactive_tag)
                for ans in answers
            ]
            prev_blob: Optional[Dict[str, Any]] = None
            if resume_from_int is not None:
                prev_blob = _load_json_optional(output_path)
                prev_rows = list((prev_blob or {}).get("answers") or []) if prev_blob else []
                merged_rows = _merge_intermediate_answer_rows_for_resume(
                    prev_rows, new_rows, resume_from_int
                )
            else:
                merged_rows = sorted(
                    new_rows,
                    key=lambda x: (int(x.get("global_index", 0)), str(x.get("question_id", ""))),
                )

            if resume_from_int is None:
                meta_out: Dict[str, Any] = {
                    **(self._validation_metadata or {}),
                    "memory_strategy": strategy,
                    "inactive_facts_mode": inactive_tag,
                }
            else:
                prev_meta: Dict[str, Any] = {}
                if isinstance(prev_blob, dict):
                    pm = prev_blob.get("metadata")
                    if isinstance(pm, dict):
                        prev_meta = dict(pm)
                meta_out = {
                    **prev_meta,
                    **(self._validation_metadata or {}),
                    "memory_strategy": strategy,
                    "inactive_facts_mode": inactive_tag,
                    "final_llm_resume_from_global_index": resume_from_int,
                }

            prev_timing: Dict[str, Any] = {}
            if resume_from_int is not None and isinstance(prev_blob, dict):
                pt = prev_blob.get("timing")
                if isinstance(pt, dict):
                    prev_timing = dict(pt)

            stats_out = _stats_from_intermediate_answer_rows(merged_rows)
            agg_calls += int(stats_out.get("final_llm_calls_total", 0) or 0)
            agg_pb += int(stats_out.get("final_llm_prompt_chars_before_clamp_total", 0) or 0)
            agg_pa += int(stats_out.get("final_llm_prompt_chars_after_clamp_total", 0) or 0)
            agg_dcc += int(stats_out.get("dialogue_context_chars_total", 0) or 0)

            payload: Dict[str, Any] = {
                "metadata": meta_out,
                "statistics": stats_out,
                "answers": merged_rows,
            }
            if resume_from_int is not None and prev_timing:
                payload["timing"] = prev_timing

            _atomic_write_validation_json(output_path, payload)

        # Align aggregate counters with merged files (sum across strategy × inactive outputs).
        if self.validation_mode == "final_llm_only" and keys_to_write:
            self.stats["final_llm_calls"] = agg_calls
            self.stats["final_llm_prompt_chars_before_clamp_total"] = agg_pb
            self.stats["final_llm_prompt_chars_after_clamp_total"] = agg_pa
            self.stats["dialogue_context_chars_total"] = agg_dcc
            self.stats["dialogue_context_chars_count"] = agg_calls

    def _flush_judge_batch(self):
        """Process accumulated answers through judge."""
        # Handle judge_only mode
        if self.validation_mode == "judge_only":
            self._process_judge_only()
            return

        # Skip if no answers or no judge client
        if not self.answer_buffer or not self.judge_client:
            self.answer_buffer.clear()
            return

        logger.info(
            "[Batch] Flushing judge batch: %d answers",
            len(self.answer_buffer)
        )

        for acc in self.answer_buffer:
            self._evaluate_single_answer(acc)

        # Clear answer buffer
        self.answer_buffer.clear()

        if self.judge_client is not None and getattr(self.judge_client, "mode", "") == "local":
            self.judge_client.unload()

        if self._pending_slot_reload_after_judge and self.pipeline is not None:
            self._pending_slot_reload_after_judge = False
            if (
                getattr(self.pipeline.config, "llm_mode", "") == "local"
                and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
            ):
                logger.info("[Batch] Reloading slot models after local judge batch")
                self.pipeline.reload_local_models()

    def _judge_one_answer_correctness(self, acc: AccumulatedAnswer) -> Tuple[float, str, Optional[str]]:
        """Exactly one judge LLM call: score predicted answer vs reference."""
        return self.judge_client.evaluate_answer(
            question=acc.question,
            predicted_answer=acc.predicted_answer,
            reference_answer=acc.reference_answer,
            question_type=acc.question_type,
        )

    def _judge_one_memory_presence(
        self,
        acc: AccumulatedAnswer,
        memory_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Exactly one judge LLM call: is reference fact present in memory_context (no predicted answer in prompt)."""
        return self.judge_client.evaluate_memory_hit(
            question=acc.question,
            reference_answer=acc.reference_answer,
            memory_context=memory_context,
        )

    def _evaluate_single_answer(self, acc: AccumulatedAnswer, memory_state_path: str = ""):
        """Evaluate one item: answer correctness is always one LLM call; MHR is a second separate LLM call when enabled."""
        n_calls = 2 if self.calculate_memory_hit_rate else 1
        logger.info(
            "[Judge] global_index=%d question_id=%s — LLM call 1/%d: answer correctness (predicted vs reference)",
            acc.global_index,
            acc.question_id,
            n_calls,
        )
        score, reasoning, judge_error = self._judge_one_answer_correctness(acc)

        memory_hit_result = None
        if self.calculate_memory_hit_rate:
            memory_context = acc.memory_context
            if memory_state_path and not memory_context:
                memory_context = self._load_memory_context_from_state(memory_state_path)

            logger.info(
                "[Judge] global_index=%d question_id=%s — LLM call 2/2: memory presence (reference fact in context; no predicted answer in prompt)",
                acc.global_index,
                acc.question_id,
            )
            memory_hit_result = self._judge_one_memory_presence(acc, memory_context)

        # Build result
        result = {
            "global_index": acc.global_index,
            "question_id": acc.question_id,
            "question": acc.question,
            "reference_answer": acc.reference_answer,
            "predicted_answer": acc.predicted_answer,
            "question_type": acc.question_type,
            "memory_strategy": acc.memory_strategy,
            "dialogue_context_chars": int(acc.dialogue_context_chars or 0),
            "score": score,
            "reasoning": reasoning,
            "judge_error": judge_error,
            "final_llm_prompt_chars_before_clamp": acc.final_llm_prompt_chars_before_clamp,
            "final_llm_prompt_chars_after_clamp": acc.final_llm_prompt_chars_after_clamp,
            "memory_hit_evaluation": memory_hit_result,
            "memory_hit": memory_hit_result.get("fact_present", False) if memory_hit_result else None,
        }
        if acc.inactive_facts_mode:
            result["inactive_facts_mode"] = acc.inactive_facts_mode
        if acc.judge_intermediate_source:
            result["judge_intermediate_source"] = acc.judge_intermediate_source
        if acc.dataset_ordinal is not None:
            result["_validation_dataset_ordinal"] = acc.dataset_ordinal
        if acc.dialogue_row_index is not None:
            result["_validation_dialogue_row_index"] = acc.dialogue_row_index

        self.results.append(result)

        # Update stats
        self.stats["total"] += 1
        self.stats["total_score"] += score
        self.stats["dialogue_context_chars_total"] += int(acc.dialogue_context_chars or 0)
        self.stats["dialogue_context_chars_count"] += 1
        if judge_error:
            self.stats["errors_judge"] += 1

        # Per-type stats
        qt = acc.question_type
        if qt in self.stats["by_type"]:
            self.stats["by_type"][qt]["count"] += 1
            self.stats["by_type"][qt]["total_score"] += score
            if judge_error:
                self.stats["by_type"][qt]["errors"] += 1

        if memory_hit_result is not None:
            if memory_hit_result.get("fact_present", False):
                self.stats["memory_hit"] += 1
            else:
                self.stats["memory_miss"] += 1
            if qt in self.stats["by_type"]:
                self.stats["by_type"][qt]["mhe_count"] += 1
                if memory_hit_result.get("fact_present", False):
                    self.stats["by_type"][qt]["mhe_hits"] += 1
                else:
                    self.stats["by_type"][qt]["mhe_misses"] += 1

        shard_out = getattr(self, "_judge_incremental_results_json_path", None)
        shard_meta = getattr(self, "_judge_shard_metadata_extra", None)
        self._write_results_json_snapshot(
            results_json_path=shard_out if shard_out is not None else None,
            metadata_extra=dict(shard_meta) if shard_meta else None,
        )

    def _load_memory_context_from_state(
        self,
        memory_state_path: str,
        *,
        include_inactive: bool = False,
    ) -> Dict[str, Any]:
        """Load memory context from saved DST state file (``dst_state.json`` slots)."""
        if not str(memory_state_path or "").strip():
            return {}
        try:
            state_file = Path(memory_state_path)
            if not state_file.exists():
                # Try with parent directory
                state_file = Path(memory_state_path) / "dst_state.json"
            if not state_file.exists():
                return {}

            with open(state_file, "r", encoding="utf-8") as f:
                state = json.load(f)

            # Reconstruct memory context from slots
            slots = state.get("slots", {})
            memory_context = {"slots": []}
            for slot_name, records in slots.items():
                messages = []
                for r in records:
                    if not include_inactive and not r.get("is_active", True):
                        continue
                    val = r.get("value")
                    messages.append({
                        "record_id": r.get("record_id"),
                        "message_text": val,
                        "value": val,
                        "source_text": r.get("source_text"),
                        "subject": r.get("subject"),
                        "relation": r.get("relation"),
                        "object": r.get("object"),
                        "created_at_datetime": str(r.get("created_at_datetime") or "").strip(),
                        "is_active": r.get("is_active", True),
                    })
                slot_data = {"slot": slot_name, "slot_label": slot_name, "messages": messages}
                if slot_data["messages"]:
                    memory_context["slots"].append(slot_data)

            return memory_context
        except Exception as e:
            logger.warning("Failed to load memory context from state: %s", e)
            return {}

    def _load_memory_states_map(self, state_dir: Path) -> Dict[int, Dict[str, Any]]:
        """Load memory states map from state directory for judge_only mode.
        
        Returns:
            Dict mapping global_index to memory context
        """
        states_map = {}
        states_file = Path(state_dir) / "memory_only_states.json"
        
        if not states_file.exists():
            logger.warning("[JudgeOnly] memory_only_states.json not found in %s", state_dir)
            return states_map
        
        try:
            with open(states_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for state in data.get("states", []):
                global_index = state.get("global_index")
                if global_index is not None:
                    # Use pipeline_state memory_context if available
                    pipeline_state = state.get("pipeline_state", {})
                    memory_context = pipeline_state.get("memory_context", {})
                    
                    # Or try to load from dst_state if memory_context is empty
                    if not memory_context:
                        memory_state_path = state.get("memory_state_path", "")
                        if memory_state_path:
                            memory_context = self._load_memory_context_from_state(memory_state_path)
                    
                    states_map[global_index] = {
                        "memory_context": memory_context,
                        "memory_state_path": state.get("memory_state_path", ""),
                        "dialogue_context_chars": int(
                            state.get("dialogue_context_chars")
                            or (pipeline_state.get("dialogue_context_chars", 0) if isinstance(pipeline_state, dict) else 0)
                            or 0
                        ),
                    }
        except Exception as e:
            logger.warning("[JudgeOnly] Failed to load memory states map: %s", e)
        
        return states_map

    def _process_judge_only(self):
        """Process judge evaluation using saved answers (judge_only mode).

        When ``len(input_answers_paths) > 1``, writes one ``validation_results.json`` per input file under
        ``persistence.output_dir/<strategy>/<inactive_tag>/`` (mirroring intermediate paths). Otherwise one file at
        ``output_dir/validation_results.json`` (legacy).

        After each judged answer, the corresponding ``validation_results.json`` is rewritten (resume-safe), same idea as
        ``final_llm_only`` rewriting ``intermediate_answers.json`` per question.
        """
        paths = list(self.input_answers_paths)
        if not paths:
            raise ValueError("input_answers_path / input_answers_paths required for judge_only mode")

        multi_shard = len(paths) > 1
        self._judge_only_total_rows = 0
        self._judge_incremental_results_json_path = None
        self._judge_shard_metadata_extra = None

        memory_states_map: Dict[int, Dict[str, Any]] = {}
        if self.input_state_dir and self.calculate_memory_hit_rate:
            logger.info("[JudgeOnly] Loading memory states from %s", self.input_state_dir)
            memory_states_map = self._load_memory_states_map(self.input_state_dir)
            logger.info("[JudgeOnly] Loaded %d memory states", len(memory_states_map))

        try:
            for pi, raw_path in enumerate(paths):
                input_path = Path(raw_path)
                if not input_path.exists():
                    raise FileNotFoundError(f"Intermediate answers file not found: {input_path}")

                if multi_shard:
                    self.results.clear()
                    self.stats = self._fresh_judge_stats()

                src_label = str(input_path).replace("\\", "/")
                shard_results_path = _validation_results_path_for_judge_shard(
                    input_path, self.persistence.output_dir
                )
                if multi_shard:
                    self._judge_incremental_results_json_path = shard_results_path
                    self._judge_shard_metadata_extra = {
                        "judge_shard_intermediate_answers": src_label,
                        "judge_shard_validation_results": str(shard_results_path).replace("\\", "/"),
                        "judge_multi_shard": True,
                    }
                else:
                    self._judge_incremental_results_json_path = None
                    self._judge_shard_metadata_extra = None

                logger.info(
                    "[JudgeOnly] (%d/%d) Loading intermediate answers from %s",
                    pi + 1,
                    len(paths),
                    input_path,
                )

                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                answers = list(data.get("answers", []))
                answers.sort(key=_validation_sort_key)
                self._judge_only_total_rows += len(answers)
                logger.info(
                    "[JudgeOnly] Loaded %d answers from this file (sorted by dataset ordinal)",
                    len(answers),
                )

                if not answers:
                    if multi_shard:
                        self._finalize_judge_by_type_averages()
                        self._write_results_json_snapshot(
                            results_json_path=shard_results_path,
                            metadata_extra=dict(self._judge_shard_metadata_extra or {}),
                        )
                        logger.info("[JudgeOnly] Wrote empty shard results: %s", shard_results_path)
                    continue

                answer_batches = [
                    answers[i:i + self.judge_batch_size]
                    for i in range(0, len(answers), self.judge_batch_size)
                ]

                for batch_idx, batch in enumerate(answer_batches):
                    logger.info(
                        "[JudgeOnly] File %d/%d batch %d/%d",
                        pi + 1,
                        len(paths),
                        batch_idx + 1,
                        len(answer_batches),
                    )

                    for ans_data in batch:
                        global_index = ans_data["global_index"]

                        memory_context = ans_data.get("memory_context") or {}
                        memory_state_path = ans_data.get("memory_state_path", "")
                        dialogue_context_chars = int(ans_data.get("dialogue_context_chars", 0) or 0)

                        if self.input_state_dir and self.calculate_memory_hit_rate:
                            map_data = memory_states_map.get(global_index)
                            if map_data:
                                if not memory_context:
                                    memory_context = map_data.get("memory_context", {})
                                    logger.info(
                                        "[JudgeOnly] memory_context empty in answers; filled from "
                                        "input_state_dir for flat index %d",
                                        global_index,
                                    )
                                if not memory_state_path:
                                    memory_state_path = (
                                        map_data.get("memory_state_path", "") or memory_state_path
                                    )
                                if not dialogue_context_chars:
                                    dialogue_context_chars = int(
                                        map_data.get("dialogue_context_chars", 0) or 0
                                    )

                        acc = AccumulatedAnswer(
                            global_index=global_index,
                            question_id=ans_data["question_id"],
                            question=ans_data["question"],
                            reference_answer=ans_data["reference_answer"],
                            question_type=ans_data.get("question_type", ""),
                            predicted_answer=ans_data["predicted_answer"],
                            memory_context=memory_context,
                            memory_strategy=str(ans_data.get("memory_strategy", "") or ""),
                            inactive_facts_mode=str(ans_data.get("inactive_facts_mode", "") or ""),
                            judge_intermediate_source=src_label,
                            dialogue_context_chars=dialogue_context_chars,
                            final_llm_prompt_chars_before_clamp=int(
                                ans_data.get("final_llm_prompt_chars_before_clamp", 0) or 0
                            ),
                            final_llm_prompt_chars_after_clamp=int(
                                ans_data.get("final_llm_prompt_chars_after_clamp", 0) or 0
                            ),
                            dataset_ordinal=ans_data.get("_validation_dataset_ordinal"),
                            dialogue_row_index=ans_data.get("_validation_dialogue_row_index"),
                        )
                        self._evaluate_single_answer(acc, memory_state_path)

                if multi_shard:
                    logger.info(
                        "[JudgeOnly] Shard complete (incremental snapshots): %s",
                        shard_results_path,
                    )
        finally:
            self._judge_incremental_results_json_path = None
            self._judge_shard_metadata_extra = None

        self.answer_buffer.clear()

        if self.judge_client is not None and getattr(self.judge_client, "mode", "") == "local":
            self.judge_client.unload()

    def _write_results_json_snapshot(
        self,
        results_json_path: Optional[Path] = None,
        metadata_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        target = results_json_path if results_json_path is not None else self._results_json_path
        if (
            target is None
            or self._validation_metadata is None
            or self._timing is None
        ):
            return
        stats_export = build_judge_statistics_export(
            self.stats,
            self.calculate_memory_hit_rate,
        )
        timing_stats = self._timing.get_stats()
        if timing_stats.get("total_time", 0) == 0 and self._timing.total_start is not None:
            timing_stats = dict(timing_stats)
            timing_stats["total_time"] = time.time() - self._timing.total_start
        meta = dict(self._validation_metadata)
        if metadata_extra:
            meta.update(metadata_extra)
        payload = {
            "metadata": meta,
            "statistics": stats_export,
            "timing": timing_stats,
            "results": list(self.results),
        }
        _atomic_write_validation_json(target, payload)

    def _write_giga_memory_dialogue_logs(self) -> None:
        """Write verbose per-dialogue logs (DST_memory pipeline test *_logs.json style)."""
        if not self.save_giga_memory_logs:
            return
        path = self.persistence.output_dir / "giga_memory_validation_logs.json"
        replaced_rows = self._memory_only_active_row_indices() if self.validation_mode == "memory_only" else set()
        prev_blob = _load_json_optional(path)
        prev_dialogues = list((prev_blob or {}).get("dialogues") or []) if prev_blob else []
        if self.validation_mode == "memory_only" and replaced_rows:
            kept = [
                d for d in prev_dialogues
                if int(d.get("dialogue_row_index", -1)) not in replaced_rows
            ]
            merged_dialogues = kept + list(self.dialogue_logs)
        else:
            merged_dialogues = list(self.dialogue_logs) if self.dialogue_logs else prev_dialogues
        if not merged_dialogues:
            return
        merged_dialogues.sort(
            key=lambda d: (
                int(d.get("dialogue_row_index", 0)),
                int(d.get("question_index_in_row", 0)),
                int(d.get("global_index", 0)),
            ),
        )
        _atomic_write_validation_json(
            path,
            {
                "metadata": self._validation_metadata,
                "note": (
                    "Verbose per-dialogue logs: write_logs + answer_without_final_llm "
                    "(same information as DST_memory pipeline test *_logs.json)."
                ),
                "dialogues": merged_dialogues,
            },
        )
        logger.info("Saved GigaMemory validation logs: %s", path)

    def finalize(self) -> Tuple[List[Dict], Dict]:
        """Finalize processing - flush any remaining items."""
        logger.info("[Batch] Finalizing - flushing remaining buffers...")

        # Handle mode-specific finalization
        if self.validation_mode == "memory_only":
            # Write final memory states
            self._write_memory_only_states()
            self._write_giga_memory_dialogue_logs()
            logger.info("[MemoryOnly] Saved %d memory states", len(self.memory_only_states))
            return self.results, self.stats

        if self.validation_mode == "final_llm_only":
            self._process_final_llm_only()
            self._write_intermediate_answers()
            logger.info("[FinalLLMOnly] Generated %d new answer row(s) this run", len(self.intermediate_answers))
            # If judge is configured, flush remaining answers
            if self.answer_buffer:
                self._flush_judge_batch()
            return self.results, self.stats

        if self.validation_mode == "judge_only":
            if not self.results:
                self._process_judge_only()
            multi_shard = len(self.input_answers_paths) > 1
            if multi_shard:
                logger.info(
                    "[JudgeOnly] Sharded outputs: %d file(s), %d judge rows total → "
                    "validation_results.json under each %s/<strategy>/<inactive_tag>/",
                    len(self.input_answers_paths),
                    self._judge_only_total_rows,
                    self.persistence.output_dir,
                )
                self.results.clear()
                self.stats = self._fresh_judge_stats()
                return self.results, self.stats
            for qt in self.stats["by_type"]:
                count = self.stats["by_type"][qt]["count"]
                if count > 0:
                    self.stats["by_type"][qt]["average_score"] = (
                        self.stats["by_type"][qt]["total_score"] / count
                    )
                mc = self.stats["by_type"][qt].get("mhe_count", 0)
                if mc > 0:
                    self.stats["by_type"][qt]["mhe_hit_rate"] = (
                        self.stats["by_type"][qt]["mhe_hits"] / mc
                    )
            self._write_results_json_snapshot()
            logger.info("[JudgeOnly] Evaluated %d answers", len(self.results))
            return self.results, self.stats

        # Full mode: flush any remaining dialogues and answers
        # Flush any remaining dialogues
        if self.dialogue_buffer:
            self._flush_final_llm_batch()

        # Flush any remaining answers
        if self.answer_buffer:
            self._flush_judge_batch()

        # Compute per-type averages
        for qt in self.stats["by_type"]:
            count = self.stats["by_type"][qt]["count"]
            if count > 0:
                self.stats["by_type"][qt]["average_score"] = (
                    self.stats["by_type"][qt]["total_score"] / count
                )
            mc = self.stats["by_type"][qt].get("mhe_count", 0)
            if mc > 0:
                self.stats["by_type"][qt]["mhe_hit_rate"] = (
                    self.stats["by_type"][qt]["mhe_hits"] / mc
                )

        self._write_results_json_snapshot()
        self._write_giga_memory_dialogue_logs()

        if getattr(self, "_pending_slot_reload_after_judge", False) and self.pipeline is not None:
            self._pending_slot_reload_after_judge = False
            if (
                getattr(self.pipeline.config, "llm_mode", "") == "local"
                and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
            ):
                logger.info("[Batch] Finalize: restoring slot models after deferred judge reload")
                self.pipeline.reload_local_models()

        return self.results, self.stats


# ============================================================================
# Main Validation Logic
# ============================================================================

def run_validation(args: argparse.Namespace) -> None:
    """Main validation entry point with batch processing."""
    # Setup logging
    log_file_path = Path(args.output_dir) / "validation.log" if args.log_file else None
    setup_logging(args.log_level, str(log_file_path) if log_file_path else None)

    # Initialize timing
    timing = TimingStats()
    timing.start_total()

    logger.info("=" * 70)
    logger.info("LongMemEval Validation v3 - Starting")
    logger.info("Validation mode: %s", args.validation_mode)
    if args.validation_mode == "final_llm_only":
        logger.info(
            "final_llm_only: this run does not execute write_to_memory — no triplet/slot/importance "
            "work in this process; only the final LLM is loaded (see FinalLLMOnlyPipelineFacade in logs)."
        )
    logger.info("=" * 70)
    logger.info("Configuration:")
    logger.info("  Dataset: %s", args.dataset_path)
    logger.info("  Output: %s", args.output_dir)
    logger.info("  Items per type: %d", args.num_items_per_type)
    logger.info("  Question types: %s", args.question_types)
    logger.info("  Final LLM batch size: %d", args.final_llm_batch_size)
    logger.info("  Judge batch size: %d", args.judge_batch_size)
    logger.info("  Calculate memory hit rate: %s", args.calculate_memory_hit_rate)
    logger.info("  Judge mode: %s", args.judge_mode)
    logger.info("  GigaMemory config: %s", args.config)
    logger.info("  final_llm_memory_strategies: %s", getattr(args, "final_llm_memory_strategies", []))
    logger.info("  memory_only_write_mode: %s", getattr(args, "memory_only_write_mode", "standard"))
    logger.info(
        "  final_llm_memory_payload_mode: %s",
        getattr(args, "final_llm_memory_payload_mode", "with_metadata"),
    )
    logger.info(
        "  inactive_facts_memory_modes: %s",
        getattr(args, "inactive_facts_memory_modes", ["active_only"]),
    )
    _iap_show = _normalize_input_answers_paths(getattr(args, "input_answers_paths", None))
    logger.info(
        "  input_answers_paths (%d): %s",
        len(_iap_show),
        _iap_show if len(_iap_show) <= 6 else _iap_show[:6] + ["..."],
    )
    logger.info(
        "  final_llm_resume_from_global_index: %s",
        getattr(args, "final_llm_resume_from_global_index", None),
    )
    logger.info(
        "  use_dataset_datetime: %s",
        getattr(args, "gm_use_dataset_datetime", None),
    )
    logger.info(
        "  force_infinite_ttl: %s",
        getattr(args, "gm_force_infinite_ttl", None),
    )
    ri_f, id_f = memory_only_row_filter_sets(args)
    if ri_f is not None or id_f is not None:
        logger.info("  memory_only row filter: indices=%s dialogue_ids=%s", ri_f, id_f)

    # Mode-specific validation
    if args.validation_mode == "final_llm_only" and not args.input_state_dir:
        logger.error("--input-state-dir is required for final_llm_only mode")
        sys.exit(1)
    if args.validation_mode == "judge_only":
        _jpaths = _normalize_input_answers_paths(getattr(args, "input_answers_paths", None))
        _single = (getattr(args, "input_answers_path", None) or "").strip()
        if not _jpaths and not _single:
            logger.error(
                "judge_only requires input_answers_path or non-empty input_answers_paths in config / CLI"
            )
            sys.exit(1)

    _rf_resume = getattr(args, "final_llm_resume_from_global_index", None)
    if _rf_resume is not None:
        if int(_rf_resume) < 0:
            logger.error(
                "final_llm_resume_from_global_index must be >= 0, got %s",
                _rf_resume,
            )
            sys.exit(1)
        if args.validation_mode != "final_llm_only":
            logger.warning(
                "final_llm_resume_from_global_index=%s applies only to validation_mode "
                "final_llm_only — ignoring for mode %s",
                _rf_resume,
                args.validation_mode,
            )

    # Load dataset with balanced sampling (for modes that need it)
    dataset = []
    if args.validation_mode in ("full", "memory_only"):
        dataset = load_dataset_balanced(
            args.dataset_path,
            args.question_types,
            args.num_items_per_type,
        )
        logger.info("Total items to process: %d", len(dataset))

    run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    results_path = Path(args.output_dir) / "validation_results.json"
    validation_metadata = {
        "validation_mode": args.validation_mode,
        "dataset_path": args.dataset_path,
        "num_items_per_type": args.num_items_per_type,
        "question_types": args.question_types,
        "final_llm_batch_size": args.final_llm_batch_size,
        "judge_batch_size": args.judge_batch_size,
        "calculate_memory_hit_rate": args.calculate_memory_hit_rate,
        "judge_mode": args.judge_mode,
        "judge_model": args.judge_model,
        "config_path": args.config,
        "timestamp": run_timestamp,
        "input_state_dir": args.input_state_dir or "",
        "input_answers_path": args.input_answers_path or "",
        "memory_only_write_mode": str(getattr(args, "memory_only_write_mode", "standard")),
        "final_llm_memory_strategies": list(getattr(args, "final_llm_memory_strategies", []) or []),
        "final_llm_memory_payload_mode": str(
            getattr(args, "final_llm_memory_payload_mode", "with_metadata")
        ),
        "memory_only_dialogue_row_indices": list(
            getattr(args, "memory_only_dialogue_row_indices", []) or []
        ),
        "memory_only_dialogue_ids": list(getattr(args, "memory_only_dialogue_ids", []) or []),
        "inactive_facts_memory_modes": list(getattr(args, "inactive_facts_memory_modes", []) or []),
        "final_llm_resume_from_global_index": getattr(
            args, "final_llm_resume_from_global_index", None
        ),
        "input_answers_paths": _normalize_input_answers_paths(getattr(args, "input_answers_paths", None)),
    }

    # Initialize persistence
    persistence = MemoryStatePersistence(args.output_dir)

    # Initialize judge client (only for modes that need it)
    judge_client = None
    if args.judge_mode != "none" and args.validation_mode in ("full", "judge_only"):
        judge_client = JudgeClient(
            mode=args.judge_mode,
            model=args.judge_model,
            api_url=args.judge_api_url,
            api_key=args.judge_api_key,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            max_context_tokens=getattr(args, "judge_max_context_tokens", 128 * 1024),
            tokenizer_model=getattr(args, "judge_tokenizer_model", ""),
            local_model_path=(args.judge_local_model_path or args.judge_model or "").strip() or None,
            enable_thinking=getattr(args, "judge_enable_thinking", True),
            load_dtype=getattr(args, "judge_load_dtype", "float16"),
            load_quantization=getattr(args, "judge_load_quantization", "none"),
            openrouter_reasoning=getattr(args, "judge_openrouter_reasoning", None),
        )

    # Build pipeline (full DST for memory phases; lightweight for final_llm_only)
    pipeline = None
    if args.validation_mode in ("full", "memory_only", "final_llm_only"):
        cli_overrides = dict(getattr(args, "_validation_shared_pipeline_overrides", {}) or {})
        cli_overrides.update(build_cli_overrides(args))
        cli_overrides["final_llm_memory_payload_mode"] = _normalize_memory_payload_mode(
            getattr(args, "final_llm_memory_payload_mode", "with_metadata")
        )
        if cli_overrides:
            logger.info(
                "  DST pipeline overrides (validation shared + giga_memory/--gm-*): %s",
                cli_overrides,
            )
        if args.validation_mode == "final_llm_only":
            pipeline = build_final_llm_only_facade(args.config, cli_overrides)
        else:
            pipeline = build_pipeline_from_config(args.config, cli_overrides)
            if (
                args.validation_mode == "memory_only"
                and str(getattr(args, "memory_only_write_mode", "standard")) == "single_path_only"
            ):
                _enable_memory_only_single_path_mode(pipeline)
        validation_metadata["giga_memory_memory_strategy"] = getattr(
            pipeline.config, "memory_strategy", ""
        )
        validation_metadata["giga_memory_llm_mode"] = getattr(pipeline.config, "llm_mode", "")
        if args.validation_mode == "final_llm_only":
            validation_metadata["pipeline_backend"] = "FinalLLMOnlyPipelineFacade"
        else:
            validation_metadata["pipeline_backend"] = "DSTMemoryPipeline"

        if pipeline is not None:
            cfg = pipeline.config
            logger.info(
                "  Effective DST conflict_rule_same_relation_updates: %s",
                getattr(cfg, "conflict_rule_same_relation_updates", None),
            )
            logger.info(
                "  Effective DST conflict_allow_multi_relation_same_object: %s",
                getattr(cfg, "conflict_allow_multi_relation_same_object", None),
            )
            logger.info(
                "  Effective DST slot_model_enable_thinking / inject_no_think / lm_format_enforcer: %s / %s / %s",
                getattr(cfg, "slot_model_enable_thinking", None),
                getattr(cfg, "slot_llm_inject_no_think_prompt", None),
                getattr(cfg, "slot_llm_lm_format_enforcer", None),
            )

    # Initialize batch processor
    batch_processor = BatchProcessor(
        pipeline=pipeline,
        judge_client=judge_client,
        final_llm_batch_size=args.final_llm_batch_size,
        judge_batch_size=args.judge_batch_size,
        calculate_memory_hit_rate=args.calculate_memory_hit_rate,
        persistence=persistence,
        results_json_path=results_path,
        validation_metadata=validation_metadata,
        timing=timing,
        validation_mode=args.validation_mode,
        input_state_dir=Path(args.input_state_dir) if args.input_state_dir else None,
        input_answers_path=(
            Path((getattr(args, "input_answers_path", None) or "").strip())
            if (getattr(args, "input_answers_path", None) or "").strip()
            else None
        ),
        input_answers_paths=_normalize_input_answers_paths(getattr(args, "input_answers_paths", None)),
        final_llm_memory_strategies=list(getattr(args, "final_llm_memory_strategies", []) or []),
        final_llm_memory_payload_mode=str(
            getattr(args, "final_llm_memory_payload_mode", "with_metadata")
        ),
        inactive_facts_memory_modes=list(getattr(args, "inactive_facts_memory_modes", []) or []),
        final_llm_resume_from_global_index=(
            getattr(args, "final_llm_resume_from_global_index", None)
            if args.validation_mode == "final_llm_only"
            else None
        ),
        save_giga_memory_logs=bool(getattr(args, "save_intermediate", True)),
    )

    # Process each item (only for modes that need dataset processing)
    if args.validation_mode in ("full", "memory_only"):
        row_filter_idx, row_filter_ids = memory_only_row_filter_sets(args)
        flat_q = 0
        for idx, item in enumerate(dataset):
            q_specs = normalize_question_specs(item)
            if args.validation_mode == "memory_only" and not memory_only_should_process_row(
                idx, item, row_filter_idx, row_filter_ids
            ):
                flat_q += len(q_specs)
                continue

            logger.info("-" * 70)
            logger.info(
                "Processing dataset row %d/%d (%d evaluation question(s)), flat indices from %d",
                idx + 1,
                len(dataset),
                len(q_specs),
                flat_q,
            )
            logger.info("Question type: %s", item.get("question_type", "unknown"))
            for j, sp in enumerate(q_specs):
                logger.info("  Question %d/%d: %s", j + 1, len(q_specs), sp.get("question", ""))

            # Count messages for timing (one memory pass per row)
            sessions = item.get("haystack_sessions", [])
            num_messages = sum(len(s) for s in sessions)

            start_time = time.time()
            try:
                batch_processor.process_single_item(item, idx, flat_q)
                flat_q += len(q_specs)
            except Exception as e:
                logger.exception("Error processing dataset row %d: %s", idx, e)
            finally:
                processing_time = time.time() - start_time
                timing.add_item(num_messages, processing_time)

    # Finalize - flush remaining buffers
    timing.end_total()
    all_results, stats = batch_processor.finalize()

    # Optional: HTML knowledge graph (requires knowledge_graph.gml under RAGU storage)
    if (
        pipeline is not None
        and getattr(args, "save_intermediate", True)
        and args.validation_mode in ("full", "memory_only")
    ):
        maybe_build_ragu_graph_html(Path(args.output_dir), pipeline)

    # Mode-specific summary
    logger.info("=" * 70)
    if args.validation_mode == "memory_only":
        logger.info("Memory-Only Processing Complete")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info("  Memory states: %s", persistence.output_dir / "memory_only_states.json")
        logger.info("  Chunks: %s", persistence.output_dir / "chunk_*/")
        logger.info("\nNext step - run final LLM:")
        logger.info("  python validate_longmemeval.py \\")
        logger.info("    --validation-mode final_llm_only \\")
        logger.info("    --input-state-dir %s \\", args.output_dir)
        logger.info("    --output-dir ./results_final_llm")
    elif args.validation_mode == "final_llm_only":
        logger.info("Final LLM Processing Complete")
        logger.info("=" * 70)
        logger.info("Output files:")
        logger.info("  Answers by strategy:")
        summary_strategies = list(getattr(args, "final_llm_memory_strategies", []) or [])
        if not summary_strategies and pipeline is not None:
            summary_strategies = [str(getattr(pipeline.config, "memory_strategy", "full_graph_json"))]
        for strategy in summary_strategies:
            logger.info("    %s: %s", strategy, persistence.output_dir / strategy / "intermediate_answers.json")
        logger.info(
            "  Final LLM prompt chars: before_clamp_total=%d after_clamp_total=%d calls=%d",
            int(stats.get("final_llm_prompt_chars_before_clamp_total", 0)),
            int(stats.get("final_llm_prompt_chars_after_clamp_total", 0)),
            int(stats.get("final_llm_calls", 0)),
        )
        logger.info("\nNext step - run judge:")
        logger.info("  python validate_longmemeval.py \\")
        logger.info("    --validation-mode judge_only \\")
        logger.info("    --input-answers-path %s \\", persistence.output_dir / "intermediate_answers.json")
        logger.info("    --output-dir ./results_judge")
    elif args.validation_mode == "judge_only":
        logger.info("Judge Evaluation Complete")
        logger.info("=" * 70)
        _print_final_stats(stats, timing.get_stats(), args, results_path)
    else:  # full mode
        logger.info("Validation Complete")
        logger.info("=" * 70)
        _print_final_stats(stats, timing.get_stats(), args, results_path)


def _print_final_stats(stats: Dict, timing_stats: Dict, args: argparse.Namespace, results_path: Path) -> None:
    """Print final statistics summary."""
    avg_score = stats["total_score"] / stats["total"] if stats["total"] > 0 else 0

    logger.info("Statistics:")
    logger.info("  Total processed: %d", stats["total"])
    logger.info("  Average score: %.3f", avg_score)
    logger.info("  Judge errors: %d", stats["errors_judge"])

    logger.info("\nPer-question-type results:")
    for qt, qt_stats in stats["by_type"].items():
        if qt_stats["count"] > 0:
            avg = qt_stats.get("average_score", 0)
            logger.info("  %s: %d items, avg score=%.3f, errors=%d",
                        qt, qt_stats["count"], avg, qt_stats["errors"])

    if args.calculate_memory_hit_rate:
        logger.info("\nMemory hit evaluation (MHE):")
        logger.info("  Hits: %d", stats["memory_hit"])
        logger.info("  Misses: %d", stats["memory_miss"])
        if stats["memory_hit"] + stats["memory_miss"] > 0:
            hit_rate = stats["memory_hit"] / (stats["memory_hit"] + stats["memory_miss"])
            logger.info("  Rate: %.2f%%", hit_rate * 100)
        logger.info("  Per-question-type MHE:")
        for qt, qt_stats in stats["by_type"].items():
            mc = qt_stats.get("mhe_count", 0)
            if mc > 0:
                mhr = qt_stats.get("mhe_hit_rate", qt_stats.get("mhe_hits", 0) / mc)
                logger.info(
                    "    %s: %d judged, hits=%d misses=%d hit_rate=%.3f",
                    qt, mc, qt_stats.get("mhe_hits", 0), qt_stats.get("mhe_misses", 0), mhr,
                )

    # Timing summary
    logger.info("\nTiming statistics:")
    logger.info("  Total time: %.2fs", timing_stats.get("total_time", 0))
    n_writes = timing_stats.get("total_user_message_writes", 0)
    if n_writes:
        logger.info("  User message writes (write_to_memory): %d", n_writes)
    if "time_per_user_message" in timing_stats:
        t = timing_stats["time_per_user_message"]
        logger.info(
            "  Per user message (write_to_memory): min=%.3fs, max=%.3fs, p50=%.3fs, p95=%.3fs, p99=%.3fs",
            t.get("min", 0), t.get("max", 0), t.get("p50", 0), t.get("p95", 0), t.get("p99", 0),
        )
    if "time_per_dialogue" in timing_stats:
        t = timing_stats["time_per_dialogue"]
        logger.info(
            "  Per dialogue (full LongMemEval item, memory pass): min=%.3fs, max=%.3fs, "
            "p50=%.3fs, p95=%.3fs, p99=%.3fs",
            t.get("min", 0), t.get("max", 0), t.get("p50", 0), t.get("p95", 0), t.get("p99", 0),
        )

    logger.info("Results saved to: %s", results_path)


def build_cli_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Build CLI overrides for GigaMemory config."""
    overrides = {}

    # Model paths
    if args.gm_importance_model_path:
        overrides["importance_model_path"] = args.gm_importance_model_path
    if args.gm_slot_model_path:
        overrides["slot_model_path"] = args.gm_slot_model_path

    # Thresholds
    if args.gm_importance_threshold is not None:
        overrides["importance_threshold"] = args.gm_importance_threshold

    # Memory strategy
    if args.gm_memory_strategy:
        overrides["memory_strategy"] = args.gm_memory_strategy
    if args.gm_graph_top_k_records is not None:
        overrides["graph_top_k_records"] = args.gm_graph_top_k_records

    # LLM settings
    if args.gm_llm_mode:
        overrides["llm_mode"] = args.gm_llm_mode
    if args.gm_llm_model:
        overrides["llm_model"] = args.gm_llm_model
    if getattr(args, "gm_llm_tokenizer_model", ""):
        overrides["llm_tokenizer_model"] = args.gm_llm_tokenizer_model
    if args.gm_llm_api_key:
        overrides["llm_api_key"] = args.gm_llm_api_key
    if args.gm_llm_load_dtype:
        overrides["llm_load_dtype"] = args.gm_llm_load_dtype
    if getattr(args, "gm_llm_load_quantization", None):
        overrides["llm_load_quantization"] = args.gm_llm_load_quantization
    if getattr(args, "gm_llm_max_context_tokens", None) is not None:
        overrides["llm_max_context_tokens"] = int(args.gm_llm_max_context_tokens)

    # RAGU settings
    if args.gm_ragu_storage_path:
        overrides["ragu_storage_path"] = args.gm_ragu_storage_path
    if args.gm_ragu_embedder_model:
        overrides["ragu_embedder_model"] = args.gm_ragu_embedder_model

    # Slot and deletion settings
    if args.gm_slot_use_stub is not None:
        overrides["slot_use_stub"] = args.gm_slot_use_stub
    if args.gm_slot_context_enabled is not None:
        overrides["slot_context_enabled"] = args.gm_slot_context_enabled
    if args.gm_triplet_deletion_mode:
        overrides["triplet_deletion_mode"] = args.gm_triplet_deletion_mode

    # Prompt language
    if args.gm_prompt_language:
        overrides["prompt_language"] = args.gm_prompt_language

    # Model unloading
    if args.gm_unload_models_before_final_llm is not None:
        overrides["unload_models_before_final_llm"] = args.gm_unload_models_before_final_llm

    if getattr(args, "gm_use_dataset_datetime", None) is not None:
        overrides["use_dataset_datetime"] = bool(args.gm_use_dataset_datetime)

    if getattr(args, "gm_force_infinite_ttl", None) is not None:
        overrides["force_infinite_ttl"] = bool(args.gm_force_infinite_ttl)

    if getattr(args, "gm_llm_enable_thinking", None) is not None:
        overrides["llm_enable_thinking"] = bool(args.gm_llm_enable_thinking)

    qslot = getattr(args, "gm_slot_llm_load_quantization", None)
    if qslot:
        overrides["slot_llm_load_quantization"] = str(qslot).strip().lower()

    if getattr(args, "gm_conflict_rule_same_relation_updates", None) is not None:
        overrides["conflict_rule_same_relation_updates"] = bool(
            args.gm_conflict_rule_same_relation_updates
        )

    if hasattr(args, "gm_openrouter_reasoning"):
        overrides["openrouter_reasoning"] = getattr(args, "gm_openrouter_reasoning")

    return overrides


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    load_dst_memory_dotenv()

    parser = argparse.ArgumentParser(
        description="Validate GigaMemory DST pipeline on LongMemEval (v3 with multiple validation modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline (default): memory -> final LLM -> judge
    python validate_longmemeval.py

    # Memory only: process dialogues and save memory state
    python validate_longmemeval.py --validation-mode memory_only

    # Final LLM only: load saved memory state and generate answers
    python validate_longmemeval.py --validation-mode final_llm_only \\
        --input-state-dir ./results_memory_only

    # Judge only: evaluate saved answers
    python validate_longmemeval.py --validation-mode judge_only \\
        --input-answers-path ./results_final_llm/intermediate_answers.json

    # Using custom config file
    python validate_longmemeval.py --config ./my_validation_config.json

    # Override specific config parameters via CLI
    python validate_longmemeval.py \\
        --val-shared-num-items-per-type 20 \\
        --val-batch-final-llm-batch-size 10 \\
        --val-judge-model openai/gpt-4o-mini
        """,
    )

    # Config file path
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "run_config.json"),
                        help="Path to validation config JSON file (default: run_config.json)")

    # Validation mode (NEW in v3)
    mode_group = parser.add_argument_group("Validation Mode (v3)")
    # default=None: do not clobber validation_mode.mode from JSON on first parse (default "full" did that).
    mode_group.add_argument("--validation-mode", type=str,
                           choices=["full", "memory_only", "final_llm_only", "judge_only"],
                           default=None,
                           help="Validation mode (omit to use config file; default in JSON/run is full)")
    mode_group.add_argument("--input-state-dir", type=str,
                           default=None,
                           help="Input directory with saved memory states (for final_llm_only mode); omit = from config")
    mode_group.add_argument("--input-answers-path", type=str,
                           default=None,
                           help="Path to intermediate answers JSON (for judge_only mode); omit = from config")
    mode_group.add_argument(
        "--input-answers-paths",
        type=str,
        default=None,
        help=(
            "judge_only: comma-separated intermediate_answers.json paths — usually one file per "
            "final_llm memory strategy you evaluated (see intermediate tree under results_*)."
        ),
    )
    mode_group.add_argument("--memory-only-output-suffix", type=str,
                           default=None,
                           help="Suffix for memory_only output dirs; omit = from config")
    mode_group.add_argument(
        "--memory-only-write-mode",
        type=str,
        default=None,
        choices=list(MEMORY_ONLY_WRITE_MODES),
        help="memory_only write path: standard | single_path_only",
    )
    mode_group.add_argument(
        "--memory-only-dialogue-row-indices",
        type=str,
        default=None,
        help=(
            "memory_only only: comma-separated row indices in the balanced dataset "
            "(same numbering as chunk_XXXX). Only those chunks/states are rebuilt; "
            "existing memory_only_states.json entries for other rows are kept."
        ),
    )
    mode_group.add_argument(
        "--memory-only-dialogue-ids",
        type=str,
        default=None,
        help=(
            "memory_only only: comma-separated dialogue_id values matching dataset rows "
            "(alternative to row indices)."
        ),
    )
    mode_group.add_argument(
        "--final-llm-memory-strategies",
        type=str,
        default=None,
        help=(
            "Comma-separated memory strategies for final_llm_only. "
            "Allowed: full_graph_json,relevant_slots_full,topk_graph_records"
        ),
    )
    mode_group.add_argument(
        "--final-llm-memory-payload-mode",
        type=str,
        default=None,
        choices=list(MEMORY_PAYLOAD_MODES),
        help="Payload mode for memory in final LLM: with_metadata | triplets_only",
    )
    mode_group.add_argument(
        "--inactive-facts-memory-modes",
        type=str,
        default=None,
        help=(
            "final_llm_only: comma-separated DST inactive-fact modes — active_only (default), "
            "with_inactive. Listing both doubles LLM calls and writes separate intermediate_answers.json paths."
        ),
    )
    mode_group.add_argument(
        "--final-llm-resume-from-global-index",
        type=int,
        default=None,
        help=(
            "final_llm_only only: merge into existing intermediate_answers.json under output_dir. "
            "Keep rows with global_index below N; re-run LLM for states with global_index >= N and overwrite "
            "the tail (same strategy × inactive_facts files). Omit for a full rewrite."
        ),
    )

    # Validation parameters (override config file)
    val_group = parser.add_argument_group("Validation Config Overrides (--val-*)")

    val_group.add_argument("--val-shared-dataset-path", type=str,
                           help="Override: dataset path")
    val_group.add_argument("--val-shared-output-dir", type=str,
                           help="Override: output directory")
    val_group.add_argument("--val-shared-num-items-per-type", type=int,
                           help="Override: number of items per question type (balanced sampling)")
    val_group.add_argument("--val-shared-question-types", type=str,
                           help="Override: comma-separated list of question types to test")
    val_group.add_argument("--val-shared-log-level", type=str,
                           choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                           help="Override: log level")
    val_group.add_argument("--val-shared-log-file", type=lambda x: x.lower() == 'true',
                           help="Override: save log to file (true/false)")
    val_group.add_argument("--val-shared-save-memory-state", type=lambda x: x.lower() == 'true',
                           help="Override: save memory state (true/false)")
    val_group.add_argument("--val-shared-save-intermediate", type=lambda x: x.lower() == 'true',
                           help="Override: save intermediate results (true/false)")

    val_group.add_argument("--val-batch-final-llm-batch-size", type=int,
                           help="Override: final LLM batch size")
    val_group.add_argument("--val-batch-judge-batch-size", type=int,
                           help="Override: judge batch size")
    val_group.add_argument("--val-batch-calculate-memory-hit-rate", type=lambda x: x.lower() == 'true',
                           help="Override: calculate memory hit rate (true/false)")

    val_group.add_argument("--val-judge-mode", type=str,
                           choices=["openrouter", "puter", "local", "none"],
                           help="Override: judge mode")
    val_group.add_argument("--val-judge-model", type=str,
                           help="Override: judge model")
    val_group.add_argument("--val-judge-api-key", type=str,
                           help="Override: judge API key")
    val_group.add_argument("--val-judge-temperature", type=float,
                           help="Override: judge temperature")
    val_group.add_argument("--val-judge-max-tokens", type=int,
                           help="Override: judge max tokens")
    val_group.add_argument("--val-judge-max-context-tokens", type=int,
                           help="Override: judge max prompt tokens before completion")
    val_group.add_argument("--val-judge-tokenizer-model", type=str,
                           help="Override: HF tokenizer id/path for judge context clamp")
    val_group.add_argument("--val-judge-local-model-path", type=str,
                           help="Override: judge local model path")

    # Legacy: direct CLI args (without --val- prefix)
    # These are required if not using config file
    legacy_group = parser.add_argument_group("Legacy CLI Args (use --val-* or config file instead)")
    legacy_group.add_argument("--dataset-path", type=str, help="[Legacy] Dataset path")
    legacy_group.add_argument("--output-dir", type=str, help="[Legacy] Output directory")
    legacy_group.add_argument("--num-items-per-type", type=int, help="[Legacy] Number of items per question type (balanced sampling)")
    legacy_group.add_argument("--question-types", type=str, help="[Legacy] Comma-separated list of question types to test")

    # Batch processing (NEW)
    parser.add_argument("--final-llm-batch-size", type=int, default=1,
                        help="Accumulate N dialogues before calling final LLM (default: 1)")
    parser.add_argument("--judge-batch-size", type=int, default=1,
                        help="Accumulate M answers before calling judge (default: 1)")

    # Memory hit rate (NEW)
    parser.add_argument("--calculate-memory-hit-rate", action="store_true", default=False,
                        help="Calculate memory hit rate metric (extra LLM calls)")

    # Judge configuration
    parser.add_argument("--judge-mode", type=str, choices=["openrouter", "puter", "local", "none"],
                        default="openrouter", help="Judge mode")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-120b:free",
                        help="Judge model for openrouter")
    parser.add_argument("--judge-api-url", type=str, default="https://openrouter.ai/api/v1",
                        help="Judge API URL")
    parser.add_argument("--judge-api-key", type=str, default="",
                        help="Judge API key (or OPENROUTER_API_KEY env var)")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="Judge max tokens")
    parser.add_argument("--judge-max-context-tokens", type=int, default=128 * 1024,
                        help="Judge max prompt tokens (0 disables clamp)")
    parser.add_argument("--judge-tokenizer-model", type=str, default="",
                        help="HF tokenizer id/path for judge context clamp")
    parser.add_argument("--judge-local-model-path", type=str, default="",
                        help="Local judge model path")

    # Output options
    parser.add_argument("--save-memory-state", action="store_true", default=True,
                        help="Save memory state after each chunk")
    parser.add_argument("--no-save-memory-state", dest="save_memory_state",
                        action="store_false", help="Disable memory state saving")
    parser.add_argument("--save-intermediate", action="store_true", default=True,
                        help="Save intermediate results")

    # Logging
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--log-file", action="store_true", default=True,
                        help="Save log to file")
    parser.add_argument("--no-log-file", dest="log_file", action="store_false",
                        help="Disable file logging")

    # GigaMemory CLI overrides (NEW)
    gm_group = parser.add_argument_group("GigaMemory Configuration Overrides")

    # Model paths
    gm_group.add_argument("--gm-importance-model-path", type=str, default="",
                          help="Path to importance classifier model")
    gm_group.add_argument("--gm-slot-model-path", type=str, default="",
                          help="Path to slot model")

    # Thresholds
    gm_group.add_argument("--gm-importance-threshold", type=float, default=None,
                          help="Importance classification threshold")

    # Memory strategy
    gm_group.add_argument("--gm-memory-strategy", type=str, default="",
                          choices=["", "full_graph_json", "relevant_slots_full", "topk_graph_records"],
                          help="Memory strategy for final LLM")
    gm_group.add_argument("--gm-graph-top-k-records", type=int, default=None,
                          help="Top-K graph records for retrieval")

    # LLM settings
    gm_group.add_argument("--gm-llm-mode", type=str, default="",
                          choices=["", "stub", "local", "openrouter", "api", "puter"],
                          help="Final LLM mode")
    gm_group.add_argument("--gm-llm-model", type=str, default="",
                          help="Final LLM model name")
    gm_group.add_argument("--gm-llm-tokenizer-model", type=str, default="",
                          help="HF tokenizer id/path for final LLM context clamp")
    gm_group.add_argument("--gm-llm-api-key", type=str, default="",
                          help="Final LLM API key")
    gm_group.add_argument(
        "--gm-llm-load-dtype",
        type=str,
        default="",
        help="For llm_mode=local: HF load dtype (float16 default in config, or bfloat16 / float32)",
    )
    gm_group.add_argument(
        "--gm-llm-load-quantization",
        type=str,
        default="",
        help="For llm_mode=local: none | 8bit | 4bit (BitsAndBytes; reduces VRAM)",
    )
    gm_group.add_argument(
        "--gm-llm-max-context-tokens",
        type=int,
        default=None,
        help="Max final-LLM prompt tokens (default 131072 from config). 0 = disable truncation.",
    )

    # RAGU settings
    gm_group.add_argument("--gm-ragu-storage-path", type=str, default="",
                          help="RAGU storage path")
    gm_group.add_argument("--gm-ragu-embedder-model", type=str, default="",
                          help="RAGU embedder model")

    # Slot and deletion settings
    gm_group.add_argument("--gm-slot-use-stub", type=lambda x: x.lower() == 'true' if x else None,
                          default=None, help="Use stub for slot operations (true/false)")
    gm_group.add_argument("--gm-slot-context-enabled", type=lambda x: x.lower() == 'true' if x else None,
                          default=None, help="Enable slot context (true/false)")
    gm_group.add_argument("--gm-triplet-deletion-mode", type=str, default="",
                          choices=["", "none", "heuristic", "llm_inline", "llm_separate"],
                          help="Triplet deletion mode")
    gm_group.add_argument(
        "--gm-conflict-rule-same-relation-updates",
        type=lambda x: x.lower() == "true" if x else None,
        default=None,
        help="Deterministic replace when same subject+relation, different object (true/false; false = LLM only)",
    )
    gm_group.add_argument(
        "--gm-slot-llm-load-quantization",
        type=str,
        default="",
        choices=["", "none", "8bit", "4bit"],
        help="BitsAndBytes quant for slot/triplet local model (CUDA + bitsandbytes).",
    )

    # Prompt language
    gm_group.add_argument("--gm-prompt-language", type=str, default="",
                          choices=["", "ru", "en"],
                          help="Prompt UI language")

    # Model unloading
    gm_group.add_argument("--gm-unload-models-before-final-llm",
                          type=lambda x: x.lower() == 'true' if x else None,
                          default=None, help="Unload models before final LLM (true/false)")

    gm_group.add_argument(
        "--gm-use-dataset-datetime",
        type=lambda x: x.lower() == "true" if x else None,
        default=None,
        help="Use LongMemEval row question_date for fact timestamps, TTL as_of, and final-LLM clock (true/false)",
    )
    gm_group.add_argument(
        "--gm-force-infinite-ttl",
        type=lambda x: x.lower() == "true" if x else None,
        default=None,
        help="Ignore model/slot TTL: all new facts get ttl=inf (no TTL expiry; default true in pipeline config)",
    )

    # Parse known args first to get config path
    args, remaining = parser.parse_known_args()

    # Load validation config file
    val_config_path = args.config if args.config else str(Path(__file__).parent / "run_config.json")
    config = load_validation_config(val_config_path)

    # Apply --val-* overrides from CLI to config
    if args.val_shared_dataset_path:
        config["shared"]["dataset_path"] = args.val_shared_dataset_path
    if args.val_shared_output_dir:
        config["shared"]["output_dir"] = args.val_shared_output_dir
    if args.val_shared_num_items_per_type is not None:
        config["shared"]["num_items_per_type"] = args.val_shared_num_items_per_type
    if args.val_shared_question_types:
        config["shared"]["question_types"] = args.val_shared_question_types.split(",")
    if args.val_shared_log_level:
        config["shared"]["log_level"] = args.val_shared_log_level
    if args.val_shared_log_file is not None:
        config["shared"]["log_file"] = args.val_shared_log_file
    if args.val_shared_save_memory_state is not None:
        config["shared"]["save_memory_state"] = args.val_shared_save_memory_state
    if args.val_shared_save_intermediate is not None:
        config["shared"]["save_intermediate"] = args.val_shared_save_intermediate

    if args.val_batch_final_llm_batch_size is not None:
        config["batch_processing"]["final_llm_batch_size"] = args.val_batch_final_llm_batch_size
    if args.val_batch_judge_batch_size is not None:
        config["batch_processing"]["judge_batch_size"] = args.val_batch_judge_batch_size
    if args.val_batch_calculate_memory_hit_rate is not None:
        config["batch_processing"]["calculate_memory_hit_rate"] = args.val_batch_calculate_memory_hit_rate

    if args.val_judge_mode:
        config["judge"]["mode"] = args.val_judge_mode
    if args.val_judge_model:
        config["judge"]["model"] = args.val_judge_model
    if args.val_judge_api_key:
        config["judge"]["api_key"] = args.val_judge_api_key
    if args.val_judge_temperature is not None:
        config["judge"]["temperature"] = args.val_judge_temperature
    if args.val_judge_max_tokens is not None:
        config["judge"]["max_tokens"] = args.val_judge_max_tokens
    if args.val_judge_max_context_tokens is not None:
        config["judge"]["max_context_tokens"] = int(args.val_judge_max_context_tokens)
    if args.val_judge_tokenizer_model:
        config["judge"]["tokenizer_model"] = args.val_judge_tokenizer_model
    if args.val_judge_local_model_path:
        config["judge"]["local_model_path"] = args.val_judge_local_model_path

    if getattr(args, "gm_llm_load_quantization", None):
        config.setdefault("giga_memory", {})
        if isinstance(config["giga_memory"], dict):
            config["giga_memory"]["llm_load_quantization"] = args.gm_llm_load_quantization
    if getattr(args, "gm_slot_llm_load_quantization", None):
        config.setdefault("giga_memory", {})
        if isinstance(config["giga_memory"], dict):
            config["giga_memory"]["slot_llm_load_quantization"] = args.gm_slot_llm_load_quantization
    if getattr(args, "gm_llm_max_context_tokens", None) is not None:
        config.setdefault("giga_memory", {})
        if isinstance(config["giga_memory"], dict):
            config["giga_memory"]["llm_max_context_tokens"] = int(args.gm_llm_max_context_tokens)

    # Handle validation mode args (only explicit CLI overrides; first-parse defaults must not erase JSON)
    if args.validation_mode is not None:
        config["validation_mode"]["mode"] = args.validation_mode
    if args.input_state_dir is not None:
        config["validation_mode"]["input_state_dir"] = args.input_state_dir
    if args.input_answers_path is not None:
        config["validation_mode"]["input_answers_path"] = args.input_answers_path
    if getattr(args, "input_answers_paths", None):
        _iap_cli = str(args.input_answers_paths).strip()
        if _iap_cli:
            config["validation_mode"]["input_answers_paths"] = _normalize_input_answers_paths(_iap_cli)
    if args.memory_only_output_suffix is not None:
        config["validation_mode"]["memory_only_output_suffix"] = args.memory_only_output_suffix
    if args.memory_only_write_mode is not None:
        config["validation_mode"]["memory_only_write_mode"] = _normalize_memory_only_write_mode(
            args.memory_only_write_mode
        )
    if args.final_llm_memory_strategies is not None:
        config["validation_mode"]["final_llm_memory_strategies"] = _normalize_memory_strategies(
            args.final_llm_memory_strategies
        )
    if args.final_llm_memory_payload_mode is not None:
        config["validation_mode"]["final_llm_memory_payload_mode"] = _normalize_memory_payload_mode(
            args.final_llm_memory_payload_mode
        )
    if getattr(args, "memory_only_dialogue_row_indices", None) is not None:
        s = str(args.memory_only_dialogue_row_indices).strip()
        config["validation_mode"]["memory_only_dialogue_row_indices"] = (
            [int(x.strip()) for x in s.split(",") if x.strip()] if s else []
        )
    if getattr(args, "memory_only_dialogue_ids", None) is not None:
        s = str(args.memory_only_dialogue_ids).strip()
        config["validation_mode"]["memory_only_dialogue_ids"] = (
            [x.strip() for x in s.split(",") if x.strip()] if s else []
        )
    if getattr(args, "inactive_facts_memory_modes", None) is not None:
        s = str(args.inactive_facts_memory_modes).strip()
        modes_in = [x.strip() for x in s.split(",") if x.strip()] if s else []
        config["validation_mode"]["inactive_facts_memory_modes"] = (
            _normalize_inactive_facts_memory_modes(modes_in)
        )
    if getattr(args, "final_llm_resume_from_global_index", None) is not None:
        config["validation_mode"]["final_llm_resume_from_global_index"] = int(
            args.final_llm_resume_from_global_index
        )

    # Handle legacy args
    if args.dataset_path:
        config["shared"]["dataset_path"] = args.dataset_path
    if args.output_dir:
        config["shared"]["output_dir"] = args.output_dir

    # Handle new balanced sampling args
    if args.num_items_per_type is not None:
        config["shared"]["num_items_per_type"] = args.num_items_per_type
    if args.question_types:
        config["shared"]["question_types"] = args.question_types.split(",")

    # Convert config to args namespace
    config_args = config_to_args(config)

    # Now parse CLI args again, using config_args as defaults
    parser.set_defaults(**vars(config_args))
    args = parser.parse_args()
    args.final_llm_memory_strategies = _normalize_memory_strategies(
        getattr(args, "final_llm_memory_strategies", [])
    )
    args.memory_only_write_mode = _normalize_memory_only_write_mode(
        getattr(args, "memory_only_write_mode", "standard")
    )
    args.final_llm_memory_payload_mode = _normalize_memory_payload_mode(
        getattr(args, "final_llm_memory_payload_mode", "with_metadata")
    )
    _ifm = getattr(args, "inactive_facts_memory_modes", [])
    if isinstance(_ifm, str):
        _ifs = _ifm.strip()
        args.inactive_facts_memory_modes = _normalize_inactive_facts_memory_modes(
            [x.strip() for x in _ifs.split(",") if x.strip()] if _ifs else []
        )
    else:
        args.inactive_facts_memory_modes = _normalize_inactive_facts_memory_modes(_ifm)

    _mori = getattr(args, "memory_only_dialogue_row_indices", [])
    if isinstance(_mori, str):
        args.memory_only_dialogue_row_indices = (
            [int(x.strip()) for x in _mori.split(",") if x.strip()] if _mori.strip() else []
        )
    else:
        args.memory_only_dialogue_row_indices = _coerce_int_list(_mori)
    _moids = getattr(args, "memory_only_dialogue_ids", [])
    if isinstance(_moids, str):
        args.memory_only_dialogue_ids = (
            [x.strip() for x in _moids.split(",") if x.strip()] if _moids.strip() else []
        )
    else:
        args.memory_only_dialogue_ids = _coerce_str_list(_moids)

    _rfgi = getattr(args, "final_llm_resume_from_global_index", None)
    if _rfgi is not None and str(_rfgi).strip() != "":
        args.final_llm_resume_from_global_index = int(_rfgi)
    else:
        args.final_llm_resume_from_global_index = None

    _iaps_raw = getattr(args, "input_answers_paths", None)
    args.input_answers_paths = _normalize_input_answers_paths(_iaps_raw)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args._validation_shared_pipeline_overrides = _pipeline_overrides_from_validation_shared(
        config.get("shared") or {}
    )

    run_validation(args)


if __name__ == "__main__":
    main()
