"""
LongMemEval validation script for GigaMemory DST pipeline - Version 2.

Advanced features:
- Batch processing for final LLM (accumulate N dialogues before answering)
- Batch processing for judge (accumulate M answers before judging)
- Memory Hit Rate metric (separate LLM call to check if fact was in context)
- Model unloading before final LLM for local mode
- Full configuration via JSON config file (mirrors DST_memory structure)

Usage:
    # Using default config file (run_config.json in same directory)
    python validate_longmemeval_v2.py

    # Using custom config
    python validate_longmemeval_v2.py --config ./my_config.json

    # Override specific parameters via CLI
    python validate_longmemeval_v2.py \
        --config ./run_config.json \
        --val-shared-start-index 20 \
        --val-shared-num-items 50

Config file structure mirrors DST_memory/run_config.json:
    {
      "shared": { ... validation dataset/output params ... },
      "batch_processing": { ... batch sizes ... },
      "judge": { ... judge configuration ... },
      "giga_memory": { ... GigaMemory pipeline config ... }
    }
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
import random


# ============================================================================
# Timing Utilities
# ============================================================================

class TimingStats:
    """Collect and compute timing statistics."""

    def __init__(self):
        self.items: List[Dict[str, float]] = []
        self.total_start: Optional[float] = None
        self.total_time: float = 0.0

    def start_total(self):
        self.total_start = time.time()

    def end_total(self):
        if self.total_start:
            self.total_time = time.time() - self.total_start

    def add_item(self, num_messages: int, processing_time: float):
        """Add timing for a single item."""
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

    def get_stats(self) -> Dict[str, Any]:
        """Get computed statistics."""
        times = [item["time"] for item in self.items]
        per_msg_times = [item["time_per_message"] for item in self.items if item["num_messages"] > 0]
        total_messages = sum(item["num_messages"] for item in self.items)

        if not times:
            return {
                "total_time": self.total_time,
                "total_items": 0,
                "total_messages": 0,
            }

        return {
            "total_time": self.total_time,
            "total_items": len(self.items),
            "total_messages": total_messages,
            "time_per_item": {
                "min": min(times),
                "max": max(times),
                "p50": self.compute_percentile(times, 50),
                "p95": self.compute_percentile(times, 95),
                "p99": self.compute_percentile(times, 99),
                "mean": sum(times) / len(times),
            },
            "time_per_message": {
                "min": min(per_msg_times) if per_msg_times else 0,
                "max": max(per_msg_times) if per_msg_times else 0,
                "p50": self.compute_percentile(per_msg_times, 50) if per_msg_times else 0,
                "p95": self.compute_percentile(per_msg_times, 95) if per_msg_times else 0,
                "p99": self.compute_percentile(per_msg_times, 99) if per_msg_times else 0,
                "mean": sum(per_msg_times) / len(per_msg_times) if per_msg_times else 0,
            },
        }


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
            "local_model_path": "",
            "unload_between_items": False,
        },
        "giga_memory": {},  # Will be merged with DST_memory defaults
    }

    if not Path(config_path).exists():
        logger.warning("Config file not found: %s, using defaults", config_path)
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        # Merge with defaults
        merged = default_config.copy()
        for section in ["shared", "batch_processing", "judge", "giga_memory"]:
            if section in user_config:
                if isinstance(merged.get(section), dict):
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
    args.judge_local_model_path = judge.get("local_model_path", "")
    args.unload_judge_between_items = judge.get("unload_between_items", False)

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
        "llm_api_key", "llm_api_url", "llm_temperature", "llm_max_tokens",
        "openrouter_http_referer", "openrouter_x_title", "slot_use_stub",
        "slot_model_path", "slot_max_slots_per_message", "ragu_storage_path",
        "ragu_embedder_model", "ttl_mode", "ttl_semantic_dedup_enabled",
        "ttl_semantic_dedup_threshold", "slot_context_enabled",
        "slot_context_max_facts", "triplet_deletion_mode", "deletion_use_pymorphy",
        "conflict_allow_multi_relation_same_object", "slot_model_enable_thinking",
        "slot_fallback_on_no_slots", "triplet_fallback_on_empty", "prompt_language",
        "unload_models_before_final_llm"
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
    Supports 'openrouter', 'local', and 'none' modes.
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
    ):
        self.mode = mode
        self.model = model
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_model_path = local_model_path
        self._local_serving = None

        logger.info(
            "JudgeClient initialized mode=%s model=%s",
            mode,
            model if mode == "openrouter" else local_model_path,
        )

    def _get_system_prompt_answer_correctness(self, question_type: str) -> str:
        """Get system prompt with 0-1 scoring criteria."""
        type_desc = QUESTION_TYPES.get(question_type, "General question answering")

        return CHAT_API_OUTPUT_POLICY + f"""You are an expert evaluator assessing answer quality.

Question Type: {question_type}
Type Description: {type_desc}

Your task: Compare the predicted answer with the reference (gold) answer and return a score from 0.0 to 1.0 representing how well the predicted answer covers the factual content of the reference.

Scoring Scale:
1.0 - Perfect match: Contains all key entities and facts from reference. Wording may differ, but meaning is identical.
0.8 - Minor inaccuracy: All key entities present, but one is slightly distorted (wrong number, approximate date, slight name variation).
0.6 - Partial answer: Covers most of reference, but one of several equally important entities is missing or replaced.
0.4 - Weak coverage: Only one correct entity from several needed mentioned, OR correct category but wrong specific fact.
0.2 - Minimal match: Thematically related to question but factually almost no overlap with reference — guessed domain but not content.
0.0 - No match: Factually incorrect, contradicts reference, or system said "I don't know" when reference exists.

Special Rules:
- For knowledge-update: If system named old/outdated fact instead of new one → 0.0 (old fact doesn't count).
- For single-session-preference: Judge if correct user fact was used, not quote accuracy. Different phrasing with correct fact = 1.0.
- For multi-session: If aggregation needed (e.g., "how many total"), partial count scores proportionally: found 2 of 4 needed entities → 0.4-0.6 depending on importance of missing ones.

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
        """Evaluate answer. Returns (score, reasoning, error). Score: 0.0 to 1.0"""
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
        """Evaluate if the needed fact was present in memory context."""
        if not reference_answer or not reference_answer.strip():
            return {"fact_present": False, "reasoning": "Empty reference answer"}

        system_msg = self._get_system_prompt_memory_hit()
        user_msg = self._get_user_prompt_memory_hit(question, reference_answer, memory_context)

        return self._call_judge(system_msg, user_msg, mode="memory_hit")

    def _call_judge(self, system_msg: str, user_msg: str, mode: str = "correctness") -> Dict[str, Any]:
        """Call judge LLM (openrouter or local)."""
        if self.mode == "openrouter":
            return self._call_openrouter(system_msg, user_msg, mode)
        elif self.mode == "local":
            return self._call_local(system_msg, user_msg, mode)
        else:
            raise ValueError(f"Unknown judge mode: {self.mode}")

    @retry_with_backoff(max_retries=3)
    def _call_openrouter_api(self, body: Dict, headers: Dict) -> str:
        """Call OpenRouter API with retry."""
        import urllib.request

        url = f"{self.api_url.rstrip('/')}/chat/completions"
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

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "tool_choice": "none",
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            raw = self._call_openrouter_api(body, headers)
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

    def _call_local(self, system_msg: str, user_msg: str, mode: str) -> Dict[str, Any]:
        """Call local model."""
        from dst_memory.clients.serving import LocalHFServing

        if self._local_serving is None:
            if not self.local_model_path:
                raise ValueError("local_model_path required for local judge mode")
            logger.info("Loading local judge model: %s", self.local_model_path)
            self._local_serving = LocalHFServing(self.local_model_path)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            response = self._local_serving.generate(messages, max_new_tokens=self.max_tokens)
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

        # Save RAGU storage
        if pipeline.ragu_processor is not None:
            ragu_storage_src = Path(pipeline.ragu_processor.kg.storage_path)
            if ragu_storage_src.exists():
                ragu_storage_dst = chunk_dir / "ragu_storage"
                if ragu_storage_dst.exists():
                    shutil.rmtree(ragu_storage_dst)
                shutil.copytree(ragu_storage_src, ragu_storage_dst)
                saved_paths["ragu_storage"] = ragu_storage_dst

        logger.info("Saved chunk state to %s", chunk_dir)
        return saved_paths


# ============================================================================
# Dataset Loading
# ============================================================================

def load_dataset_balanced(
    dataset_path: str,
    question_types: List[str],
    num_per_type: int
) -> List[Dict[str, Any]]:
    """Load dataset with balanced sampling across question types."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Group by question type
    by_type: Dict[str, List[Dict]] = {qt: [] for qt in question_types}
    for item in data:
        qt = item.get("question_type", "")
        if qt in by_type:
            by_type[qt].append(item)

    # Sample from each type
    result = []
    type_counts = {}
    for qt in question_types:
        available = len(by_type[qt])
        to_sample = min(num_per_type, available)
        sampled = by_type[qt][:to_sample]
        result.extend(sampled)
        type_counts[qt] = len(sampled)
        logger.info("Type %s: sampled %d/%d available", qt, to_sample, available)

    logger.info("Total loaded: %d items", len(result))
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


# ============================================================================
# Pipeline Building
# ============================================================================

def build_pipeline_from_config(config_path: str, cli_overrides: Optional[Dict] = None):
    """Build DSTMemoryPipeline from config file with optional CLI overrides."""
    from dst_memory import PipelineConfig
    from dst_memory.core.pipeline import DSTMemoryPipeline
    from dst_memory.storage.ragu_graph_processor import build_ragu_processor

    file_cfg = load_run_config(config_path)
    shared = shared_section(file_cfg)

    if cli_overrides:
        shared.update(cli_overrides)

    cfg = PipelineConfig(
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
        llm_max_tokens=int(shared.get("llm_max_tokens", 1024)),
        llm_temperature=float(shared.get("llm_temperature", 0.0)),
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
        slot_model_enable_thinking=shared.get("slot_model_enable_thinking", False),
        slot_fallback_on_no_slots=shared.get("slot_fallback_on_no_slots", True),
        triplet_fallback_on_empty=shared.get("triplet_fallback_on_empty", True),
        prompt_language=shared.get("prompt_language", "ru"),
        unload_models_before_final_llm=shared.get("unload_models_before_final_llm", True),
    )

    logger.info("Initializing RAGU backend...")
    _kg, ragu_processor = build_ragu_processor(
        embedder_model=cfg.ragu_embedder_model,
        storage_path=cfg.ragu_storage_path or None,
    )

    return DSTMemoryPipeline(cfg, ragu_processor=ragu_processor)


# ============================================================================
# Batch Processing Classes
# ============================================================================

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


@dataclass
class AccumulatedAnswer:
    """Accumulated answer ready for judge evaluation."""
    global_index: int
    question_id: str
    question: str
    reference_answer: str
    predicted_answer: str
    memory_context: Dict[str, Any]


class BatchProcessor:
    """
    Handles batch processing for final LLM and judge.

    Flow:
    1. Process dialogues through memory pipeline (write_to_memory for all sessions)
    2. Accumulate processed dialogues (without calling final LLM)
    3. When batch is full (or at end), call final LLM for all accumulated dialogues
    4. Accumulate answers
    5. When judge batch is full (or at end), call judge for evaluation
    """

    def __init__(
        self,
        pipeline: "DSTMemoryPipeline",
        judge_client: Optional[JudgeClient],
        final_llm_batch_size: int,
        judge_batch_size: int,
        calculate_memory_hit_rate: bool,
        persistence: MemoryStatePersistence,
    ):
        self.pipeline = pipeline
        self.judge_client = judge_client
        self.final_llm_batch_size = final_llm_batch_size
        self.judge_batch_size = judge_batch_size
        self.calculate_memory_hit_rate = calculate_memory_hit_rate
        self.persistence = persistence

        # Accumulators
        self.dialogue_buffer: List[AccumulatedDialogue] = []
        self.answer_buffer: List[AccumulatedAnswer] = []

        # Results storage
        self.results: List[Dict[str, Any]] = []

        # Statistics
        self.stats = {
            "total": 0,
            "total_score": 0.0,
            "errors_final_llm": 0,
            "errors_judge": 0,
            "memory_hit": 0,
            "memory_miss": 0,
            "by_type": {qt: {"count": 0, "total_score": 0.0, "errors": 0}
                        for qt in QUESTION_TYPES.keys()},
        }

    def process_single_item(
        self,
        item: Dict[str, Any],
        global_index: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Process a single item through memory pipeline.
        Does NOT call final LLM immediately - accumulates for batch processing.

        Returns:
            Result dict if this was the last item and batch was flushed,
            None otherwise (result will be added later in batch)
        """
        dialogue_id = f"longmemeval_{global_index}_{item.get('question_id', 'unknown')}"
        question = item.get("question", "")
        reference_answer = item.get("answer", "")
        question_type = item.get("question_type", "")
        sessions = item.get("haystack_sessions", [])

        # Extract and process user messages
        user_messages = extract_user_messages_from_sessions(sessions)

        logger.info(
            "[Batch] Processing memory for item %d: %d sessions, %d messages, type=%s",
            global_index, len(sessions), len(user_messages), question_type,
        )

        # Process through memory pipeline
        write_logs = []
        from dst_memory.core.models import Message

        for msg in user_messages:
            log = self.pipeline.write_to_memory(dialogue_id, Message(role="user", content=msg))
            write_logs.append(log)

        # Get memory state (without calling final LLM)
        answer_details = self.pipeline.answer_without_final_llm(dialogue_id, question)

        # Save memory state
        self.persistence.save_chunk_state(f"{global_index:04d}", self.pipeline, dialogue_id)

        # Create accumulated dialogue
        acc_dialogue = AccumulatedDialogue(
            global_index=global_index,
            dialogue_id=dialogue_id,
            question_id=item.get("question_id", ""),
            question=question,
            reference_answer=reference_answer,
            question_type=question_type,
            pipeline_state={
                "write_logs": write_logs,
                "memory_slots": answer_details.get("memory_slots", []),
                "expired_facts": answer_details.get("expired_facts", []),
                "deleted_facts_with_reasons": answer_details.get("deleted_facts_with_reasons", []),
                "use_memory": answer_details.get("use_memory", False),
                "memory_context": answer_details.get("memory_context_for_final_llm", {}),
                "final_llm_prompt": answer_details.get("final_llm_prompt", []),
            },
        )

        self.dialogue_buffer.append(acc_dialogue)

        # Clear memory for next dialogue
        self.pipeline.clear_memory(dialogue_id)

        # Check if we should flush final LLM batch
        if len(self.dialogue_buffer) >= self.final_llm_batch_size:
            self._flush_final_llm_batch()

        return None  # Result will be added by batch processing

    def _flush_final_llm_batch(self):
        """Process accumulated dialogues through final LLM."""
        if not self.dialogue_buffer:
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

            if self.pipeline.config.llm_mode != "stub":
                # Rebuild memory context
                memory_context = acc.pipeline_state["memory_context"]
                recent_pairs = []  # We don't have recent pairs in batch mode

                # Call final LLM
                try:
                    predicted_answer = self.pipeline.final_llm.generate(
                        question=acc.question,
                        memory_context=memory_context,
                        recent_pairs=recent_pairs,
                    )
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
                predicted_answer=predicted_answer,
                memory_context=acc.pipeline_state["memory_context"],
            )

            self.answer_buffer.append(acc_answer)

        # Clear dialogue buffer
        self.dialogue_buffer.clear()

        # Reload models if they were unloaded
        if (
            self.pipeline.config.llm_mode == "local"
            and getattr(self.pipeline.config, "unload_models_before_final_llm", True)
        ):
            logger.info("[Batch] Reloading models after final LLM processing...")
            self.pipeline.reload_local_models()

        # Check if we should flush judge batch
        if len(self.answer_buffer) >= self.judge_batch_size:
            self._flush_judge_batch()

    def _flush_judge_batch(self):
        """Process accumulated answers through judge."""
        if not self.answer_buffer or not self.judge_client:
            self.answer_buffer.clear()
            return

        logger.info(
            "[Batch] Flushing judge batch: %d answers",
            len(self.answer_buffer)
        )

        for acc in self.answer_buffer:
            # Evaluate answer correctness (0-1 score)
            score, reasoning, judge_error = self.judge_client.evaluate_answer(
                question=acc.question,
                predicted_answer=acc.predicted_answer,
                reference_answer=acc.reference_answer,
                question_type=acc.question_type,
            )

            # Evaluate memory hit rate if requested
            memory_hit_result = None
            if self.calculate_memory_hit_rate:
                memory_hit_result = self.judge_client.evaluate_memory_hit(
                    question=acc.question,
                    reference_answer=acc.reference_answer,
                    memory_context=acc.memory_context,
                )

            # Build result
            result = {
                "global_index": acc.global_index,
                "question_id": acc.question_id,
                "question": acc.question,
                "reference_answer": acc.reference_answer,
                "predicted_answer": acc.predicted_answer,
                "question_type": acc.question_type,
                "score": score,
                "reasoning": reasoning,
                "judge_error": judge_error,
                "memory_hit_evaluation": memory_hit_result,
                "memory_hit": memory_hit_result.get("fact_present", False) if memory_hit_result else None,
            }

            self.results.append(result)

            # Update stats
            self.stats["total"] += 1
            self.stats["total_score"] += score
            if judge_error:
                self.stats["errors_judge"] += 1

            # Per-type stats
            qt = acc.question_type
            if qt in self.stats["by_type"]:
                self.stats["by_type"][qt]["count"] += 1
                self.stats["by_type"][qt]["total_score"] += score
                if judge_error:
                    self.stats["by_type"][qt]["errors"] += 1

            if memory_hit_result:
                if memory_hit_result.get("fact_present", False):
                    self.stats["memory_hit"] += 1
                else:
                    self.stats["memory_miss"] += 1

        # Clear answer buffer
        self.answer_buffer.clear()

    def finalize(self) -> Tuple[List[Dict], Dict]:
        """Finalize processing - flush any remaining items."""
        logger.info("[Batch] Finalizing - flushing remaining buffers...")

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
    logger.info("LongMemEval Validation v2 - Starting")
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

    # Load dataset with balanced sampling
    dataset = load_dataset_balanced(
        args.dataset_path,
        args.question_types,
        args.num_items_per_type,
    )

    logger.info("Total items to process: %d", len(dataset))

    # Initialize persistence
    persistence = MemoryStatePersistence(args.output_dir)

    # Initialize judge client
    judge_client = None
    if args.judge_mode != "none":
        judge_client = JudgeClient(
            mode=args.judge_mode,
            model=args.judge_model,
            api_url=args.judge_api_url,
            api_key=args.judge_api_key,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            local_model_path=args.judge_local_model_path,
        )

    # Build pipeline
    cli_overrides = build_cli_overrides(args)
    pipeline = build_pipeline_from_config(args.config, cli_overrides)

    # Initialize batch processor
    batch_processor = BatchProcessor(
        pipeline=pipeline,
        judge_client=judge_client,
        final_llm_batch_size=args.final_llm_batch_size,
        judge_batch_size=args.judge_batch_size,
        calculate_memory_hit_rate=args.calculate_memory_hit_rate,
        persistence=persistence,
    )

    # Process each item
    for idx, item in enumerate(dataset):
        global_idx = idx

        logger.info("-" * 70)
        logger.info("Processing item %d/%d (global: %d)", idx + 1, len(dataset), global_idx)
        logger.info("Question type: %s", item.get("question_type", "unknown"))
        logger.info("Question: %s", item.get("question", ""))

        # Count messages for timing
        sessions = item.get("haystack_sessions", [])
        num_messages = sum(len(s) for s in sessions)

        start_time = time.time()
        try:
            batch_processor.process_single_item(item, global_idx)
        except Exception as e:
            logger.exception("Error processing item %d: %s", global_idx, e)
        finally:
            processing_time = time.time() - start_time
            timing.add_item(num_messages, processing_time)

    # Finalize - flush remaining buffers
    timing.end_total()
    all_results, stats = batch_processor.finalize()

    # Compute average score
    avg_score = stats["total_score"] / stats["total"] if stats["total"] > 0 else 0

    # Final summary
    logger.info("=" * 70)
    logger.info("Validation Complete")
    logger.info("=" * 70)
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
        logger.info("\nMemory Hit Rate:")
        logger.info("  Hits: %d", stats["memory_hit"])
        logger.info("  Misses: %d", stats["memory_miss"])
        if stats["memory_hit"] + stats["memory_miss"] > 0:
            hit_rate = stats["memory_hit"] / (stats["memory_hit"] + stats["memory_miss"])
            logger.info("  Rate: %.2f%%", hit_rate * 100)

    # Timing summary
    timing_stats = timing.get_stats()
    logger.info("\nTiming statistics:")
    logger.info("  Total time: %.2fs", timing_stats.get("total_time", 0))
    if "time_per_item" in timing_stats:
        t = timing_stats["time_per_item"]
        logger.info("  Per item: min=%.3fs, max=%.3fs, p50=%.3fs, p95=%.3fs, p99=%.3fs",
                    t.get("min", 0), t.get("max", 0), t.get("p50", 0),
                    t.get("p95", 0), t.get("p99", 0))

    if stats["total"] > 0:
        accuracy = stats["correct"] / stats["total"]
        logger.info("  Accuracy: %.2f%%", accuracy * 100)

    # Save final results
    results_path = Path(args.output_dir) / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "dataset_path": args.dataset_path,
                    "num_items_per_type": args.num_items_per_type,
                    "question_types": args.question_types,
                    "final_llm_batch_size": args.final_llm_batch_size,
                    "judge_batch_size": args.judge_batch_size,
                    "calculate_memory_hit_rate": args.calculate_memory_hit_rate,
                    "judge_mode": args.judge_mode,
                    "judge_model": args.judge_model,
                    "config_path": args.config,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "statistics": stats,
                "timing": timing_stats,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
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
    if args.gm_llm_api_key:
        overrides["llm_api_key"] = args.gm_llm_api_key

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

    return overrides


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    load_dst_memory_dotenv()

    parser = argparse.ArgumentParser(
        description="Validate GigaMemory DST pipeline on LongMemEval (v2 with batch processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using default config file (run_config.json in same directory)
    python validate_longmemeval.py

    # Using custom config file
    python validate_longmemeval.py --config ./my_validation_config.json

    # Override specific config parameters via CLI
    python validate_longmemeval.py \\
        --val-shared-num-items-per-type 20 \\
        --val-batch-final-llm-batch-size 10 \\
        --val-judge-model openai/gpt-4o-mini

    # Legacy: using only CLI args (without config file)
    python validate_longmemeval.py \\
        --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \\
        --output-dir ./results \\
        --num-items-per-type 10
        """,
    )

    # Config file path
    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "run_config.json"),
                        help="Path to validation config JSON file (default: run_config.json)")

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
                           choices=["openrouter", "local", "none"],
                           help="Override: judge mode")
    val_group.add_argument("--val-judge-model", type=str,
                           help="Override: judge model")
    val_group.add_argument("--val-judge-api-key", type=str,
                           help="Override: judge API key")
    val_group.add_argument("--val-judge-temperature", type=float,
                           help="Override: judge temperature")
    val_group.add_argument("--val-judge-max-tokens", type=int,
                           help="Override: judge max tokens")
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
    parser.add_argument("--judge-mode", type=str, choices=["openrouter", "local", "none"],
                        default="openrouter", help="Judge mode")
    parser.add_argument("--judge-model", type=str, default="openai/gpt-oss-120b:free",
                        help="Judge model for openrouter")
    parser.add_argument("--judge-api-url", type=str, default="https://openrouter.ai/api/v1",
                        help="Judge API URL")
    parser.add_argument("--judge-api-key", type=str, default="",
                        help="Judge API key (or OPENROUTER_API_KEY env var)")
    parser.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="Judge max tokens")
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
                          choices=["", "stub", "local", "openrouter", "api"],
                          help="Final LLM mode")
    gm_group.add_argument("--gm-llm-model", type=str, default="",
                          help="Final LLM model name")
    gm_group.add_argument("--gm-llm-api-key", type=str, default="",
                          help="Final LLM API key")

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

    # Prompt language
    gm_group.add_argument("--gm-prompt-language", type=str, default="",
                          choices=["", "ru", "en"],
                          help="Prompt UI language")

    # Model unloading
    gm_group.add_argument("--gm-unload-models-before-final-llm",
                          type=lambda x: x.lower() == 'true' if x else None,
                          default=None, help="Unload models before final LLM (true/false)")

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
    if args.val_judge_local_model_path:
        config["judge"]["local_model_path"] = args.val_judge_local_model_path

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

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_validation(args)


if __name__ == "__main__":
    main()
