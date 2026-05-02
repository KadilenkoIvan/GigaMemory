"""
LongMemEval validation script for GigaMemory DST pipeline.

This script tests the full DST_memory pipeline on the LongMemEval dataset.
It processes sessions sequentially, saves memory state between chunks, and evaluates
answers using an LLM-as-judge approach.

Usage:
    python validate_longmemeval.py \
        --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \
        --output-dir ./results \
        --start-index 0 \
        --num-items 10 \
        --judge-mode openrouter \
        --config ../../DST_memory/run_config.json
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import asdict
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

    Supports two modes:
    - 'local': Uses a local model (loads model, evaluates, then unloads)
    - 'openrouter': Uses OpenRouter API (same as FinalLLMClient)
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

    def _get_system_prompt(self) -> str:
        return (
            "You are an expert evaluator assessing the correctness of answers.\n"
            "Your task is to compare a predicted answer with a reference (gold) answer\n"
            "and determine if they convey the same information, even if worded differently.\n\n"
            "Evaluation criteria:\n"
            "1. Semantic equivalence: Do both answers convey the same core information?\n"
            "2. Factual correctness: Is the predicted answer factually accurate based on the reference?\n"
            "3. No hallucinations: Does the predicted answer introduce false information?\n\n"
            "Respond with ONLY a JSON object in this exact format:\n"
            '{"correct": true/false, "reasoning": "brief explanation"}'
        )

    def _get_user_prompt(self, question: str, predicted: str, reference: str) -> str:
        return (
            f"Question: {question}\n\n"
            f"Predicted Answer: {predicted}\n\n"
            f"Reference Answer: {reference}\n\n"
            "Evaluate if the predicted answer is semantically equivalent to the reference."
        )

    def evaluate(
        self, question: str, predicted_answer: str, reference_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate if predicted_answer is semantically equivalent to reference_answer.

        Returns:
            {"correct": bool, "reasoning": str}
        """
        if not predicted_answer or not predicted_answer.strip():
            return {"correct": False, "reasoning": "Empty predicted answer"}

        if not reference_answer or not reference_answer.strip():
            return {"correct": False, "reasoning": "Empty reference answer"}

        system_msg = self._get_system_prompt()
        user_msg = self._get_user_prompt(question, predicted_answer, reference_answer)

        if self.mode == "openrouter":
            return self._evaluate_openrouter(system_msg, user_msg)
        elif self.mode == "local":
            return self._evaluate_local(system_msg, user_msg)
        else:
            raise ValueError(f"Unknown judge mode: {self.mode}")

    def _evaluate_openrouter(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        """Evaluate using OpenRouter API."""
        import urllib.request
        import urllib.error

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.api_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            logger.error("Judge HTTP error: %s", e)
            return {"correct": False, "reasoning": f"HTTP error: {e.code}"}
        except Exception as e:
            logger.error("Judge request error: %s", e)
            return {"correct": False, "reasoning": f"Request error: {e}"}

        try:
            data = json.loads(raw)
            content = data["choices"][0]["message"]["content"]
            # Try to parse JSON from the response
            # Sometimes LLMs wrap JSON in markdown code blocks
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

            result = json.loads(json_str)
            return {
                "correct": bool(result.get("correct", False)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        except Exception as e:
            logger.error("Failed to parse judge response: %s", e)
            return {"correct": False, "reasoning": f"Parse error: {e}"}

    def _evaluate_local(self, system_msg: str, user_msg: str) -> Dict[str, Any]:
        """Evaluate using local model (loads model, evaluates, then keeps loaded)."""
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
            # Try to parse JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()

            result = json.loads(json_str)
            return {
                "correct": bool(result.get("correct", False)),
                "reasoning": str(result.get("reasoning", "No reasoning provided")),
            }
        except Exception as e:
            logger.error("Local judge evaluation error: %s", e)
            return {"correct": False, "reasoning": f"Local eval error: {e}"}

    def unload(self):
        """Unload local model to free memory."""
        if self._local_serving is not None:
            logger.info("Unloading local judge model")
            # Force garbage collection
            import gc
            self._local_serving = None
            gc.collect()
            if hasattr(os, 'system'):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


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
        """
        Save memory state for a processed chunk.

        Returns:
            Dict with paths to saved files.
        """
        chunk_dir = self.output_dir / f"chunk_{chunk_id}"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = {}

        # 1. Save DST state (slots, deleted_facts, etc.)
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

        # 2. Save RAGU storage if available
        if pipeline.ragu_processor is not None:
            # RAGU stores data in its storage folder
            # We need to copy the entire storage directory
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

def load_longmemeval_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load LongMemEval dataset from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter relevant question types
    filtered = [
        item for item in data
        if item.get("question_type") in RELEVANT_QUESTION_TYPES
    ]

    logger.info(
        "Loaded LongMemEval dataset: %d total, %d relevant (types: %s)",
        len(data),
        len(filtered),
        ", ".join(RELEVANT_QUESTION_TYPES),
    )

    return filtered


def extract_user_messages_from_sessions(sessions: List[List[Dict]]) -> List[str]:
    """Extract all user messages from a list of sessions."""
    user_messages = []
    for session in sessions:
        for turn in session:
            if turn.get("role", "").lower() == "user":
                content = turn.get("content", "").strip()
                if content:
                    user_messages.append(content)
    return user_messages


# ============================================================================
# Pipeline Builder (from run.py)
# ============================================================================

def build_pipeline_from_config(config_path: str, cli_overrides: Optional[Dict] = None):
    """Build DSTMemoryPipeline from config file with optional CLI overrides."""
    import argparse

    from dst_memory import PipelineConfig
    from dst_memory.core.pipeline import DSTMemoryPipeline
    from dst_memory.storage.ragu_graph_processor import build_ragu_processor

    # Load config
    file_cfg = load_run_config(config_path)
    shared = shared_section(file_cfg)

    # Merge CLI overrides
    if cli_overrides:
        shared.update(cli_overrides)

    # Build PipelineConfig
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
    )

    logger.info("Initializing RAGU backend...")
    _kg, ragu_processor = build_ragu_processor(
        embedder_model=cfg.ragu_embedder_model,
        storage_path=cfg.ragu_storage_path or None,
    )

    return DSTMemoryPipeline(cfg, ragu_processor=ragu_processor)


# ============================================================================
# Main Validation Logic
# ============================================================================

def process_single_item(
    pipeline: "DSTMemoryPipeline",
    item: Dict[str, Any],
    dialogue_id: str,
    judge_client: Optional[JudgeClient],
    no_final_llm: bool = False,
) -> Dict[str, Any]:
    """
    Process a single LongMemEval item through the pipeline.

    Args:
        pipeline: Initialized DSTMemoryPipeline
        item: LongMemEval data item
        dialogue_id: Unique dialogue ID for this item
        judge_client: Optional judge client for evaluation
        no_final_llm: If True, skip final LLM call

    Returns:
        Result dictionary with all processing metadata
    """
    question = item.get("question", "")
    reference_answer = item.get("answer", "")
    question_type = item.get("question_type", "")
    sessions = item.get("haystack_sessions", [])

    # Extract and process all user messages
    user_messages = extract_user_messages_from_sessions(sessions)

    logger.info(
        "Processing item %s: %d sessions, %d user messages, type=%s",
        dialogue_id,
        len(sessions),
        len(user_messages),
        question_type,
    )

    # Write all user messages to memory
    write_logs = []
    for idx, msg in enumerate(user_messages):
        from dst_memory.core.models import Message

        log = pipeline.write_to_memory(dialogue_id, Message(role="user", content=msg))
        write_logs.append({"message_idx": idx, "log": log})

    # Get answer from pipeline
    if no_final_llm:
        answer_result = pipeline.answer_without_final_llm(dialogue_id, question)
        predicted_answer = "[no_final_llm_mode]"
    else:
        predicted_answer = pipeline.answer(dialogue_id, question)
        answer_result = pipeline.answer_without_final_llm(dialogue_id, question)

    # Evaluate with judge if available
    judge_result = None
    if judge_client is not None:
        judge_result = judge_client.evaluate(question, predicted_answer, reference_answer)

    # Build result
    result = {
        "question_id": item.get("question_id", ""),
        "question_type": question_type,
        "question": question,
        "reference_answer": reference_answer,
        "predicted_answer": predicted_answer,
        "num_sessions": len(sessions),
        "num_user_messages": len(user_messages),
        "write_logs_summary": {
            "total_messages": len(write_logs),
            "saved_messages": sum(
                1 for wl in write_logs if wl["log"].get("saved", False)
            ),
        },
        "memory_state": {
            "slots": pipeline.dst.slots_with_messages(dialogue_id),
            "expired_facts": pipeline.dst.expired_facts(dialogue_id),
            "deleted_facts_with_reasons": pipeline.dst.deleted_facts_with_reasons(dialogue_id),
        },
        "answer_details": {
            "use_memory": answer_result.get("use_memory", False),
            "memory_strategy": answer_result.get("memory_gate", {}).get(
                "memory_strategy", "unknown"
            ),
            "retrieved": answer_result.get("retrieved", []),
        },
        "judge_evaluation": judge_result,
        "correct": judge_result.get("correct", False) if judge_result else None,
    }

    return result


def run_validation(args: argparse.Namespace) -> None:
    """Main validation entry point."""
    # Setup logging
    log_file = Path(args.output_dir) / "validation.log"
    setup_logging(args.log_level, str(log_file) if args.log_file else None)

    logger.info("=" * 70)
    logger.info("LongMemEval Validation Starting")
    logger.info("=" * 70)
    logger.info("Dataset: %s", args.dataset_path)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Start index: %d", args.start_index)
    logger.info("Number of items: %d", args.num_items)
    logger.info("Judge mode: %s", args.judge_mode)
    logger.info("Config: %s", args.config)

    # Load dataset
    dataset = load_longmemeval_dataset(args.dataset_path)

    if args.start_index >= len(dataset):
        logger.error(
            "Start index %d exceeds dataset size %d", args.start_index, len(dataset)
        )
        return

    # Calculate slice
    end_index = min(args.start_index + args.num_items, len(dataset))
    dataset_slice = dataset[args.start_index:end_index]

    logger.info(
        "Processing items %d to %d (total %d)",
        args.start_index,
        end_index - 1,
        len(dataset_slice),
    )

    # Initialize persistence
    persistence = MemoryStatePersistence(args.output_dir)

    # Initialize judge client if requested
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
    cli_overrides = {}
    if args.ragu_storage_path:
        cli_overrides["ragu_storage_path"] = args.ragu_storage_path
    if args.no_final_llm:
        cli_overrides["no_final_llm"] = True

    pipeline = build_pipeline_from_config(args.config, cli_overrides)

    # Process each item
    all_results = []
    stats = {"total": 0, "correct": 0, "incorrect": 0, "no_judge": 0}

    for idx, item in enumerate(dataset_slice):
        global_idx = args.start_index + idx
        dialogue_id = f"longmemeval_{global_idx}_{item.get('question_id', 'unknown')}"

        logger.info("-" * 70)
        logger.info(
            "Processing item %d/%d (global index: %d)",
            idx + 1,
            len(dataset_slice),
            global_idx,
        )

        try:
            result = process_single_item(
                pipeline=pipeline,
                item=item,
                dialogue_id=dialogue_id,
                judge_client=judge_client,
                no_final_llm=args.no_final_llm,
            )
            result["global_index"] = global_idx
            result["slice_index"] = idx
            all_results.append(result)

            # Update stats
            stats["total"] += 1
            if result["correct"] is True:
                stats["correct"] += 1
            elif result["correct"] is False:
                stats["incorrect"] += 1
            else:
                stats["no_judge"] += 1

            # Save intermediate result
            if args.save_intermediate:
                interim_path = Path(args.output_dir) / f"result_{global_idx:04d}.json"
                with open(interim_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

            # Save chunk state
            if args.save_memory_state:
                persistence.save_chunk_state(
                    chunk_id=f"{global_idx:04d}",
                    pipeline=pipeline,
                    dialogue_id=dialogue_id,
                )

            # Clear memory for next item
            pipeline.clear_memory(dialogue_id)

            # If using local judge, optionally unload between items to save memory
            if args.judge_mode == "local" and args.unload_judge_between_items:
                judge_client.unload()

        except Exception as e:
            logger.exception("Error processing item %d: %s", global_idx, e)
            all_results.append({
                "global_index": global_idx,
                "slice_index": idx,
                "error": str(e),
                "correct": False,
            })
            stats["incorrect"] += 1

    # Final summary
    logger.info("=" * 70)
    logger.info("Validation Complete")
    logger.info("=" * 70)
    logger.info("Total items processed: %d", stats["total"])
    logger.info("Correct: %d", stats["correct"])
    logger.info("Incorrect: %d", stats["incorrect"])
    if stats["no_judge"] > 0:
        logger.info("No judge evaluation: %d", stats["no_judge"])

    if stats["total"] > 0 and stats["no_judge"] < stats["total"]:
        accuracy = stats["correct"] / (stats["total"] - stats["no_judge"])
        logger.info("Accuracy: %.2f%%", accuracy * 100)

    # Save final results
    results_path = Path(args.output_dir) / "validation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "dataset_path": args.dataset_path,
                    "start_index": args.start_index,
                    "num_items": args.num_items,
                    "judge_mode": args.judge_mode,
                    "judge_model": args.judge_model,
                    "config_path": args.config,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "statistics": stats,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Results saved to: %s", results_path)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    load_dst_memory_dotenv()

    parser = argparse.ArgumentParser(
        description="Validate GigaMemory DST pipeline on LongMemEval dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process first 10 items with openrouter judge
    python validate_longmemeval.py \\
        --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \\
        --output-dir ./results \\
        --start-index 0 \\
        --num-items 10 \\
        --judge-mode openrouter

    # Process items 20-30 with local judge
    python validate_longmemeval.py \\
        --dataset-path ../../LongMemEval/longmemeval_s_cleaned.json \\
        --output-dir ./results_local \\
        --start-index 20 \\
        --num-items 10 \\
        --judge-mode local \\
        --judge-local-model-path "meta-llama/Llama-3.2-1B-Instruct"
        """,
    )

    # Dataset and output
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to LongMemEval dataset JSON file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results and memory states",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Start index in the filtered dataset (0-based)",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=10,
        help="Number of items to process",
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default=str(dst_memory_path / "run_config.json"),
        help="Path to DST_memory run_config.json",
    )

    # Memory state saving
    parser.add_argument(
        "--save-memory-state",
        action="store_true",
        default=True,
        help="Save memory state after each chunk (default: True)",
    )
    parser.add_argument(
        "--no-save-memory-state",
        dest="save_memory_state",
        action="store_false",
        help="Disable saving memory state",
    )
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        default=True,
        help="Save intermediate results after each item (default: True)",
    )
    parser.add_argument(
        "--ragu-storage-path",
        type=str,
        default="",
        help="Custom path for RAGU storage",
    )

    # Judge configuration
    parser.add_argument(
        "--judge-mode",
        type=str,
        choices=["openrouter", "local", "none"],
        default="openrouter",
        help="Judge mode: openrouter (API), local (local model), or none (no evaluation)",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="openai/gpt-oss-120b:free",
        help="Model name for openrouter judge",
    )
    parser.add_argument(
        "--judge-api-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API URL for openrouter judge",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default="",
        help="API key for openrouter judge (or use OPENROUTER_API_KEY env var)",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for judge LLM",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="Max tokens for judge LLM",
    )
    parser.add_argument(
        "--judge-local-model-path",
        type=str,
        default="",
        help="Path to local model for local judge mode",
    )
    parser.add_argument(
        "--unload-judge-between-items",
        action="store_true",
        default=False,
        help="Unload local judge model between items to save memory",
    )

    # Pipeline options
    parser.add_argument(
        "--no-final-llm",
        action="store_true",
        default=False,
        help="Skip final LLM call (useful for testing memory without generation)",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument(
        "--log-file",
        action="store_true",
        default=True,
        help="Save log to file (default: True)",
    )
    parser.add_argument(
        "--no-log-file",
        dest="log_file",
        action="store_false",
        help="Disable saving log to file",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    run_validation(args)


if __name__ == "__main__":
    main()
