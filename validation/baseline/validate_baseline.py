"""
Baseline validation for LongMemEval dataset.

Two baseline strategies:
1. full_context: Pass ALL user and assistant messages to final LLM
2. recent_10_plus_user: Pass last 10 user/assistant pairs + remaining user messages

Metrics collected same way as GigaMemory for fair comparison:
- Accuracy (correct / total)
- Batch processing for optimization

Usage:
    python validate_baseline.py --config ./run_config.json
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Setup logging
def setup_logging(level: str, log_file: Optional[str] = None) -> None:
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
# Configuration
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from JSON file with defaults."""
    defaults = {
        "shared": {
            "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
            "output_dir": "./results",
            "start_index": 0,
            "num_items": 10,
            "log_level": "INFO",
            "log_file": True,
        },
        "baseline": {
            "strategy": "full_context",  # or "recent_10_plus_user"
            "final_llm_batch_size": 1,
            "judge_batch_size": 1,
        },
        "final_llm": {
            "mode": "openrouter",  # "openrouter", "local", "stub"
            "model": "openai/gpt-oss-120b:free",
            "api_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "temperature": 0.0,
            "max_tokens": 1024,
            "local_model_path": "",
        },
        "judge": {
            "mode": "openrouter",
            "model": "openai/gpt-oss-120b:free",
            "api_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "temperature": 0.0,
            "max_tokens": 1024,
            "local_model_path": "",
        },
    }

    if not Path(config_path).exists():
        logger.warning("Config not found: %s, using defaults", config_path)
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        # Merge
        for section in ["shared", "baseline", "final_llm", "judge"]:
            if section in user_config:
                defaults[section].update(user_config[section])

        return defaults
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return defaults


# ============================================================================
# Dataset Loading
# ============================================================================

RELEVANT_TYPES = [
    "single-session-user",
    "single-session-preference",
    "multi-session",
    "knowledge-update",
]


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load and filter LongMemEval dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [item for item in data if item.get("question_type") in RELEVANT_TYPES]
    logger.info("Loaded dataset: %d total, %d relevant", len(data), len(filtered))
    return filtered


def extract_context_full(sessions: List[List[Dict]]) -> List[Dict[str, str]]:
    """Extract ALL user and assistant messages from sessions."""
    context = []
    for session in sessions:
        for turn in session:
            role = turn.get("role", "").lower()
            content = turn.get("content", "").strip()
            if content and role in ("user", "assistant"):
                context.append({"role": role, "content": content})
    return context


def extract_context_recent_10_plus_user(sessions: List[List[Dict]]) -> List[Dict[str, str]]:
    """
    Extract context with strategy:
    - Last 10 user/assistant pairs
    - All remaining user messages from earlier sessions
    """
    # First, collect all turns with metadata
    all_turns = []
    for session_idx, session in enumerate(sessions):
        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "").lower()
            content = turn.get("content", "").strip()
            if content and role in ("user", "assistant"):
                all_turns.append({
                    "role": role,
                    "content": content,
                    "session_idx": session_idx,
                    "turn_idx": turn_idx,
                })

    if not all_turns:
        return []

    # Find last 10 pairs (20 turns if alternating)
    # A "pair" is user followed by assistant
    recent_pairs = []
    remaining_user = []

    # Start from the end and find complete pairs
    i = len(all_turns) - 1
    pairs_found = 0

    while i >= 0 and pairs_found < 10:
        # Look for assistant followed by user (going backwards)
        if all_turns[i]["role"] == "assistant" and i > 0:
            if all_turns[i - 1]["role"] == "user":
                # Found a pair
                recent_pairs.insert(0, all_turns[i - 1])  # user first
                recent_pairs.insert(0, all_turns[i])        # then assistant
                pairs_found += 1
                i -= 2
                continue
        i -= 1

    # If we didn't find 10 pairs, take what we have and continue
    # Now collect all user messages that are NOT in recent_pairs
    recent_indices = {(t["session_idx"], t["turn_idx"]) for t in recent_pairs}

    for turn in all_turns:
        if turn["role"] == "user":
            key = (turn["session_idx"], turn["turn_idx"])
            if key not in recent_indices:
                remaining_user.append({
                    "role": "user",
                    "content": turn["content"],
                })

    # Combine: remaining user messages + recent pairs
    context = remaining_user + recent_pairs

    logger.debug(
        "Context built: %d early user messages + %d recent pairs",
        len(remaining_user), len(recent_pairs) // 2
    )
    return context


# ============================================================================
# Final LLM Client (simplified from GigaMemory)
# ============================================================================

class FinalLLMClient:
    """Final LLM client for baseline (no memory context needed)."""

    def __init__(
        self,
        mode: str = "openrouter",
        api_url: str = "",
        api_key: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        local_model_path: str = "",
    ):
        self.mode = mode
        self.api_url = api_url or "https://openrouter.ai/api/v1"
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_model_path = local_model_path
        self._local_serving = None

        logger.info("FinalLLM mode=%s model=%s", mode, model if mode != "local" else local_model_path)

    def build_messages(
        self,
        question: str,
        context: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Build messages list with context and question."""
        # Format context as conversation history
        context_text = ""
        for turn in context:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            context_text += f"{role_label}: {turn['content']}\n\n"

        system = (
            "You are a helpful assistant answering questions based on conversation history.\n"
            "Use ONLY the information from the conversation to answer.\n"
            "Answer concisely and accurately."
        )

        user = (
            f"Conversation history:\n\n{context_text}\n"
            f"Question: {question}\n\n"
            f"Answer based on the conversation history above."
        )

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(self, question: str, context: List[Dict[str, str]]) -> str:
        """Generate answer using final LLM."""
        messages = self.build_messages(question, context)

        if self.mode == "stub":
            return f"[STUB] Answer to: {question[:50]}..."

        if self.mode == "openrouter":
            return self._call_openrouter(messages)

        if self.mode == "local":
            return self._call_local(messages)

        raise ValueError(f"Unknown mode: {self.mode}")

    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenRouter API."""
        import urllib.request
        import urllib.error

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
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
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error("Final LLM call failed: %s", e)
            return f"[ERROR: {e}]"

    def _call_local(self, messages: List[Dict[str, str]]) -> str:
        """Call local model."""
        # Import here to avoid dependency issues
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            if self._local_serving is None:
                logger.info("Loading local model: %s", self.local_model_path)
                tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                model = AutoModelForCausalLM.from_pretrained(
                    self.local_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )
                self._local_serving = (model, tokenizer)

            model, tokenizer = self._local_serving

            # Format messages
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return response.strip()

        except Exception as e:
            logger.error("Local LLM failed: %s", e)
            return f"[ERROR: {e}]"


# ============================================================================
# Judge Client (same as GigaMemory)
# ============================================================================

class JudgeClient:
    """LLM-as-Judge for baseline (same implementation as GigaMemory)."""

    def __init__(
        self,
        mode: str = "openrouter",
        model: str = "openai/gpt-oss-120b:free",
        api_url: str = "https://openrouter.ai/api/v1",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        local_model_path: str = "",
    ):
        self.mode = mode
        self.model = model
        self.api_url = api_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_model_path = local_model_path

    def evaluate(self, question: str, predicted: str, reference: str) -> Dict[str, Any]:
        """Evaluate answer correctness."""
        if not predicted or not reference:
            return {"correct": False, "reasoning": "Empty answer"}

        system = (
            "You are an expert evaluator. Compare predicted answer with reference.\n"
            "Respond ONLY with JSON: {\"correct\": true/false, \"reasoning\": \"...\"}"
        )

        user = (
            f"Question: {question}\n\n"
            f"Predicted: {predicted}\n\n"
            f"Reference: {reference}\n\n"
            "Are they semantically equivalent?"
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self.mode == "none":
            return {"correct": False, "reasoning": "Judge disabled"}

        try:
            if self.mode == "openrouter":
                response = self._call_openrouter(messages)
            else:
                response = self._call_local(messages)

            # Parse JSON
            result = json.loads(response)
            return {
                "correct": bool(result.get("correct", False)),
                "reasoning": str(result.get("reasoning", "No reasoning")),
            }
        except Exception as e:
            logger.error("Judge failed: %s", e)
            return {"correct": False, "reasoning": f"Error: {e}"}

    def _call_openrouter(self, messages: List[Dict[str, str]]) -> str:
        import urllib.request

        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
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

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_local(self, messages: List[Dict[str, str]]) -> str:
        # Simplified - would need full implementation
        return '{"correct": false, "reasoning": "Local judge not fully implemented"}'


# ============================================================================
# Batch Processing
# ============================================================================

@dataclass
class AccumulatedItem:
    """Item accumulated for batch processing."""
    global_index: int
    question_id: str
    question: str
    reference_answer: str
    context: List[Dict[str, str]]
    question_type: str


class BatchProcessor:
    """Batch processor for baseline (simplified from GigaMemory)."""

    def __init__(
        self,
        final_llm: FinalLLMClient,
        judge: JudgeClient,
        final_llm_batch_size: int,
        judge_batch_size: int,
    ):
        self.final_llm = final_llm
        self.judge = judge
        self.final_llm_batch_size = final_llm_batch_size
        self.judge_batch_size = judge_batch_size

        self.item_buffer: List[AccumulatedItem] = []
        self.answer_buffer: List[Tuple[AccumulatedItem, str]] = []
        self.results: List[Dict[str, Any]] = []

        self.stats = {"total": 0, "correct": 0, "incorrect": 0}

    def add_item(
        self,
        item: Dict[str, Any],
        global_index: int,
        context: List[Dict[str, str]],
    ) -> None:
        """Add item to buffer."""
        acc = AccumulatedItem(
            global_index=global_index,
            question_id=item.get("question_id", ""),
            question=item.get("question", ""),
            reference_answer=item.get("answer", ""),
            context=context,
            question_type=item.get("question_type", ""),
        )
        self.item_buffer.append(acc)

        if len(self.item_buffer) >= self.final_llm_batch_size:
            self._flush_final_llm_batch()

    def _flush_final_llm_batch(self) -> None:
        """Process accumulated items through final LLM."""
        if not self.item_buffer:
            return

        logger.info("[Batch] Processing %d items through final LLM", len(self.item_buffer))

        for item in self.item_buffer:
            try:
                answer = self.final_llm.generate(item.question, item.context)
                logger.info("[Item %d] Answer: %s...", item.global_index, answer[:100])
                self.answer_buffer.append((item, answer))
            except Exception as e:
                logger.error("[Item %d] Failed: %s", item.global_index, e)
                self.answer_buffer.append((item, f"[ERROR: {e}]"))

        self.item_buffer.clear()

        if len(self.answer_buffer) >= self.judge_batch_size:
            self._flush_judge_batch()

    def _flush_judge_batch(self) -> None:
        """Process accumulated answers through judge."""
        if not self.answer_buffer or self.judge.mode == "none":
            self.answer_buffer.clear()
            return

        logger.info("[Batch] Judging %d answers", len(self.answer_buffer))

        for item, predicted in self.answer_buffer:
            judge_result = self.judge.evaluate(
                item.question, predicted, item.reference_answer
            )

            self.results.append({
                "global_index": item.global_index,
                "question_id": item.question_id,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "predicted_answer": predicted,
                "question_type": item.question_type,
                "correct": judge_result.get("correct", False),
                "judge_evaluation": judge_result,
            })

            self.stats["total"] += 1
            if judge_result.get("correct", False):
                self.stats["correct"] += 1
            else:
                self.stats["incorrect"] += 1

        self.answer_buffer.clear()

    def finalize(self) -> Tuple[List[Dict], Dict]:
        """Flush remaining items."""
        if self.item_buffer:
            self._flush_final_llm_batch()
        if self.answer_buffer:
            self._flush_judge_batch()

        return self.results, self.stats


# ============================================================================
# Main Validation
# ============================================================================

def run_validation(config: Dict[str, Any]) -> None:
    """Main validation entry point."""
    shared = config["shared"]
    baseline = config["baseline"]
    final_llm_cfg = config["final_llm"]
    judge_cfg = config["judge"]

    # Create output directory first (before logging setup)
    output_path = Path(shared["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging (now directory exists)
    log_file = (
        output_path / "validation.log"
        if shared.get("log_file", True)
        else None
    )
    setup_logging(shared.get("log_level", "INFO"), str(log_file) if log_file else None)

    logger.info("=" * 70)
    logger.info("Baseline Validation Starting")
    logger.info("=" * 70)
    logger.info("Strategy: %s", baseline["strategy"])
    logger.info("Dataset: %s", shared["dataset_path"])
    logger.info("Output: %s", shared["output_dir"])
    logger.info("Final LLM batch: %d", baseline["final_llm_batch_size"])
    logger.info("Judge batch: %d", baseline["judge_batch_size"])

    # Load dataset
    dataset = load_dataset(shared["dataset_path"])
    start = shared.get("start_index", 0)
    end = min(start + shared.get("num_items", 10), len(dataset))
    dataset_slice = dataset[start:end]

    logger.info("Processing items %d to %d (%d total)", start, end - 1, len(dataset_slice))

    # Initialize clients
    final_llm = FinalLLMClient(
        mode=final_llm_cfg["mode"],
        api_url=final_llm_cfg.get("api_url", ""),
        api_key=final_llm_cfg.get("api_key", ""),
        model=final_llm_cfg.get("model", ""),
        temperature=final_llm_cfg.get("temperature", 0.0),
        max_tokens=final_llm_cfg.get("max_tokens", 1024),
        local_model_path=final_llm_cfg.get("local_model_path", ""),
    )

    judge = JudgeClient(
        mode=judge_cfg["mode"],
        model=judge_cfg.get("model", ""),
        api_url=judge_cfg.get("api_url", ""),
        api_key=judge_cfg.get("api_key", ""),
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 1024),
        local_model_path=judge_cfg.get("local_model_path", ""),
    )

    # Initialize batch processor
    processor = BatchProcessor(
        final_llm=final_llm,
        judge=judge,
        final_llm_batch_size=baseline["final_llm_batch_size"],
        judge_batch_size=baseline["judge_batch_size"],
    )

    # Select context extraction function
    extract_fn = (
        extract_context_full
        if baseline["strategy"] == "full_context"
        else extract_context_recent_10_plus_user
    )

    # Process items
    for idx, item in enumerate(dataset_slice):
        global_idx = start + idx
        sessions = item.get("haystack_sessions", [])

        logger.info("-" * 70)
        logger.info("Processing item %d/%d (global: %d)", idx + 1, len(dataset_slice), global_idx)

        # Extract context based on strategy
        context = extract_fn(sessions)
        logger.info("Extracted %d context turns", len(context))

        processor.add_item(item, global_idx, context)

    # Finalize
    results, stats = processor.finalize()

    # Summary
    logger.info("=" * 70)
    logger.info("Validation Complete")
    logger.info("=" * 70)
    logger.info("Total: %d", stats["total"])
    logger.info("Correct: %d", stats["correct"])
    logger.info("Incorrect: %d", stats["incorrect"])
    if stats["total"] > 0:
        logger.info("Accuracy: %.2f%%", (stats["correct"] / stats["total"]) * 100)

    # Save results (directory already created at start)
    results_file = output_path / "validation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "strategy": baseline["strategy"],
                    "dataset_path": shared["dataset_path"],
                    "start_index": start,
                    "num_items": len(dataset_slice),
                    "final_llm_mode": final_llm_cfg["mode"],
                    "final_llm_model": final_llm_cfg.get("model", ""),
                    "judge_mode": judge_cfg["mode"],
                    "judge_model": judge_cfg.get("model", ""),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                "statistics": stats,
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info("Results saved to: %s", results_file)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline validation for LongMemEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full context baseline (default)
    python validate_baseline.py --config ./run_config.json

    # Recent 10 + user messages baseline
    python validate_baseline.py --config ./run_config.json --strategy recent_10_plus_user

    # With specific output directory
    python validate_baseline.py \
        --config ./run_config.json \
        --output-dir ./results_recent10
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "run_config.json"),
        help="Path to config file (default: run_config.json)",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["full_context", "recent_10_plus_user"],
        help="Baseline strategy (overrides config)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.strategy:
        config["baseline"]["strategy"] = args.strategy
    if args.output_dir:
        config["shared"]["output_dir"] = args.output_dir

    # Run
    run_validation(config)


if __name__ == "__main__":
    main()
