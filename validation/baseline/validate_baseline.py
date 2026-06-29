"""
Baseline validation for LongMemEval dataset.

Two baseline strategies:
1. full_context: Pass ALL user and assistant messages to final LLM
2. recent_10_plus_user: Pass last 10 user/assistant pairs + remaining user messages

Features:
- Timing metrics (total, per-message, min/max/p50/p95/p99)
- Retry logic (3 attempts) for HTTP errors
- Judge scoring 0-1 scale with detailed criteria
- Per-question-type metrics
- Balanced sampling across question types
- Haystack dialogue_context_chars + strategy context chars before/after tokenizer clamp (aligned with GigaMemory_full non-MHR stats)
- OpenRouter: ``reasoning`` только если явно задан в JSON; иначе поле **не** отправляется (иначе часть endpoint’ов отвечает 400 «Reasoning is mandatory … cannot be disabled» при попытке выключить)
- Несколько вопросов на строку: непустой список ``questions`` (как в validate_longmemeval)

Usage:
    python validate_baseline.py --config ./run_config.json
"""

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.error
import urllib.request

_repo_root_baseline = Path(__file__).resolve().parents[2]
_dst_memory_for_policy = _repo_root_baseline / "DST_memory"
if str(_dst_memory_for_policy) not in sys.path:
    sys.path.insert(0, str(_dst_memory_for_policy))
from dst_memory.clients.llm_client import CHAT_API_OUTPUT_POLICY  # noqa: E402

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
PUTER_OPENAI_BASE_URL = "https://api.puter.com/puterai/openai/v1"


def _messages_char_count(messages: List[Dict[str, str]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _strip_provider_thinking_blocks(text: str) -> str:
    """
    Remove common interleaved thinking / scratchpad blocks some APIs embed in ``content``.

    OpenRouter Qwen thinking-style responses may still leak fenced blocks even when
    ``reasoning.max_tokens`` is 0; judge/final answers must not treat those as the reply.
    """
    if not text or not text.strip():
        return text
    t = text
    patterns = [
        # DeepSeek / Qwen-style fenced thinking blocks (OpenRouter may leak these into ``content``)
        (
            r"\x3cthink\x3e[\s\S]*?\x3c/think\x3e",
            re.DOTALL | re.IGNORECASE,
        ),
        (
            r"\x3cthinking\x3e[\s\S]*?\x3c/thinking\x3e",
            re.DOTALL | re.IGNORECASE,
        ),
        (
            r"\x3credacted_reasoning\x3e[\s\S]*?\x3c/redacted_reasoning\x3e",
            re.DOTALL | re.IGNORECASE,
        ),
        (
            r"\x3credacted_thinking\x3e[\s\S]*?\x3c/redacted_thinking\x3e",
            re.DOTALL | re.IGNORECASE,
        ),
        (r"\[think\][\s\S]*?\[/think\]", re.DOTALL | re.IGNORECASE),
    ]
    for _ in range(8):
        prev = t
        for pat, flags in patterns:
            t = re.sub(pat, "", t, flags=flags)
        if t == prev:
            break
    return t.strip()


def _normalize_openai_assistant_text(message: Any) -> str:
    """
    Build a single string from an OpenAI-style assistant message.

    Handles null content (some reasoning / tool models), list-shaped content,
    optional reasoning-only fields when ``content`` is empty, and strips
    interleaved thinking fences occasionally returned inside ``content``.
    """
    if not isinstance(message, dict):
        return ""
    raw = message.get("content")
    out = ""
    if isinstance(raw, str):
        out = _strip_provider_thinking_blocks(raw.strip())
    elif isinstance(raw, list):
        parts: List[str] = []
        for p in raw:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p.get("text"), str):
                parts.append(p["text"])
        out = _strip_provider_thinking_blocks("".join(parts).strip())
    elif raw is not None:
        out = _strip_provider_thinking_blocks(str(raw).strip())

    if out:
        return out

    for key in ("reasoning", "reasoning_content", "thinking"):
        v = message.get(key)
        if isinstance(v, str) and v.strip():
            return _strip_provider_thinking_blocks(v.strip())
    return ""


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
                    err_body = ""
                    try:
                        if getattr(e, "fp", None) is not None:
                            raw = e.read()
                            if raw:
                                err_body = raw.decode("utf-8", errors="replace")
                    except Exception:
                        err_body = ""
                    if e.code in (400, 429, 500, 502, 503, 504):
                        wait_time = backoff_base * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning("HTTP %d error, retrying in %.1fs (attempt %d/%d)",
                                       e.code, wait_time, attempt + 1, max_retries)
                        if err_body:
                            logger.warning("HTTP %d response body: %s", e.code, err_body[:1000])
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
# Configuration
# ============================================================================

# Question type definitions with descriptions for judge
QUESTION_TYPES = {
    "single-session-user": "User mentioned a fact about themselves in one session - system should remember it",
    "single-session-preference": "User previously shared their preferences - system should use them when answering a new request",
    "multi-session": "Facts about user are scattered across multiple sessions - system should collect them together",
    "knowledge-update": "User provided new fact contradicting old one - system should return the current one",
}

RELEVANT_TYPES = list(QUESTION_TYPES.keys())


def resolve_openrouter_reasoning(section: Dict[str, Any], mode: str) -> Optional[Dict[str, Any]]:
    """
    Optional OpenRouter ``reasoning`` block on chat completions.
    https://openrouter.ai/docs/guides/best-practices/reasoning-tokens

    Many routed endpoints **require** reasoning and respond with HTTP 400 if the client sends
    ``effort: none``, ``max_tokens: 0``, etc. Therefore we **do not** inject defaults — only send
    ``reasoning`` when the JSON explicitly includes it.

    - ``reasoning`` key absent → omit field (same practical effect as omitting disable knobs).
    - ``"reasoning": null`` → omit field.
    - Non-empty object → passed through as-is (use only when the model docs say it is safe).
    """
    if str(mode or "").lower() != "openrouter":
        return None
    if "reasoning" not in section:
        return None
    raw = section.get("reasoning")
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw) if raw else None
    logger.warning("Ignoring invalid openrouter reasoning (expected object or null): %s", raw)
    return None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config from JSON file with defaults."""
    defaults = {
        "shared": {
            "dataset_path": "../../LongMemEval/longmemeval_s_cleaned.json",
            "output_dir": "./results",
            "start_index": 0,
            "num_items_per_type": 10,
            "question_types": list(QUESTION_TYPES.keys()),
            "log_level": "INFO",
            "log_file": True,
        },
        "baseline": {
            "strategy": "full_context",
            "final_llm_batch_size": 1,
            "judge_batch_size": 1,
        },
        "final_llm": {
            "mode": "openrouter",
            "model": "openai/gpt-oss-120b:free",
            "tokenizer_model": "",
            "api_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "temperature": 0.0,
            "max_tokens": 1024,
            "local_model_path": "",
            "load_dtype": "float16",
            "load_quantization": "none",
            "max_context_tokens": 131072,
        },
        "judge": {
            "mode": "openrouter",
            "model": "openai/gpt-oss-120b:free",
            "api_url": "https://openrouter.ai/api/v1",
            "api_key": "",
            "temperature": 0.0,
            "max_tokens": 1024,
            "local_model_path": "",
            "load_dtype": "float16",
            "load_quantization": "none",
        },
    }

    if not Path(config_path).exists():
        logger.warning("Config not found: %s, using defaults", config_path)
        return defaults

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = json.load(f)

        for section in ["shared", "baseline", "final_llm", "judge"]:
            if section in user_config:
                defaults[section].update(user_config[section])

        return defaults
    except Exception as e:
        logger.error("Failed to load config: %s", e)
        return defaults


# ============================================================================
# Dataset Loading with Balanced Sampling
# ============================================================================

def load_dataset_balanced(dataset_path: str, question_types: List[str],
                          num_per_type: int) -> List[Dict[str, Any]]:
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


def normalize_question_specs(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Список вопросов для одной строки датасета (как в validate_longmemeval).

    Если ``questions`` — непустой список dict, у каждого ожидаются ``question_id``,
    ``question`` и ``answer`` или ``reference_answer``. Поле ``question_type`` в
    элементе при наличии переопределяет тип строки для judge. Иначе используются
    поля строки ``question_id`` / ``question`` / ``answer`` / ``question_type``.
    """
    row_qt = str(item.get("question_type", "") or "")
    raw = item.get("questions")
    if isinstance(raw, list) and len(raw) > 0:
        specs: List[Dict[str, str]] = []
        for i, q in enumerate(raw):
            if not isinstance(q, dict):
                continue
            qid = str(q.get("question_id", "") or "").strip() or f"q_{i}"
            qtext = str(q.get("question", "") or "")
            ref = q.get("answer", q.get("reference_answer", ""))
            sub_qt = str(q.get("question_type", "") or "").strip() or row_qt
            specs.append({
                "question_id": qid,
                "question": qtext,
                "reference_answer": str(ref if ref is not None else ""),
                "question_type": sub_qt,
            })
        if specs:
            return specs
    return [{
        "question_id": str(item.get("question_id", "") or ""),
        "question": str(item.get("question", "") or ""),
        "reference_answer": str(item.get("answer", "") or ""),
        "question_type": row_qt,
    }]


def extract_context_full(sessions: List[List[Dict]]) -> List[Dict[str, str]]:
    """Extract ALL user and assistant messages from sessions."""
    context = []
    for session in sessions:
        for turn in session:
            role = turn.get("role", "").lower()
            content = (turn.get("content") or "").strip()
            if content and role in ("user", "assistant"):
                context.append({"role": role, "content": content})
    return context


def haystack_dialogue_chars(sessions: List[List[Dict]]) -> int:
    """
    Total UTF-8 character length of all turns in haystack_sessions (user + assistant).
    Matches validation/GigaMemory_full.validate_longmemeval dialogue_context_chars.
    """
    n = 0
    for session in sessions:
        if not isinstance(session, list):
            continue
        for turn in session:
            if not isinstance(turn, dict):
                continue
            n += len(str(turn.get("content") or ""))
    return n


def context_turns_char_count(context: List[Dict[str, str]]) -> int:
    """Character sum of ``content`` for baseline context turns passed to the final LLM."""
    return sum(len(str(t.get("content") or "")) for t in context)


def extract_context_recent_10_plus_user(sessions: List[List[Dict]]) -> List[Dict[str, str]]:
    """Extract: last 10 pairs + remaining user messages."""
    all_turns = []
    for session_idx, session in enumerate(sessions):
        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "").lower()
            content = (turn.get("content") or "").strip()
            if content and role in ("user", "assistant"):
                all_turns.append({
                    "role": role,
                    "content": content,
                    "session_idx": session_idx,
                    "turn_idx": turn_idx,
                })

    if not all_turns:
        return []

    recent_pairs = []
    i = len(all_turns) - 1
    pairs_found = 0

    while i >= 0 and pairs_found < 10:
        if all_turns[i]["role"] == "assistant" and i > 0:
            if all_turns[i - 1]["role"] == "user":
                recent_pairs.insert(0, all_turns[i - 1])
                recent_pairs.insert(0, all_turns[i])
                pairs_found += 1
                i -= 2
                continue
        i -= 1

    recent_indices = {(t["session_idx"], t["turn_idx"]) for t in recent_pairs}
    remaining_user = []

    for turn in all_turns:
        if turn["role"] == "user":
            key = (turn["session_idx"], turn["turn_idx"])
            if key not in recent_indices:
                remaining_user.append({"role": "user", "content": turn["content"]})

    context = remaining_user + recent_pairs
    return [{"role": t["role"], "content": t["content"]} for t in context]


# ============================================================================
# Final LLM Client with Retry
# ============================================================================

class FinalLLMClient:
    """Final LLM client with retry logic."""

    def __init__(
        self,
        mode: str = "openrouter",
        api_url: str = "",
        api_key: str = "",
        model: str = "",
        tokenizer_model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        local_model_path: str = "",
        load_dtype: str = "float16",
        load_quantization: str = "none",
        max_context_tokens: int = 131072,
        openrouter_reasoning: Optional[Dict[str, Any]] = None,
    ):
        self.mode = mode
        self.api_url = api_url or "https://openrouter.ai/api/v1"
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("PUTER_API_KEY", "")
        )
        if self.mode == "puter" and (not self.api_url or "openrouter.ai" in self.api_url):
            self.api_url = PUTER_OPENAI_BASE_URL
        self.model = model
        self.tokenizer_model = (tokenizer_model or "").strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_model_path = local_model_path or ""
        self.load_dtype = load_dtype or "float16"
        self.load_quantization = (load_quantization or "none").strip().lower()
        self.max_context_tokens = int(max_context_tokens)
        self._hf_serving = None
        self._tok_limit: Any = None  # AutoTokenizer or False
        self._last_prompt_chars_before_clamp = 0
        self._last_prompt_chars_after_clamp = 0
        self._last_conversation_chars_before_clamp = 0
        self._last_conversation_chars_after_clamp = 0
        self.openrouter_reasoning = openrouter_reasoning

        # local: from_pretrained(local_model_path or model) — HF repo id or disk path
        _resolved = (self.local_model_path or "").strip() or (self.model or "").strip()
        logger.info(
            "FinalLLM mode=%s pretrained=%s tokenizer_model=%s load_dtype=%s load_quantization=%s max_context_tokens=%s",
            mode,
            _resolved or "(empty)",
            self.tokenizer_model or "(auto)",
            self.load_dtype,
            self.load_quantization,
            self.max_context_tokens,
        )

    def build_messages(self, question: str, context: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Build messages with full context (no truncation)."""
        context_text = ""
        for turn in context:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            context_text += f"{role_label}: {turn['content']}\n\n"

        system = (
            CHAT_API_OUTPUT_POLICY
            + "You are a helpful assistant answering questions based on conversation history.\n"
            "Use ONLY the information from the conversation to answer.\n"
            "Answer concisely and accurately without reasoning."
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

    @retry_with_backoff(max_retries=3)
    def _call_api(self, messages: List[Dict[str, str]]) -> str:
        """Call API with retry."""
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if self.mode == "openrouter" and self.openrouter_reasoning:
            body["reasoning"] = self.openrouter_reasoning

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
            choices = data.get("choices") or []
            if not choices:
                return ""
            text = _normalize_openai_assistant_text(choices[0].get("message") or {})
            if not text:
                logger.warning(
                    "Final LLM returned empty assistant text (model=%s); raw message keys=%s",
                    self.model,
                    list((choices[0].get("message") or {}).keys()),
                )
            return text

    def _get_limit_tokenizer(self):
        if self._tok_limit is False:
            return None
        if self._tok_limit is not None:
            return self._tok_limit
        pid = (self.tokenizer_model or "").strip() or self._local_pretrained_id()
        if not pid:
            self._tok_limit = False
            return None
        try:
            from transformers import AutoTokenizer

            self._tok_limit = AutoTokenizer.from_pretrained(pid, trust_remote_code=True)
            return self._tok_limit
        except Exception as e:
            logger.warning("Tokenizer for max_context_tokens unavailable: %s", e)
            self._tok_limit = False
            return None

    def generate(self, question: str, context: List[Dict[str, str]]) -> Tuple[str, Optional[str]]:
        """Generate answer. Returns (answer, error)."""
        ctx_full: List[Dict[str, str]] = list(context)
        self._last_conversation_chars_before_clamp = context_turns_char_count(ctx_full)
        messages_before = self.build_messages(question, ctx_full)
        self._last_prompt_chars_before_clamp = _messages_char_count(messages_before)
        ctx: List[Dict[str, str]] = list(context)
        if self.max_context_tokens > 0 and self.mode != "stub":
            tok = self._get_limit_tokenizer()
            if tok is not None:
                from dst_memory.clients.context_limit import truncate_baseline_dialogue_turns

                ctx = truncate_baseline_dialogue_turns(
                    question,
                    ctx,
                    self.build_messages,
                    tok,
                    self.max_context_tokens,
                )
        self._last_conversation_chars_after_clamp = context_turns_char_count(ctx)
        messages = self.build_messages(question, ctx)
        self._last_prompt_chars_after_clamp = _messages_char_count(messages)

        if self.mode == "stub":
            return f"[STUB] Answer to: {question[:50]}...", None

        if self.mode in ("openrouter", "puter", "api"):
            try:
                return self._call_api(messages), None
            except Exception as e:
                logger.error("Final LLM failed after retries: %s", e)
                return "", str(e)

        if self.mode == "local":
            return self._call_local(messages)

        return "", f"Unknown mode: {self.mode}"

    def get_last_prompt_char_stats(self) -> Dict[str, int]:
        return {
            "before_clamp_chars": int(self._last_prompt_chars_before_clamp),
            "after_clamp_chars": int(self._last_prompt_chars_after_clamp),
            "conversation_chars_before_clamp": int(self._last_conversation_chars_before_clamp),
            "conversation_chars_after_clamp": int(self._last_conversation_chars_after_clamp),
        }

    def _local_pretrained_id(self) -> str:
        return (self.local_model_path or "").strip() or (self.model or "").strip()

    def release_local_model(self) -> None:
        """Drop local HF weights (call before loading judge on the same GPU)."""
        if self._hf_serving is not None:
            try:
                self._hf_serving.release()
            except Exception as e:
                logger.warning("FinalLLM LocalHFServing.release failed: %s", e)
            self._hf_serving = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _call_local(self, messages: List[Dict[str, str]]) -> Tuple[str, Optional[str]]:
        try:
            from dst_memory.clients.llm_client import _torch_dtype_from_string
            from dst_memory.clients.serving import GenerationConfig, LocalHFServing

            pretrained = self._local_pretrained_id()
            if not pretrained:
                return "", (
                    "local mode: set final_llm.model (HF repo id) or final_llm.local_model_path "
                    "(directory); both were empty"
                )

            if self._hf_serving is None:
                td = _torch_dtype_from_string(self.load_dtype)
                logger.info("Loading local final LLM: %s dtype=%s quant=%s", pretrained, td, self.load_quantization)
                self._hf_serving = LocalHFServing(
                    pretrained,
                    torch_dtype=td,
                    enable_thinking=False,
                    load_quantization=self.load_quantization,
                )
                self._tok_limit = self._hf_serving.tokenizer

            gen_cfg = GenerationConfig(
                max_new_tokens=int(self.max_tokens),
                do_sample=float(self.temperature) > 0.0,
                temperature=float(self.temperature) if float(self.temperature) > 0.0 else 1.0,
            )
            text = self._hf_serving.generate_chat(messages, generation_config=gen_cfg)
            return (text or "").strip(), None

        except Exception as e:
            return "", str(e)


# ============================================================================
# Judge Client with 0-1 Scoring
# ============================================================================

class JudgeClient:
    """LLM-as-Judge with 0-1 scoring scale."""

    def __init__(
        self,
        mode: str = "openrouter",
        model: str = "openai/gpt-oss-120b:free",
        api_url: str = "https://openrouter.ai/api/v1",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        local_model_path: str = "",
        load_dtype: str = "float16",
        load_quantization: str = "none",
        openrouter_reasoning: Optional[Dict[str, Any]] = None,
    ):
        self.mode = mode
        self.model = model
        self.api_url = api_url or "https://openrouter.ai/api/v1"
        self.api_key = (
            api_key
            or os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("PUTER_API_KEY", "")
        )
        if self.mode == "puter" and (not self.api_url or "openrouter.ai" in self.api_url):
            self.api_url = PUTER_OPENAI_BASE_URL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.local_model_path = local_model_path or ""
        self.load_dtype = load_dtype or "float16"
        self.load_quantization = (load_quantization or "none").strip().lower()
        self._hf_serving = None
        self.openrouter_reasoning = openrouter_reasoning

        _resolved = (self.local_model_path or "").strip() or (self.model or "").strip()
        logger.info(
            "Judge mode=%s pretrained=%s load_dtype=%s load_quantization=%s",
            mode,
            _resolved or "(empty)",
            self.load_dtype,
            self.load_quantization,
        )

    def _local_pretrained_id(self) -> str:
        return (self.local_model_path or "").strip() or (self.model or "").strip()

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

    @retry_with_backoff(max_retries=3)
    def _call_judge_api(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call judge API with retry."""
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if self.mode == "openrouter" and self.openrouter_reasoning:
            body["reasoning"] = self.openrouter_reasoning

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
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError("Judge response has no choices")
            content = _normalize_openai_assistant_text(choices[0].get("message") or {})
            if not content:
                raise RuntimeError("Judge returned empty assistant content")

            # Parse JSON
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()

            result = json.loads(json_str)
            return {
                "score": float(result.get("score", 0)),
                "reasoning": str(result.get("reasoning", "No reasoning")),
            }

    def release_local_model(self) -> None:
        if self._hf_serving is not None:
            try:
                self._hf_serving.release()
            except Exception as e:
                logger.warning("Judge LocalHFServing.release failed: %s", e)
            self._hf_serving = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _call_judge_local(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        from dst_memory.clients.llm_client import _torch_dtype_from_string
        from dst_memory.clients.serving import GenerationConfig, LocalHFServing

        pretrained = self._local_pretrained_id()
        if not pretrained:
            raise RuntimeError(
                "local mode: set judge.model (HF repo id) or judge.local_model_path; both were empty"
            )

        if self._hf_serving is None:
            td = _torch_dtype_from_string(self.load_dtype)
            logger.info("Loading local judge model: %s dtype=%s quant=%s", pretrained, td, self.load_quantization)
            self._hf_serving = LocalHFServing(
                pretrained,
                torch_dtype=td,
                enable_thinking=False,
                load_quantization=self.load_quantization,
            )

        gen_cfg = GenerationConfig(
            max_new_tokens=int(self.max_tokens),
            do_sample=float(self.temperature) > 0.0,
            temperature=float(self.temperature) if float(self.temperature) > 0.0 else 1.0,
        )
        content = self._hf_serving.generate_chat(messages, generation_config=gen_cfg).strip()
        if not content:
            raise RuntimeError("Judge (local) returned empty text")

        json_str = content
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].split("```")[0].strip()

        result = json.loads(json_str)
        return {
            "score": float(result.get("score", 0)),
            "reasoning": str(result.get("reasoning", "No reasoning")),
        }

    def evaluate(self, question: str, predicted: str, reference: str,
                 question_type: str) -> Tuple[float, str, Optional[str]]:
        """
        Evaluate answer. Returns (score, reasoning, error).
        Score: 0.0 to 1.0
        """
        if not predicted or not reference:
            return 0.0, "Empty answer or reference", None

        if self.mode == "none":
            return 0.0, "Judge disabled", None

        system = self._get_system_prompt_answer_correctness(question_type)
        user = (
            f"Question: {question}\n\n"
            f"Reference Answer: {reference}\n\n"
            f"Predicted Answer: {predicted}\n\n"
            f"Score the predicted answer's coverage of the reference."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        try:
            if self.mode == "local":
                result = self._call_judge_local(messages)
            else:
                result = self._call_judge_api(messages)
            return result["score"], result["reasoning"], None
        except Exception as e:
            logger.error("Judge failed after retries: %s", e)
            return 0.0, f"Error: {e}", str(e)


# ============================================================================
# Batch Processing with Timing
# ============================================================================

def _atomic_write_validation_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


@dataclass
class AccumulatedItem:
    """Один прогон final LLM + judge; при нескольких вопросах в строке их несколько."""
    global_index: int
    dialogue_row_index: int
    question_sub_index: int
    question_id: str
    question: str
    reference_answer: str
    question_type: str
    context: List[Dict[str, str]]
    num_messages: int
    dialogue_context_chars: int = 0


class BatchProcessor:
    def __init__(
        self,
        final_llm: FinalLLMClient,
        judge: JudgeClient,
        final_llm_batch_size: int,
        judge_batch_size: int,
        timing: TimingStats,
        results_json_path: Optional[Path] = None,
        validation_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.final_llm = final_llm
        self.judge = judge
        self.final_llm_batch_size = final_llm_batch_size
        self.judge_batch_size = judge_batch_size
        self.timing = timing
        self._results_json_path = results_json_path
        self._validation_metadata = validation_metadata

        self.item_buffer: List[AccumulatedItem] = []
        self.answer_buffer: List[
            Tuple[AccumulatedItem, str, Optional[str], int, int, int, int]
        ] = []
        self.results: List[Dict[str, Any]] = []

        self.stats = {
            "total": 0,
            "errors_final_llm": 0,
            "errors_judge": 0,
            "final_llm_calls": 0,
            "final_llm_prompt_chars_before_clamp_total": 0,
            "final_llm_prompt_chars_after_clamp_total": 0,
            "dialogue_context_chars_total": 0,
            "dialogue_context_chars_count": 0,
            "conversation_context_chars_before_clamp_total": 0,
            "conversation_context_chars_after_clamp_total": 0,
            "by_type": {qt: {"count": 0, "total_score": 0.0, "errors": 0}
                        for qt in QUESTION_TYPES.keys()},
        }

    def add_item(
        self,
        item: Dict[str, Any],
        global_idx: int,
        context: List[Dict[str, str]],
        num_messages: int,
        *,
        dialogue_row_index: int,
        question_sub_index: int,
        dialogue_context_chars: int = 0,
    ) -> None:
        start_time = time.time()

        acc = AccumulatedItem(
            global_index=global_idx,
            dialogue_row_index=dialogue_row_index,
            question_sub_index=question_sub_index,
            question_id=str(item.get("question_id", "") or ""),
            question=str(item.get("question", "") or ""),
            reference_answer=str(item.get("answer", "") or ""),
            question_type=str(item.get("question_type", "") or ""),
            context=context,
            num_messages=num_messages,
            dialogue_context_chars=int(dialogue_context_chars),
        )
        self.item_buffer.append(acc)

        if len(self.item_buffer) >= self.final_llm_batch_size:
            self._flush_final_llm_batch()

        processing_time = time.time() - start_time
        self.timing.add_item(num_messages, processing_time)

    def _flush_final_llm_batch(self) -> None:
        if not self.item_buffer:
            return

        logger.info("[Batch] Processing %d items through final LLM", len(self.item_buffer))

        for item in self.item_buffer:
            answer, error = self.final_llm.generate(item.question, item.context)
            char_stats = self.final_llm.get_last_prompt_char_stats()
            prompt_chars_before = int(char_stats.get("before_clamp_chars", 0))
            prompt_chars_after = int(char_stats.get("after_clamp_chars", 0))
            conv_before = int(char_stats.get("conversation_chars_before_clamp", 0))
            conv_after = int(char_stats.get("conversation_chars_after_clamp", 0))
            self.stats["final_llm_calls"] += 1
            self.stats["final_llm_prompt_chars_before_clamp_total"] += prompt_chars_before
            self.stats["final_llm_prompt_chars_after_clamp_total"] += prompt_chars_after
            self.stats["conversation_context_chars_before_clamp_total"] += conv_before
            self.stats["conversation_context_chars_after_clamp_total"] += conv_after

            if error:
                logger.error("[Item %d] Final LLM error: %s", item.global_index, error)
                self.stats["errors_final_llm"] += 1

            logger.info("[Item %d] Answer: %s", item.global_index, answer)
            self.answer_buffer.append(
                (item, answer, error, prompt_chars_before, prompt_chars_after, conv_before, conv_after)
            )

        self.item_buffer.clear()

        if (
            self.final_llm.mode == "local"
            and self.judge.mode == "local"
            and hasattr(self.final_llm, "release_local_model")
        ):
            logger.info("[Batch] Releasing local final LLM before local judge (VRAM)")
            self.final_llm.release_local_model()

        if len(self.answer_buffer) >= self.judge_batch_size:
            self._flush_judge_batch()

    def _flush_judge_batch(self) -> None:
        if not self.answer_buffer or self.judge.mode == "none":
            self.answer_buffer.clear()
            return

        logger.info("[Batch] Judging %d answers", len(self.answer_buffer))

        for (
            item,
            predicted,
            final_llm_error,
            prompt_chars_before,
            prompt_chars_after,
            conv_chars_before,
            conv_chars_after,
        ) in self.answer_buffer:
            score, reasoning, judge_error = self.judge.evaluate(
                item.question, predicted, item.reference_answer, item.question_type
            )

            if judge_error:
                logger.error("[Item %d] Judge error: %s", item.global_index, judge_error)
                self.stats["errors_judge"] += 1

            # Update per-type stats
            qt = item.question_type
            if qt in self.stats["by_type"]:
                self.stats["by_type"][qt]["count"] += 1
                self.stats["by_type"][qt]["total_score"] += score
                if judge_error or final_llm_error:
                    self.stats["by_type"][qt]["errors"] += 1

            self.results.append({
                "global_index": item.global_index,
                "dialogue_row_index": item.dialogue_row_index,
                "question_sub_index": item.question_sub_index,
                "question_id": item.question_id,
                "question": item.question,
                "reference_answer": item.reference_answer,
                "predicted_answer": predicted,
                "question_type": item.question_type,
                "score": score,
                "reasoning": reasoning,
                "final_llm_error": final_llm_error,
                "judge_error": judge_error,
                "dialogue_context_chars": int(item.dialogue_context_chars or 0),
                "conversation_context_chars_before_clamp": conv_chars_before,
                "conversation_context_chars_after_clamp": conv_chars_after,
                "final_llm_prompt_chars_before_clamp": prompt_chars_before,
                "final_llm_prompt_chars_after_clamp": prompt_chars_after,
            })

            self.stats["total"] += 1
            self.stats["dialogue_context_chars_total"] += int(item.dialogue_context_chars or 0)
            self.stats["dialogue_context_chars_count"] += 1

            self._write_results_json_snapshot()

        self.answer_buffer.clear()

        if self.judge.mode == "local" and hasattr(self.judge, "release_local_model"):
            logger.info("[Batch] Releasing local judge model after judge batch")
            self.judge.release_local_model()

    def _write_results_json_snapshot(self) -> None:
        if self._results_json_path is None or self._validation_metadata is None:
            return
        by_type = copy.deepcopy(self.stats["by_type"])
        for st in by_type.values():
            if isinstance(st, dict) and st.get("count", 0) > 0:
                st["average_score"] = st["total_score"] / st["count"]
        avg_score = (
            sum(r["score"] for r in self.results) / len(self.results)
            if self.results else 0.0
        )
        timing_stats = self.timing.get_stats()
        if timing_stats.get("total_time", 0) == 0 and self.timing.total_start is not None:
            timing_stats = dict(timing_stats)
            timing_stats["total_time"] = time.time() - self.timing.total_start

        calls = int(self.stats["final_llm_calls"] or 0)
        d_count = int(self.stats["dialogue_context_chars_count"] or 0)
        d_total = int(self.stats["dialogue_context_chars_total"] or 0)
        conv_b_total = int(self.stats["conversation_context_chars_before_clamp_total"] or 0)
        conv_a_total = int(self.stats["conversation_context_chars_after_clamp_total"] or 0)
        prompt_b_total = int(self.stats["final_llm_prompt_chars_before_clamp_total"] or 0)
        prompt_a_total = int(self.stats["final_llm_prompt_chars_after_clamp_total"] or 0)

        payload = {
            "metadata": self._validation_metadata,
            "statistics": {
                "total": self.stats["total"],
                "errors_final_llm": self.stats["errors_final_llm"],
                "errors_judge": self.stats["errors_judge"],
                "average_score": avg_score,
                "final_llm_calls": calls,
                "final_llm_prompt_chars_before_clamp_total": prompt_b_total,
                "final_llm_prompt_chars_after_clamp_total": prompt_a_total,
                "final_llm_prompt_chars_before_clamp_avg": (
                    prompt_b_total / calls if calls > 0 else 0.0
                ),
                "final_llm_prompt_chars_after_clamp_avg": (
                    prompt_a_total / calls if calls > 0 else 0.0
                ),
                "dialogue_context_chars": {
                    "count": d_count,
                    "total": d_total,
                    "avg": (d_total / d_count) if d_count > 0 else 0.0,
                },
                "conversation_context_chars": {
                    "calls": calls,
                    "before_clamp_total": conv_b_total,
                    "after_clamp_total": conv_a_total,
                    "before_clamp_avg": (conv_b_total / calls) if calls > 0 else 0.0,
                    "after_clamp_avg": (conv_a_total / calls) if calls > 0 else 0.0,
                },
                "final_llm_prompt_chars": {
                    "calls": calls,
                    "before_clamp_total": prompt_b_total,
                    "after_clamp_total": prompt_a_total,
                    "before_clamp_avg": (prompt_b_total / calls) if calls > 0 else 0.0,
                    "after_clamp_avg": (prompt_a_total / calls) if calls > 0 else 0.0,
                },
                "by_type": by_type,
            },
            "timing": timing_stats,
            "results": list(self.results),
        }
        _atomic_write_validation_json(self._results_json_path, payload)

    def finalize(self) -> Tuple[List[Dict], Dict, Dict]:
        if self.item_buffer:
            self._flush_final_llm_batch()
        if self.answer_buffer:
            self._flush_judge_batch()

        # Compute per-type averages
        for qt in self.stats["by_type"]:
            count = self.stats["by_type"][qt]["count"]
            if count > 0:
                self.stats["by_type"][qt]["average_score"] = (
                    self.stats["by_type"][qt]["total_score"] / count
                )

        self._write_results_json_snapshot()
        return self.results, self.stats, self.timing.get_stats()


# ============================================================================
# Main Validation
# ============================================================================

def run_validation(config: Dict[str, Any]) -> None:
    shared = config["shared"]
    baseline = config["baseline"]
    final_llm_cfg = config["final_llm"]
    judge_cfg = config["judge"]

    # Create output directory
    output_path = Path(shared["output_dir"])
    output_path.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = output_path / "validation.log" if shared.get("log_file", True) else None
    setup_logging(shared.get("log_level", "INFO"), str(log_file) if log_file else None)

    # Initialize timing
    timing = TimingStats()
    timing.start_total()

    logger.info("=" * 70)
    logger.info("Baseline Validation Starting")
    logger.info("=" * 70)
    logger.info("Strategy: %s", baseline["strategy"])
    logger.info("Dataset: %s", shared["dataset_path"])
    logger.info("Output: %s", shared["output_dir"])
    logger.info("Question types: %s", shared.get("question_types", list(QUESTION_TYPES.keys())))
    logger.info("Items per type: %d", shared.get("num_items_per_type", 10))

    # Load dataset with balanced sampling
    dataset = load_dataset_balanced(
        shared["dataset_path"],
        shared.get("question_types", list(QUESTION_TYPES.keys())),
        shared.get("num_items_per_type", 10),
    )

    logger.info("Total items to process: %d", len(dataset))

    run_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    results_path = output_path / "validation_results.json"
    fin_reasoning = resolve_openrouter_reasoning(final_llm_cfg, final_llm_cfg.get("mode", ""))
    judge_reasoning = resolve_openrouter_reasoning(judge_cfg, judge_cfg.get("mode", ""))
    validation_metadata = {
        "strategy": baseline["strategy"],
        "dataset_path": shared["dataset_path"],
        "num_items": len(dataset),
        "question_types": shared.get("question_types", list(QUESTION_TYPES.keys())),
        "final_llm_mode": final_llm_cfg["mode"],
        "final_llm_model": final_llm_cfg.get("model", ""),
        "judge_mode": judge_cfg["mode"],
        "judge_model": judge_cfg.get("model", ""),
        "openrouter_reasoning_final_llm": fin_reasoning,
        "openrouter_reasoning_judge": judge_reasoning,
        "timestamp": run_timestamp,
    }

    # Initialize clients
    final_llm = FinalLLMClient(
        mode=final_llm_cfg["mode"],
        api_url=final_llm_cfg.get("api_url", ""),
        api_key=final_llm_cfg.get("api_key", ""),
        model=final_llm_cfg.get("model", ""),
        tokenizer_model=final_llm_cfg.get("tokenizer_model", ""),
        temperature=final_llm_cfg.get("temperature", 0.0),
        max_tokens=final_llm_cfg.get("max_tokens", 1024),
        local_model_path=final_llm_cfg.get("local_model_path", ""),
        load_dtype=final_llm_cfg.get("load_dtype", "float16"),
        load_quantization=final_llm_cfg.get("load_quantization", "none"),
        max_context_tokens=int(final_llm_cfg.get("max_context_tokens", 131072)),
        openrouter_reasoning=fin_reasoning,
    )

    judge = JudgeClient(
        mode=judge_cfg["mode"],
        model=judge_cfg.get("model", ""),
        api_url=judge_cfg.get("api_url", ""),
        api_key=judge_cfg.get("api_key", ""),
        temperature=judge_cfg.get("temperature", 0.0),
        max_tokens=judge_cfg.get("max_tokens", 1024),
        local_model_path=judge_cfg.get("local_model_path", ""),
        load_dtype=judge_cfg.get("load_dtype", "float16"),
        load_quantization=judge_cfg.get("load_quantization", "none"),
        openrouter_reasoning=judge_reasoning,
    )

    # Initialize processor
    processor = BatchProcessor(
        final_llm=final_llm,
        judge=judge,
        final_llm_batch_size=baseline["final_llm_batch_size"],
        judge_batch_size=baseline["judge_batch_size"],
        timing=timing,
        results_json_path=results_path,
        validation_metadata=validation_metadata,
    )

    # Select extraction function
    extract_fn = (
        extract_context_full
        if baseline["strategy"] == "full_context"
        else extract_context_recent_10_plus_user
    )

    # Process items (one row may yield several questions via ``questions``[])
    eval_seq = 0
    for idx, item in enumerate(dataset):
        sessions = item.get("haystack_sessions", [])
        num_messages = sum(len(s) for s in sessions)
        specs = normalize_question_specs(item)

        logger.info("-" * 70)
        logger.info("Processing row %d/%d (%d question(s))", idx + 1, len(dataset), len(specs))
        logger.info("Question type: %s", item.get("question_type", "unknown"))
        if len(specs) == 1:
            logger.info("Question: %s", specs[0].get("question", ""))
        else:
            for si, sp in enumerate(specs):
                logger.info("  [%d] id=%s q=%s", si, sp.get("question_id", ""), sp.get("question", "")[:120])

        context = extract_fn(sessions)
        dialogue_dataset_chars = haystack_dialogue_chars(sessions)
        logger.info("Extracted %d context turns from %d messages",
                    len(context), num_messages)

        for sub_i, spec in enumerate(specs):
            row_item = dict(item)
            row_item["question_id"] = spec["question_id"]
            row_item["question"] = spec["question"]
            row_item["answer"] = spec["reference_answer"]
            row_item["question_type"] = spec.get("question_type", item.get("question_type", ""))
            processor.add_item(
                row_item,
                eval_seq,
                context,
                num_messages,
                dialogue_row_index=idx,
                question_sub_index=sub_i,
                dialogue_context_chars=dialogue_dataset_chars,
            )
            eval_seq += 1

    # Finalize
    timing.end_total()
    results, stats, timing_stats = processor.finalize()

    # Summary
    logger.info("=" * 70)
    logger.info("Validation Complete")
    logger.info("=" * 70)
    logger.info("Total processed: %d", stats["total"])
    logger.info("Final LLM errors: %d", stats["errors_final_llm"])
    logger.info("Judge errors: %d", stats["errors_judge"])
    logger.info(
        "Final LLM prompt chars: before_clamp_total=%d after_clamp_total=%d calls=%d",
        stats.get("final_llm_prompt_chars_before_clamp_total", 0),
        stats.get("final_llm_prompt_chars_after_clamp_total", 0),
        stats.get("final_llm_calls", 0),
    )
    dc_n = int(stats.get("dialogue_context_chars_count", 0) or 0)
    dc_tot = int(stats.get("dialogue_context_chars_total", 0) or 0)
    logger.info(
        "Haystack dialogue chars (dataset row, all turns): total=%d count=%d avg=%.1f",
        dc_tot,
        dc_n,
        (dc_tot / dc_n) if dc_n > 0 else 0.0,
    )
    fcalls = int(stats.get("final_llm_calls", 0) or 0)
    cb = int(stats.get("conversation_context_chars_before_clamp_total", 0) or 0)
    ca = int(stats.get("conversation_context_chars_after_clamp_total", 0) or 0)
    logger.info(
        "Extracted context chars (strategy turns; token clamp): before_total=%d after_total=%d calls=%d",
        cb,
        ca,
        fcalls,
    )

    # Per-type summary
    logger.info("\nPer-question-type results:")
    for qt, qt_stats in stats["by_type"].items():
        if qt_stats["count"] > 0:
            avg = qt_stats.get("average_score", 0)
            logger.info("  %s: %d items, avg score=%.3f, errors=%d",
                        qt, qt_stats["count"], avg, qt_stats["errors"])

    # Overall average score
    total_score = sum(r["score"] for r in results)
    avg_score = total_score / len(results) if results else 0
    logger.info("\nOverall average score: %.3f", avg_score)

    # Timing summary
    logger.info("\nTiming statistics:")
    logger.info("  Total time: %.2fs", timing_stats.get("total_time", 0))
    if "time_per_item" in timing_stats:
        t = timing_stats["time_per_item"]
        logger.info("  Per item: min=%.3fs, max=%.3fs, p50=%.3fs, p95=%.3fs, p99=%.3fs",
                    t.get("min", 0), t.get("max", 0), t.get("p50", 0),
                    t.get("p95", 0), t.get("p99", 0))
    if "time_per_message" in timing_stats:
        t = timing_stats["time_per_message"]
        logger.info("  Per message: min=%.3fs, max=%.3fs, p50=%.3fs, p95=%.3fs, p99=%.3fs",
                    t.get("min", 0), t.get("max", 0), t.get("p50", 0),
                    t.get("p95", 0), t.get("p99", 0))

    logger.info("\nResults saved to: %s", results_path)


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Baseline validation for LongMemEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file
    python validate_baseline.py --config ./run_config.json

    # Full context strategy
    python validate_baseline.py --strategy full_context --output-dir ./results_full

    # Recent 10 + user strategy
    python validate_baseline.py --strategy recent_10_plus_user --output-dir ./results_recent10
        """,
    )

    parser.add_argument("--config", type=str,
                        default=str(Path(__file__).parent / "run_config.json"),
                        help="Path to config file")

    parser.add_argument("--strategy", type=str,
                        choices=["full_context", "recent_10_plus_user"],
                        help="Baseline strategy (overrides config)")

    parser.add_argument("--output-dir", type=str,
                        help="Output directory (overrides config)")

    args = parser.parse_args()

    config = load_config(args.config)

    if args.strategy:
        config["baseline"]["strategy"] = args.strategy
    if args.output_dir:
        config["shared"]["output_dir"] = args.output_dir

    run_validation(config)


if __name__ == "__main__":
    main()
