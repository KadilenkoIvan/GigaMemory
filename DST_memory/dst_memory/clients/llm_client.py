from __future__ import annotations

import importlib
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from ..prompts.loader import normalize_prompt_language
from .context_limit import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    clamp_chat_messages_to_max_tokens,
)

logger = logging.getLogger(__name__)

# Bilingual prefix for judge / baseline evaluator API calls (validation scripts import this).
# Final LLM uses language-specific policy from `prompts.<ru|en>.final_llm_messages`.
CHAT_API_OUTPUT_POLICY = (
    "IMPORTANT — follow strictly:\n"
    "- Do not use tools, function calls, plugins, browsing, code execution, or any external APIs. "
    "Reply with a single plain-text assistant message only (no tool calls).\n"
    "- Keep any chain-of-thought extremely brief: "
    "the budget for the output tokens is small — you should have enough tokens for an answer, not just for reasoning.\n\n"
)


def _normalize_assistant_message_text(message: Any) -> str:
    """Turn OpenAI-style assistant `message` dict into a single string (handles null/list/reasoning)."""
    if not isinstance(message, dict):
        return ""
    raw = message.get("content")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts: List[str] = []
        for p in raw:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "text" and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p.get("text"), str):
                parts.append(p["text"])
        return "".join(parts).strip()
    if raw is not None:
        return str(raw).strip()
    for key in ("reasoning", "reasoning_content", "thinking"):
        v = message.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _torch_dtype_from_string(name: str):
    """Map config string to torch.dtype for local HF loading."""
    import torch

    n = (name or "float16").lower().strip()
    if n in ("float16", "fp16", "half"):
        return torch.float16
    if n in ("bfloat16", "bf16"):
        return torch.bfloat16
    return torch.float32


class FinalLLMClient:
    def __init__(
        self,
        mode: str = "stub",
        api_url: str = "",
        api_key: str = "",
        model: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        http_referer: str = "",
        x_title: str = "",
        prompt_language: str = "ru",
        load_dtype: str = "float16",
        enable_thinking: bool = True,
        load_quantization: str = "none",
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        attn_implementation: str = "eager",
        use_sliding_window: bool = False,
        sliding_window: Optional[int] = None,
    ):
        self.mode = (mode or "stub").lower().strip()
        self.api_url = (api_url or "").rstrip("/")
        self.api_key = (api_key or "").strip() or (
            os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        ).strip()
        self.model = model or ""
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.http_referer = http_referer or ""
        self.x_title = x_title or ""
        self.load_dtype = load_dtype or "float16"
        self.load_quantization = (load_quantization or "none").strip().lower()
        self.max_context_tokens = int(max_context_tokens)
        self.attn_implementation = (attn_implementation or "eager").strip() or "eager"
        self.use_sliding_window = bool(use_sliding_window)
        self.sliding_window = int(sliding_window) if sliding_window is not None else None
        self.enable_thinking = bool(enable_thinking)
        self._tokenizer_limit: Any = None  # lazy AutoTokenizer, or False if unavailable
        self._prompt_lang = normalize_prompt_language(prompt_language)
        self._final_llm_prompts = importlib.import_module(
            f"dst_memory.prompts.{self._prompt_lang}.final_llm_messages"
        )
        # Last sent messages (system + user) — populated on every generate() call.
        # Used for logging to *_logs.json.
        self._last_prompt_messages: List[Dict[str, str]] = []
        self._local_serving: Any = None
        logger.info(
            "FinalLLMClient initialized mode=%s model=%s temperature=%s prompt_language=%s "
            "load_dtype=%s load_quantization=%s attn_implementation=%s use_sliding_window=%s "
            "sliding_window=%s max_context_tokens=%s enable_thinking=%s",
            self.mode,
            self.model or "(none)",
            temperature,
            self._prompt_lang,
            self.load_dtype,
            self.load_quantization,
            self.attn_implementation,
            self.use_sliding_window,
            self.sliding_window,
            self.max_context_tokens,
            self.enable_thinking,
        )

    def release_local_serving(self) -> None:
        """Drop local HF model to free VRAM (called when unloading/reloading pipeline locals)."""
        if self._local_serving is None:
            return
        import gc

        logger.info("Releasing local final LLM (HF) from memory")
        if self._local_serving is not None and hasattr(self._local_serving, "release"):
            try:
                self._local_serving.release()
            except Exception as e:
                logger.warning("release() on local serving failed: %s", e)
        self._local_serving = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _tokenizer_for_prompt_limit(self):
        """Tokenizer used only to measure/clamp prompt length (lighter than full model load)."""
        if self._local_serving is not None and getattr(self._local_serving, "tokenizer", None):
            return self._local_serving.tokenizer
        if self._tokenizer_limit is False:
            return None
        if self._tokenizer_limit is not None:
            return self._tokenizer_limit
        if not (self.model or "").strip():
            self._tokenizer_limit = False
            return None
        try:
            from transformers import AutoTokenizer

            self._tokenizer_limit = AutoTokenizer.from_pretrained(
                self.model.strip(),
                trust_remote_code=True,
            )
            return self._tokenizer_limit
        except Exception as e:
            logger.warning("Could not load tokenizer for max_context_tokens: %s", e)
            self._tokenizer_limit = False
            return None

    def build_messages(
        self,
        question: str,
        memory_context: Any,
        recent_pairs: Optional[List[Dict[str, str]]] = None,
        clock_display: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Build the chat messages list without sending. Exposed for logging."""
        import datetime

        if clock_display is not None and str(clock_display).strip():
            now_str = str(clock_display).strip()
        else:
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        pm = self._final_llm_prompts
        system = pm.chat_api_output_policy() + pm.final_llm_system_prompt(now_str)

        mem_block = json.dumps(memory_context or {}, ensure_ascii=False, indent=2)
        pairs_block = (
            json.dumps(recent_pairs, ensure_ascii=False, indent=2)
            if recent_pairs
            else "[]"
        )
        user = pm.final_llm_user_prompt(now_str, mem_block, pairs_block, question)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(
        self,
        question: str,
        memory_context: Any,
        recent_pairs: Optional[List[Dict[str, str]]] = None,
        clock_display: Optional[str] = None,
    ) -> str:
        messages = self.build_messages(
            question, memory_context, recent_pairs or [], clock_display=clock_display
        )
        if self.mode != "stub" and self.max_context_tokens > 0:
            tok = self._tokenizer_for_prompt_limit()
            if tok is not None:
                messages = clamp_chat_messages_to_max_tokens(
                    tok,
                    messages,
                    self.max_context_tokens,
                    enable_thinking=self.enable_thinking,
                )
        self._last_prompt_messages = messages

        logger.info(
            "FinalLLM generate mode=%s question_len=%d has_memory=%s pairs=%d",
            self.mode,
            len(question),
            bool(memory_context),
            len(recent_pairs or []),
        )
        # Full prompt at DEBUG level (console / file handler)
        logger.debug(
            "FinalLLM full prompt:\nSYSTEM:\n%s\n\nUSER:\n%s",
            messages[0]["content"],
            messages[1]["content"],
        )

        if self.mode == "stub":
            compact = json.dumps(memory_context, ensure_ascii=False)[:300] if memory_context else "no-memory"
            return f"[STUB_ANSWER] q='{question}' | memory='{compact}'"

        if self.mode == "local":
            from .serving import GenerationConfig, LocalHFServing

            if not self.model.strip():
                raise ValueError(
                    "llm_model is empty; for llm_mode=local set llm_model to a HuggingFace model id "
                    "or a local directory path (e.g. Qwen/Qwen2.5-7B-Instruct)."
                )
            if self._local_serving is None:
                td = _torch_dtype_from_string(self.load_dtype)
                logger.info(
                    "Loading local final LLM: path_or_id=%s torch_dtype=%s",
                    self.model.strip(),
                    td,
                )
                self._local_serving = LocalHFServing(
                    self.model.strip(),
                    torch_dtype=td,
                    enable_thinking=self.enable_thinking,
                    load_quantization=self.load_quantization,
                    attn_implementation=self.attn_implementation,
                    use_sliding_window=self.use_sliding_window,
                    sliding_window=self.sliding_window,
                )
            gen_cfg = GenerationConfig(
                max_new_tokens=int(self.max_tokens),
                do_sample=float(self.temperature) > 0.0,
                temperature=float(self.temperature) if float(self.temperature) > 0.0 else 1.0,
            )
            return self._local_serving.generate_chat(messages, generation_config=gen_cfg)

        if self.mode in ("api", "openrouter"):
            return self._openai_compatible_chat(messages)

        raise ValueError(f"Unknown llm_mode: {self.mode}")

    def _chat_url(self) -> str:
        base = self.api_url
        if not base:
            base = "https://openrouter.ai/api/v1"
        if base.endswith("/chat/completions"):
            return base
        return f"{base.rstrip('/')}/chat/completions"

    def _openai_compatible_chat(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        """Call API with retry logic for transient errors."""
        import time
        import random

        if not self.api_key.strip():
            raise ValueError(
                "llm_api_key is empty; set it in run_config.json or OPENROUTER_API_KEY."
            )
        if not self.model.strip():
            raise ValueError("llm_model is empty; set it in run_config.json or --llm-model.")

        body = {
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "messages": messages,
            "tool_choice": "none",
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer.strip():
            headers["Referer"] = self.http_referer.strip()
        if self.x_title.strip():
            headers["X-OpenRouter-Title"] = self.x_title.strip()

        last_exception = None
        for attempt in range(max_retries):
            req = urllib.request.Request(
                self._chat_url(),
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8")

                data = json.loads(raw)
                err = data.get("error")
                if err:
                    msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"Final LLM API error: {msg}")

                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError("Final LLM response has no choices")
                message = choices[0].get("message") or {}
                text = _normalize_assistant_message_text(message)
                if not text:
                    logger.warning(
                        "Final LLM returned blank assistant text (model=%s); message keys=%s",
                        self.model or "(none)",
                        list(message.keys()),
                    )
                    raise RuntimeError("Final LLM returned blank text")
                return text

            except urllib.error.HTTPError as e:
                last_exception = e
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("Final LLM HTTP %d, retrying in %.1fs (attempt %d/%d)",
                                   e.code, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
                detail = e.read().decode("utf-8", errors="replace")
                logger.error("Final LLM HTTP %s: %s", e.code, detail[:500])
                raise RuntimeError(f"Final LLM request failed: HTTP {e.code}") from e
            except urllib.error.URLError as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("Final LLM URL error, retrying in %.1fs (attempt %d/%d): %s",
                                   wait_time, attempt + 1, max_retries, e)
                    time.sleep(wait_time)
                    continue
                logger.error("Final LLM URL error: %s", e)
                raise RuntimeError(f"Final LLM request failed: {e}") from e
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    logger.warning("Final LLM error, retrying in %.1fs (attempt %d/%d): %s",
                                   wait_time, attempt + 1, max_retries, e)
                    time.sleep(wait_time)
                    continue
                raise

        raise last_exception if last_exception else RuntimeError("Final LLM failed after retries")
