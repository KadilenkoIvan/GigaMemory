"""
Limit prompt length in tokenizer space to avoid CUDA OOM on very long inputs.

Default cap: 128k tokens (input prompt only; generation uses max_new_tokens separately).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)

# Maximum prompt tokens (chat template → encode) before inference.
DEFAULT_MAX_CONTEXT_TOKENS = 128 * 1024


def count_prompt_tokens_chat(
    tokenizer,
    messages: List[Dict[str, str]],
    *,
    enable_thinking: bool = False,
) -> int:
    """Token count for the same string LocalHFServing.generate_chat would build (pre-generation)."""
    msgs = list(messages)
    try:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
        )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return len(ids)


def truncate_baseline_dialogue_turns(
    question: str,
    context: List[Dict[str, str]],
    build_messages: Callable[[str, List[Dict[str, str]]], List[Dict[str, str]]],
    tokenizer,
    max_prompt_tokens: int,
) -> List[Dict[str, str]]:
    """
    Drop oldest user/assistant turns until the full chat template is <= max_prompt_tokens.
    If one huge turn remains, shrinks that turn's text from the left in coarse steps.
    """
    if max_prompt_tokens <= 0:
        return list(context)

    ctx: List[Dict[str, str]] = [dict(t) for t in context]
    dropped = 0
    for _ in range(len(context) + 5):
        messages = build_messages(question, ctx)
        n = count_prompt_tokens_chat(tokenizer, messages, enable_thinking=False)
        if n <= max_prompt_tokens:
            if dropped:
                logger.warning(
                    "Context truncated: removed %d oldest turn(s); prompt_tokens=%d (cap=%d)",
                    dropped,
                    n,
                    max_prompt_tokens,
                )
            return ctx
        if len(ctx) > 1:
            ctx = ctx[1:]
            dropped += 1
            continue
        # Single (or last) turn still too long — trim content from the start
        if not ctx:
            break
        content = str(ctx[0].get("content") or "")
        if len(content) < 256:
            logger.error(
                "Prompt still %d tokens with short content; cap=%d — check tokenizer/model",
                n,
                max_prompt_tokens,
            )
            return ctx
        step = max(256, len(content) // 10)
        ctx[0] = {**ctx[0], "content": content[step:]}
        logger.warning(
            "Context truncated: shortened oldest turn from the left (step=%d chars); tokens≈%d cap=%d",
            step,
            n,
            max_prompt_tokens,
        )
    return ctx


def clamp_chat_messages_to_max_tokens(
    tokenizer,
    messages: List[Dict[str, str]],
    max_prompt_tokens: int,
    *,
    enable_thinking: bool = False,
) -> List[Dict[str, str]]:
    """
    Shorten the last user message from the left until the template fits in max_prompt_tokens.
    Used for GigaMemory final-LLM messages (system + long user with memory JSON).
    """
    if max_prompt_tokens <= 0 or len(messages) < 2:
        return [dict(m) for m in messages]

    out: List[Dict[str, str]] = [dict(m) for m in messages]
    user_idx = next((i for i in range(len(out) - 1, -1, -1) if out[i].get("role") == "user"), -1)
    if user_idx < 0:
        return out

    saw_truncation = False
    for _ in range(200):
        n = count_prompt_tokens_chat(tokenizer, out, enable_thinking=enable_thinking)
        if n <= max_prompt_tokens:
            if saw_truncation:
                logger.warning(
                    "Final LLM prompt clamped: prompt_tokens=%d (cap=%d)",
                    n,
                    max_prompt_tokens,
                )
            return out
        c = str(out[user_idx].get("content") or "")
        if len(c) < 200:
            break
        step = max(200, len(c) // 12)
        out[user_idx]["content"] = (
            "[... truncated from the start to fit max_context_tokens ...]\n" + c[step:]
        )
        saw_truncation = True

    n = count_prompt_tokens_chat(tokenizer, out, enable_thinking=enable_thinking)
    if n > max_prompt_tokens:
        logger.error("Could not fit prompt under max_prompt_tokens=%d (still ~%d)", max_prompt_tokens, n)
    return out
