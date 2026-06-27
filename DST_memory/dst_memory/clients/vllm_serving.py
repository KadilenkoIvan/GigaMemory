"""vLLM slot serving client — calls a running vLLM OpenAI-compatible server.

vLLM docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
Structured output via guided_json: vLLM's built-in constrained decoding,
no lm-format-enforcer needed.
"""

from __future__ import annotations

import logging
from typing import Any

from .serving import GenerationConfig

logger = logging.getLogger(__name__)


class VLLMSlotServing:
    """Drop-in replacement for LocalHFServing backed by an external vLLM server.

    The server must be started separately (see README / ``make vllm``).
    Structured JSON output is enforced via ``guided_json`` in the request
    extra_body — equivalent to lm-format-enforcer but handled server-side
    by vLLM's xgrammar/outlines backend.

    Setting ``use_lm_format_enforcer = True`` (class-level) causes all slot
    clients (SlotSelectClient, TripletExtractionClient, …) to pass their
    JSON schemas into GenerationConfig.lm_enforcer_json_schema, which this
    class then forwards to vLLM as ``guided_json``.
    """

    use_lm_format_enforcer: bool = True

    def __init__(
        self,
        model: str,
        api_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        enable_thinking: bool = False,
        max_retries: int = 3,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for VLLMSlotServing; "
                "it should already be installed as a core dependency."
            ) from e

        self.model = model
        self.enable_thinking = enable_thinking
        self.device = "vllm"
        self._client = OpenAI(
            base_url=api_url, api_key=api_key, max_retries=max_retries
        )
        logger.info(
            "VLLMSlotServing ready model=%s api_url=%s",
            model,
            api_url,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig | None = None,
    ) -> str:
        """Generate a response via vLLM's OpenAI-compatible chat completions API."""
        if generation_config is None:
            generation_config = GenerationConfig()

        temperature = (
            float(generation_config.temperature) if generation_config.do_sample else 0.0
        )

        # extra_body carries vLLM-specific options not in the OpenAI schema.
        # chat_template_kwargs.enable_thinking=False disables Qwen3 reasoning
        # so the whole token budget goes to the actual JSON answer, not to a
        # <think> block that the reasoning parser strips into reasoning_content.
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

        schema = getattr(generation_config, "lm_enforcer_json_schema", None)
        if schema is not None:
            extra_body["guided_json"] = schema

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": generation_config.max_new_tokens,
            "temperature": temperature,
            "extra_body": extra_body,
        }

        logger.debug(
            "VLLMSlotServing request model=%s guided_json=%s thinking=%s temperature=%.2f tokens=%d",
            self.model,
            schema is not None,
            self.enable_thinking,
            temperature,
            generation_config.max_new_tokens,
        )

        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        text = (message.content or "").strip()
        reasoning = getattr(message, "reasoning_content", None)
        finish_reason = response.choices[0].finish_reason

        if not text:
            # Empty content almost always means the budget went to reasoning
            # or the JSON was cut off — surface both so it is diagnosable.
            logger.warning(
                "VLLMSlotServing EMPTY content finish_reason=%s reasoning_len=%s reasoning=%.300s",
                finish_reason,
                len(reasoning) if reasoning else 0,
                reasoning or "",
            )
        elif finish_reason == "length":
            logger.warning(
                "VLLMSlotServing TRUNCATED (finish_reason=length) content=%.300s",
                text,
            )
        else:
            logger.info(
                "VLLMSlotServing response finish_reason=%s reasoning_len=%s content=%.400s",
                finish_reason,
                len(reasoning) if reasoning else 0,
                text,
            )
        return text
