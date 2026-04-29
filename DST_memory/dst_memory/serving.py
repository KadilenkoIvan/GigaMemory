import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .slot_model_path import resolve_slot_model_path

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    max_new_tokens: int = 300
    do_sample: bool = False
    # NOTE: In HF Transformers, `temperature` is only used when `do_sample=True`.
    # Default 1.0 avoids confusing "temperature=0" semantics.
    temperature: float = 1.0
    top_p: float = 1.0


class LocalHFServing:
    """
    Minimal serving wrapper around HF CausalLM.
    Loads model once and provides chat-style generation.

    enable_thinking:
        Controls the thinking/reasoning mode for models that support it
        (Qwen3, Qwen3.5 and similar hybrid-thinking models).
        False (default) — disable thinking. Passed as enable_thinking=False to
        apply_chat_template; also injects '/no_think' into the system prompt as
        a belt-and-suspenders fallback for older tokenizer versions.
        True — let the model think (may produce verbose reasoning output).
    """

    def __init__(
        self,
        model_path_or_id: str,
        torch_dtype: torch.dtype = torch.float16,
        enable_thinking: bool = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_thinking = enable_thinking
        resolved = resolve_slot_model_path(model_path_or_id)
        logger.info(
            "Serving loading model=%s (resolved=%s) device=%s dtype=%s enable_thinking=%s",
            model_path_or_id,
            resolved,
            self.device,
            torch_dtype,
            enable_thinking,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()
        try:
            first_param = next(self.model.parameters())
            model_device = first_param.device
            param_dtype = first_param.dtype
        except StopIteration:
            model_device = "unknown"
            param_dtype = None
        logger.info(
            "Serving model loaded on device=%s param_dtype=%s (requested load dtype=%s)",
            model_device,
            param_dtype,
            torch_dtype,
        )

    @staticmethod
    def _inject_no_think(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Prepend '/no_think' to the first system message (or create one) so that
        Qwen3/3.5 models that respect the prompt-level directive skip their thinking
        phase even when apply_chat_template doesn't support enable_thinking.
        """
        msgs = [dict(m) for m in messages]
        for msg in msgs:
            if msg.get("role") == "system":
                if not msg["content"].startswith("/no_think"):
                    msg["content"] = "/no_think\n" + msg["content"]
                return msgs
        # No system message found — insert one
        msgs.insert(0, {"role": "system", "content": "/no_think"})
        return msgs

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        if generation_config is None:
            generation_config = GenerationConfig()

        # --- Thinking mode control ---
        # Primary: pass enable_thinking to apply_chat_template (Qwen3/3.5 tokenizers support this).
        # Fallback: inject /no_think into the system prompt for older tokenizer versions.
        msgs = messages
        if not self.enable_thinking:
            msgs = self._inject_no_think(messages)

        try:
            text = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            # Tokenizer does not support enable_thinking — rely on /no_think fallback only
            text = self.tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or eos_id
        # Transformers only uses temperature/top_p when sampling is enabled.
        # Passing them with do_sample=False triggers warnings ("will be ignored").
        gen_kwargs = {
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.do_sample,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }
        if generation_config.do_sample:
            gen_kwargs["temperature"] = generation_config.temperature
            gen_kwargs["top_p"] = generation_config.top_p
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_kwargs,
            )
        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return result.strip()
