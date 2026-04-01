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
    """

    def __init__(self, model_path_or_id: str, torch_dtype: torch.dtype = torch.float16):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = resolve_slot_model_path(model_path_or_id)
        logger.info(
            "Serving loading model=%s (resolved=%s) device=%s dtype=%s",
            model_path_or_id,
            resolved,
            self.device,
            torch_dtype,
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

    def generate_chat(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None,
    ) -> str:
        if generation_config is None:
            generation_config = GenerationConfig()
        text = self.tokenizer.apply_chat_template(
            messages,
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
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return result.strip()

