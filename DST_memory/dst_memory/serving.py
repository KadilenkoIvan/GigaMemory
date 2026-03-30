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
    temperature: float = 0.0
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
            model_device = next(self.model.parameters()).device
        except StopIteration:
            model_device = "unknown"
        logger.info("Serving model loaded on device=%s", model_device)

    def generate_chat(
        self, messages: List[Dict[str, str]], gen: Optional[GenerationConfig] = None
    ) -> str:
        if gen is None:
            gen = GenerationConfig()
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None) or eos_id
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen.max_new_tokens,
                do_sample=gen.do_sample,
                temperature=gen.temperature,
                top_p=gen.top_p,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return result.strip()

