import gc
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..slots.slot_model_path import resolve_slot_model_path

logger = logging.getLogger(__name__)

def _patched_update_causal_mask(self, attention_mask, *args, **kwargs):
    return None

def _normalize_attn_implementation(raw: Optional[str]) -> str:
    """HF ``attn_implementation`` for ``AutoModelForCausalLM.from_pretrained`` (e.g. eager, sdpa, flash_attention_2)."""
    s = (raw or "eager").strip()
    return s if s else "eager"


def _from_pretrained_causal_lm(
    resolved: str,
    *,
    torch_dtype=None,
    quantization_config=None,
    device_map=None,
    attn_implementation: str,
):
    """Load causal LM; retry without ``attn_implementation`` if the installed transformers is too old."""
    kwargs: Dict[str, object] = {
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = device_map
    else:
        kwargs["torch_dtype"] = torch_dtype
    try:
        return AutoModelForCausalLM.from_pretrained(resolved, **kwargs)
    except TypeError as e:
        logger.warning(
            "from_pretrained rejected attn_implementation=%r (%s); retrying without it",
            attn_implementation,
            e,
        )
        kwargs.pop("attn_implementation", None)
        return AutoModelForCausalLM.from_pretrained(resolved, **kwargs)


def _normalize_load_quantization(raw: Optional[str]) -> str:
    """Return one of: none, 8bit, 4bit."""
    q = (raw or "none").lower().strip()
    if q in ("", "none", "off", "false", "no", "fp16", "float16", "bf16", "bfloat16", "fp32", "float32"):
        return "none"
    if q in ("8bit", "int8", "8-bit", "bnb8"):
        return "8bit"
    if q in ("4bit", "int4", "4-bit", "bnb4"):
        return "4bit"
    raise ValueError(
        f"Unknown load_quantization={raw!r}; use none, 8bit, or 4bit (requires bitsandbytes)."
    )


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
        load_quantization: str = "none",
        attn_implementation: str = "eager",
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_thinking = enable_thinking
        self._quant = _normalize_load_quantization(load_quantization)
        self._attn = _normalize_attn_implementation(attn_implementation)
        resolved = resolve_slot_model_path(model_path_or_id)
        logger.info(
            "Serving loading model=%s (resolved=%s) device=%s dtype=%s quant=%s attn=%s enable_thinking=%s",
            model_path_or_id,
            resolved,
            self.device,
            torch_dtype,
            self._quant,
            self._attn,
            enable_thinking,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(resolved, trust_remote_code=True)

        if self._quant != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as e:
                raise ImportError(
                    "load_quantization requires transformers with BitsAndBytesConfig; "
                    "install bitsandbytes (and a compatible GPU driver)."
                ) from e

            compute = torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else torch.float16
            if self._quant == "8bit":
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            self.model = _from_pretrained_causal_lm(
                resolved,
                quantization_config=bnb,
                device_map="auto",
                attn_implementation=self._attn,
            )
        else:
            self.model = _from_pretrained_causal_lm(
                resolved,
                torch_dtype=torch_dtype,
                attn_implementation=self._attn,
            ).to(self.device)
            
        self.model.eval()
        import types
        self.model._update_causal_mask = types.MethodType(
            _patched_update_causal_mask, self.model
        )
        
        try:
            first_param = next(self.model.parameters())
            model_device = first_param.device
            param_dtype = first_param.dtype
        except StopIteration:
            model_device = "unknown"
            param_dtype = None
        logger.info(
            "Serving model loaded on device=%s param_dtype=%s (requested load dtype=%s quant=%s)",
            model_device,
            param_dtype,
            torch_dtype,
            self._quant,
        )

    def release(self) -> None:
        """Drop weights from VRAM/RAM (call before loading another large local model)."""
        logger.info("LocalHFServing.release() — dropping model and tokenizer")
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None  # type: ignore[assignment]
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

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

        try:
            in_dev = next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            in_dev = torch.device(self.device)
        inputs = self.tokenizer(text, return_tensors="pt").to(in_dev)
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
                input_ids=inputs["input_ids"],
                **gen_kwargs,
            )
        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return result.strip()
