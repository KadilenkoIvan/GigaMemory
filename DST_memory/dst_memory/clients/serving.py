import gc
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# torch and transformers are heavy optional deps — imported lazily inside LocalHFServing
# so that the rest of the codebase (and tests in stub mode) can be imported without GPU.
if TYPE_CHECKING:
    import torch

from ..slots.slot_model_path import resolve_slot_model_path

logger = logging.getLogger(__name__)


def _normalize_load_quantization(raw: str | None) -> str:
    """Return one of: none, 8bit, 4bit."""
    q = (raw or "none").lower().strip()
    if q in (
        "",
        "none",
        "off",
        "false",
        "no",
        "fp16",
        "float16",
        "bf16",
        "bfloat16",
        "fp32",
        "float32",
    ):
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
    # When LocalHFServing.use_lm_format_enforcer is True, pass a JSON Schema dict to constrain decoding.
    lm_enforcer_json_schema: dict[str, Any] | None = None


class LocalHFServing:
    """
    Minimal serving wrapper around HF CausalLM.
    Loads model once and provides chat-style generation.

    enable_thinking:
        Controls the thinking/reasoning mode for models that support it
        (Qwen3, Qwen3.5 and similar hybrid-thinking models).
        False — passes enable_thinking=False to apply_chat_template.
        True — let the model think (may produce verbose reasoning output).

    inject_no_think_prompt:
        When True (legacy default) and enable_thinking is False, prepends ``/no_think``
        via :meth:`_inject_no_think`. Set False to rely on chat-template ``enable_thinking`` only.

    use_lm_format_enforcer:
        When True, :meth:`generate_chat` may apply ``lm-format-enforcer`` if
        :class:`GenerationConfig` includes ``lm_enforcer_json_schema`` (requires optional dependency).
    """

    def __init__(
        self,
        model_path_or_id: str,
        torch_dtype: Any = None,  # defaults to torch.float16 after lazy import
        enable_thinking: bool = False,
        load_quantization: str = "none",
        inject_no_think_prompt: bool = True,
        use_lm_format_enforcer: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch_dtype is None:
            torch_dtype = torch.float16
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.enable_thinking = enable_thinking
        self.inject_no_think_prompt = bool(inject_no_think_prompt)
        self.use_lm_format_enforcer = bool(use_lm_format_enforcer)
        self._lm_enforcer_prefix_cache: dict[str, Any] = {}
        self._quant = _normalize_load_quantization(load_quantization)
        if self.use_lm_format_enforcer:
            try:
                import lmformatenforcer  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "LocalHFServing(use_lm_format_enforcer=True) requires `pip install lm-format-enforcer`."
                ) from e
        resolved = resolve_slot_model_path(model_path_or_id)
        logger.info(
            "Serving loading model=%s (resolved=%s) device=%s dtype=%s quant=%s enable_thinking=%s "
            "inject_no_think_prompt=%s use_lm_format_enforcer=%s",
            model_path_or_id,
            resolved,
            self.device,
            torch_dtype,
            self._quant,
            enable_thinking,
            self.inject_no_think_prompt,
            self.use_lm_format_enforcer,
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

            compute = (
                torch_dtype
                if torch_dtype in (torch.float16, torch.bfloat16)
                else torch.float16
            )
            if self._quant == "8bit":
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            else:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                trust_remote_code=True,
                quantization_config=bnb,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved,
                trust_remote_code=True,
                torch_dtype=torch_dtype,
            ).to(
                self.device
            )  # type: ignore[arg-type]
        self.model.eval()
        try:
            first_param = next(self.model.parameters())
            model_device = first_param.device
            param_dtype = first_param.dtype
        except StopIteration:
            model_device = "unknown"  # type: ignore[assignment]
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
        import torch

        logger.info("LocalHFServing.release() — dropping model and tokenizer")
        self._lm_enforcer_prefix_cache.clear()
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
    def _inject_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
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

    def _prefix_allowed_tokens_fn_for_schema(self, schema: dict[str, Any]):
        key = json.dumps(schema, sort_keys=True, ensure_ascii=False)
        if key in self._lm_enforcer_prefix_cache:
            return self._lm_enforcer_prefix_cache[key]
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(schema)
        fn = build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
        self._lm_enforcer_prefix_cache[key] = fn
        return fn

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        generation_config: GenerationConfig | None = None,
    ) -> str:
        import torch

        if generation_config is None:
            generation_config = GenerationConfig()

        # --- Thinking mode control ---
        if not self.enable_thinking and self.inject_no_think_prompt:
            msgs = self._inject_no_think(messages)
        else:
            msgs = messages

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
        schema = getattr(generation_config, "lm_enforcer_json_schema", None)
        if self.use_lm_format_enforcer and schema:
            gen_kwargs["prefix_allowed_tokens_fn"] = (
                self._prefix_allowed_tokens_fn_for_schema(schema)
            )
        with torch.no_grad():
            outputs = self.model.generate(  # type: ignore[misc]
                **inputs,
                **gen_kwargs,  # type: ignore[arg-type]
            )
        result = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return result.strip()  # type: ignore[union-attr]
