from typing import List
import logging

logger = logging.getLogger(__name__)


class FinalLLMClient:
    def __init__(
        self,
        mode: str = "stub",
        api_url: str = "",
        api_key: str = "",
        temperature: float = 0.0,
    ):
        self.mode = mode
        self.api_url = api_url
        self.api_key = api_key
        self.temperature = temperature
        logger.info("FinalLLMClient initialized mode=%s temperature=%s", mode, temperature)

    def generate(self, question: str, memory_lines: List[str]) -> str:
        logger.info(
            "FinalLLM generate mode=%s question_len=%d memory_lines=%d",
            self.mode,
            len(question),
            len(memory_lines),
        )
        if self.mode == "stub":
            joined = "; ".join(memory_lines[:3]) if memory_lines else "no-memory"
            return f"[STUB_ANSWER] q='{question}' | memory='{joined}'"

        if self.mode == "local":
            # TODO: implement local LLM call (vLLM/transformers/llama.cpp backend).
            # Pass self.temperature (default 0) and do_sample=False for deterministic decoding.
            raise NotImplementedError("TODO: local LLM backend is not implemented yet.")

        if self.mode == "api":
            # TODO: implement remote API call (temperature=self.temperature, typically 0).
            raise NotImplementedError("TODO: API LLM backend is not implemented yet.")

        raise ValueError(f"Unknown llm_mode: {self.mode}")
