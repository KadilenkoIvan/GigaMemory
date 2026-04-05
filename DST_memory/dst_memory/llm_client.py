from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import List

logger = logging.getLogger(__name__)


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
        logger.info(
            "FinalLLMClient initialized mode=%s model=%s temperature=%s",
            self.mode,
            self.model or "(none)",
            temperature,
        )

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
            raise NotImplementedError("TODO: local LLM backend is not implemented yet.")

        if self.mode in ("api", "openrouter"):
            return self._openai_compatible_chat(question, memory_lines)

        raise ValueError(f"Unknown llm_mode: {self.mode}")

    def _chat_url(self) -> str:
        base = self.api_url
        if not base:
            base = "https://openrouter.ai/api/v1"
        if base.endswith("/chat/completions"):
            return base
        return f"{base.rstrip('/')}/chat/completions"

    def _openai_compatible_chat(self, question: str, memory_lines: List[str]) -> str:
        if not self.api_key.strip():
            raise ValueError(
                "llm_api_key is empty; set it in run_config.json or OPENROUTER_API_KEY."
            )
        if not self.model.strip():
            raise ValueError("llm_model is empty; set it in run_config.json or --llm-model.")

        system = (
            "Ты помощник в диалоге с долговременной памятью в виде слотов. "
            "Отвечай по-русски, кратко и по делу. "
            "Если в блоке памяти есть релевантные факты — опирайся на них; "
            "если память пуста или не относится к вопросу — отвечай из общих знаний, не выдумывай факты о пользователе."
        )
        mem_block = "\n".join(f"- {line}" for line in memory_lines) if memory_lines else "(память пуста)"
        user = f"Фрагменты памяти (слоты):\n{mem_block}\n\nВопрос пользователя:\n{question}"

        body = {
            "model": self.model,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer.strip():
            headers["Referer"] = self.http_referer.strip()
        if self.x_title.strip():
            headers["X-OpenRouter-Title"] = self.x_title.strip()

        req = urllib.request.Request(
            self._chat_url(),
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            logger.error("Final LLM HTTP %s: %s", e.code, detail[:500])
            raise RuntimeError(f"Final LLM request failed: HTTP {e.code}") from e
        except urllib.error.URLError as e:
            logger.error("Final LLM URL error: %s", e)
            raise RuntimeError(f"Final LLM request failed: {e}") from e

        data = json.loads(raw)
        err = data.get("error")
        if err:
            msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            raise RuntimeError(f"Final LLM API error: {msg}")

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Final LLM response has no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise RuntimeError("Final LLM response has empty content")
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    parts.append(p.get("text") or "")
            content = "".join(parts).strip()
        text = str(content).strip()
        if not text:
            raise RuntimeError("Final LLM returned blank text")
        return text
