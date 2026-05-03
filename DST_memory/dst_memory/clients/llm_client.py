from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

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
        # Last sent messages (system + user) — populated on every generate() call.
        # Used for logging to *_logs.json.
        self._last_prompt_messages: List[Dict[str, str]] = []
        logger.info(
            "FinalLLMClient initialized mode=%s model=%s temperature=%s",
            self.mode,
            self.model or "(none)",
            temperature,
        )

    def build_messages(
        self,
        question: str,
        memory_context: Any,
        recent_pairs: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        """Build the chat messages list without sending. Exposed for logging."""
        import datetime
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        system = (
            "Ты — персональный ассистент с долговременной памятью о пользователе.\n"
            "Отвечай по-русски, кратко и по делу.\n\n"

            "## Что такое память\n"
            "Долговременная память реализована в виде графа знаний (knowledge graph). "
            "Каждый факт — это направленное ребро графа в форме триплета:\n"
            "  субъект  →[связь]→  объект\n"
            "Например: «пользователь —[работает в]→ Яндекс».\n"
            "Факты накапливаются последовательно по ходу диалога: каждый новый "
            "разговор добавляет, уточняет или заменяет узлы и рёбра графа. "
            "Объекты одного факта могут быть субъектами другого, образуя цепочки: "
            "«пользователь —[есть жена]→ Мария —[работает в]→ Сбер». "
            "Это позволяет отвечать на составные вопросы, проходя по цепи связей.\n\n"

            "## Структура памяти\n"
            "Граф разделён на тематические подграфы — слоты. "
            "Каждый слот хранит факты об одной области жизни пользователя и "
            "является независимым срезом графа.\n"
            "Таблица слотов (ключ → тема):\n"
            "  IDENTITY=Личность, FAMILY=Семья, FRIENDS=Друзья, ROMANCE=Романтика,\n"
            "  WORK=Работа, EDUCATION=Образование, FINANCE=Финансы,\n"
            "  HEALTH=Здоровье, MENTAL_HEALTH=Психическое состояние,\n"
            "  HABITS=Привычки, PREFERENCES=Предпочтения, HOBBIES=Хобби, SPORTS=Спорт,\n"
            "  FOOD=Еда, HOME=Дом/Жильё, LOCATION=Местоположение, TRAVEL=Путешествия,\n"
            "  PETS=Питомцы, TECH=Техника, VEHICLES=Транспорт,\n"
            "  SCHEDULE=Расписание, GOALS=Цели/Планы, EVENTS=События.\n\n"

            "## Как читать блок памяти\n"
            "Поле \"slots\" — список подграфов-слотов. Каждый слот:\n"
            "  - \"slot\" и \"slot_label\": канонический ключ слота на английском (напр. FAMILY).\n"
            "  - \"messages\": список рёбер графа этого слота. Каждое ребро:\n"
            "      • subject, relation, object — узлы и тип связи;\n"
            "      • created_at_datetime — момент когда факт был добавлен в граф;\n"
            "      • ttl — время жизни факта (\"inf\" = бессрочно).\n\n"

            "## Правила применения памяти\n"
            "  1. Опирайся на факты из графа если они релевантны вопросу.\n"
            "  2. Прослеживай связи между фактами разных слотов — ответ может "
            "требовать объединения данных из нескольких подграфов.\n"
            "  3. При нескольких слотах с разными данными — структурируй ответ по тематикам.\n"
            "  4. При противоречии двух фактов — более свежий "
            "(created_at_datetime позднее) имеет приоритет.\n"
            "  5. Если память пуста или нерелевантна — отвечай из общих знаний, "
            "  6. При ответе - не упоминай слоты, память и как она работает, "
            "не выдумывай факты о пользователе.\n\n"
            f"Текущее время: {now_str}."
        )

        mem_block = json.dumps(memory_context or {}, ensure_ascii=False, indent=2)
        pairs_block = (
            json.dumps(recent_pairs, ensure_ascii=False, indent=2)
            if recent_pairs
            else "[]"
        )
        user = (
            f"Текущая дата и время: {now_str}\n\n"
            "Контекст памяти (JSON):\n"
            f"{mem_block}\n\n"
            "Последние пары user/assistant (JSON):\n"
            f"{pairs_block}\n\n"
            "Текущий запрос пользователя:\n"
            f"{question}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def generate(
        self,
        question: str,
        memory_context: Any,
        recent_pairs: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        messages = self.build_messages(question, memory_context, recent_pairs or [])
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
            raise NotImplementedError("TODO: local LLM backend is not implemented yet.")

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
