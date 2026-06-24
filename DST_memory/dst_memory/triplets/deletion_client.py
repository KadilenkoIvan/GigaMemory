"""
Клиент для Варианта B: отдельный LLM-вызов детекции удалений.

Используется когда triplet_deletion_mode="llm_separate".
Вызывается ПОСЛЕ обычной экстракции триплетов (без контекста слота).
Получает текущие факты + сообщение → возвращает список DeletionSignal.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..clients.serving import GenerationConfig, LocalHFServing
from ..prompts.loader import PromptModules, load_prompt_modules
from ..prompts.parsers import parse_deletion_response

logger = logging.getLogger(__name__)


class TripletDeletionClient:
    """
    Отдельный LLM-вызов для детекции устаревших фактов (Вариант B).

    Всегда получает контекст текущих фактов слота, даже если
    slot_context_enabled=False для основной экстракции триплетов.
    """

    def __init__(
        self,
        *,
        use_stub: bool,
        serving: LocalHFServing | None = None,
        max_retries: int = 1,
        prompt_language: str = "ru",
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.max_retries = max_retries
        self.prompt_language = prompt_language
        self._prompt_modules: PromptModules | None = None

        if not use_stub and serving is None:
            raise ValueError(
                "TripletDeletionClient requires serving when use_stub is False"
            )
        if use_stub:
            logger.info("TripletDeletionClient: STUB mode (no deletions)")
        else:
            logger.info(
                "TripletDeletionClient: LLM mode device=%s", serving.device  # type: ignore[union-attr]
            )

    def detect_deletions(
        self,
        user_message: str,
        slot_name: str,
        existing_triplets: list[str],
    ) -> list:
        """
        Определить какие факты нужно удалить на основе нового сообщения.

        Parameters
        ----------
        user_message : str
            Новое сообщение пользователя.
        slot_name : str
            Канонический ключ слота.
        existing_triplets : list of str
            Активные факты в формате "subject | relation | object".

        Returns
        -------
        List[DeletionSignal]
        """
        from .triplet_client import DeletionSignal

        if self.use_stub or not existing_triplets:
            return []

        assert self.serving is not None
        if self._prompt_modules is None:
            self._prompt_modules = load_prompt_modules(self.prompt_language)
        messages = self._prompt_modules.deletion_messages.build_deletion_messages(
            user_message, slot_name, existing_triplets
        )
        cfg = GenerationConfig(max_new_tokens=256, temperature=0.0, do_sample=False)

        for attempt in range(self.max_retries + 1):
            raw = self.serving.generate_chat(messages, cfg)
            logger.info(
                "DeletionClient slot=%s attempt=%d raw: %s",
                slot_name,
                attempt,
                raw[:400],
            )
            try:
                items = parse_deletion_response(raw)
                result = [
                    DeletionSignal(
                        subject=it["subject"],
                        relation=it["relation"],
                        object=it["object"],
                    )
                    for it in items
                ]
                logger.info(
                    "DeletionClient slot=%s: %d deletion(s) detected",
                    slot_name,
                    len(result),
                )
                return result
            except Exception as exc:
                logger.warning(
                    "DeletionClient parse failed attempt=%d slot=%s: %s | raw=%r",
                    attempt,
                    slot_name,
                    exc,
                    raw[:200],
                )

        logger.warning("DeletionClient: all attempts failed, returning empty deletions")
        return []
