"""
Вариант C: эвристический детектор отрицания/удаления фактов.

Не требует LLM-вызова. Используется когда slot_context_enabled=false
и triplet_deletion_mode="heuristic".

Работа:
  1. Ищет в сообщении пользователя паттерны отрицания/удаления.
  2. Если найдены — извлекает ключевые слова из сообщения.
  3. Сравнивает с полями object/relation активных записей в слоте.
  4. Каскадное совпадение: сначала точное (subj+rel+obj), потом по (subj+rel).

Нормализация с pymorphy2: "Москве" → "москва", "живёт" → "жить".
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from .models import FactRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Паттерны отрицания/удаления для русского языка
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS: list[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"больше\s+не\b",
    r"уже\s+не\b",
    r"не\s+(живу|живёт|живет|работаю|работает|хожу|ходит|делаю|делает|учусь|учится)\b",
    r"перестал[аи]?\b",
    r"бросил[аи]?\b",
    r"расстал[иась]+\b",
    r"разошл[иась]+\b",
    r"разорвал[аи]?\s+отношения",
    r"уволился|уволилась|уволен[аы]?\b",
    r"умер|умерла|скончал[ся|ась]+\b",
    r"продал[аи]?\b",
    r"потерял[аи]?\b",
    r"нет\s+у\s+меня\b",
    r"у\s+меня\s+нет\b",
    r"больше\s+не\s+занимаюсь",
    r"не\s+занимаюсь\b",
    r"перебрался|перебралась\b",
    r"переехал[аи]?\b",
    r"съехал[аи]?\b",
]]

# Стоп-слова, которые не несут смыслового значения при сравнении
_STOP_WORDS: Set[str] = {
    "в", "на", "из", "к", "у", "по", "за", "до", "от", "об", "под",
    "над", "при", "через", "без", "для", "с", "со", "и", "или", "но",
    "не", "уже", "больше", "теперь", "что", "как", "это", "меня",
    "мне", "я", "он", "она", "они", "мы", "ты",
    "да", "нет", "всё", "все", "того",
}


class NegationDeletionDetector:
    """
    Эвристический детектор удалений на основе паттернов отрицания.

    Parameters
    ----------
    use_pymorphy : bool
        Использовать pymorphy2 для лемматизации слов при сравнении.
        Если False — сравнение по сырым нормализованным (lowercase) формам.
    """

    def __init__(self, use_pymorphy: bool = False):
        self.use_pymorphy = use_pymorphy
        self._morph: Optional[object] = None
        if use_pymorphy:
            try:
                import pymorphy2  # type: ignore
                self._morph = pymorphy2.MorphAnalyzer()
                logger.info("NegationDeletionDetector: pymorphy2 loaded")
            except ImportError:
                logger.warning(
                    "NegationDeletionDetector: pymorphy2 not installed, "
                    "falling back to lowercase-only normalization"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_deletions(
        self,
        user_message: str,
        active_records: "List[FactRecord]",
    ) -> "List[_DeletionSignalLike]":
        """
        Определить, какие активные записи следует удалить на основе
        паттернов отрицания в сообщении пользователя.

        Returns
        -------
        List of (subject, relation, object) named tuples for deletion.
        """
        from .triplet_client import DeletionSignal

        if not active_records:
            return []

        if not self._has_negation(user_message):
            logger.debug(
                "NegationDetector: no negation patterns found in message: %.80s",
                user_message,
            )
            return []

        msg_words = self._normalize_text(user_message)
        logger.debug("NegationDetector: negation found, msg_words=%s", msg_words)

        deletions: List[DeletionSignal] = []
        seen_ids: Set[int] = set()

        # --- Проход 1: точное совпадение subj+rel+obj ---
        for rec in active_records:
            if rec.record_id in seen_ids:
                continue
            obj_words = self._normalize_text(rec.object)
            rel_words = self._normalize_text(rec.relation)
            # Объект или связь упомянуты в сообщении
            if (obj_words & msg_words) or (rel_words & msg_words and obj_words & msg_words):
                deletions.append(DeletionSignal(
                    subject=rec.subject,
                    relation=rec.relation,
                    object=rec.object,
                ))
                seen_ids.add(rec.record_id)
                logger.info(
                    "NegationDetector pass1: marking for deletion record_id=%d [%s|%s|%s]",
                    rec.record_id, rec.subject, rec.relation, rec.object,
                )

        # --- Проход 2 (каскадный): только subject+relation если объект не нашли,
        #     но relation явно упомянут в сообщении ---
        if not deletions:
            for rec in active_records:
                if rec.record_id in seen_ids:
                    continue
                rel_words = self._normalize_text(rec.relation)
                if rel_words & msg_words:
                    deletions.append(DeletionSignal(
                        subject=rec.subject,
                        relation=rec.relation,
                        object=rec.object,
                    ))
                    seen_ids.add(rec.record_id)
                    logger.info(
                        "NegationDetector pass2 (cascade): marking record_id=%d [%s|%s|%s]",
                        rec.record_id, rec.subject, rec.relation, rec.object,
                    )

        return deletions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _has_negation(self, text: str) -> bool:
        """Проверить наличие паттернов отрицания/удаления в тексте."""
        for pattern in _NEGATION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _normalize_text(self, text: str) -> Set[str]:
        """
        Извлечь нормализованные значимые слова из текста.
        Если pymorphy2 загружен — лемматизировать.
        """
        raw = re.sub(r"[^\w\s]", " ", text.lower())
        tokens = raw.split()
        result: Set[str] = set()
        for tok in tokens:
            if tok in _STOP_WORDS:
                continue
            if self._morph is not None:
                parsed = self._morph.parse(tok)  # type: ignore
                if parsed:
                    normal = parsed[0].normal_form
                    if normal not in _STOP_WORDS:
                        result.add(normal)
                        continue
            result.add(tok)
        return result
