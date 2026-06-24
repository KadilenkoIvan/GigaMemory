"""Промпты memory gate: русские few-shot, чередование user + assistant."""

from __future__ import annotations

from typing import Dict, List

from ...slots.ontology import CANONICAL_TO_RU_LABEL
from .prompt_fewshots import (
    MEMORY_GATE_FEWSHOT,
    MEMORY_GATE_FEWSHOT_VECTOR,
    memory_gate_user_block,
)


def _slots_ru(canonical: list[str]) -> list[str]:
    return [CANONICAL_TO_RU_LABEL.get(s, s) for s in canonical]


def build_memory_gate_messages(
    user_message: str,
    slot_names: list[str],
    *,
    for_vector_context: bool = False,
) -> list[dict[str, str]]:
    extra_block = ""
    if for_vector_context:
        extra_block = (
            " ЕСЛИ ПАМЯТЬ ДЛЯ ОТВЕТА НУЖНА, НО НИ ОДИН КОНКРЕТНЫЙ СЛОТ ИЗ СПИСКА ЯВНО НЕ ПОДХОДИТ — "
            "укажи use_memory: true и slots: []."
        )

    system = (
        "ТЫ ПОМОЩНИК ДЛЯ ВЫБОРА РЕЛЕВАНТНОЙ ДОЛГОВРЕМЕННОЙ ПАМЯТИ.\n"
        "ДАНЫ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ И СПИСОК ИМЁН СЛОТОВ (ТОЛЬКО НАЗВАНИЯ, БЕЗ СОДЕРЖИМОГО).\n"
        "РЕШИ, НУЖНО ЛИ ДЛЯ ОТВЕТА ИСПОЛЬЗОВАТЬ СОХРАНЁННУЮ ПАМЯТЬ.\n"
        "ЕСЛИ ВОПРОС АБСТРАКТНЫЙ, ОБЩИЙ ИЛИ БЕЗ КАКОГО-ЛИБО ЛИЧНОГО КОНТЕКСТА — ПАМЯТЬ НЕ НУЖНА.\n"
        "В ОСТАЛЬНЫХ СЛУЧАЯХ — ВЫБИРАЙ ВСЕ СЛОТЫ, КОТОРЫЕ ХОТЬ КОСВЕННО ПОМОГУТ ДАТЬ ЛУЧШИЙ ОТВЕТ.\n"
        "ПРАВИЛО: ЛУЧШЕ ВЫБРАТЬ ЛИШНИЙ СЛОТ, ЧЕМ ПРОПУСТИТЬ ПОТЕНЦИАЛЬНО ВАЖНЫЙ.\n"
        "ПЕРЕЧИСЛИ ИМЕНА ВЫБРАННЫХ СЛОТОВ ТОЧНО КАК В СПИСКЕ.\n"
        "ОТВЕТ СТРОГО ОДИН JSON-ОБЪЕКТ, БЕЗ MARKDOWN, БЕЗ ПОЯСНЕНИЙ:\n"
        '{"use_memory": true или false, "slots": ["ИМЯ_СЛОТА", ...]}\n'
        'ПРИ use_memory=false МАССИВ "slots" ДОЛЖЕН БЫТЬ [].'
    )

    few_shot: list[dict[str, str]] = []
    for question, slots_block, assistant_json in MEMORY_GATE_FEWSHOT:
        slot_list = [s.strip() for s in slots_block.split("\n") if s.strip()]
        few_shot.append(
            {
                "role": "user",
                "content": memory_gate_user_block(question, slot_list, ""),
            }
        )
        few_shot.append({"role": "assistant", "content": assistant_json})

    if for_vector_context:
        for question, slots_block, assistant_json in MEMORY_GATE_FEWSHOT_VECTOR:
            slot_list = [s.strip() for s in slots_block.split("\n") if s.strip()]
            few_shot.append(
                {
                    "role": "user",
                    "content": memory_gate_user_block(
                        question, slot_list, extra_block.strip()
                    ),
                }
            )
            few_shot.append({"role": "assistant", "content": assistant_json})

    final_user = memory_gate_user_block(
        user_message, _slots_ru(slot_names), extra_block
    )

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": final_user}]
    )
