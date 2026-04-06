"""Промпты memory gate: русские few-shot, чередование user + assistant."""

from __future__ import annotations

from typing import Dict, List

from .prompt_fewshots_ru import MEMORY_GATE_FEWSHOT, MEMORY_GATE_FEWSHOT_VECTOR, memory_gate_user_block


def build_memory_gate_messages(
    user_message: str,
    slot_names: List[str],
    *,
    for_vector_context: bool = False,
) -> List[Dict[str, str]]:
    extra_block = ""
    if for_vector_context:
        extra_block = (
            " ЕСЛИ ПАМЯТЬ ДЛЯ ОТВЕТА НУЖНА, НО НИ ОДИН КОНКРЕТНЫЙ СЛОТ ИЗ СПИСКА ЯВНО НЕ ПОДХОДИТ — "
            'укажи use_memory: true и slots: [].'
        )

    system = (
        "ТЫ ПОМОЩНИК ДЛЯ ВЫБОРА РЕЛЕВАНТНОЙ ДОЛГОВРЕМЕННОЙ ПАМЯТИ.\n"
        "ДАНЫ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ И СПИСОК ИМЁН СЛОТОВ (ТОЛЬКО НАЗВАНИЯ, БЕЗ СОДЕРЖИМОГО).\n"
        "РЕШИ, НУЖНО ЛИ ДЛЯ ОТВЕТА ПОДСТАВЛЯТЬ СОХРАНЁННЫЕ ФАКТЫ.\n"
        "ЕСЛИ ВОПРОС ОБЩИЙ, ОФФТОП ИЛИ БЕЗ ЛИЧНОГО КОНТЕКСТА — ПАМЯТЬ НЕ НУЖНА.\n"
        "ЕСЛИ ЯВНО НУЖНЫ ОДИН ИЛИ НЕСКОЛЬКО СЛОТОВ — ПЕРЕЧИСЛИ ИХ ИМЕНА ТОЧНО КАК В СПИСКЕ.\n"
        "ОТВЕТ СТРОГО ОДИН JSON-ОБЪЕКТ, БЕЗ MARKDOWN, БЕЗ ПОЯСНЕНИЙ:\n"
        '{"use_memory": true или false, "slots": ["ИМЯ_СЛОТА", ...]}\n'
        'ПРИ use_memory=false МАССИВ "slots" ДОЛЖЕН БЫТЬ [].'
    )

    few_shot: List[Dict[str, str]] = []
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
                    "content": memory_gate_user_block(question, slot_list, extra_block.strip()),
                }
            )
            few_shot.append({"role": "assistant", "content": assistant_json})

    final_user = memory_gate_user_block(user_message, slot_names, extra_block)

    return [{"role": "system", "content": system}] + few_shot + [{"role": "user", "content": final_user}]
