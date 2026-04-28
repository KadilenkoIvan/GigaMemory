"""Промпты извлечения триплетов (русские few-shot, system без markdown)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .ontology import CANONICAL_TO_RU_LABEL, DEFAULT_USER_SLOTS, RU_SLOT_LABELS_ORDERED
from .prompt_fewshots_ru import (
    triplet_per_slot_few_shot_messages,
    triplet_single_pass_few_shot_messages,
)


def build_triplet_messages(
    user_message: str,
    *,
    slot_name: str | None = None,
    include_slot: bool = False,
    ontology_slots: List[str] | None = None,
    max_triplets: int = 12,
    ttl_mode: str = "mode2",
) -> List[Dict[str, Any]]:
    _ = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_ru_json = json.dumps(RU_SLOT_LABELS_ORDERED, ensure_ascii=False)

    ru_slot = CANONICAL_TO_RU_LABEL.get(slot_name, slot_name) if slot_name else None

    slot_header = ""
    if slot_name and ru_slot:
        slot_header = (
            f"ТЕКУЩИЙ СЛОТ: {ru_slot} ({slot_name}).\n"
            f"ИЗВЛЕКАЙ ТОЛЬКО ФАКТЫ, ОТНОСЯЩИЕСЯ К СЛОТУ «{ru_slot}».\n"
            f"ЕСЛИ В СООБЩЕНИИ НЕТ ФАКТОВ ДЛЯ СЛОТА «{ru_slot}» — ВЕРНИ {{\"triplets\":[]}}.\n"
            "В JSON НЕ УКАЗЫВАЙ ПОЛЕ slot — СЛОТ ЗАДАЁТСЯ СИСТЕМОЙ.\n\n"
        )

    use_ttl = (ttl_mode == "mode2")

    if use_ttl:
        if include_slot:
            output_schema = (
                '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер","ttl":"1y"}]}'
            )
        else:
            output_schema = (
                '{"triplets":[{"subject":"пользователь","relation":"работает как","object":"водитель такси","ttl":"1y"}]}'
            )
        ttl_block = (
            "\nДОПОЛНИТЕЛЬНО К КАЖДОМУ ТРИПЛЕТУ ДОБАВЛЯЙ ПОЛЕ TTL (время жизни факта).\n"
            "ДОПУСТИМЫЕ ЗНАЧЕНИЯ TTL: 1d, 3d, 10d, 2w, 3w, 1m, 3m, 6m, 1y, inf\n"
            "ПРАВИЛА ВЫБОРА TTL:\n"
            "  inf  — имя, пол, национальность, члены семьи, питомцы, устойчивые привычки (кофе по утрам)\n"
            "  1y   — работа, учёба, жильё, здоровье (диагнозы), авто, местоположение\n"
            "  6m   — хобби, спорт, предпочтения, психическое состояние, знакомства\n"
            "  3m   — цели, романтические отношения, финансовые планы\n"
            "  1m   — расписание, планы на ближайшее будущее, еда\n"
            "  2w   — конкретные события (был на свадьбе, сдал экзамен)\n"
            "  1d   — сиюминутные состояния (пьяная, устала, злюсь)\n"
        )
    else:
        if include_slot:
            output_schema = (
                '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер"}]}'
            )
        else:
            output_schema = (
                '{"triplets":[{"subject":"пользователь","relation":"работает как","object":"водитель такси"}]}'
            )
        ttl_block = ""

    system = (
        slot_header
        + "ТЫ СИСТЕМА ИЗВЛЕЧЕНИЯ ФАКТОВ ИЗ РЕПЛИКИ ПОЛЬЗОВАТЕЛЯ.\n"
        "ПРЕДСТАВЬ ФАКТЫ КАК ТРИПЛЕТЫ: СУБЪЕКТ, СВЯЗЬ, ОБЪЕКТ.\n"
        "СУБЪЕКТ, СВЯЗЬ И ОБЪЕКТ ПИШИ СТРОЧНЫМИ БУКВАМИ (lowercase).\n"
        "НЕ ИСПОЛЬЗУЙ СИМВОЛ ПОДЧЁРКИВАНИЯ «_» — РАЗДЕЛЯЙ СЛОВА ТОЛЬКО ПРОБЕЛАМИ.\n"
        "ДЛЯ ФАКТОВ О САМОМ ПОЛЬЗОВАТЕЛЕ ИСПОЛЬЗУЙ СУБЪЕКТ: пользователь.\n"

        "ЕСЛИ УПОМЯНУТА СВЯЗАННАЯ СУЩНОСТЬ (член семьи, коллега, питомец и т.п.):\n"
        "  1. ДОБАВЬ ТРИПЛЕТ СВЯЗИ: пользователь → отношение → роль сущности "
        "(например: пользователь → есть сын → сын пользователя).\n"
        "  2. ВСЕ СВОЙСТВА ЭТОЙ СУЩНОСТИ (имя, возраст, и т.п.) ВЕШАЙ НА РОЛЬ КАК ОТДЕЛЬНЫЕ ТРИПЛЕТЫ "
        "(например: сын пользователя → имя → миша).\n"
        "  3. СУБЪЕКТОМ ВСЕГДА ЯВЛЯЕТСЯ РОЛЬ, НИКОГДА ИМЯ.\n"
        "     ЗАПРЕЩЕНО: {\"subject\":\"миша\",\"relation\":\"имя\",\"object\":\"миша\"}\n"
        "     ЗАПРЕЩЕНО: {\"subject\":\"миша\",\"relation\":\"возраст\",\"object\":\"1\"}\n"
        "     ВЕРНО:     {\"subject\":\"сын пользователя\",\"relation\":\"имя\",\"object\":\"миша\"}\n"
        "     ВЕРНО:     {\"subject\":\"сын пользователя\",\"relation\":\"возраст\",\"object\":\"1\"}\n"

        "КАЖДЫЙ ТРИПЛЕТ ДОЛЖЕН БЫТЬ САМОДОСТАТОЧНЫМ: читаться и пониматься независимо от других триплетов.\n"
        "RELATION ДОЛЖЕН ОДНОЗНАЧНО ОПИСЫВАТЬ СМЫСЛ БЕЗ КОНТЕКСТА СОСЕДНИХ ТРИПЛЕТОВ.\n"
        "     ЗАПРЕЩЕНО: {\"subject\":\"пользователь\",\"relation\":\"частота\",\"object\":\"раз в неделю\"}\n"
        "     ВЕРНО:     {\"subject\":\"пользователь\",\"relation\":\"частота рыбалки\",\"object\":\"раз в неделю\"}\n"

        "РАЗРЕШИ КОРЕФЕРЕНЦИЮ ВНУТРИ СООБЩЕНИЯ.\n"
        "НЕ ВЫДУМЫВАЙ ФАКТЫ.\n"
        "ИГНОРИРУЙ ЧИСТЫЕ ЭМОЦИИ БЕЗ ПРОВЕРЯЕМЫХ ФАКТОВ.\n"
        + ttl_block
        + f"ОНТОЛОГИЯ СЛОТОВ (СПРАВОЧНО): {slots_ru_json}\n"
        "ОТВЕТ ТОЛЬКО ВАЛИДНЫЙ JSON. БЕЗ MARKDOWN. БЕЗ ТЕКСТА ВНЕ JSON.\n"
        "СХЕМА ОТВЕТА:\n"
        f"{output_schema}\n"
        f"МАКСИМУМ ТРИПЛЕТОВ: {max_triplets}.\n"
        'ЕСЛИ НЕТ УСТОЙЧИВЫХ ФАКТОВ: {"triplets":[]}.'
    )

    def user_turn_no_slot(msg: str) -> str:
        return f"Сообщение пользователя:\n{msg}\n\nИзвлеки триплеты."

    def user_turn_with_slot(msg: str) -> str:
        return (
            f"Слот: {ru_slot}\n"
            f"Сообщение пользователя:\n{msg}\n\n"
            f"Извлеки триплеты только для слота «{ru_slot}»."
        )

    if include_slot:
        few_shot = triplet_single_pass_few_shot_messages(user_turn_no_slot, use_ttl=use_ttl)
        user_turn = user_turn_no_slot
    elif slot_name and ru_slot:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_with_slot, slot_name, use_ttl=use_ttl
        )
        user_turn = user_turn_with_slot
    else:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_no_slot, None, use_ttl=use_ttl
        )
        user_turn = user_turn_no_slot

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
