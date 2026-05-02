"""
Промпты для Варианта B: отдельный LLM-вызов детекции удалений.

Используется когда triplet_deletion_mode="llm_separate".
Модель получает: текущие факты слота + новое сообщение пользователя.
Возвращает: {"delete": [{subject, relation, object}, ...]}

Этот вызов ВСЕГДА получает контекст текущих фактов (иначе бессмысленен),
даже если slot_context_enabled=False для экстракции триплетов.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..slots.ontology import CANONICAL_TO_RU_LABEL


# ---------------------------------------------------------------------------
# Few-shot примеры для детекции удалений
# ---------------------------------------------------------------------------

_DELETION_FEWSHOTS: List[tuple[str, str, str, str]] = [
    # (slot_ru, existing_lines, user_message, assistant_json)
    (
        "ЛОКАЦИЯ",
        "пользователь | место жительства | москва",
        "я больше не живу в Москве, съехал",
        '{"delete":[{"subject":"пользователь","relation":"место жительства","object":"москва"}]}',
    ),
    (
        "РАБОТА",
        "пользователь | работает как | инженер\nпользователь | место работы | яндекс",
        "уволился из Яндекса месяц назад",
        '{"delete":[{"subject":"пользователь","relation":"место работы","object":"яндекс"}]}',
    ),
    (
        "РОМАНТИКА",
        "пользователь | встречается с | девушка пользователя\nдевушка пользователя | имя | катя",
        "мы расстались с Катей",
        '{"delete":[{"subject":"пользователь","relation":"встречается с","object":"девушка пользователя"},{"subject":"девушка пользователя","relation":"имя","object":"катя"}]}',
    ),
    (
        "ПИТОМЦЫ",
        "пользователь | есть кот | кот пользователя\nкот пользователя | имя | рыжик",
        "просто спасибо, понял",
        '{"delete":[]}',
    ),
    (
        "ПРИВЫЧКИ",
        "пользователь | курит | да\nпользователь | количество | пачка в день",
        "бросил курить три недели назад",
        '{"delete":[{"subject":"пользователь","relation":"курит","object":"да"},{"subject":"пользователь","relation":"количество","object":"пачка в день"}]}',
    ),
]


def build_deletion_messages(
    user_message: str,
    slot_name: str,
    existing_triplets: List[str],
) -> List[Dict[str, Any]]:
    """
    Построить чат-сообщения для отдельного LLM-вызова детекции удалений.

    Parameters
    ----------
    user_message : str
        Новое сообщение пользователя.
    slot_name : str
        Канонический ключ слота (LOCATION, WORK, ...).
    existing_triplets : list of str
        Текущие активные факты в слоте в формате "subject | relation | object".
    """
    ru_slot = CANONICAL_TO_RU_LABEL.get(slot_name, slot_name) if slot_name else slot_name
    facts_block = "\n".join(existing_triplets) if existing_triplets else "(нет фактов)"

    system = (
        f"СЛОТ: {ru_slot} ({slot_name}).\n"
        "ТЫ СИСТЕМА ДЕТЕКЦИИ УСТАРЕВШИХ ФАКТОВ.\n"
        "ПОЛУЧАЕШЬ ТЕКУЩИЕ ФАКТЫ СЛОТА И НОВОЕ СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЯ.\n"
        "ОПРЕДЕЛИ, КАКИЕ ФАКТЫ СЛЕДУЕТ УДАЛИТЬ, ИСХОДЯ ИЗ НОВОГО СООБЩЕНИЯ.\n\n"
        "ПРАВИЛА:\n"
        "  - Удаляй только факты, которые явно противоречат или отменяются новым сообщением.\n"
        "  - Если пользователь сообщил об изменении факта — удаляй СТАРЫЙ факт.\n"
        "  - Если пользователь явно сказал что факт больше не актуален — удаляй его.\n"
        "  - Если новое сообщение не отменяет никакие факты — возвращай пустой delete.\n"
        "  - НЕ удаляй факты на основе предположений.\n"
        "  - НЕ ВЫДУМЫВАЙ факты — только те, что присутствуют в списке.\n\n"
        "ОТВЕТ ТОЛЬКО ВАЛИДНЫЙ JSON. БЕЗ MARKDOWN. БЕЗ ТЕКСТА ВНЕ JSON.\n"
        'СХЕМА: {"delete":[{"subject":"...","relation":"...","object":"..."}]}\n'
        'ЕСЛИ НИЧЕГО НЕ УДАЛЯТЬ: {"delete":[]}'
    )

    def _user_turn(slot_ru: str, facts: str, msg: str) -> str:
        return (
            f"Слот: {slot_ru}\n"
            f"Текущие факты:\n{facts}\n\n"
            f"Новое сообщение: {msg}\n\n"
            "Какие факты нужно удалить?"
        )

    few_shots: List[Dict[str, Any]] = []
    for fs_slot, fs_facts, fs_msg, fs_ans in _DELETION_FEWSHOTS:
        few_shots.append({"role": "user", "content": _user_turn(fs_slot, fs_facts, fs_msg)})
        few_shots.append({"role": "assistant", "content": fs_ans})

    return (
        [{"role": "system", "content": system}]
        + few_shots
        + [{"role": "user", "content": _user_turn(ru_slot, facts_block, user_message)}]
    )


def parse_deletion_response(text: str) -> List[Dict[str, str]]:
    """
    Распарсить ответ LLM в список объектов для удаления.

    Returns
    -------
    List of {"subject": ..., "relation": ..., "object": ...} dicts.
    Raises ValueError on parse failure.
    """
    blob = (text or "").strip()
    if blob.startswith("```"):
        lines = blob.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        blob = "\n".join(lines).strip()

    try:
        obj = json.loads(blob)
    except Exception:
        start = blob.find("{")
        if start < 0:
            raise ValueError(f"No JSON object found in: {blob[:200]!r}")
        depth = 0
        for i in range(start, len(blob)):
            c = blob[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    obj = json.loads(blob[start:i + 1])
                    break
        else:
            raise ValueError(f"Unbalanced JSON in: {blob[:200]!r}")

    if not isinstance(obj, dict) or "delete" not in obj:
        raise ValueError(f"Expected {{\"delete\": [...]}} but got: {blob[:200]!r}")

    items = obj["delete"]
    if not isinstance(items, list):
        raise ValueError(f"\"delete\" field is not a list: {blob[:200]!r}")

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        s = str(item.get("subject", "")).strip().lower()
        r = str(item.get("relation", "")).strip().lower()
        o = str(item.get("object", "")).strip().lower()
        if s and r and o:
            result.append({"subject": s, "relation": r, "object": o})
    return result
