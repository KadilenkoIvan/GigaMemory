"""
Сообщения для модели слотов: system, few-shot, финальный user.

Текущая версия использует фиксированную онтологию слотов и ожидает
строго JSON-ответ со списком выбранных слотов.
"""

from __future__ import annotations

import json
from typing import Any

from .slot_ontology import SLOT_LABEL_BY_ID, SLOT_ID_BY_LABEL, SLOT_IDS, slot_catalog_markdown


def build_messages(
    message: str,
    max_s: int,
    slots: list[str],
    force_at_least_one: bool = False,
) -> list[dict[str, Any]]:
    system = (
        "You are a long-term memory router for a conversational assistant.\n"
        "Given a user message, select which memory slots should be updated.\n"
        "Slots are fixed. Output must be strictly valid JSON, no markdown, no extra text."
    )

    def user_turn(msg: str, existing_slots: list[str]) -> str:
        # existing_slots may contain internal ids; expose prompt labels in uppercase.
        existing_labels: list[str] = []
        for s in existing_slots:
            key = str(s).strip()
            if key in SLOT_IDS:
                # map id -> label if possible, else keep as upper
                existing_labels.append(SLOT_LABEL_BY_ID.get(key, key.upper()))
            else:
                existing_labels.append(key.upper())
        slots_json = json.dumps(existing_labels, ensure_ascii=False)
        allowed_labels = sorted(SLOT_ID_BY_LABEL.keys())
        allowed_json = json.dumps(allowed_labels, ensure_ascii=False)
        return (
            f"Active slots (already have stored records): {slots_json}\n"
            f"Allowed slots (ontology): {allowed_json}\n"
            f"Max slots in answer: {max_s}\n\n"
            f"Сообщение:\n```text\n{msg}\n```\n\n"
            "Slot catalog:\n"
            f"{slot_catalog_markdown()}\n\n"
            "Rules:\n"
            "- Select ONLY from the allowed ontology labels.\n"
            "- Select slots only if the message contains stable personal facts/preferences/relations/habits/plans.\n"
            "- If nothing useful for memory is present, return an empty list.\n"
            + (
                "- IMPORTANT: the message was pre-filtered as IMPORTANT for memory. "
                "If it contains any stable personal fact, you MUST choose at least one slot.\n"
                if force_at_least_one
                else ""
            )
            + "\n"
            "Return format строго:\n"
            '{"slot_assignments":["<SLOT_LABEL>", "..."]}'
        )

    few_shot = [
        {
            "role": "user",
            "content": user_turn(
                "Исполнилось 45, пошёл паспорт менять. В паспорте фамилию написали с ошибкой.",
                [],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["IDENTITY"]}'},
        {
            "role": "user",
            "content": user_turn(
                "жена расстроилась, мы уже 20 лет женаты",
                [],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["FAMILY"]}'},
        {
            "role": "user",
            "content": user_turn(
                "по пятницам с мужиками в футбол играю уже лет 15",
                [],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["SPORTS"]}'},
        {
            "role": "user",
            "content": user_turn(
                "вчера с семьёй из Твери вернулись, в следующий раз думаем в Нижний Новгород",
                ["family_relationships"],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["TRAVEL"]}'},
        {
            "role": "user",
            "content": user_turn(
                "окей, понял, спасибо",
                ["family_relationships"],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":[]}'},
    ]

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(message, slots)}]
    )


def parse_response(response_text: str) -> list[str]:
    """Парсит ответ модели, возвращает список слотов (может быть пустым)."""
    data = json.loads(response_text.strip())
    return data.get("slot_assignments", [])