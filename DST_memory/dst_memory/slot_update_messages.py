"""
Prompts for slot content update (add/update/delete by record_id).
"""

from __future__ import annotations

import json
from typing import Any


def build_update_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:
    system = (
        "Вы — модуль управления содержимым слота памяти пользователя.\n"
        "Слот — это категория (например: транспорт, спорт, семья). Внутри слота хранятся атомарные записи.\n"
        "Ваша задача — на основе нового сообщения пользователя обновить атомарные записи слота.\n"
        "Возвращайте только JSON.\n\n"
        "Формат JSON строго:\n"
        '{ "operations": [ {"op":"add","value":"..."}, {"op":"update","id":1,"value":"..."}, {"op":"delete","id":2} ] }\n\n'
        "Правила:\n"
        "1) Извлекайте только полезную, атомарную информацию. Не сохраняйте весь текст сообщения.\n"
        "2) add — новый факт; update — изменить существующий факт по id; delete — удалить устаревший факт по id.\n"
        "3) Минимизируйте количество операций.\n"
        "4) Если сообщение не добавляет полезной информации для этого слота — operations должен быть пустым списком.\n"
    )

    def user_turn(s: str, records: list[dict[str, Any]]) -> str:
        records_json = json.dumps(records, ensure_ascii=False)
        return (
            f"Слот: {slot_name}\n"
            f"Текущие записи слота: {records_json}\n\n"
            "Сообщение:\n"
            f"```text\n{s}\n```\n"
        )

    few_shot = [
        # delete
        {
            "role": "user",
            "content": user_turn(
                "Я продал свою Ладу",
                [{"id": 1, "value": "машина: lada granta"}],
            ),
        },
        {"role": "assistant", "content": '{"operations":[{"op":"delete","id":1}]}'},
        # update
        {
            "role": "user",
            "content": user_turn(
                "Сменил машину, теперь у меня Kia Rio",
                [{"id": 1, "value": "машина: toyota camry"}],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":1,"value":"машина: kia rio"}]}',
        },
        # add
        {
            "role": "user",
            "content": user_turn(
                "У меня машина Hyundai Solaris",
                [],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"add","value":"машина: hyundai solaris"}]}',
        },
        # no-op
        {
            "role": "user",
            "content": user_turn(
                "Окей, понял",
                [{"id": 1, "value": "машина: kia rio"}],
            ),
        },
        {"role": "assistant", "content": '{"operations":[]}'},
    ]

    final_user = {"role": "user", "content": user_turn(user_message, existing_records)}

    return [{"role": "system", "content": system}] + few_shot + [final_user]

