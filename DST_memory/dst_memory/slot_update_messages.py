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
        "Вы — модуль управления содержимым слота памяти пользователя. "
        "Слот — это категория (например: транспорт, спорт, семья). "
        "Внутри слота хранятся атомарные записи — короткие факты о пользователе. "
        "На основе нового сообщения обновите записи слота. "
        "Возвращайте только JSON."
    )

    def user_turn(s: str, records: list[dict[str, Any]]) -> str:
        records_json = json.dumps(records, ensure_ascii=False)
        return (
            f"Слот: {slot_name}\n"
            f"Текущие записи: {records_json}\n\n"
            f"Сообщение:\n```text\n{s}\n```\n\n"
            "Формат ответа — строго JSON, никакого текста до или после:\n"
            '{"operations":[<операция>, <операция>, ...]}\n\n'
            "Каждая операция — один из четырёх вариантов:\n"
            '{"op":"add","value":"..."}               — добавить новый факт\n'
            '{"op":"update","id":<id>,"value":"..."}  — заменить существующий факт по id\n'
            '{"op":"delete","id":<id>}                — удалить устаревший факт по id\n'
            '{"op":"nothing"}                         — ничего не менять\n\n'
            "Можно вернуть несколько операций в одном ответе, например:\n"
            '{"operations":[{"op":"update","id":1,"value":"..."},{"op":"add","value":"..."}]}\n\n'
            "Правила:\n"
            "— Если несколько фактов изменились — верни операцию для каждого.\n"
            "— nothing возвращай только одну, без других операций рядом.\n"
            "— Извлекай только атомарные факты. Минимизируй количество операций."
        )

    few_shot = [
        # Новый слот — записей нет, информация важная → сразу add
        {
            "role": "user",
            "content": user_turn(
                "я вегетарианец уже лет пять",
                [],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"add","value":"питание: вегетарианец"}]}',
        },
        # Новый слот — записей нет, сообщение нейтральное → nothing
        {
            "role": "user",
            "content": user_turn(
                "хорошо, спасибо",
                [],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"nothing"}]}',
        },
        # add — новый факт к одной существующей записи
        {
            "role": "user",
            "content": user_turn(
                "ещё у меня есть права на мотоцикл",
                [{"id": 1, "value": "машина: hyundai solaris"}],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"add","value":"мотоцикл: есть права"}]}',
        },
        # update — один факт изменился среди нескольких записей
        {
            "role": "user",
            "content": user_turn(
                "сменил машину, теперь езжу на kia rio",
                [
                    {"id": 1, "value": "машина: toyota camry"},
                    {"id": 2, "value": "мотоцикл: есть права"},
                    {"id": 3, "value": "велосипед: городской"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":1,"value":"машина: kia rio"}]}',
        },
        # delete — пользователь отказался от одного факта, остальные не трогаем
        {
            "role": "user",
            "content": user_turn(
                "продал мотоцикл",
                [
                    {"id": 1, "value": "машина: kia rio"},
                    {"id": 2, "value": "мотоцикл: есть права"},
                    {"id": 3, "value": "велосипед: городской"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"delete","id":2}]}',
        },
        # update + add — один факт изменился и появился новый
        {
            "role": "user",
            "content": user_turn(
                "пересел на ford focus, и ещё взял самокат для коротких поездок",
                [
                    {"id": 1, "value": "машина: kia rio"},
                    {"id": 2, "value": "велосипед: городской"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":1,"value":"машина: ford focus"},{"op":"add","value":"самокат: для коротких поездок"}]}',
        },
        # update + update — изменились сразу два существующих факта
        {
            "role": "user",
            "content": user_turn(
                "теперь бегаю не по утрам а по вечерам, и хожу в бассейн раз в неделю вместо двух",
                [
                    {"id": 1, "value": "бег: по утрам"},
                    {"id": 2, "value": "плавание: 2 раза в неделю"},
                    {"id": 3, "value": "велосипед: по выходным"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":1,"value":"бег: по вечерам"},{"op":"update","id":2,"value":"плавание: 1 раз в неделю"}]}',
        },
        # delete + add — старый факт устарел, появился новый
        {
            "role": "user",
            "content": user_turn(
                "бросил бегать, теперь хожу в бассейн",
                [{"id": 1, "value": "спорт: бег по утрам"}],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"delete","id":1},{"op":"add","value":"спорт: плавание в бассейне"}]}',
        },
        # nothing — сообщение не относится к слоту
        {
            "role": "user",
            "content": user_turn(
                "окей, понял",
                [
                    {"id": 1, "value": "машина: ford focus"},
                    {"id": 2, "value": "велосипед: городской"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"nothing"}]}',
        },
    ]

    final_user = {"role": "user", "content": user_turn(user_message, existing_records)}
    return [{"role": "system", "content": system}] + few_shot + [final_user]


def parse_update_response(response_text: str) -> list[dict[str, Any]]:
    """Парсит ответ модели, возвращает список операций."""
    data = json.loads(response_text.strip())
    return data.get("operations", [])