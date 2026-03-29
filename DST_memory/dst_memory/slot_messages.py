"""
Сообщения для модели слотов: system, few-shot, финальный user (структура из промпт-спеки).
"""

from __future__ import annotations

import json
from typing import Any


def build_messages(message: str, max_s: int, slots: list[str]) -> list[dict[str, Any]]:
    system = (
        "Вы — эксперт по извлечению информации о пользователе из его сообщений. "
        "Слот — это широкая сфера жизни человека: еда, спорт, семья, работа, здоровье, "
        "транспорт, хобби, питомцы, финансы, покупки и подобные категории."
    )

    def user_turn(msg: str, existing_slots: list[str]) -> str:
        slots_json = json.dumps(existing_slots, ensure_ascii=False)
        return (
            f"Существующие слоты: {slots_json}\n"
            f"Максимум слотов в ответе: {max_s}\n\n"
            f"Сообщение:\n```text\n{msg}\n```\n\n"
            "Задача: определи слоты, которые раскрывают что-то важное о самом пользователе "
            "(его предпочтения, привычки, интересы, жизненные обстоятельства).\n"
            "Если сообщение — просто факт, вопрос или действие без личной окраски, "
            "верни пустой список.\n"
            "Используй существующие слоты дословно. "
            "Новый слот — только если ни один не подходит. "
            "Слот — одно ключевое слово-существительное (широкая категория). "
            "Второе слово добавляй только если без него смысл категории теряется. "
            "Плохо: «личный интерес», «сфера хобби», «домашний питомец». "
            "Хорошо: «интересы», «хобби», «питомцы». "
            "Ответ — только JSON."
        )

    few_shot = [
        # Личное отношение → слот
        {
            "role": "user",
            "content": user_turn("обожаю эстрагон, кладу его вообще везде", ["еда"]),
        },
        {"role": "assistant", "content": '{"slot_assignments":["еда"]}'},
        # Просто действие без личной окраски → пусто
        {
            "role": "user",
            "content": user_turn("добавил эстрагон в соус", ["еда"]),
        },
        {"role": "assistant", "content": '{"slot_assignments":[]}'},
        # Личные обстоятельства → несколько слотов
        {
            "role": "user",
            "content": user_turn(
                "у меня такой стресс на работе, не могу нормально спать",
                ["работа", "здоровье"],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["работа","здоровье"]}'},
        # Нейтральный вопрос → пусто
        {
            "role": "user",
            "content": user_turn("а что, завтра дождь будет?", ["хобби"]),
        },
        {"role": "assistant", "content": '{"slot_assignments":[]}'},
        # Личный питомец → лаконичный слот (не "домашний питомец")
        {
            "role": "user",
            "content": user_turn(
                "мой кот Барсик вообще не переносит других животных, вечно шипит",
                ["питомцы", "семья"],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["питомцы"]}'},
        # Новый слот с размытым сообщением → одно слово, не "личный интерес"
        {
            "role": "user",
            "content": user_turn(
                "люблю читать про историю, особенно про средневековье", ["работа"]
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["хобби"]}'},
        # Несколько сфер одновременно
        {
            "role": "user",
            "content": user_turn(
                "в выходные едем с женой в горы, я давно хотел попробовать треккинг",
                ["семья", "хобби", "спорт"],
            ),
        },
        {"role": "assistant", "content": '{"slot_assignments":["семья","спорт"]}'},
        # Новый слот (не было в списке)
        {
            "role": "user",
            "content": user_turn("купил велик, теперь каждое утро катаюсь до работы", []),
        },
        {"role": "assistant", "content": '{"slot_assignments":["спорт","транспорт"]}'},
        # Короткая нейтральная реплика → пусто
        {
            "role": "user",
            "content": user_turn("окей, понял", ["работа"]),
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