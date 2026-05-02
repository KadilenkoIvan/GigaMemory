"""
Slot content update (add/update/delete by record_id) — English UI.
Principle: one record per entity/topic.
"""
from __future__ import annotations

import json
from typing import Any

from .prompt_fewshots import SLOT_UPDATE_EXTRA_FEWSHOT


def build_update_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:

    system = (
        "You manage records inside a user-memory slot. "
        "A slot is a category (e.g. transport, sport, family). "
        "Each record describes one entity or topic. "
        "Update records from the new message. "
        "Return JSON only."
    )

    def user_turn(s: str, records: list[dict[str, Any]]) -> str:
        records_json = json.dumps(records, ensure_ascii=False)
        return (
            f"Slot: {slot_name}\n"
            f"Current records: {records_json}\n\n"
            f"Message:\n{s}\n\n"
            "Response format — JSON only, no text before or after:\n"
            '{"operations":[<op>, <op>, ...]}\n\n'
            "Each operation is one of:\n"
            '{"op":"add","value":"..."}               — add a new record\n'
            '{"op":"update","id":<id>,"value":"..."}  — update record by id\n'
            '{"op":"delete","id":<id>}                — delete obsolete record\n'
            '{"op":"nothing"}                         — no change\n\n'
            "Multiple operations allowed:\n"
            '{"operations":[{"op":"update","id":1,"value":"..."},{"op":"add","value":"..."}]}\n\n'
            "Rules:\n"
            "— Return nothing alone, without other ops.\n"
            "— If several entities changed — one operation per entity.\n\n"
            "Record value structure:\n"
            "— Format: «label: fact1, fact2, ...» — label is entity or topic.\n"
            "  Examples: «жена: Людмила, любит Чехова», «рыбалка: раз в неделю, зимой тоже».\n\n"
            "— ONE SUBJECT = ONE RECORD. Facts about the same entity stay together.\n"
            "  In «family»: «жена: Людмила, любит Чехова» — one record.\n"
            "  New fact about same entity → update, not add.\n\n"
            "— DIFFERENT SUBJECTS = DIFFERENT RECORDS.\n"
            "  «жена: Людмила» and «сын: Ромка» — two separate adds.\n\n"
            "— No emotions or lyricism. Verifiable facts only.\n"
            "— Record must be understandable without the original message.\n"
            "— If entity already exists — update, not add.\n"
        )

    few_shot = [
        {
            "role": "user",
            "content": user_turn(
                "Married 20 years, have son Romka and daughter Anya",
                [],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"add","value":"семейное положение: женат 20 лет"},'
                       '{"op":"add","value":"сын: Ромка"},'
                       '{"op":"add","value":"дочь: Аня"}'
                       ']}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Daughter moved to Kazan and got married",
                [
                    {"id": 1, "value": "семейное положение: женат 20 лет"},
                    {"id": 2, "value": "дочь: Аня"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":2,"value":"дочь: Аня, живёт в Казани, замужем"}]}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Son is in university now, wife teaches at a school",
                [
                    {"id": 1, "value": "жена: Людмила"},
                    {"id": 2, "value": "сын: Ромка"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"update","id":1,"value":"жена: Людмила, работает учителем в школе"},'
                       '{"op":"update","id":2,"value":"сын: Ромка, учится в институте"}'
                       ']}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Okay, thanks",
                [],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"nothing"}]}',
        },
        {
            "role": "user",
            "content": user_turn(
                "I really don't get Chekhov though",
                [
                    {"id": 1, "value": "рыбалка: с детства, раз в неделю, даже зимой"},
                    {"id": 2, "value": "чтение: любит читать, любимый автор — Достоевский"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":2,"value":"чтение: любит читать, любимый автор — Достоевский, не любит Чехова"}]}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Went to Lyube — total disappointment, Rastorguev is not what he was",
                [
                    {"id": 1, "value": "рыбалка: с детства, раз в неделю, даже зимой"},
                    {"id": 2, "value": "чтение: любит читать, любимый автор — Достоевский"},
                    {"id": 3, "value": "концерты: Аукцыон с сыном; органный с женой — не понравился"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":3,"value":"концерты: Аукцыон с сыном; органный с женой — не понравился; Любэ — не понравился"}]}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Just got back from Kostroma, next time thinking Kazan",
                [
                    {"id": 1, "value": "маршруты: Тверь с семьёй"},
                    {"id": 2, "value": "предпочтения: путешествия по России"},
                    {"id": 3, "value": "планы: Нижний Новгород с семьёй"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"update","id":1,"value":"маршруты: Тверь с семьёй, Кострома с семьёй"},'
                       '{"op":"update","id":3,"value":"планы: Казань с семьёй"}'
                       ']}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Switched car to Ford Focus, also got a scooter for short trips",
                [
                    {"id": 1, "value": "машина: kia rio"},
                    {"id": 2, "value": "велосипед: городской"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"update","id":1,"value":"машина: ford focus"},'
                       '{"op":"add","value":"самокат: для коротких поездок"}'
                       ']}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Stopped running, now I swim at the pool",
                [{"id": 1, "value": "спорт: бег по утрам"}],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"delete","id":1},'
                       '{"op":"add","value":"спорт: плавание в бассейне"}'
                       ']}',
        },
        {
            "role": "user",
            "content": user_turn(
                "Okay, understood",
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

    for msg, records, assistant_json in SLOT_UPDATE_EXTRA_FEWSHOT:
        few_shot.append({"role": "user", "content": user_turn(msg, records)})
        few_shot.append({"role": "assistant", "content": assistant_json})

    final_user = {"role": "user", "content": user_turn(user_message, existing_records)}
    return [{"role": "system", "content": system}] + few_shot + [final_user]
