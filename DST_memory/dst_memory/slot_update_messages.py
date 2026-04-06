"""
Prompts for slot content update (add/update/delete by record_id).
Principle: one record per entity/topic. All facts about "жена" → one record.
All facts about "рыбалка" → one record. New info about same entity → update, not add.
"""
from __future__ import annotations

import json
from typing import Any

from .prompt_fewshots_ru import SLOT_UPDATE_EXTRA_FEWSHOT


def build_update_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:

    system = (
        "Вы — модуль управления содержимым слота памяти пользователя. "
        "Слот — это категория (например: транспорт, спорт, семья). "
        "Внутри слота хранятся записи — каждая запись описывает одну сущность или тему. "
        "На основе нового сообщения обновите записи слота. "
        "Возвращайте только JSON."
    )

    def user_turn(s: str, records: list[dict[str, Any]]) -> str:
        records_json = json.dumps(records, ensure_ascii=False)
        return (
            f"Слот: {slot_name}\n"
            f"Текущие записи: {records_json}\n\n"
            f"Сообщение:\n{s}\n\n"
            "Формат ответа — строго JSON, никакого текста до или после:\n"
            '{"operations":[<операция>, <операция>, ...]}\n\n'
            "Каждая операция — один из четырёх вариантов:\n"
            '{"op":"add","value":"..."}               — добавить новую запись\n'
            '{"op":"update","id":<id>,"value":"..."}  — обновить существующую запись по id\n'
            '{"op":"delete","id":<id>}                — удалить устаревшую запись по id\n'
            '{"op":"nothing"}                         — ничего не менять\n\n'
            "Можно вернуть несколько операций:\n"
            '{"operations":[{"op":"update","id":1,"value":"..."},{"op":"add","value":"..."}]}\n\n'
            "Правила:\n"
            "— nothing возвращай только одну, без других операций рядом.\n"
            "— Если несколько сущностей изменились — верни операцию для каждой.\n\n"
            "Структура значения записи:\n"
            "— Формат: «метка: факт1, факт2, ...» — метка обозначает сущность или тему.\n"
            "  Примеры: «жена: Людмила, любит Чехова», «рыбалка: раз в неделю, зимой тоже».\n\n"
            "— ОДИН СУБЪЕКТ = ОДНА ЗАПИСЬ. Все факты об одной сущности хранятся вместе.\n"
            "  В слоте «семья»: «жена: Людмила, любит Чехова» — одна запись, не две.\n"
            "  В слоте «хобби»: «рыбалка: с детства, раз в неделю» — одна запись.\n"
            "  Новый факт о той же сущности → update существующей записи, не add новой.\n\n"
            "— РАЗНЫЕ СУБЪЕКТЫ = РАЗНЫЕ ЗАПИСИ. Жена и сын — разные люди, разные записи.\n"
            "  «жена: Людмила» и «сын: Ромка» — два отдельных add.\n\n"
            "— Не сохраняй эмоции, оценки и лирику. Только проверяемые факты.\n"
            "  Плохо: «любовь как в первый день»  Хорошо: «семейное положение: женат 20 лет»\n\n"
            "— Запись должна быть понятна без исходного сообщения.\n"
            "  Плохо: «Тверь»  Хорошо: «маршруты: Тверь с семьёй, Кострома с семьёй»\n\n"
            "— Проверяй дубли: если сущность уже есть в записях — update, не add.\n"
        )

    few_shot = [
        # Новый слот — первое сообщение содержит несколько сущностей → несколько add
        {
            "role": "user",
            "content": user_turn(
                "женат уже 20 лет, есть сын Ромка и дочь Аня",
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
        # Новый факт об уже известной сущности → update, не add
        {
            "role": "user",
            "content": user_turn(
                "дочка переехала в Казань и вышла замуж",
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
        # Два факта о разных сущностях → два update
        {
            "role": "user",
            "content": user_turn(
                "сын теперь в институте, а жена стала преподавать в школе",
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
        # Нейтральное сообщение → nothing
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
        # Хобби: новый факт об уже известном хобби → update (не создавать 'литература' отдельно)
        {
            "role": "user",
            "content": user_turn(
                "а вот Чехова совсем не понимаю",
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
        # Хобби: несколько концертов → один record, update а не add
        {
            "role": "user",
            "content": user_turn(
                "сходил на Любэ — полное разочарование, Расторгуев уже не тот",
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
        # Путешествия: маршруты — одна запись, update при новой поездке
        {
            "role": "user",
            "content": user_turn(
                "только вернулись из Костромы, в следующий раз думаем в Казань",
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
        # update + add одновременно
        {
            "role": "user",
            "content": user_turn(
                "сменил машину на ford focus, и ещё взял самокат для коротких поездок",
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
        # delete + add — старое устарело
        {
            "role": "user",
            "content": user_turn(
                "бросил бегать, теперь хожу в бассейн",
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

    for msg, records, assistant_json in SLOT_UPDATE_EXTRA_FEWSHOT:
        few_shot.append({"role": "user", "content": user_turn(msg, records)})
        few_shot.append({"role": "assistant", "content": assistant_json})

    final_user = {"role": "user", "content": user_turn(user_message, existing_records)}
    return [{"role": "system", "content": system}] + few_shot + [final_user]


def parse_update_response(response_text: str) -> list[dict[str, Any]]:
    """Парсит ответ модели, возвращает список операций."""
    data = json.loads(response_text.strip())
    return data.get("operations", [])
