"""
Prompts for slot content update (add/update/delete by record_id).
"""
from __future__ import annotations

import json
from typing import Any

from .slot_ontology import SLOT_BY_ID, SLOT_LABEL_BY_ID


def build_update_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:

    system = (
        "You are a memory update module for a dialogue state slot.\n"
        "A slot stores atomic user facts as short strings.\n"
        "Given the new user message and existing records, return a list of update operations.\n"
        "Output must be strictly valid JSON, no markdown, no extra text."
    )

    def user_turn(s: str, records: list[dict[str, Any]]) -> str:
        records_json = json.dumps(records, ensure_ascii=False)
        slot_label = SLOT_LABEL_BY_ID.get(slot_name, slot_name.upper())
        slot_desc = SLOT_BY_ID.get(slot_name).description if slot_name in SLOT_BY_ID else ""
        return (
            f"SLOT_LABEL: {slot_label}\n"
            f"SLOT_ID: {slot_name}\n"
            f"SLOT_SCOPE: {slot_desc}\n"
            f"Existing records: {records_json}\n\n"
            f"Сообщение:\n```text\n{s}\n```\n\n"
            "Return format (strict JSON, nothing before/after):\n"
            '{"operations":[<операция>, <операция>, ...]}\n\n'
            "Each operation is one of:\n"
            '{"op":"add","value":"..."}               — добавить новый факт\n'
            '{"op":"update","id":<id>,"value":"..."}  — заменить существующий факт по id\n'
            '{"op":"delete","id":<id>}                — удалить устаревший факт по id\n'
            '{"op":"nothing"}                         — ничего не менять\n\n'
            "You may return multiple operations, e.g.:\n"
            '{"operations":[{"op":"update","id":1,"value":"..."},{"op":"add","value":"..."}]}\n\n'
            "Rules:\n"
            "- Multiple operations are allowed (add/update/delete).\n"
            "- If you return nothing, it must be the only operation.\n"
            "- If the slot is chosen but the message has no facts for this slot, return nothing.\n"
            "- CRITICAL: extract ONLY facts that belong to THIS SLOT (SLOT_LABEL / SLOT_SCOPE).\n"
            "  If the message contains facts for other slots, IGNORE them here.\n\n"
            "Requirements:\n"
            "— АТОМАРНОСТЬ: один факт = одна запись. Не склеивай несколько фактов через запятую или точку с запятой.\n"
            "  Плохо: «женат 20 лет, есть сын, любит путешествия»\n"
            "  Хорошо: три отдельных записи — «семейное положение: женат 20 лет», «дети: сын», «хобби: путешествия»\n"
            "— ТОЛЬКО ФАКТЫ О ПОЛЬЗОВАТЕЛЕ: не сохраняй эмоции, оценки и лирику.\n"
            "  Плохо: «любовь как в первый день», «показывают всякую хрень»\n"
            "  Хорошо: «семейное положение: женат 20 лет»\n"
            "— КОНТЕКСТ В ЗАПИСИ: запись должна быть понятна без исходного сообщения.\n"
            "  Плохо: «Тверь», «Россия, Турция»\n"
            "  Хорошо: «путешествия: был в Твери с семьёй», «предпочтения: путешествия по России, не Турция»\n"
            "— РАЗДЕЛЯЙ ФАКТЫ О РАЗНЫХ ЛЮДЯХ: если факт о члене семьи — указывай кто.\n"
            "  Плохо: «литературные предпочтения: жена - Чехов, я - нет»\n"
            "  Хорошо: две записи — «литература: не любит Чехова», «жена: любит Чехова»\n"
            "— СТРУКТУРА ЗНАЧЕНИЯ (метка темы + факт): в начале строки укажи, о чём запись, затем двоеточие и суть.\n"
            "  В слоте «семья» разные люди — разные строки: «дочь: …», «сын: …», «жена: …», «отец: …».\n"
            "  В других слотах аналогично: «работа: …», «спорт: …», «питомцы: …» — чтобы записи не сливались в один текст.\n"
            "— НЕТ ДУБЛЕЙ: перед добавлением проверяй, нет ли уже похожего факта в записях.\n"
            "  Если факт уже есть — update, не add.\n"
            "— НЕ СКЛЕИВАЙ РАЗНЫЕ ТЕМЫ В UPDATE: update должен менять ТОЛЬКО тот же факт, что и старая запись.\n"
            "  Если в сообщении появился новый факт другого типа — это add отдельной записью, а не дописывание к старой.\n"
        )

    few_shot = [
        # Новый слот — записей нет, несколько фактов → несколько атомарных add
        {
            "role": "user",
            "content": user_turn(
                "женат уже 20 лет, есть сын Ромка, живём дружно",
                [],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"add","value":"семейное положение: женат 20 лет"},'
                       '{"op":"add","value":"дети: сын Ромка"}'
                       ']}',
        },
        # Новый слот — сообщение нейтральное → nothing
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
        # Факты о разных людях → отдельные записи
        {
            "role": "user",
            "content": user_turn(
                "моя жена любит Чехова, я его вообще не понимаю",
                [
                    {"id": 1, "value": "семейное положение: женат"},
                    {"id": 2, "value": "дети: сын Ромка"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"add","value":"литература: не любит Чехова"},'
                       '{"op":"add","value":"жена: любит Чехова"}'
                       ']}',
        },
        # Эмоции и оценки — не сохраняем, только факт
        {
            "role": "user",
            "content": user_turn(
                "ходили с сыном в кино, показывают всякую хрень, зря сходили",
                [
                    {"id": 1, "value": "семейное положение: женат"},
                    {"id": 2, "value": "дети: сын Ромка"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"nothing"}]}',
        },
        # Запись с контекстом, не просто топоним
        {
            "role": "user",
            "content": user_turn(
                "вчера с семьёй из Твери вернулись, жена всё в Турцию хочет, а я говорю — по России ещё не всё посмотрели",
                [
                    {"id": 1, "value": "семейное положение: женат"},
                    {"id": 2, "value": "дети: сын Ромка"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":['
                       '{"op":"add","value":"путешествия: был с семьёй в Твери"},'
                       '{"op":"add","value":"предпочтения: путешествия по России, не за границу"},'
                       '{"op":"add","value":"жена: хочет в Турцию"}'
                       ']}',
        },
        # Проверка на дубль — факт уже есть, делаем update а не add
        {
            "role": "user",
            "content": user_turn(
                "кстати, мы с женой уже 25 лет вместе, не 20",
                [
                    {"id": 1, "value": "семейное положение: женат 20 лет"},
                    {"id": 2, "value": "дети: сын Ромка"},
                ],
            ),
        },
        {
            "role": "assistant",
            "content": '{"operations":[{"op":"update","id":1,"value":"семейное положение: женат 25 лет"}]}',
        },
        # update + add — один факт изменился, появился новый
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
            "content": '{"operations":['
                       '{"op":"update","id":1,"value":"машина: ford focus"},'
                       '{"op":"add","value":"самокат: для коротких поездок"}'
                       ']}',
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

    final_user = {"role": "user", "content": user_turn(user_message, existing_records)}
    return [{"role": "system", "content": system}] + few_shot + [final_user]


def parse_update_response(response_text: str) -> list[dict[str, Any]]:
    """Парсит ответ модели, возвращает список операций."""
    data = json.loads(response_text.strip())
    return data.get("operations", [])


def build_operation_classification_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
) -> list[dict[str, Any]]:
    system = (
        "Вы — классификатор операции обновления памяти. "
        "Верните строго одно слово из списка: add, update, delete, nothing."
    )

    def user_turn(s: str, records: list[dict[str, Any]]) -> str:
        return (
            f"Слот: {slot_name}\n"
            f"Текущие записи: {json.dumps(records, ensure_ascii=False)}\n\n"
            f"Сообщение:\n```text\n{s}\n```\n\n"
            "Нужно выбрать ровно одну операцию: add / update / delete / nothing.\n"
            "Ответ: только одно слово."
        )

    few_shot = [
        {
            "role": "user",
            "content": user_turn(
                "у меня теперь ford focus вместо kia rio",
                [{"id": 1, "value": "машина: kia rio"}],
            ),
        },
        {"role": "assistant", "content": "update"},
        {
            "role": "user",
            "content": user_turn(
                "бросил бегать, теперь хожу в бассейн",
                [{"id": 1, "value": "спорт: бег по утрам"}],
            ),
        },
        {"role": "assistant", "content": "delete"},
        {
            "role": "user",
            "content": user_turn(
                "в выходные снова поедем в Тверь",
                [{"id": 1, "value": "маршруты: Нижний Новгород"}],
            ),
        },
        {"role": "assistant", "content": "add"},
        {
            "role": "user",
            "content": user_turn("окей, понял", [{"id": 1, "value": "спорт: футбол по пятницам"}]),
        },
        {"role": "assistant", "content": "nothing"},
    ]
    return [{"role": "system", "content": system}] + few_shot + [
        {"role": "user", "content": user_turn(user_message, existing_records)}
    ]


def build_target_record_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
    op: str,
) -> list[dict[str, Any]]:
    system = "Вы выбираете target record id для операции update/delete. Ответ: только целое число id."

    def user_turn(s: str, records: list[dict[str, Any]], operation: str) -> str:
        return (
            f"Слот: {slot_name}\n"
            f"Операция: {operation}\n"
            f"Записи: {json.dumps(records, ensure_ascii=False)}\n\n"
            f"Сообщение:\n```text\n{s}\n```\n\n"
            "Выберите единственный id записи, которая должна быть изменена. Ответ: только id."
        )

    few_shot = [
        {
            "role": "user",
            "content": user_turn(
                "мы с женой уже 25 лет вместе, не 20",
                [{"id": 1, "value": "семейное положение: женат 20 лет"}, {"id": 2, "value": "сын: Ромка"}],
                "update",
            ),
        },
        {"role": "assistant", "content": "1"},
        {
            "role": "user",
            "content": user_turn(
                "бегать перестал совсем",
                [{"id": 3, "value": "спорт: бег по утрам"}, {"id": 4, "value": "спорт: плавание"}],
                "delete",
            ),
        },
        {"role": "assistant", "content": "3"},
    ]
    return [{"role": "system", "content": system}] + few_shot + [
        {"role": "user", "content": user_turn(user_message, existing_records, op)}
    ]


def build_single_value_messages(
    slot_name: str,
    existing_records: list[dict[str, Any]],
    user_message: str,
    op: str,
    target_id: int | None,
) -> list[dict[str, Any]]:
    system = (
        "Вы формируете ОДНО значение записи памяти. "
        "Ответ — строго JSON вида {\"value\":\"...\"} без пояснений."
    )
    target = f"{target_id}" if target_id is not None else "null"
    user = (
        f"Слот: {slot_name}\n"
        f"Операция: {op}\n"
        f"Target id: {target}\n"
        f"Текущие записи: {json.dumps(existing_records, ensure_ascii=False)}\n\n"
        f"Сообщение:\n```text\n{user_message}\n```\n\n"
        "Сформируй одно атомарное значение для записи. "
        "Не добавляй эмоций и лишнего контекста."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_triplets_messages(
    slot_name: str,
    user_message: str,
    value: str,
) -> list[dict[str, Any]]:
    system = (
        "You extract knowledge graph artifacts from a user memory fact.\n"
        "Return strictly valid JSON with TWO fields: entities[] and relations[]. No markdown.\n"
        "Schema:\n"
        "{\n"
        '  \"entities\": [{\"entity_name\": str, \"entity_type\": str, \"description\": str}],\n'
        '  \"relations\": [{\"source_entity\": str, \"target_entity\": str, \"relation_type\": str, \"description\": str, \"relationship_strength\": int}]\n'
        "}\n"
        "Rules:\n"
        "- Use English UPPER_CASE for entity_type and UPPER_SNAKE_CASE for relation_type.\n"
        "- Always include an entity for the user: entity_name = \"USER\", entity_type = \"PERSON\".\n"
        "- Entities and relations must be grounded in the text; no hallucination.\n"
        "- relation endpoints MUST match entity_name values exactly.\n"
        "- Prefer 1-5 entities and 1-5 relations.\n"
    )
    user = (
        f"Слот: {slot_name}\n"
        f"Сообщение пользователя:\n```text\n{user_message}\n```\n\n"
        f"Сохраненное значение:\n```text\n{value}\n```\n\n"
        "Extract entities and relations for THIS memory fact. "
        "If nothing can be extracted, return empty lists."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]