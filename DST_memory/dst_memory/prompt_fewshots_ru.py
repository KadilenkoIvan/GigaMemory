"""
Русскоязычные few-shot для промптов DST_memory (чередование user + assistant).
Слоты в slot_assignments: русские метки (СЕМЬЯ, РАБОТА) — резолвятся в ontology.RU_SLOT_TO_CANONICAL.
В memory gate списки слотов — канонические английские ключи (как в DialogueMemoryState).
Связи в триплетах: русские метки, ВЕРХНИЙ_РЕГИСТР_С_ПОДЧЁРКИВАНИЕМ.
Субъект пользователя по умолчанию: ПОЛЬЗОВАТЕЛЬ.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# --- Выбор слотов ---
SLOT_SELECT_FEWSHOT: List[Tuple[str, str]] = [
    ("мы с женой женаты десять лет, у нас есть сын", '{"slot_assignments":["СЕМЬЯ"]}'),
    ("работаю водителем такси и по выходным играю в футбол", '{"slot_assignments":["РАБОТА","СПОРТ"]}'),
    ("окей, понял, спасибо", '{"slot_assignments":[]}'),
    ("у меня гипертония, пью таблетки по назначению врача", '{"slot_assignments":["ЗДОРОВЬЕ"]}'),
    ("каждый день откладываю часть зарплаты", '{"slot_assignments":["ФИНАНСЫ"]}'),
    ("закончил НГУ, думаю про магистратуру", '{"slot_assignments":["ОБРАЗОВАНИЕ","ЦЕЛИ"]}'),
    ("живу в Новосибирске, раньше жил в Омске", '{"slot_assignments":["ЛОКАЦИЯ"]}'),
    ("в августе едем в Сочи с семьёй", '{"slot_assignments":["ПУТЕШЕСТВИЯ","СЕМЬЯ"]}'),
    ("у меня кот Барсик, боится собак", '{"slot_assignments":["ПИТОМЦЫ"]}'),
    ("меня зовут Иван, мне двадцать семь", '{"slot_assignments":["ЛИЧНОСТЬ"]}'),
    ("встречаюсь с девушкой полтора года", '{"slot_assignments":["РОМАНТИКА"]}'),
    ("мы с лучшим другом Димой дружим со школы", '{"slot_assignments":["ДРУЗЬЯ"]}'),
    ("люблю читать фантастику по вечерам", '{"slot_assignments":["ХОББИ","ПРЕДПОЧТЕНИЯ"]}'),
    ("не ем мясо, но ем рыбу", '{"slot_assignments":["ЕДА"]}'),
    ("сегодня на работе завал, ненавижу смены", '{"slot_assignments":[]}'),
    ("купил макбук для работы", '{"slot_assignments":["ТЕХНИКА","РАБОТА"]}'),
    ("машина киа рио, до этого был логан", '{"slot_assignments":["ТРАНСПОРТ"]}'),
    ("встаю в шесть утра по будням", '{"slot_assignments":["РАСПИСАНИЕ","ПРИВЫЧКИ"]}'),
    ("хочу через год сменить профессию на продукт", '{"slot_assignments":["ЦЕЛИ","РАБОТА"]}'),
    ("в прошлом месяце выступал на конференции", '{"slot_assignments":["СОБЫТИЯ","РАБОТА"]}'),
    ("сильная тревога перед экзаменами", '{"slot_assignments":["ПСИХИКА"]}'),
    ("сделал ремонт на кухне, живём в двушке", '{"slot_assignments":["ДОМ"]}'),
    ("курю по пачке в день, стыдно", '{"slot_assignments":["ПРИВЫЧКИ","ЗДОРОВЬЕ"]}'),
    ("аллергия на арахис с детства", '{"slot_assignments":["ЗДОРОВЬЕ","ЕДА"]}'),
]


def slot_select_few_shot_messages(user_turn_fn) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in SLOT_SELECT_FEWSHOT:
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    return out


def _t(items: List[dict]) -> str:
    return json.dumps({"triplets": items}, ensure_ascii=False)


# Общие примеры для per-slot (без поля slot в JSON)
TRIPLET_PER_SLOT_SHARED: List[Tuple[str, str]] = [
    (
        "мы с женой семь лет в браке, есть сын Артём",
        _t(
            [
                {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖЕНАТ_С", "object": "ЖЕНА"},
                {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_РЕБЁНКА", "object": "СЫН_АРТЁМ"},
            ]
        ),
    ),
    (
        "жена получила повышение, она теперь руководит отделом",
        _t(
            [
                {"subject": "ЖЕНА", "relation": "ПОЛУЧИЛА", "object": "ПОВЫШЕНИЕ"},
                {"subject": "ЖЕНА", "relation": "РУКОВОДИТ", "object": "ОТДЕЛ"},
            ]
        ),
    ),
    (
        "просто спасибо",
        '{"triplets":[]}',
    ),
]


# Дополнительные пары под конкретный канонический слот (англ. ключ)
TRIPLET_PER_SLOT_BY_SLOT: Dict[str, List[Tuple[str, str]]] = {
    "FAMILY": [
        (
            "дочка переехала в Казань",
            _t([{"subject": "ДОЧЬ", "relation": "ЖИВЁТ_В", "object": "КАЗАНЬ"}]),
        ),
        (
            "мы с сыном ходим на бокс два раза в неделю",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "С_СЫНОМ", "object": "БОКС"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЧАСТОТА", "object": "ДВА_РАЗА_В_НЕДЕЛЮ"},
                ]
            ),
        ),
    ],
    "WORK": [
        (
            "устроился в Яндекс аналитиком данных",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАБОТАЕТ_В", "object": "ЯНДЕКС"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДОЛЖНОСТЬ", "object": "АНАЛИТИК_ДАННЫХ"},
                ]
            ),
        ),
        (
            "уволился с прошлой работы в марте",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "УВОЛИЛСЯ", "object": "МАРТ"}]),
        ),
    ],
    "PETS": [
        (
            "кот Барсик не переносит других котов",
            _t(
                [
                    {"subject": "КОТ_БАРСИК", "relation": "НЕ_ПЕРЕНОСИТ", "object": "ДРУГИЕ_КОТЫ"},
                ]
            ),
        ),
        (
            "собаку зовут Луна, гуляю с ней утром",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_СОБАКУ", "object": "ЛУНА"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ВЫГУЛ", "object": "УТРО"},
                ]
            ),
        ),
    ],
    "FOOD": [
        (
            "не ем глютен и лактозу",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИСКЛЮЧАЕТ", "object": "ГЛЮТЕН"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИСКЛЮЧАЕТ", "object": "ЛАКТОЗА"},
                ]
            ),
        ),
        (
            "люблю борщ и соленья",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "НРАВИТСЯ", "object": "БОРЩ_И_СОЛЕНЬЯ"}]),
        ),
    ],
    "HEALTH": [
        (
            "диабет второго типа, на учёте у эндокринолога",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДИАГНОЗ", "object": "ДИАБЕТ_2_ТИПА"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "НА_УЧЁТЕ", "object": "ЭНДОКРИНОЛОГ"},
                ]
            ),
        ),
    ],
    "MENTAL_HEALTH": [
        (
            "ходил к психотерапевту полгода, стало легче",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ТЕРАПИЯ", "object": "ПОЛГОДА"},
                ]
            ),
        ),
    ],
    "EDUCATION": [
        (
            "магистратура в НГУ, кафедра ИИ",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "УЧИТСЯ_В", "object": "НГУ_МАГИСТРАТУРА"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "КАФЕДРА", "object": "ИИ"},
                ]
            ),
        ),
    ],
    "SPORTS": [
        (
            "бегаю десять километров по воскресеньям",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЗАНИМАЕТСЯ", "object": "БЕГ_10КМ_ВОСКРЕСЕНЬЕ"}]),
        ),
    ],
    "LOCATION": [
        (
            "переехал в Красноярск из Иркутска",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖИВЁТ_В", "object": "КРАСНОЯРСК"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАНЕЕ_ЖИЛ_В", "object": "ИРКУТСК"},
                ]
            ),
        ),
    ],
    "FINANCE": [
        (
            "ипотека в Сбере, платёж сорок тысяч",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИПОТЕКА", "object": "СБЕР"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ПЛАТЁЖ", "object": "40000_РУБ"},
                ]
            ),
        ),
    ],
    "VEHICLES": [
        (
            "поменял масло, езжу на форде",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "АВТО", "object": "FORD"}]),
        ),
    ],
    "TRAVEL": [
        (
            "в сентябре лечу в Токио",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ПОЕЗДКА", "object": "ТОКИО_СЕНТЯБРЬ"}]),
        ),
    ],
    "HOBBIES": [
        (
            "собираю виниловые пластинки пятнадцать лет",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "КОЛЛЕКЦИОНИРУЕТ", "object": "ВИНИЛ"}]),
        ),
    ],
    "TECH": [
        (
            "основной телефон самсунг, ноутбук леново",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ТЕЛЕФОН", "object": "SAMSUNG"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "НОУТБУК", "object": "LENOVO"},
                ]
            ),
        ),
    ],
    "SCHEDULE": [
        (
            "по вторникам и четвергам до девяти на работе",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ГРАФИК", "object": "ВТ_ЧТ_ДО_21_00"},
                ]
            ),
        ),
    ],
    "GOALS": [
        (
            "хочу сдать IELTS на семь с половиной",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЦЕЛЬ", "object": "IELTS_7_5"}]),
        ),
    ],
    "EVENTS": [
        (
            "на прошлой неделе был на свадьбе у кузена",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "БЫЛ_НА", "object": "СВАДЬБА_КУЗЕН"}]),
        ),
    ],
    "HOME": [
        (
            "снял однушку на окраине, пятый этаж",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖИЛЬЁ", "object": "ОДНУШКА_ОКРАИНА"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЭТАЖ", "object": "5"},
                ]
            ),
        ),
    ],
    "IDENTITY": [
        (
            "меня зовут Алексей, мне тридцать два",
            _t(
                [
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЯ", "object": "АЛЕКСЕЙ"},
                    {"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ВОЗРАСТ", "object": "32"},
                ]
            ),
        ),
    ],
    "ROMANCE": [
        (
            "мы с партнёром живём вместе второй год",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "СОВМЕСТНОЕ_ПРОЖИВАНИЕ", "object": "ПАРТНЁР_2_ГОДА"}]),
        ),
    ],
    "FRIENDS": [
        (
            "каждую пятницу встречаемся с компанией в баре",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ВСТРЕЧА", "object": "КОМПАНИЯ_ПЯТНИЦА_БАР"}]),
        ),
    ],
    "HABITS": [
        (
            "лечь спать до полуночи стараюсь каждый день",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ПРИВЫЧКА_СОН", "object": "ДО_00_00"}]),
        ),
    ],
    "PREFERENCES": [
        (
            "не люблю сладкое в кофе",
            _t([{"subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "НЕ_ЛЮБИТ", "object": "САХАР_В_КОФЕ"}]),
        ),
    ],
}


def triplet_per_slot_few_shot_messages(
    user_turn_fn, slot_name: str | None
) -> List[Dict[str, Any]]:
    rows: List[Tuple[str, str]] = list(TRIPLET_PER_SLOT_SHARED)
    if slot_name and slot_name in TRIPLET_PER_SLOT_BY_SLOT:
        rows.extend(TRIPLET_PER_SLOT_BY_SLOT[slot_name])
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in rows:
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    return out


# Single-pass: в JSON есть поле slot — русские метки
TRIPLET_SINGLE_PASS_FEWSHOT: List[Tuple[str, str]] = [
    (
        "женат, сын в школе, работаю инженером",
        _t(
            [
                {"slot": "СЕМЬЯ", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖЕНАТ", "object": "ДА"},
                {"slot": "СЕМЬЯ", "subject": "СЫН", "relation": "УЧИТСЯ_В", "object": "ШКОЛА"},
                {"slot": "РАБОТА", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАБОТАЕТ_КАК", "object": "ИНЖЕНЕР"},
            ]
        ),
    ),
    (
        "кот Мурзик, не ем сахар",
        _t(
            [
                {"slot": "ПИТОМЦЫ", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ", "object": "КОТ_МУРЗИК"},
                {"slot": "ЕДА", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИСКЛЮЧАЕТ", "object": "САХАР"},
            ]
        ),
    ),
    (
        "погода сегодня супер",
        '{"triplets":[]}',
    ),
    (
        "переехал в Екатеринбург, ипотека в ВТБ",
        _t(
            [
                {"slot": "ЛОКАЦИЯ", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖИВЁТ_В", "object": "ЕКАТЕРИНБУРГ"},
                {"slot": "ФИНАНСЫ", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИПОТЕКА", "object": "ВТБ"},
            ]
        ),
    ),
    (
        "бегаю марафоны и не пью алкоголь",
        _t(
            [
                {"slot": "СПОРТ", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЗАНИМАЕТСЯ", "object": "МАРАФОНЫ"},
                {"slot": "ПРИВЫЧКИ", "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "НЕ_УПОТРЕБЛЯЕТ", "object": "АЛКОГОЛЬ"},
            ]
        ),
    ),
]


def triplet_single_pass_few_shot_messages(user_turn_fn) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in TRIPLET_SINGLE_PASS_FEWSHOT:
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    return out


# Memory gate: слоты в списке — как в состоянии (англ. ключи)
MEMORY_GATE_FEWSHOT: List[Tuple[str, str, str]] = [
    ("как зовут мою жену?", "FAMILY\nWORK", '{"use_memory": true, "slots": ["FAMILY"]}'),
    ("что такое квантовая механика?", "FAMILY\nWORK", '{"use_memory": false, "slots": []}'),
    ("напомни, где я работаю", "WORK\nLOCATION", '{"use_memory": true, "slots": ["WORK"]}'),
    ("как дела?", "FAMILY\nHEALTH", '{"use_memory": false, "slots": []}'),
    ("расскажи про моих питомцев", "PETS\nFOOD", '{"use_memory": true, "slots": ["PETS"]}'),
    ("сравни мой график и цели", "SCHEDULE\nGOALS\nWORK", '{"use_memory": true, "slots": ["SCHEDULE", "GOALS", "WORK"]}'),
    ("что посмотреть вечером из сериалов", "HOBBIES\nPREFERENCES", '{"use_memory": true, "slots": ["HOBBIES", "PREFERENCES"]}'),
    ("сколько будет два плюс два", "EDUCATION\nWORK", '{"use_memory": false, "slots": []}'),
]


def memory_gate_user_block(question: str, slot_names: List[str], extra: str) -> str:
    slots_text = (
        "\n".join(f"- {name}" for name in slot_names)
        if slot_names
        else "(нет ни одного слота с сохранёнными данными)"
    )
    return (
        f"Сообщение пользователя:\n{question}\n\n"
        f"Имена слотов памяти:\n{slots_text}\n{extra}"
    )


MEMORY_GATE_FEWSHOT_VECTOR: List[Tuple[str, str, str]] = [
    (
        "что я говорил про отпуск",
        "TRAVEL\nWORK",
        '{"use_memory": true, "slots": []}',
    ),
    (
        "какой у меня график",
        "SCHEDULE\nFAMILY",
        '{"use_memory": true, "slots": ["SCHEDULE"]}',
    ),
]


# Slot update — дополнительные пары (к основным в slot_update_messages)
SLOT_UPDATE_EXTRA_FEWSHOT: List[Tuple[str, list, str]] = [
    (
        "дочка теперь в другой школе",
        [{"id": 1, "value": "дочь: Маша, школа №5"}],
        '{"operations":[{"op":"update","id":1,"value":"дочь: Маша, другая школа"}]}',
    ),
    (
        "забыл, ничего нового",
        [{"id": 1, "value": "машина: киа"}],
        '{"operations":[{"op":"nothing"}]}',
    ),
    (
        "продал велик, купил самокат",
        [{"id": 1, "value": "велосипед: городской"}],
        '{"operations":[{"op":"delete","id":1},{"op":"add","value":"самокат: для поездок"}]}',
    ),
    (
        "жена сменила фамилию на Иванова",
        [{"id": 1, "value": "жена: Мария Петрова"}],
        '{"operations":[{"op":"update","id":1,"value":"жена: Мария Иванова"}]}',
    ),
    (
        "сын больше не ходит в секцию",
        [{"id": 1, "value": "сын: секция плавания"}],
        '{"operations":[{"op":"delete","id":1}]}',
    ),
]


# ---------------------------------------------------------------------------
# Triplet Conflict Resolution few-shots
# ---------------------------------------------------------------------------
# Формат запроса: slot, existing_triplets (list with record_id), new_triplets (indexed list)
# Формат ответа: {"deactivate":[record_ids], "skip_new":[new_indices]}
# Если нет конфликтов — {"deactivate":[], "skip_new":[]}
# ---------------------------------------------------------------------------

def _cr(existing: list, new_triplets: list, answer: str) -> tuple:
    """Helper: returns (existing_json, new_json, answer_json)."""
    return (json.dumps(existing, ensure_ascii=False),
            json.dumps(new_triplets, ensure_ascii=False),
            answer)


CONFLICT_RESOLUTION_FEWSHOT: List[Tuple[str, str, str]] = [
    # Смена работы: РАБОТАЕТ_В меняется → деактивировать старый
    _cr(
        existing=[{"record_id": 1, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАБОТАЕТ_В", "object": "ЯНДЕКС"},
                  {"record_id": 2, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДОЛЖНОСТЬ", "object": "АНАЛИТИК_ДАННЫХ"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАБОТАЕТ_В", "object": "СБЕР"}],
        answer='{"deactivate":[1],"skip_new":[]}',
    ),
    # Переезд: старый город → деактивировать
    _cr(
        existing=[{"record_id": 5, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖИВЁТ_В", "object": "МОСКВА"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖИВЁТ_В", "object": "ТОМСК"}],
        answer='{"deactivate":[5],"skip_new":[]}',
    ),
    # Развод: ЖЕНАТ_С → В_РАЗВОДЕ — другой relation, деактивировать «женат»
    _cr(
        existing=[{"record_id": 3, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ЖЕНАТ_С", "object": "ЛЮДМИЛА"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "В_РАЗВОДЕ", "object": "ЛЮДМИЛА"}],
        answer='{"deactivate":[3],"skip_new":[]}',
    ),
    # Дубль: тот же факт — пропустить новый
    _cr(
        existing=[{"record_id": 7, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_КОТА", "object": "БАРСИК"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_КОТА", "object": "БАРСИК"}],
        answer='{"deactivate":[],"skip_new":[0]}',
    ),
    # Новый факт, не противоречащий ничему — оставить всё
    _cr(
        existing=[{"record_id": 2, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАБОТАЕТ_В", "object": "ЯНДЕКС"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДОЛЖНОСТЬ", "object": "ТИМЛИД"}],
        answer='{"deactivate":[],"skip_new":[]}',
    ),
    # Новая машина: старая → деактивировать, новый тип транспорта → добавить
    _cr(
        existing=[{"record_id": 8, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_МАШИНУ", "object": "KIA_RIO"},
                  {"record_id": 9, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_ВЕЛОСИПЕД", "object": "ГОРОДСКОЙ"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_МАШИНУ", "object": "FORD_FOCUS"},
                      {"idx": 1, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_САМОКАТ", "object": "ДЛЯ_ГОРОДА"}],
        answer='{"deactivate":[8],"skip_new":[]}',
    ),
    # Смена диагноза: обновить диагноз, оставить врача
    _cr(
        existing=[{"record_id": 11, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДИАГНОЗ", "object": "ГИПЕРТОНИЯ"},
                  {"record_id": 12, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "НА_УЧЁТЕ", "object": "КАРДИОЛОГ"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДИАГНОЗ", "object": "ГИПЕРТОНИЯ_2_СТЕПЕНИ"}],
        answer='{"deactivate":[11],"skip_new":[]}',
    ),
    # Смена должности при той же компании
    _cr(
        existing=[{"record_id": 4, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "РАБОТАЕТ_В", "object": "СБЕР"},
                  {"record_id": 6, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДОЛЖНОСТЬ", "object": "АНАЛИТИК"}],
        new_triplets=[{"idx": 0, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ДОЛЖНОСТЬ", "object": "СТАРШИЙ_АНАЛИТИК"}],
        answer='{"deactivate":[6],"skip_new":[]}',
    ),
    # Нет конфликтов, нет дублей — ничего не деактивировать
    _cr(
        existing=[{"record_id": 10, "subject": "ПОЛЬЗОВАТЕЛЬ", "relation": "ИМЕЕТ_КОТА", "object": "БАРСИК"}],
        new_triplets=[{"idx": 0, "subject": "КОТ_БАРСИК", "relation": "ПОРОДА", "object": "МЕЙН-КУН"}],
        answer='{"deactivate":[],"skip_new":[]}',
    ),
]
