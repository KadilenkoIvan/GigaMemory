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


# Общие примеры для per-slot (без указания слота в user-turn — базовый формат).
# Субъект, связь и объект — строчными буквами без символа «_».
# Постобработка в TripletExtractionClient._normalize_field переведёт их в UPPER_CASE с _.
TRIPLET_PER_SLOT_SHARED: List[Tuple[str, str]] = [
    (
        "мы с женой семь лет в браке, есть сын Артём",
        _t(
            [
                {"subject": "пользователь", "relation": "есть жена", "object": "жена пользователя"},
                # срок брака — факт об отношениях пользователя, не свойство жены
                {"subject": "пользователь", "relation": "лет в браке", "object": "7"},
                {"subject": "пользователь", "relation": "есть сын", "object": "сын пользователя"},
                {"subject": "сын пользователя", "relation": "имя", "object": "артём"},
            ]
        ),
    ),
    (
        "жена получила повышение, она теперь руководит отделом",
        _t(
            [
                {"subject": "жена пользователя", "relation": "получила", "object": "повышение"},
                {"subject": "жена пользователя", "relation": "руководит", "object": "отдел"},
            ]
        ),
    ),
    (
        "просто спасибо",
        '{"triplets":[]}',
    ),
]


TRIPLET_PER_SLOT_BY_SLOT: Dict[str, List[Tuple[str, str]]] = {
    "FAMILY": [
        (
            "у меня есть дочь Маша, ей пять лет",
            _t(
                [
                    {"subject": "пользователь", "relation": "есть дочь", "object": "дочь пользователя"},
                    {"subject": "дочь пользователя", "relation": "имя", "object": "маша"},
                    {"subject": "дочь пользователя", "relation": "возраст", "object": "5"},
                ]
            ),
        ),
        (
            "мы с сыном ходим на бокс два раза в неделю",
            _t(
                [
                    {"subject": "пользователь", "relation": "занимается", "object": "бокс"},
                    {"subject": "пользователь", "relation": "частота бокса", "object": "два раза в неделю"},
                    {"subject": "пользователь", "relation": "занимается вместе с", "object": "сын пользователя"},
                    # сын тоже занимается боксом — факт о нём самом
                    {"subject": "сын пользователя", "relation": "занимается", "object": "бокс"},
                ]
            ),
        ),
        (
            "устроился в яндекс аналитиком",
            '{"triplets":[]}',
        ),
    ],
    "WORK": [
        (
            "устроился в Яндекс аналитиком данных",
            _t(
                [
                    {"subject": "пользователь", "relation": "работает в", "object": "яндекс"},
                    {"subject": "пользователь", "relation": "должность", "object": "аналитик данных"},
                ]
            ),
        ),
        (
            "уволился с прошлой работы в марте",
            _t([{"subject": "пользователь", "relation": "уволился", "object": "март"}]),
        ),
        (
            "у меня кот барсик",
            '{"triplets":[]}',
        ),
    ],
    "PETS": [
        (
            "у меня кот, зовут Барсик, не переносит других котов",
            _t(
                [
                    {"subject": "пользователь", "relation": "есть кот", "object": "кот пользователя"},
                    {"subject": "кот пользователя", "relation": "имя", "object": "барсик"},
                    {"subject": "кот пользователя", "relation": "не переносит", "object": "другие коты"},
                ]
            ),
        ),
        (
            "собаку зовут Луна, гуляю с ней утром",
            _t(
                [
                    {"subject": "пользователь", "relation": "есть собака", "object": "собака пользователя"},
                    {"subject": "собака пользователя", "relation": "имя", "object": "луна"},
                    {"subject": "пользователь", "relation": "выгул собаки", "object": "утро"},
                ]
            ),
        ),
        (
            "я работаю в банке",
            '{"triplets":[]}',
        ),
    ],
    "FOOD": [
        (
            "не ем глютен и лактозу",
            _t(
                [
                    {"subject": "пользователь", "relation": "исключает", "object": "глютен"},
                    {"subject": "пользователь", "relation": "исключает", "object": "лактоза"},
                ]
            ),
        ),
        (
            "люблю борщ и соленья",
            _t([{"subject": "пользователь", "relation": "нравится", "object": "борщ и соленья"}]),
        ),
        (
            "я работаю в банке и у меня кот барсик",
            '{"triplets":[]}',
        ),
    ],
    "HEALTH": [
        (
            "диабет второго типа, на учёте у эндокринолога",
            _t(
                [
                    {"subject": "пользователь", "relation": "диагноз", "object": "диабет 2 типа"},
                    {"subject": "пользователь", "relation": "на учёте", "object": "эндокринолог"},
                ]
            ),
        ),
        (
            "гипертония, слежу за давлением каждый день",
            _t(
                [
                    {"subject": "пользователь", "relation": "диагноз", "object": "гипертония"},
                    {"subject": "пользователь", "relation": "контролирует", "object": "давление"},
                    # "каждый день" — важный факт о регулярности, не должен теряться
                    {"subject": "пользователь", "relation": "частота контроля", "object": "каждый день"},
                ]
            ),
        ),
        (
            "сегодня на работе было всё окей",
            '{"triplets":[]}',
        ),
    ],
    "MENTAL_HEALTH": [
        (
            "ходил к психотерапевту полгода, стало легче",
            _t(
                [
                    # факт посещения специалиста не должен теряться — только длительность
                    {"subject": "пользователь", "relation": "ходил к", "object": "психотерапевт"},
                    {"subject": "пользователь", "relation": "длительность терапии", "object": "полгода"},
                ]
            ),
        ),
        (
            "часто тревожность перед выступлениями",
            _t(
                [
                    {"subject": "пользователь", "relation": "часто чувствует", "object": "тревожность перед выступлениями"},
                ]
            ),
        ),
    ],
    "EDUCATION": [
        (
            "магистратура в НГУ, кафедра ИИ",
            _t(
                [
                    {"subject": "пользователь", "relation": "учится в", "object": "нгу магистратура"},
                    {"subject": "пользователь", "relation": "кафедра", "object": "ии"},
                ]
            ),
        ),
        (
            "закончил НГУ по специальности прикладная математика",
            _t(
                [
                    {"subject": "пользователь", "relation": "закончил", "object": "нгу"},
                    {"subject": "пользователь", "relation": "специализация", "object": "прикладная математика"},
                ]
            ),
        ),
    ],
    "SPORTS": [
        (
            "бегаю десять километров по воскресеньям",
            _t([{"subject": "пользователь", "relation": "занимается", "object": "бег 10 км воскресенье"}]),
        ),
        (
            "каждую субботу играю в футбол с друзьями",
            _t([{"subject": "пользователь", "relation": "играет", "object": "футбол по субботам"}]),
        ),
        (
            "ну да, спорт это полезно, но сегодня лень",
            '{"triplets":[]}',
        ),
    ],
    "LOCATION": [
        (
            "переехал в Красноярск из Иркутска",
            _t(
                [
                    {"subject": "пользователь", "relation": "живёт в", "object": "красноярск"},
                    {"subject": "пользователь", "relation": "ранее жил в", "object": "иркутск"},
                ]
            ),
        ),
        (
            "сейчас живу в Москве, снял квартиру в Митино",
            _t(
                [
                    {"subject": "пользователь", "relation": "живёт в", "object": "москва"},
                    {"subject": "пользователь", "relation": "район", "object": "митино"},
                ]
            ),
        ),
    ],
    "FINANCE": [
        (
            "ипотека в Сбере, платёж сорок тысяч",
            _t(
                [
                    {"subject": "пользователь", "relation": "ипотека", "object": "сбер"},
                    {"subject": "пользователь", "relation": "платёж", "object": "40000 руб"},
                ]
            ),
        ),
        (
            "откладываю 20 процентов зарплаты на подушку безопасности",
            _t(
                [
                    {"subject": "пользователь", "relation": "откладывает", "object": "20 процентов зарплаты"},
                    {"subject": "пользователь", "relation": "цель накоплений", "object": "подушка безопасности"},
                ]
            ),
        ),
    ],
    "VEHICLES": [
        (
            "поменял масло, езжу на форде",
            _t([{"subject": "пользователь", "relation": "авто", "object": "форд"}]),
        ),
        (
            "у меня kia rio, а до этого была renault logan",
            _t(
                [
                    {"subject": "пользователь", "relation": "авто", "object": "kia rio"},
                    {"subject": "пользователь", "relation": "было авто", "object": "renault logan"},
                ]
            ),
        ),
    ],
    "TRAVEL": [
        (
            "в сентябре лечу в Токио",
            _t([{"subject": "пользователь", "relation": "поездка", "object": "токио сентябрь"}]),
        ),
        (
            "в августе планируем поездку в Казань с семьей",
            _t(
                [
                    {"subject": "пользователь", "relation": "планирует поездку", "object": "казань август"},
                    # "с семьей" — важный факт о составе поездки, не должен теряться
                    {"subject": "пользователь", "relation": "едет с", "object": "семья"},
                ]
            ),
        ),
    ],
    "HOBBIES": [
        (
            "собираю виниловые пластинки уже пятнадцать лет",
            _t(
                [
                    {"subject": "пользователь", "relation": "коллекционирует", "object": "винил"},
                    {"subject": "пользователь", "relation": "стаж хобби", "object": "15 лет"},
                ]
            ),
        ),
        (
            "увлекаюсь фотографией, снимаю на зеркалку",
            _t(
                [
                    {"subject": "пользователь", "relation": "хобби", "object": "фотография"},
                    {"subject": "пользователь", "relation": "снимает на", "object": "зеркальная камера"},
                ]
            ),
        ),
    ],
    "TECH": [
        (
            "основной телефон самсунг, ноутбук леново",
            _t(
                [
                    {"subject": "пользователь", "relation": "телефон", "object": "samsung"},
                    {"subject": "пользователь", "relation": "ноутбук", "object": "lenovo"},
                ]
            ),
        ),
        (
            "пользуюсь айфоном и макбуком для работы",
            _t(
                [
                    # "для работы" — важный контекст, не должен теряться
                    {"subject": "пользователь", "relation": "использует для работы", "object": "айфон"},
                    {"subject": "пользователь", "relation": "использует для работы", "object": "макбук"},
                ]
            ),
        ),
    ],
    "SCHEDULE": [
        (
            "по вторникам и четвергам до девяти на работе",
            _t(
                [
                    {"subject": "пользователь", "relation": "график", "object": "вт чт до 21 00"},
                ]
            ),
        ),
        (
            "по будням встаю в 6 утра и ложусь около 23:00",
            _t(
                [
                    {"subject": "пользователь", "relation": "время подъёма", "object": "6 утра"},
                    {"subject": "пользователь", "relation": "время отхода ко сну", "object": "23 00"},
                ]
            ),
        ),
    ],
    "GOALS": [
        (
            "хочу сдать IELTS на семь с половиной",
            _t([{"subject": "пользователь", "relation": "цель", "object": "ielts 7 5"}]),
        ),
        (
            "хочу через год перейти в продуктовую аналитику",
            _t([{"subject": "пользователь", "relation": "цель", "object": "перейти в продуктовую аналитику через год"}]),
        ),
    ],
    "EVENTS": [
        (
            "на прошлой неделе был на свадьбе у кузена",
            _t([{"subject": "пользователь", "relation": "был на", "object": "свадьба кузена"}]),
        ),
        (
            "в прошлом месяце выступал на конференции Data Fest",
            _t([{"subject": "пользователь", "relation": "выступал на", "object": "конференция data fest"}]),
        ),
    ],
    "HOME": [
        (
            "снял однушку на окраине, пятый этаж",
            _t(
                [
                    {"subject": "пользователь", "relation": "жильё", "object": "однушка окраина"},
                    {"subject": "пользователь", "relation": "этаж", "object": "5"},
                ]
            ),
        ),
        (
            "живу в двухкомнатной квартире, недавно сделал ремонт кухни",
            _t(
                [
                    {"subject": "пользователь", "relation": "жильё", "object": "двухкомнатная квартира"},
                    {"subject": "пользователь", "relation": "сделал", "object": "ремонт кухни"},
                ]
            ),
        ),
    ],
    "IDENTITY": [
        (
            "меня зовут Алексей, мне тридцать два",
            _t(
                [
                    {"subject": "пользователь", "relation": "имя", "object": "алексей"},
                    {"subject": "пользователь", "relation": "возраст", "object": "32"},
                ]
            ),
        ),
        (
            "я программист из Екатеринбурга, мне 28",
            _t(
                [
                    {"subject": "пользователь", "relation": "профессия", "object": "программист"},
                    {"subject": "пользователь", "relation": "живёт в", "object": "екатеринбург"},
                    {"subject": "пользователь", "relation": "возраст", "object": "28"},
                ]
            ),
        ),
    ],
    "ROMANCE": [
        (
            "мы с партнёром живём вместе второй год",
            _t(
                [
                    {"subject": "пользователь", "relation": "есть партнёр", "object": "партнёр пользователя"},
                    {"subject": "пользователь", "relation": "живёт вместе с", "object": "партнёр пользователя"},
                    {"subject": "пользователь", "relation": "совместное проживание длится", "object": "2 года"},
                ]
            ),
        ),
        (
            "встречаюсь с девушкой уже полтора года",
            _t(
                [
                    {"subject": "пользователь", "relation": "есть девушка", "object": "девушка пользователя"},
                    {"subject": "пользователь", "relation": "встречается с девушкой", "object": "полтора года"},
                ]
            ),
        ),
    ],
    "FRIENDS": [
        (
            "каждую пятницу встречаемся с компанией в баре",
            _t([{"subject": "пользователь", "relation": "встреча с друзьями", "object": "пятница бар"}]),
        ),
        (
            "мы с лучшим другом Димой дружим со школы",
            _t(
                [
                    {"subject": "пользователь", "relation": "есть лучший друг", "object": "лучший друг пользователя"},
                    {"subject": "лучший друг пользователя", "relation": "имя", "object": "дима"},
                    # "дружим со школы" — взаимный факт об отношениях, субъект — пользователь
                    {"subject": "пользователь", "relation": "дружба с лучшим другом с", "object": "школа"},
                ]
            ),
        ),
    ],
    "HABITS": [
        (
            "лечь спать до полуночи стараюсь каждый день",
            _t([{"subject": "пользователь", "relation": "привычка сон", "object": "до 00 00"}]),
        ),
        (
            "каждое утро пью кофе и читаю новости",
            _t([{"subject": "пользователь", "relation": "утренняя привычка", "object": "кофе и новости"}]),
        ),
    ],
    "PREFERENCES": [
        (
            "не люблю сладкое в кофе",
            _t([{"subject": "пользователь", "relation": "не любит", "object": "сахар в кофе"}]),
        ),
        (
            "предпочитаю тёмные интерфейсы и минимализм в дизайне",
            _t(
                [
                    {"subject": "пользователь", "relation": "предпочитает", "object": "тёмные интерфейсы"},
                    {"subject": "пользователь", "relation": "предпочитает", "object": "минимализм"},
                ]
            ),
        ),
    ],
}


def triplet_per_slot_few_shot_messages(
    shared_user_turn_fn,
    per_slot_user_turn_fn,
    slot_name: str | None,
) -> List[Dict[str, Any]]:
    """
    Build few-shot messages for per-slot triplet extraction.

    shared_user_turn_fn  — used for TRIPLET_PER_SLOT_SHARED (no slot hint in user-turn,
                           demonstrates the basic triplet format).
    per_slot_user_turn_fn — used for TRIPLET_PER_SLOT_BY_SLOT entries (slot hint present,
                            demonstrates slot-scoped extraction and negative examples).
    """
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in TRIPLET_PER_SLOT_SHARED:
        out.append({"role": "user", "content": shared_user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    if slot_name and slot_name in TRIPLET_PER_SLOT_BY_SLOT:
        for msg, assistant_json in TRIPLET_PER_SLOT_BY_SLOT[slot_name]:
            out.append({"role": "user", "content": per_slot_user_turn_fn(msg)})
            out.append({"role": "assistant", "content": assistant_json})
    return out


# Single-pass: в JSON есть поле slot — русские метки.
# Субъект, связь и объект — строчными буквами без «_».
TRIPLET_SINGLE_PASS_FEWSHOT: List[Tuple[str, str]] = [
    (
        "женат, сын в школе, работаю инженером",
        _t(
            [
                {"slot": "СЕМЬЯ", "subject": "пользователь", "relation": "женат", "object": "да"},
                {"slot": "СЕМЬЯ", "subject": "сын", "relation": "учится в", "object": "школа"},
                {"slot": "РАБОТА", "subject": "пользователь", "relation": "работает как", "object": "инженер"},
            ]
        ),
    ),
    (
        "кот Мурзик, не ем сахар",
        _t(
            [
                {"slot": "ПИТОМЦЫ", "subject": "пользователь", "relation": "имеет", "object": "кот мурзик"},
                {"slot": "ЕДА", "subject": "пользователь", "relation": "исключает", "object": "сахар"},
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
                {"slot": "ЛОКАЦИЯ", "subject": "пользователь", "relation": "живёт в", "object": "екатеринбург"},
                {"slot": "ФИНАНСЫ", "subject": "пользователь", "relation": "ипотека", "object": "втб"},
            ]
        ),
    ),
    (
        "бегаю марафоны и не пью алкоголь",
        _t(
            [
                {"slot": "СПОРТ", "subject": "пользователь", "relation": "занимается", "object": "марафоны"},
                {"slot": "ПРИВЫЧКИ", "subject": "пользователь", "relation": "не употребляет", "object": "алкоголь"},
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
