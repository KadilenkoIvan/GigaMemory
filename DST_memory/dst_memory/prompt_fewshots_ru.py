"""
Русскоязычные few-shot для промптов DST_memory (чередование user + assistant).

ФОРМАТ ТРИПЛЕТОВ:
  subject / relation / object — строчные буквами с пробелами (lowercase).
  НЕ ИСПОЛЬЗУЙ UPPER_CASE_UNDERSCORE — это нечитаемо для Meno-Lite-0.1.
  Правильно:  "subject": "пользователь",  "relation": "есть собака",  "object": "собака пользователя"
  Неверно:    "subject": "ПОЛЬЗОВАТЕЛЬ",  "relation": "ЕСТЬ_СОБАКА",  "object": "СОБАКА_ПОЛЬЗОВАТЕЛЯ"

Слоты в slot_assignments: русские метки (СЕМЬЯ, РАБОТА) — резолвятся в ontology.RU_SLOT_TO_CANONICAL.
В memory gate списки слотов — канонические английские ключи (как в DialogueMemoryState).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Выбор слотов
# ---------------------------------------------------------------------------

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
    ("вернулась только что из бара, я такая пьяненькая", '{"slot_assignments":["СОБЫТИЯ"]}'),
    ("взяли из питомника щенка, золотистый ретривер", '{"slot_assignments":["ПИТОМЦЫ"]}'),
    ("младшей сестрёнке нужен в школе пересказ гранатового браслета", '{"slot_assignments":["СЕМЬЯ"]}'),
    ("простудился, сижу с температурой 38", '{"slot_assignments":["ЗДОРОВЬЕ"]}'),
    ("взяли на стажировку в яндекс на 2 месяца", '{"slot_assignments":["РАБОТА"]}'),
    ("ура, пятница!", '{"slot_assignments":[]}'),
]


def slot_select_few_shot_messages(user_turn_fn) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in SLOT_SELECT_FEWSHOT:
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    return out


# ---------------------------------------------------------------------------
# Хелперы для сборки JSON триплетов
# ---------------------------------------------------------------------------

def _t(items: List[dict]) -> str:
    return json.dumps({"triplets": items}, ensure_ascii=False)


def _t_ttl(items: List[dict], ttl_map: Dict[int, str] | None = None) -> str:
    """Same as _t, but adds ttl field from ttl_map (index → ttl value)."""
    result = []
    for i, item in enumerate(items):
        entry = dict(item)
        if ttl_map and i in ttl_map:
            entry["ttl"] = ttl_map[i]
        elif "ttl" not in entry:
            entry["ttl"] = "inf"
        result.append(entry)
    return json.dumps({"triplets": result}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Общие примеры для per-slot (без указания слота в user-turn).
# Субъект, связь и объект — строчными буквами с пробелами.
# ---------------------------------------------------------------------------

TRIPLET_PER_SLOT_SHARED_BASE: List[Tuple[str, List[dict]]] = [
    (
        "мы с женой семь лет в браке, есть сын Артём",
        [
            {"subject": "пользователь", "relation": "женат", "object": "жена пользователя", "ttl": "inf"},
            {"subject": "жена пользователя", "relation": "стаж брака", "object": "7 лет", "ttl": "inf"},
            {"subject": "пользователь", "relation": "есть сын", "object": "сын пользователя", "ttl": "inf"},
            {"subject": "сын пользователя", "relation": "имя", "object": "артём", "ttl": "inf"},
        ],
    ),
    (
        "жена получила повышение, она теперь руководит отделом",
        [
            {"subject": "жена пользователя", "relation": "получила", "object": "повышение", "ttl": "6m"},
            {"subject": "жена пользователя", "relation": "руководит", "object": "отдел", "ttl": "1y"},
        ],
    ),
    (
        "Как приготовить блины?",
        [],
    ),
]


def _build_shared(use_ttl: bool) -> List[Tuple[str, str]]:
    out = []
    for msg, items in TRIPLET_PER_SLOT_SHARED_BASE:
        if not items:
            out.append((msg, '{"triplets":[]}'))
        elif use_ttl:
            out.append((msg, _t(items)))
        else:
            out.append((msg, _t([{k: v for k, v in item.items() if k != "ttl"} for item in items])))
    return out


# ---------------------------------------------------------------------------
# Per-slot few-shot examples (slot-scoped extraction with negative examples).
# ---------------------------------------------------------------------------

TRIPLET_PER_SLOT_BY_SLOT_BASE: Dict[str, List[Tuple[str, List[dict]]]] = {
    "FAMILY": [
        (
            "у меня есть сын Эдуард, ему нет ещё года",
            [
                {"subject": "пользователь", "relation": "есть", "object": "сын пользователя", "ttl": "inf"},
                {"subject": "сын пользователя", "relation": "имя", "object": "эдуард", "ttl": "inf"},
                {"subject": "сын пользователя", "relation": "возраст", "object": "менее одного года", "ttl": "1y"},
            ],
        ),
        (
            "мы с сыном ходим на бокс два раза в неделю",
            [
                {"subject": "пользователь", "relation": "занимается боксом с", "object": "сын пользователя", "ttl": "6m"},
                {"subject": "пользователь", "relation": "частота бокса", "object": "2 раза в неделю", "ttl": "6m"},
                {"subject": "сын пользователя", "relation": "занимается", "object": "бокс", "ttl": "6m"},
            ],
        ),
        (
            "младшей сестрёнке нужно в школе выучить наизусть золотую рыбку",
            [
                {"subject": "пользователь", "relation": "есть сестра", "object": "сестра пользователя", "ttl": "inf"},
                {"subject": "сестра пользователя", "relation": "учится", "object": "школа", "ttl": "1y"},
                {"subject": "сестра пользователя", "relation": "задание", "object": "выучить наизусть золотая рыбка", "ttl": "1w"},
            ],
        ),
        (
            "устроился в яндекс аналитиком",
            [],
        ),
    ],
    "WORK": [
        (
            "устроился в Яндекс аналитиком данных",
            [
                {"subject": "пользователь", "relation": "работает в", "object": "яндекс", "ttl": "1y"},
                {"subject": "яндекс", "relation": "должность", "object": "аналитик данных", "ttl": "1y"},
            ],
        ),
        (
            "уволился с прошлой работы в марте",
            [
                {"subject": "пользователь", "relation": "уволился", "object": "прошлая работа", "ttl": "1y"},
                {"subject": "прошлая работа", "relation": "дата увольнения", "object": "март", "ttl": "1y"},
            ],
        ),
        (
            "взяли на двухмесячную стажировку в Газпром, начну в июне",
            [
                {"subject": "пользователь", "relation": "стажировка", "object": "газпром", "ttl": "3m"},
                {"subject": "газпром", "relation": "длительность стажировки", "object": "2 месяца", "ttl": "3m"},
                {"subject": "газпром", "relation": "начало стажировки", "object": "июнь", "ttl": "3m"},
            ],
        ),
    ],
    "PETS": [
        (
            "у меня кот, зовут Барсик, не переносит других котов",
            [
                {"subject": "пользователь", "relation": "есть кот", "object": "кот пользователя", "ttl": "inf"},
                {"subject": "кот пользователя", "relation": "имя", "object": "барсик", "ttl": "inf"},
                {"subject": "кот пользователя", "relation": "не переносит", "object": "другие коты", "ttl": "6m"},
            ],
        ),
        (
            "забрали из питомника щенка, золотистый ретривер, назвали рыжик",
            [
                {"subject": "пользователь", "relation": "есть собака", "object": "собака пользователя", "ttl": "inf"},
                {"subject": "собака пользователя", "relation": "порода", "object": "золотистый ретривер", "ttl": "inf"},
                {"subject": "собака пользователя", "relation": "имя", "object": "рыжик", "ttl": "inf"},
            ],
        ),
        (
            "присматриваю за хомяком соседки пока та в больнице, наверное дней десять",
            [
                {"subject": "пользователь", "relation": "временно содержит питомца", "object": "хомяк соседки", "ttl": "10d"},
            ],
        ),
        (
            "собаку зовут Луна, гуляю с ней утром",
            [
                {"subject": "пользователь", "relation": "есть собака", "object": "собака пользователя", "ttl": "inf"},
                {"subject": "собака пользователя", "relation": "имя", "object": "луна", "ttl": "inf"},
                {"subject": "собака пользователя", "relation": "время прогулки", "object": "утро", "ttl": "inf"},
            ],
        ),
    ],
    "FOOD": [
        (
            "не ем глютен и лактозу",
            [
                {"subject": "пользователь", "relation": "исключает из рациона", "object": "глютен", "ttl": "inf"},
                {"subject": "пользователь", "relation": "исключает из рациона", "object": "лактоза", "ttl": "inf"},
            ],
        ),
        (
            "люблю борщ и соленья",
            [
                {"subject": "пользователь", "relation": "нравится", "object": "борщ", "ttl": "1y"},
                {"subject": "пользователь", "relation": "нравится", "object": "соленья", "ttl": "1y"},
            ],
        ),
        (
            "в Англии еда совсем не понравилась, пресно и скучно",
            [
                {"subject": "пользователь", "relation": "не нравится еда", "object": "английская кухня", "ttl": "6m"},
                {"subject": "английская кухня", "relation": "вкус", "object": "пресно", "ttl": "6m"},
                {"subject": "английская кухня", "relation": "вкус", "object": "скучно", "ttl": "6m"},
            ],
        ),
    ],
    "HEALTH": [
        (
            "диабет второго типа, на учёте у эндокринолога",
            [
                {"subject": "пользователь", "relation": "диагноз", "object": "диабет 2 типа", "ttl": "inf"},
                {"subject": "пользователь", "relation": "на учёте", "object": "эндокринолог", "ttl": "1y"},
            ],
        ),
        (
            "гипертония, слежу за давлением каждый день",
            [
                {"subject": "пользователь", "relation": "диагноз", "object": "гипертония", "ttl": "inf"},
                {"subject": "пользователь", "relation": "контролирует", "object": "давление", "ttl": "inf"},
                {"subject": "давление", "relation": "частота контроля", "object": "каждый день", "ttl": "inf"},
            ],
        ),
        # Нетипичный пример: HEALTH обычно хранит хронические диагнозы (1y+),
        # но острая болезнь — временный факт с коротким TTL
        (
            "заболел гриппом, температура 38.7, пью жаропонижающее",
            [
                {"subject": "пользователь", "relation": "болен", "object": "грипп", "ttl": "10d"},
                {"subject": "пользователь", "relation": "температура", "object": "38.7", "ttl": "3d"},
                {"subject": "пользователь", "relation": "принимает", "object": "жаропонижающее", "ttl": "3d"},
            ],
        ),
        (
            "сегодня на работе было всё окей",
            [],
        ),
    ],
    "MENTAL_HEALTH": [
        (
            "ходил к психотерапевту полгода, стало легче",
            [
                {"subject": "пользователь", "relation": "посещал", "object": "психотерапевт", "ttl": "6m"},
                {"subject": "психотерапевт", "relation": "длительность терапии", "object": "полгода", "ttl": "6m"},
            ],
        ),
        (
            "часто тревожность перед выступлениями",
            [
                {"subject": "пользователь", "relation": "часто чувствует", "object": "тревожность перед выступлениями", "ttl": "6m"},
            ],
        ),
    ],
    "EDUCATION": [
        (
            "Я сейчас на магистратуре в НГУ, кафедра ИИ",
            [
                {"subject": "пользователь", "relation": "учится в", "object": "нгу", "ttl": "1y"},
                {"subject": "пользователь", "relation": "уровень обучения", "object": "магистратура", "ttl": "1y"},
                {"subject": "пользователь", "relation": "кафедра", "object": "искусственный интеллект", "ttl": "1y"},
            ],
        ),
        (
            "закончил НГУ по специальности прикладная математика",
            [
                {"subject": "пользователь", "relation": "закончил", "object": "нгу", "ttl": "inf"},
                {"subject": "пользователь", "relation": "специализация", "object": "прикладная математика", "ttl": "inf"},
            ],
        ),
    ],
    "SPORTS": [
        (
            "бегаю десять километров по воскресеньям",
            [
                {"subject": "пользователь", "relation": "занимается", "object": "бег", "ttl": "6m"},
                {"subject": "бег", "relation": "дистанция", "object": "10 км", "ttl": "6m"},
                {"subject": "бег", "relation": "дата бега", "object": "воскресенье", "ttl": "6m"},
            ],
        ),
        (
            "каждую субботу играю в футбол с друзьями",
            [
                {"subject": "пользователь", "relation": "играет", "object": "футбол", "ttl": "6m"},
                {"subject": "футбол", "relation": "частота", "object": "каждую субботу", "ttl": "6m"},
                {"subject": "футбол", "relation": "вместе с", "object": "друзья", "ttl": "6m"},
            ],
        ),
        (
            "ну да, спорт это полезно, но сегодня лень",
            [],
        ),
    ],
    "LOCATION": [
        (
            "переехал в Красноярск из Иркутска",
            [
                {"subject": "пользователь", "relation": "живёт в", "object": "красноярск", "ttl": "1y"},
                {"subject": "пользователь", "relation": "ранее жил в", "object": "иркутск", "ttl": "1y"},
            ],
        ),
        (
            "сейчас живу в Москве, снял квартиру в Митино",
            [
                {"subject": "пользователь", "relation": "живёт в", "object": "москва", "ttl": "1y"},
                {"subject": "пользователь", "relation": "район проживания", "object": "митино", "ttl": "1y"},
            ],
        ),
    ],
    "FINANCE": [
        (
            "ипотека в Сбере, платёж сорок тысяч",
            [
                {"subject": "пользователь", "relation": "платит", "object": "ипотека", "ttl": "1y"},
                {"subject": "ипотека", "relation": "банк", "object": "сбер", "ttl": "1y"},
                {"subject": "ипотека", "relation": "платёж", "object": "40000 рублей", "ttl": "1y"},
            ],
        ),
        (
            "откладываю 20 процентов зарплаты на подушку безопасности",
            [
                {"subject": "пользователь", "relation": "откладывает", "object": "20 процентов зарплаты", "ttl": "3m"},
                {"subject": "пользователь", "relation": "цель накоплений", "object": "подушка безопасности", "ttl": "3m"},
            ],
        ),
    ],
    "VEHICLES": [
        (
            "поменял масло, езжу на форде",
            [
                {"subject": "пользователь", "relation": "авто", "object": "форд", "ttl": "1y"},
            ],
        ),
        (
            "у меня kia rio, а до этого была renault logan",
            [
                {"subject": "пользователь", "relation": "авто", "object": "kia rio", "ttl": "1y"},
                {"subject": "пользователь", "relation": "было авто", "object": "renault logan", "ttl": "1y"},
            ],
        ),
    ],
    "TRAVEL": [
        (
            "в сентябре лечу в Токио",
            [
                {"subject": "пользователь", "relation": "поездка", "object": "токио", "ttl": "3m"},
                {"subject": "токио", "relation": "дата", "object": "сентябрь", "ttl": "3m"},
            ],
        ),
        (
            "в августе планируем поездку в Казань с семьей",
            [
                {"subject": "пользователь", "relation": "планирует поездку", "object": "казань", "ttl": "3m"},
                {"subject": "казань", "relation": "дата", "object": "август", "ttl": "3m"},
                {"subject": "казань", "relation": "едет с", "object": "семья", "ttl": "3m"},
            ],
        ),
        # Путешествие упоминается вскользь — атрибуты (впечатления) вешаем на место, не на пользователя
        (
            "последний раз была в Англии — еда вообще не понравилась",
            [
                {"subject": "пользователь", "relation": "была", "object": "англия", "ttl": "3m"},
                {"subject": "англия", "relation": "впечатление от еды", "object": "не понравилась", "ttl": "3m"},
            ],
        ),
        # Отрицательный пример — нет факта о путешествии
        (
            "мне нравится японская кухня",
            [],
        ),
    ],
    "HOBBIES": [
        (
            "собираю виниловые пластинки уже пятнадцать лет",
            [
                {"subject": "пользователь", "relation": "коллекционирует", "object": "винил", "ttl": "inf"},
                {"subject": "винил", "relation": "стаж коллекционирования", "object": "15 лет", "ttl": "inf"},
            ],
        ),
        (
            "увлекаюсь фотографией, снимаю на зеркалку",
            [
                {"subject": "пользователь", "relation": "хобби", "object": "фотография", "ttl": "1y"},
                {"subject": "фотография", "relation": "снимает на", "object": "зеркальная камера", "ttl": "1y"},
            ],
        ),
    ],
    "TECH": [
        (
            "основной телефон самсунг, ноутбук леново",
            [
                {"subject": "пользователь", "relation": "телефон", "object": "samsung", "ttl": "6m"},
                {"subject": "пользователь", "relation": "ноутбук", "object": "lenovo", "ttl": "6m"},
            ],
        ),
        (
            "пользуюсь айфоном и макбуком для работы",
            [
                {"subject": "пользователь", "relation": "использует для работы", "object": "айфон", "ttl": "6m"},
                {"subject": "пользователь", "relation": "использует для работы", "object": "макбук", "ttl": "6m"},
            ],
        ),
    ],
    "SCHEDULE": [
        (
            "по вторникам и четвергам до девяти на работе",
            [
                {"subject": "пользователь", "relation": "работает", "object": "вт, чт до 21:00", "ttl": "1m"},
            ],
        ),
        (
            "по будням встаю в 6 утра и ложусь около 23:00",
            [
                {"subject": "пользователь", "relation": "время подъёма по будням", "object": "06:00", "ttl": "1m"},
                {"subject": "пользователь", "relation": "время отхода ко сну", "object": "23:00", "ttl": "1m"},
            ],
        ),
    ],
    "GOALS": [
        (
            "хочу сдать IELTS на семь с половиной",
            [
                {"subject": "пользователь", "relation": "цель", "object": "ielts 7.5", "ttl": "3m"},
            ],
        ),
        (
            "хочу через год перейти в продуктовую аналитику",
            [
                {"subject": "пользователь", "relation": "цель", "object": "перейти в продуктовую аналитику", "ttl": "1y"},
                {"subject": "перейти в продуктовую аналитику", "relation": "срок", "object": "через год", "ttl": "1y"},
            ],
        ),
        # Нетипичный пример: GOALS обычно 3m, но давняя мечта — бессрочно
        (
            "с детства мечтаю побывать на Байкале, всю жизнь об этом думаю",
            [
                {"subject": "пользователь", "relation": "мечтает побывать", "object": "байкал", "ttl": "inf"},
                {"subject": "байкал", "relation": "мечтает побывать", "object": "с детства", "ttl": "inf"},
            ],
        ),
    ],
    "EVENTS": [
        (
            "на прошлой неделе был на свадьбе у кузена",
            [
                {"subject": "пользователь", "relation": "посетил", "object": "свадьба кузена", "ttl": "2w"},
            ],
        ),
        (
            "в прошлом месяце выступал на конференции Data Fest",
            [
                {"subject": "пользователь", "relation": "выступил", "object": "конференция data fest", "ttl": "3m"},
            ],
        ),
        # Ситуативные события с коротким TTL
        (
            "вернулась только что из бара, я такая пьяненькая",
            [
                {"subject": "пользователь", "relation": "вернулась из", "object": "бар", "ttl": "1d"},
                {"subject": "пользователь", "relation": "состояние", "object": "пьяная", "ttl": "1d"},
            ],
        ),
        # Нетипичный пример: событие с долгосрочным последствием — сам факт события 2w,
        # но результирующий статус (учёная степень) — inf, и он лучше идёт в EDUCATION
        (
            "сегодня защитил диссертацию, теперь я кандидат наук!",
            [
                {"subject": "пользователь", "relation": "защитил", "object": "диссертация", "ttl": "2w"},
                {"subject": "пользователь", "relation": "учёная степень", "object": "кандидат наук", "ttl": "inf"},
            ],
        ),
    ],
    "HOME": [
        (
            "снял однушку на окраине, пятый этаж",
            [
                {"subject": "пользователь", "relation": "жильё", "object": "однушка", "ttl": "1y"},
                {"subject": "однушка", "relation": "местоположение", "object": "окраина", "ttl": "1y"},
                {"subject": "однушка", "relation": "этаж", "object": "5", "ttl": "1y"},
            ],
        ),
        (
            "живу в двухкомнатной квартире, недавно сделал ремонт кухни",
            [
                {"subject": "пользователь", "relation": "жильё", "object": "двухкомнатная квартира", "ttl": "1y"},
                {"subject": "двухкомнатная квартира", "relation": "сделал ремонт", "object": "кухня", "ttl": "6m"},
            ],
        ),
    ],
    "IDENTITY": [
        (
            "меня зовут Алексей, мне тридцать два",
            [
                {"subject": "пользователь", "relation": "имя", "object": "алексей", "ttl": "inf"},
                {"subject": "пользователь", "relation": "возраст", "object": "32", "ttl": "1y"},
            ],
        ),
        (
            "я мужчина, мне двадцать восемь, по национальности татарин",
            [
                {"subject": "пользователь", "relation": "пол", "object": "мужчина", "ttl": "inf"},
                {"subject": "пользователь", "relation": "возраст", "object": "28", "ttl": "1y"},
                {"subject": "пользователь", "relation": "национальность", "object": "татарин", "ttl": "inf"},
            ],
        ),
    ],
    "ROMANCE": [
        (
            "мы с партнёром живём вместе второй год",
            [
                {"subject": "пользователь", "relation": "есть партнёр", "object": "партнёр пользователя", "ttl": "1y"},
                {"subject": "пользователь", "relation": "живёт вместе с", "object": "партнёр пользователя", "ttl": "1y"},
                {"subject": "партнёр пользователя", "relation": "совместное проживание", "object": "2 года", "ttl": "1y"},
            ],
        ),
        (
            "встречаюсь с девушкой уже полтора года",
            [
                {"subject": "пользователь", "relation": "есть девушка", "object": "девушка пользователя", "ttl": "1y"},
                {"subject": "девушка пользователя", "relation": "стаж отношений", "object": "полтора года", "ttl": "1y"},
            ],
        ),
    ],
    "FRIENDS": [
        (
            "каждую пятницу встречаемся с компанией в баре",
            [
                {"subject": "пользователь", "relation": "ходит с друзьями", "object": "бар", "ttl": "6m"},
                {"subject": "бар", "relation": "вместе с", "object": "друзья", "ttl": "6m"},
            ],
        ),
        (
            "мы с лучшим другом Димой дружим со школы",
            [
                {"subject": "пользователь", "relation": "есть лучший друг", "object": "лучший друг пользователя", "ttl": "inf"},
                {"subject": "лучший друг пользователя", "relation": "имя", "object": "дима", "ttl": "inf"},
                {"subject": "лучший друг пользователя", "relation": "начало дружбы", "object": "школа", "ttl": "inf"},
            ],
        ),
    ],
    "HABITS": [
        (
            "лечь спать до полуночи стараюсь каждый день",
            [
                {"subject": "пользователь", "relation": "привычка сон", "object": "до 00:00", "ttl": "inf"},
            ],
        ),
        (
            "курю уже 15 лет",
            [
                {"subject": "пользователь", "relation": "вредная привычка", "object": "курение", "ttl": "inf"},
                {"subject": "курение", "relation": "стаж", "object": "15 лет", "ttl": "inf"},
            ],
        ),
    ],
    "PREFERENCES": [
        (
            "не люблю сладкое в кофе",
            [
                {"subject": "пользователь", "relation": "не любит", "object": "сладкий кофе", "ttl": "inf"},
            ],
        ),
        (
            "предпочитаю тёмные интерфейсы и минимализм в дизайне",
            [
                {"subject": "пользователь", "relation": "предпочитает", "object": "тёмные интерфейсы", "ttl": "6m"},
                {"subject": "пользователь", "relation": "предпочитает", "object": "минимализм", "ttl": "6m"},
            ],
        ),
    ],
}


def _build_per_slot(
    slot_name: str | None,
    use_ttl: bool,
) -> List[Tuple[str, str]]:
    entries = TRIPLET_PER_SLOT_BY_SLOT_BASE.get(slot_name or "", [])
    out: List[Tuple[str, str]] = []
    for msg, items in entries:
        if not items:
            out.append((msg, '{"triplets":[]}'))
        elif use_ttl:
            out.append((msg, _t(items)))
        else:
            out.append((msg, _t([{k: v for k, v in item.items() if k != "ttl"} for item in items])))
    return out


def triplet_per_slot_few_shot_messages(
    shared_user_turn_fn,
    per_slot_user_turn_fn,
    slot_name: str | None,
    use_ttl: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in _build_shared(use_ttl):
        out.append({"role": "user", "content": shared_user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    if slot_name:
        for msg, assistant_json in _build_per_slot(slot_name, use_ttl):
            out.append({"role": "user", "content": per_slot_user_turn_fn(msg)})
            out.append({"role": "assistant", "content": assistant_json})
    return out


# ---------------------------------------------------------------------------
# Single-pass few-shot (все слоты за один проход, slot field в JSON)
# ---------------------------------------------------------------------------

TRIPLET_SINGLE_PASS_BASE: List[Tuple[str, List[dict]]] = [
    (
        "женат, сын в школе, работаю инженером",
        [
            {"slot": "СЕМЬЯ", "subject": "пользователь", "relation": "семейное положение", "object": "женат", "ttl": "inf"},
            {"slot": "СЕМЬЯ", "subject": "пользователь", "relation": "есть сын", "object": "сын пользователя", "ttl": "inf"},
            {"slot": "СЕМЬЯ", "subject": "сын пользователя", "relation": "учится в", "object": "школа", "ttl": "1y"},
            {"slot": "РАБОТА", "subject": "пользователь", "relation": "работает как", "object": "инженер", "ttl": "1y"},
        ],
    ),
    (
        "у меня кот Мурзик, не ем сахар",
        [
            {"slot": "ПИТОМЦЫ", "subject": "пользователь", "relation": "есть кот", "object": "кот пользователя", "ttl": "inf"},
            {"slot": "ПИТОМЦЫ", "subject": "кот пользователя", "relation": "имя", "object": "мурзик", "ttl": "inf"},
            {"slot": "ЕДА", "subject": "пользователь", "relation": "исключает", "object": "сахар", "ttl": "inf"},
        ],
    ),
    (
        "погода сегодня супер",
        [],
    ),
    (
        "переехал в Екатеринбург, ипотека в ВТБ",
        [
            {"slot": "ЛОКАЦИЯ", "subject": "пользователь", "relation": "живёт в", "object": "екатеринбург", "ttl": "1y"},
            {"slot": "ФИНАНСЫ", "subject": "пользователь", "relation": "ипотека", "object": "втб", "ttl": "1y"},
        ],
    ),
    (
        "бегаю марафоны и не пью алкоголь",
        [
            {"slot": "СПОРТ", "subject": "пользователь", "relation": "занимается", "object": "марафоны", "ttl": "6m"},
            {"slot": "ПРИВЫЧКИ", "subject": "пользователь", "relation": "не употребляет", "object": "алкоголь", "ttl": "inf"},
        ],
    ),
    (
        "вернулась только что из бара, я такая пьяненькая",
        [
            {"slot": "СОБЫТИЯ", "subject": "пользователь", "relation": "вернулась из", "object": "бар", "ttl": "1d"},
            {"slot": "СОБЫТИЯ", "subject": "пользователь", "relation": "состояние", "object": "выпившая", "ttl": "1d"},
        ],
    ),
    # Нетипичный пример: слот ЗДОРОВЬЕ обычно хранит долгосрочные диагнозы (inf, 1y),
    # но временная болезнь — короткий TTL 3d–10d
    (
        "простудился, сижу дома с температурой 37.8",
        [
            {"slot": "ЗДОРОВЬЕ", "subject": "пользователь", "relation": "болен", "object": "простуда", "ttl": "10d"},
            {"slot": "ЗДОРОВЬЕ", "subject": "пользователь", "relation": "температура", "object": "37.8", "ttl": "3d"},
        ],
    ),
    # Нетипичный пример: слот ЦЕЛИ обычно 3m, но давняя мечта — inf
    (
        "всегда хотел научиться играть на гитаре, с самого детства",
        [
            {"slot": "ЦЕЛИ", "subject": "пользователь", "relation": "давняя мечта", "object": "научиться играть на гитаре", "ttl": "inf"},
        ],
    ),
]


def triplet_single_pass_few_shot_messages(
    user_turn_fn,
    use_ttl: bool = False,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg, items in TRIPLET_SINGLE_PASS_BASE:
        if not items:
            assistant_json = '{"triplets":[]}'
        elif use_ttl:
            assistant_json = _t(items)
        else:
            assistant_json = _t([{k: v for k, v in item.items() if k != "ttl"} for item in items])
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    return out


# ---------------------------------------------------------------------------
# Memory gate few-shots
# ---------------------------------------------------------------------------

MEMORY_GATE_FEWSHOT: List[Tuple[str, str, str]] = [
    # Прямые личные вопросы — основной + смежные слоты
    ("как зовут мою жену?", "FAMILY\nWORK\nROMANCE", '{"use_memory": true, "slots": ["FAMILY", "ROMANCE"]}'),
    ("что такое квантовая механика?", "FAMILY\nWORK", '{"use_memory": false, "slots": []}'),
    ("напомни, где я работаю и как добраться", "WORK\nLOCATION\nVEHICLES", '{"use_memory": true, "slots": ["WORK", "LOCATION", "VEHICLES"]}'),
    ("как дела?", "FAMILY\nHEALTH", '{"use_memory": false, "slots": []}'),
    ("расскажи про моих питомцев", "PETS\nFOOD\nHEALTH", '{"use_memory": true, "slots": ["PETS", "FOOD", "HEALTH"]}'),
    ("сравни мой график и цели", "SCHEDULE\nGOALS\nWORK\nHABITS", '{"use_memory": true, "slots": ["SCHEDULE", "GOALS", "WORK", "HABITS"]}'),
    ("что посмотреть вечером из сериалов", "HOBBIES\nPREFERENCES\nEVENTS", '{"use_memory": true, "slots": ["HOBBIES", "PREFERENCES"]}'),
    ("сколько будет два плюс два", "EDUCATION\nWORK", '{"use_memory": false, "slots": []}'),
    # Косвенные вопросы — включаем все смежные слоты
    ("посоветуй, что поесть сегодня", "FOOD\nHEALTH\nHABITS\nPREFERENCES", '{"use_memory": true, "slots": ["FOOD", "HEALTH", "HABITS", "PREFERENCES"]}'),
    ("хочу заняться спортом, что посоветуешь?", "SPORTS\nHEALTH\nSCHEDULE\nGOALS", '{"use_memory": true, "slots": ["SPORTS", "HEALTH", "SCHEDULE", "GOALS"]}'),
    ("могу ли я позволить себе новую машину?", "FINANCE\nVEHICLES\nWORK", '{"use_memory": true, "slots": ["FINANCE", "VEHICLES", "WORK"]}'),
    ("напомни про мои планы", "GOALS\nSCHEDULE\nTRAVEL\nEVENTS", '{"use_memory": true, "slots": ["GOALS", "SCHEDULE", "TRAVEL", "EVENTS"]}'),
    ("расскажи о моём здоровье", "HEALTH\nMENTAL_HEALTH\nHABITS\nSPORTS\nFOOD", '{"use_memory": true, "slots": ["HEALTH", "MENTAL_HEALTH", "HABITS", "SPORTS", "FOOD"]}'),
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


# ---------------------------------------------------------------------------
# Slot update extra few-shots
# ---------------------------------------------------------------------------

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
# Context-aware triplet few-shots (slot_context_enabled=True, Вариант A)
# Формат user-turn: текущие факты + сообщение
# Формат assistant: {"triplets":[...], "delete":[...]}
# ---------------------------------------------------------------------------

_CONTEXT_FEWSHOTS: List[Tuple[str, List[str], str, dict]] = [
    # --- Переезд с указанием нового места (обновление + история) ---
    (
        "переехал в Сызрань с семьёй",
        ["пользователь | место жительства | москва"],
        "HOME",
        {
            "triplets": [
                {"subject": "пользователь", "relation": "место жительства", "object": "сызрань", "ttl": "1y"},
                {"subject": "пользователь", "relation": "бывшее место жительства", "object": "москва", "ttl": "1y"},
            ],
            "delete": [
                {"subject": "пользователь", "relation": "место жительства", "object": "москва"},
            ],
        },
    ),
    # --- Явный отказ от места без нового (удаление + история) ---
    (
        "я больше не живу в Москве, съехал",
        ["пользователь | место жительства | москва"],
        "HOME",
        {
            "triplets": [
                {"subject": "пользователь", "relation": "бывшее место жительства", "object": "москва", "ttl": "1y"},
            ],
            "delete": [
                {"subject": "пользователь", "relation": "место жительства", "object": "москва"},
            ],
        },
    ),
    # --- Смена работы (обновление + история компании) ---
    (
        "уволился из Яндекса, теперь работаю в Сбере старшим аналитиком",
        [
            "пользователь | работает в | яндекс",
            "пользователь | должность | аналитик данных",
        ],
        "WORK",
        {
            "triplets": [
                {"subject": "пользователь", "relation": "работает в", "object": "сбер", "ttl": "1y"},
                {"subject": "пользователь", "relation": "должность", "object": "старший аналитик", "ttl": "1y"},
                {"subject": "пользователь", "relation": "прежнее место работы", "object": "яндекс", "ttl": "1y"},
            ],
            "delete": [
                {"subject": "пользователь", "relation": "работает в", "object": "яндекс"},
                {"subject": "пользователь", "relation": "должность", "object": "аналитик данных"},
            ],
        },
    ),
    # --- Бросил курить (удаление + факт о прошлой привычке) ---
    (
        "бросил курить месяц назад",
        [
            "пользователь | курит | да",
            "пользователь | количество сигарет | пачка в день",
        ],
        "HABITS",
        {
            "triplets": [
                {"subject": "пользователь", "relation": "бросил курить", "object": "да", "ttl": "inf"},
            ],
            "delete": [
                {"subject": "пользователь", "relation": "курит", "object": "да"},
                {"subject": "пользователь", "relation": "количество сигарет", "object": "пачка в день"},
            ],
        },
    ),
    # --- Новый факт, ничего не нужно удалять ---
    (
        "купил велосипед для поездок по городу",
        ["пользователь | место жительства | екатеринбург"],
        "TECH",
        {
            "triplets": [
                {"subject": "пользователь", "relation": "есть велосипед", "object": "велосипед пользователя", "ttl": "1y"},
                {"subject": "велосипед пользователя", "relation": "тип", "object": "городской", "ttl": "1y"},
            ],
            "delete": [],
        },
    ),
    # --- Нет новых фактов и нечего удалять ---
    (
        "всё ок, спасибо",
        ["пользователь | работает в | сбер"],
        "WORK",
        {"triplets": [], "delete": []},
    ),
]


def _ctx_assistant_json(data: dict, use_ttl: bool) -> str:
    """Сериализация ответа для context few-shot."""
    triplets = data.get("triplets", [])
    delete = data.get("delete", [])
    if not use_ttl:
        triplets = [{k: v for k, v in t.items() if k != "ttl"} for t in triplets]
    return json.dumps({"triplets": triplets, "delete": delete}, ensure_ascii=False)


def triplet_context_few_shot_messages(
    user_turn_fn,
    slot_name: str | None = None,
    use_ttl: bool = True,
) -> List[Dict[str, Any]]:
    """
    Few-shot примеры для context-aware режима экстракции (slot_context_enabled=True).
    Модель видит текущие факты и должна выдать {"triplets":[...], "delete":[...]}.

    Фильтрует примеры по slot_name если указан, иначе берёт первые 3 общих.
    """
    out: List[Dict[str, Any]] = []
    selected = []

    if slot_name:
        for msg, facts, fs_slot, data in _CONTEXT_FEWSHOTS:
            if fs_slot == slot_name:
                selected.append((msg, facts, data))
        # Если примеров для слота нет — берём первые 2 общих (переезд + нет фактов)
        if not selected:
            selected = [
                (_CONTEXT_FEWSHOTS[0][0], _CONTEXT_FEWSHOTS[0][1], _CONTEXT_FEWSHOTS[0][3]),
                (_CONTEXT_FEWSHOTS[4][0], _CONTEXT_FEWSHOTS[4][1], _CONTEXT_FEWSHOTS[4][3]),
                (_CONTEXT_FEWSHOTS[5][0], _CONTEXT_FEWSHOTS[5][1], _CONTEXT_FEWSHOTS[5][3]),
            ]
    else:
        # Single-pass без слота: берём разнообразный набор
        selected = [
            (_CONTEXT_FEWSHOTS[0][0], _CONTEXT_FEWSHOTS[0][1], _CONTEXT_FEWSHOTS[0][3]),
            (_CONTEXT_FEWSHOTS[2][0], _CONTEXT_FEWSHOTS[2][1], _CONTEXT_FEWSHOTS[2][3]),
            (_CONTEXT_FEWSHOTS[5][0], _CONTEXT_FEWSHOTS[5][1], _CONTEXT_FEWSHOTS[5][3]),
        ]

    for msg, facts, data in selected:
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": _ctx_assistant_json(data, use_ttl)})
    return out


# ---------------------------------------------------------------------------
# Conflict Resolution few-shots
# Формат запроса: slot, existing_triplets (list with record_id), new_triplets (indexed list)
# Формат ответа: {"deactivate":[record_ids], "skip_new":[new_indices]}
# ВАЖНО: triplets теперь в lowercase с пробелами (без UPPER_CASE_UNDERSCORE)
# ---------------------------------------------------------------------------

def _cr(existing: list, new_triplets: list, answer: str) -> tuple:
    return (json.dumps(existing, ensure_ascii=False),
            json.dumps(new_triplets, ensure_ascii=False),
            answer)


CONFLICT_RESOLUTION_FEWSHOT: List[Tuple[str, str, str]] = [
    # Смена работы
    _cr(
        existing=[{"record_id": 1, "subject": "пользователь", "relation": "работает в", "object": "яндекс"},
                  {"record_id": 2, "subject": "пользователь", "relation": "должность", "object": "аналитик данных"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "работает в", "object": "сбер"}],
        answer='{"deactivate":[1],"skip_new":[]}',
    ),
    # Переезд
    _cr(
        existing=[{"record_id": 5, "subject": "пользователь", "relation": "живёт в", "object": "москва"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "живёт в", "object": "томск"}],
        answer='{"deactivate":[5],"skip_new":[]}',
    ),
    # Смена семейного положения
    _cr(
        existing=[{"record_id": 3, "subject": "пользователь", "relation": "семейное положение", "object": "женат"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "семейное положение", "object": "разведён"}],
        answer='{"deactivate":[3],"skip_new":[]}',
    ),
    # Дубль — пропустить новый
    _cr(
        existing=[{"record_id": 7, "subject": "пользователь", "relation": "есть кот", "object": "кот пользователя"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "есть кот", "object": "кот пользователя"}],
        answer='{"deactivate":[],"skip_new":[0]}',
    ),
    # Новый факт — оставить всё
    _cr(
        existing=[{"record_id": 2, "subject": "пользователь", "relation": "работает в", "object": "яндекс"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "должность", "object": "тимлид"}],
        answer='{"deactivate":[],"skip_new":[]}',
    ),
    # Новая машина
    _cr(
        existing=[{"record_id": 8, "subject": "пользователь", "relation": "авто", "object": "kia rio"},
                  {"record_id": 9, "subject": "пользователь", "relation": "есть велосипед", "object": "городской"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "авто", "object": "ford focus"},
                      {"idx": 1, "subject": "пользователь", "relation": "есть самокат", "object": "для города"}],
        answer='{"deactivate":[8],"skip_new":[]}',
    ),
    # Смена диагноза
    _cr(
        existing=[{"record_id": 11, "subject": "пользователь", "relation": "диагноз", "object": "гипертония"},
                  {"record_id": 12, "subject": "пользователь", "relation": "на учёте", "object": "кардиолог"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "диагноз", "object": "гипертония 2 степени"}],
        answer='{"deactivate":[11],"skip_new":[]}',
    ),
    # Смена должности
    _cr(
        existing=[{"record_id": 4, "subject": "пользователь", "relation": "работает в", "object": "сбер"},
                  {"record_id": 6, "subject": "пользователь", "relation": "должность", "object": "аналитик"}],
        new_triplets=[{"idx": 0, "subject": "пользователь", "relation": "должность", "object": "старший аналитик"}],
        answer='{"deactivate":[6],"skip_new":[]}',
    ),
    # Нет конфликтов
    _cr(
        existing=[{"record_id": 10, "subject": "пользователь", "relation": "есть кот", "object": "кот пользователя"}],
        new_triplets=[{"idx": 0, "subject": "кот пользователя", "relation": "порода", "object": "мейн кун"}],
        answer='{"deactivate":[],"skip_new":[]}',
    ),
]
