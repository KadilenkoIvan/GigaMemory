"""
English UI few-shots for DST_memory prompts (alternating user + assistant).

TRIPLET FORMAT (English UI — lemmas in English, lowercase, spaces not underscores):
  Example: "subject": "user", "relation": "has cat", "object": "user's cat"

slot_assignments: canonical English slot ids (FAMILY, WORK, …).
Memory gate slot lists: canonical English keys (DialogueMemoryState).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Slot selection
# ---------------------------------------------------------------------------

SLOT_SELECT_FEWSHOT: List[Tuple[str, str]] = [
    ("My wife and I have been married ten years, we have a son", '{"slot_assignments":["FAMILY"]}'),
    ("I drive a taxi for work and play football on weekends", '{"slot_assignments":["WORK","SPORTS","HOBBIES"]}'),
    ("Okay, got it, thanks", '{"slot_assignments":[]}'),
    ("I have hypertension, I take pills as prescribed", '{"slot_assignments":["HEALTH"]}'),
    ("I set aside part of my salary every day", '{"slot_assignments":["FINANCE","HABITS"]}'),
    ("Finished NSU, thinking about a master's", '{"slot_assignments":["EDUCATION","GOALS"]}'),
    ("I live in Novosibirsk, used to live in Omsk", '{"slot_assignments":["LOCATION","HOME"]}'),
    ("We're going to Sochi with the family in August", '{"slot_assignments":["TRAVEL","FAMILY"]}'),
    ("I have a cat Barsik, he's scared of dogs", '{"slot_assignments":["PETS"]}'),
    ("My name is Ivan, I'm twenty-seven", '{"slot_assignments":["IDENTITY"]}'),
    ("I've been dating my girlfriend for a year and a half", '{"slot_assignments":["ROMANCE"]}'),
    ("Dima is my best friend, we've been friends since school", '{"slot_assignments":["FRIENDS"]}'),
    ("I love reading sci-fi in the evenings", '{"slot_assignments":["HOBBIES","PREFERENCES"]}'),
    ("I don't eat meat but I love fish", '{"slot_assignments":["FOOD","PREFERENCES"]}'),
    ("Work was crazy today, I hate shifts", '{"slot_assignments":["WORK"]}'),
    ("Bought a MacBook for work", '{"slot_assignments":["TECH","WORK"]}'),
    ("Car is Kia Rio, before that I had a Logan", '{"slot_assignments":["VEHICLES"]}'),
    ("I wake up at six on weekdays", '{"slot_assignments":["SCHEDULE","HABITS"]}'),
    ("I want to switch to product in a year", '{"slot_assignments":["GOALS","WORK"]}'),
    ("Last month I spoke at a conference", '{"slot_assignments":["EVENTS"]}'),
    ("Strong anxiety before exams", '{"slot_assignments":["MENTAL_HEALTH"]}'),
    ("Renovated the kitchen, we live in a two-room flat", '{"slot_assignments":["HOME"]}'),
    ("I smoke a pack a day, ashamed of it", '{"slot_assignments":["HABITS","HEALTH","MENTAL_HEALTH"]}'),
    ("Peanut allergy since childhood", '{"slot_assignments":["HEALTH","FOOD"]}'),
    ("We got a golden retriever puppy from a shelter", '{"slot_assignments":["PETS"]}'),
    ("My little sister needs a book report on The Garnet Bracelet for school", '{"slot_assignments":["FAMILY"]}'),
    ("Caught a cold, I have a temperature of 38", '{"slot_assignments":["HEALTH"]}'),
    ("Got a two-month internship at Yandex", '{"slot_assignments":["WORK"]}'),
    ("Yay, Friday!", '{"slot_assignments":[]}')
]

def slot_select_few_shot_messages(user_turn_fn, lowercase_slots: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg, assistant_json in SLOT_SELECT_FEWSHOT:
        if lowercase_slots:
            obj = json.loads(assistant_json)
            obj["slot_assignments"] = [s.lower() for s in obj["slot_assignments"]]
            assistant_json = json.dumps(obj, ensure_ascii=False)
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": assistant_json})
    return out


# ---------------------------------------------------------------------------
# Helpers to serialize triplet JSON
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
# Shared per-slot examples (user turn does not name a slot).
# subject, relation, object: lowercase English lemmas.
# ---------------------------------------------------------------------------

TRIPLET_PER_SLOT_SHARED_BASE: List[Tuple[str, List[dict]]] = [
    (
        "My wife and I have been married seven years, we have a son Artyom",
        [
            {"subject": "user", "relation": "married to", "object": "user's wife", "ttl": "inf"},
            {"subject": "user's wife", "relation": "years married", "object": "7 years", "ttl": "inf"},
            {"subject": "user", "relation": "has son", "object": "user's son", "ttl": "inf"},
            {"subject": "user's son", "relation": "name", "object": "artyom", "ttl": "inf"},
        ],
    ),
    (
        "My wife got a promotion, she now runs a department",
        [
            {"subject": "user's wife", "relation": "received", "object": "promotion", "ttl": "6m"},
            {"subject": "user's wife", "relation": "runs", "object": "department", "ttl": "1y"},
        ],
    ),
    (
        "How do you make pancakes?",
        [],
    ),
    (
        "Before my shift job I went hunting with my father every weekend, then for almost two years we couldn't, "
        "but this spring we started going again",
        [
            {"subject": "user", "relation": "used to do", "object": "hunting", "ttl": "inf"},
            {"subject": "user", "relation": "enjoys", "object": "hunting", "ttl": "6m"},
            {"subject": "hunting", "relation": "together with", "object": "user's father", "ttl": "6m"},
        ],
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
            "I have a son Eduard, he's not one yet",
            [
                {"subject": "user", "relation": "has", "object": "user's son", "ttl": "inf"},
                {"subject": "user's son", "relation": "name", "object": "eduard", "ttl": "inf"},
                {"subject": "user's son", "relation": "age", "object": "under one year", "ttl": "1y"},
            ],
        ),
        (
            "My son and I go to boxing twice a week",
            [
                {"subject": "user", "relation": "does boxing with", "object": "user's son", "ttl": "6m"},
                {"subject": "user", "relation": "boxing frequency", "object": "twice a week", "ttl": "6m"},
                {"subject": "user's son", "relation": "does", "object": "boxing", "ttl": "6m"},
            ],
        ),
        (
            "My little sister needs to memorize The Golden Fish by heart for school",
            [
                {"subject": "user", "relation": "has sister", "object": "user's sister", "ttl": "inf"},
                {"subject": "user's sister", "relation": "studies at", "object": "school", "ttl": "1y"},
                {"subject": "user's sister", "relation": "assignment", "object": "memorize the golden fish", "ttl": "1w"},
            ],
        ),
        (
            "Got hired at Yandex as an analyst",
            [],
        ),
        (
            "I got a job at Yandex as an analyst",
            [],
        ),
        (
            "my son, his name is Vanya, has an exam today, and I was given a new laptop at work, a Macbook Air",
            [
                {"subject": "user", "relation": "has son", "object": "user's son", "ttl": "inf"},
                {"subject": "user's son", "relation": "name", "object": "vanya", "ttl": "inf"},
                {"subject": "user's son", "relation": "event", "object": "exam", "ttl": "1w"},
            ],
        ),
    ],
    "WORK": [
        (
            "Got hired at Yandex as a data analyst",
            [
                {"subject": "user", "relation": "works at", "object": "yandex", "ttl": "1y"},
                {"subject": "yandex", "relation": "role", "object": "data analyst", "ttl": "1y"},
            ],
        ),
        (
            "Left my previous job in March",
            [
                {"subject": "user", "relation": "left job", "object": "previous job", "ttl": "1y"},
                {"subject": "previous job", "relation": "quit date", "object": "march", "ttl": "1y"},
            ],
        ),
        (
            "Got a two-month internship at Gazprom, starting in June",
            [
                {"subject": "user", "relation": "internship at", "object": "gazprom", "ttl": "3m"},
                {"subject": "gazprom", "relation": "internship length", "object": "2 months", "ttl": "3m"},
                {"subject": "gazprom", "relation": "internship start", "object": "june", "ttl": "3m"},
            ],
        ),
        (
            "I watched a movie yesterday and went to bed early",
            [],
        ),
        (
            "I got a job as an analyst at Sber, and my son just graduated from college and is looking for a job",
            [
                {"subject": "user", "relation": "works at", "object": "sber", "ttl": "1y"},
                {"subject": "sber", "relation": "position", "object": "analyst", "ttl": "1y"},
            ],
        ),
    ],
    "PETS": [
        (
            "I have a cat named Barsik, he can't stand other cats",
            [
                {"subject": "user", "relation": "has cat", "object": "user's cat", "ttl": "inf"},
                {"subject": "user's cat", "relation": "name", "object": "barsik", "ttl": "inf"},
                {"subject": "user's cat", "relation": "dislikes", "object": "other cats", "ttl": "6m"},
            ],
        ),
        (
            "We got a puppy from a shelter, golden retriever, named him Ryzhik",
            [
                {"subject": "user", "relation": "has dog", "object": "user's dog", "ttl": "inf"},
                {"subject": "user's dog", "relation": "breed", "object": "golden retriever", "ttl": "inf"},
                {"subject": "user's dog", "relation": "name", "object": "ryzhik", "ttl": "inf"},
            ],
        ),
        (
            "Watching my neighbor's hamster while she's in hospital, maybe ten days",
            [
                {"subject": "user", "relation": "temporarily caring for pet", "object": "neighbor's hamster", "ttl": "10d"},
            ],
        ),
        (
            "The dog is named Luna, I walk her in the morning",
            [
                {"subject": "user", "relation": "has dog", "object": "user's dog", "ttl": "inf"},
                {"subject": "user's dog", "relation": "name", "object": "luna", "ttl": "inf"},
                {"subject": "user's dog", "relation": "walk time", "object": "morning", "ttl": "inf"},
            ],
        ),
        (
            "My son and I go to the pool twice a week",
            [],
        ),
        (
            "My cat Barsik has an appointment next week, and I have an interview that day at Yandex",
            [
                {"subject": "user", "relation": "has cat", "object": "user's cat", "ttl": "inf"},
                {"subject": "user's cat", "relation": "name", "object": "barsik", "ttl": "inf"},
                {"subject": "user's cat", "relation": "event", "object": "appointment", "ttl": "1w"},
            ],
        ),
    ],
    "FOOD": [
        (
            "I don't eat gluten or lactose",
            [
                {"subject": "user", "relation": "avoids", "object": "gluten", "ttl": "inf"},
                {"subject": "user", "relation": "avoids", "object": "lactose", "ttl": "inf"},
            ],
        ),
        (
            "I love borscht and pickles",
            [
                {"subject": "user", "relation": "likes", "object": "borscht", "ttl": "1y"},
                {"subject": "user", "relation": "likes", "object": "pickles", "ttl": "1y"},
            ],
        ),
        (
            "Food in England was awful, bland and boring",
            [
                {"subject": "user", "relation": "dislikes food", "object": "english cuisine", "ttl": "6m"},
                {"subject": "english cuisine", "relation": "taste", "object": "bland", "ttl": "6m"},
                {"subject": "english cuisine", "relation": "taste", "object": "boring", "ttl": "6m"},
            ],
        ),
        (
            "I like to go hunting with my friends from time to time",
            [],
        ),
        (
            "I don't eat gluten, and I'm moving to Syzran tomorrow, so I'm afraid there won't be any gluten-free food there",
            [
                {"subject": "user", "relation": "excludes from diet", "object": "gluten", "ttl": "inf"},
            ],
        ),
    ],
    "HEALTH": [
        (
            "Type 2 diabetes, seeing an endocrinologist",
            [
                {"subject": "user", "relation": "diagnosis", "object": "type 2 diabetes", "ttl": "inf"},
                {"subject": "user", "relation": "under care of", "object": "endocrinologist", "ttl": "1y"},
            ],
        ),
        (
            "Hypertension, I check my blood pressure every day",
            [
                {"subject": "user", "relation": "diagnosis", "object": "hypertension", "ttl": "inf"},
                {"subject": "user", "relation": "monitors", "object": "blood pressure", "ttl": "inf"},
                {"subject": "blood pressure", "relation": "check frequency", "object": "every day", "ttl": "inf"},
            ],
        ),
        (
            "Got the flu, temperature 38.7, taking fever reducer",
            [
                {"subject": "user", "relation": "sick with", "object": "flu", "ttl": "10d"},
                {"subject": "user", "relation": "temperature", "object": "38.7", "ttl": "3d"},
                {"subject": "user", "relation": "takes", "object": "fever reducer", "ttl": "3d"},
            ],
        ),
        (
            "everything was fine at work today",
            [],
        ),
        (
            "I have type 2 diabetes, but my son doesn't",
            [
                {"subject": "user", "relation": "diagnosis", "object": "type 2 diabetes", "ttl": "inf"},
            ],
        ),
    ],
    "MENTAL_HEALTH": [
        (
            "Saw a therapist for six months, feeling better",
            [
                {"subject": "user", "relation": "saw", "object": "therapist", "ttl": "6m"},
                {"subject": "therapist", "relation": "therapy duration", "object": "six months", "ttl": "6m"},
            ],
        ),
        (
            "Often anxious before public speaking",
            [
                {"subject": "user", "relation": "often feels", "object": "anxiety before talks", "ttl": "6m"},
            ],
        ),
        (
            "my son's birthday is tomorrow",
            [],
        ),
        (
            "I'm very anxious before a presentation, I bought a new phone, and I'll read from it",
            [
                {"subject": "user", "relation": "often feels", "object": "anxiety before presentations", "ttl": "6m"},
            ],
        )
    ],
    "EDUCATION": [
        (
            "I'm doing a master's at NSU, AI department",
            [
                {"subject": "user", "relation": "studies at", "object": "nsu", "ttl": "1y"},
                {"subject": "user", "relation": "degree level", "object": "master's", "ttl": "1y"},
                {"subject": "user", "relation": "department", "object": "artificial intelligence", "ttl": "1y"},
            ],
        ),
        (
            "Graduated from NSU in applied mathematics",
            [
                {"subject": "user", "relation": "graduated from", "object": "nsu", "ttl": "inf"},
                {"subject": "user", "relation": "major", "object": "applied mathematics", "ttl": "inf"},
            ],
        ),
        (
            "I went to the store and cooked dinner, and my girlfriend liked it",
            [],
        ),
        (
            "I'm a master's student at the Moscow Institute of Physics and Technology, and we go to guitar workshops on weekends",
            [
                {"subject": "user", "relation": "studies at", "object": "mipt", "ttl": "1y"},
                {"subject": "user", "relation": "level of study", "object": "master's degree", "ttl": "1y"},
            ],
        ),
        (
            "Completed a degree in corporate finance, it will realy helped me to find a job",
            [
                {"subject": "user", "relation": "has degree in", "object": "corporate finance", "ttl": "inf"},
                {"subject": "corporate finance", "relation": "help", "object": "find a job", "ttl": "inf"},
            ],
        ),
    ],
    "SPORTS": [
        (
            "I run ten kilometers on Sundays",
            [
                {"subject": "user", "relation": "does", "object": "running", "ttl": "6m"},
                {"subject": "running", "relation": "distance", "object": "10 km", "ttl": "6m"},
                {"subject": "running", "relation": "day", "object": "sunday", "ttl": "6m"},
            ],
        ),
        (
            "Every Saturday I play football with friends",
            [
                {"subject": "user", "relation": "plays", "object": "football", "ttl": "6m"},
                {"subject": "football", "relation": "frequency", "object": "every saturday", "ttl": "6m"},
                {"subject": "football", "relation": "with", "object": "friends", "ttl": "6m"},
            ],
        ),
        (
            "Yeah sport is great but I'm lazy today",
            [],
        ),
        (
            "I watched a TV series at home today",
            [],
        ),
        (
            "I play volleyball every Saturday, and I also work at Yandex",
            [
                {"subject": "user", "relation": "plays", "object": "volleyball", "ttl": "6m"},
                {"subject": "volleyball", "relation": "frequency", "object": "every Saturday", "ttl": "6m"},
            ],
        ),
    ],
    "LOCATION": [
        (
            "Moved to Krasnoyarsk from Irkutsk",
            [
                {"subject": "user", "relation": "lives in", "object": "krasnoyarsk", "ttl": "1y"},
                {"subject": "user", "relation": "previously lived in", "object": "irkutsk", "ttl": "1y"},
            ],
        ),
        (
            "I live in Moscow now, renting in Mitino",
            [
                {"subject": "user", "relation": "lives in", "object": "moscow", "ttl": "1y"},
                {"subject": "user", "relation": "neighborhood", "object": "mitino", "ttl": "1y"},
            ],
        ),
        (
            "I went to work today, but ended up spending 2 hours in traffic",
            [],
        ),
        (
            "I live in Moscow and walk my dog outside every day",
            [
                {"subject": "user", "relation": "lives in", "object": "moscow", "ttl": "1y"},
            ],
        ),
    ],
    "FINANCE": [
        (
            "Mortgage at Sber, payment forty thousand",
            [
                {"subject": "user", "relation": "pays", "object": "mortgage", "ttl": "1y"},
                {"subject": "mortgage", "relation": "bank", "object": "sber", "ttl": "1y"},
                {"subject": "mortgage", "relation": "payment", "object": "40000 rubles", "ttl": "1y"},
            ],
        ),
        (
            "I save twenty percent of salary for an emergency fund",
            [
                {"subject": "user", "relation": "saves", "object": "twenty percent of salary", "ttl": "3m"},
                {"subject": "user", "relation": "savings goal", "object": "emergency fund", "ttl": "3m"},
            ],
        ),
        (
            "I'm watching a new TV series and I'm not leaving the house",
            [],
        ),
        (
            "I'm saving 20% of my salary, and I'm thinking about buying a car in a year so that my wife and I can travel",
            [
                {"subject": "user", "relation": "saving", "object": "20% of salary", "ttl": "3m"},
                {"subject": "user", "relation": "saving goal", "object": "car", "ttl": "1y"},
            ],
        ),
    ],
    "VEHICLES": [
        (
            "Changed the oil, I drive a Ford",
            [
                {"subject": "user", "relation": "car", "object": "ford", "ttl": "1y"},
            ],
        ),
        (
            "I have a Kia Rio, before that a Renault Logan",
            [
                {"subject": "user", "relation": "car", "object": "kia rio", "ttl": "1y"},
                {"subject": "user", "relation": "previous car", "object": "renault logan", "ttl": "1y"},
            ],
        ),
        (
            "Жене сегодня 67 лет, что можно подарить?",
            [],
        ),
        (
            "у меня киа рио, специально взял, чтобы детей на бокс возить",
            [
                {"subject": "пользователь", "relation": "авто", "object": "киа рио", "ttl": "1 год"},
            ],
        ),
    ],
    "TRAVEL": [
        (
            "Flying to Tokyo in September",
            [
                {"subject": "user", "relation": "trip", "object": "tokyo", "ttl": "3m"},
                {"subject": "tokyo", "relation": "date", "object": "september", "ttl": "3m"},
            ],
        ),
        (
            "Planning a trip to Kazan with family in August",
            [
                {"subject": "user", "relation": "plans trip", "object": "kazan", "ttl": "3m"},
                {"subject": "kazan", "relation": "date", "object": "august", "ttl": "3m"},
                {"subject": "kazan", "relation": "travels with", "object": "family", "ttl": "3m"},
            ],
        ),
        (
            "Last time I was in England the food was awful",
            [
                {"subject": "user", "relation": "visited", "object": "england", "ttl": "3m"},
                {"subject": "england", "relation": "food impression", "object": "did not like", "ttl": "3m"},
            ],
        ),
        (
            "I like Japanese cuisine",
            [],
        ),
        (
            "I'm flying to Tokyo in September, and I'm finishing my master's degree in August",
            [
                {"subject": "user", "relation": "trip", "object": "tokyo", "ttl": "3m"},
                {"subject": "tokyo", "relation": "date", "object": "september", "ttl": "3m"},
            ],
        ),
    ],
    "HOBBIES": [
        (
            "I've been collecting vinyl for fifteen years",
            [
                {"subject": "user", "relation": "collects", "object": "vinyl", "ttl": "inf"},
                {"subject": "vinyl", "relation": "collecting since", "object": "15 years", "ttl": "inf"},
            ],
        ),
        (
            "Into photography, shoot with a DSLR",
            [
                {"subject": "user", "relation": "hobby", "object": "photography", "ttl": "1y"},
                {"subject": "photography", "relation": "shoots with", "object": "dslr", "ttl": "1y"},
            ],
        ),
        (
            "the weather is great today and I'm in a good mood, I think I'll go play basketball",
            [],
        ),
        (
            "I'm into photography, but I'm allergic to cold, so I won't be able to take photos today",
            [
                {"subject": "user", "relation": "hobby", "object": "photography", "ttl": "1y"},
            ],
        ),
    ],
    "TECH": [
        (
            "Main phone Samsung, laptop Lenovo",
            [
                {"subject": "user", "relation": "phone", "object": "samsung", "ttl": "6m"},
                {"subject": "user", "relation": "laptop", "object": "lenovo", "ttl": "6m"},
            ],
        ),
        (
            "I use an iPhone and MacBook for work",
            [
                {"subject": "user", "relation": "uses for work", "object": "iphone", "ttl": "6m"},
                {"subject": "user", "relation": "uses for work", "object": "macbook", "ttl": "6m"},
            ],
        ),
        (
            "взял отпуск на неделю и уехал в Тверь",
            [],
        ),
        (
            "основной телефон самсунг, а ещё у меня две кошки",
            [
                {"subject": "пользователь", "relation": "телефон", "object": "самсунг", "ttl": "6 м"},
            ],
        ),
    ],
    "SCHEDULE": [
        (
            "Tuesdays and Thursdays I'm at work until nine",
            [
                {"subject": "user", "relation": "work schedule", "object": "tue thu until 21:00", "ttl": "1m"},
            ],
        ),
        (
            "On weekdays I wake at 6 and sleep around 23:00",
            [
                {"subject": "user", "relation": "weekday wake time", "object": "06:00", "ttl": "1m"},
                {"subject": "user", "relation": "bedtime", "object": "23:00", "ttl": "1m"},
            ],
        ),
        (
            "I like coffee in the morning",
            [],
        ),
        (
            "I get up at 6 a.m. on weekdays, just in time to walk the dog",
            [
                {"subject": "user", "relation": "weekday wake-up time", "object": "06:00", "ttl": "1m"},
            ],
        ),
    ],
    "GOALS": [
        (
            "I want to pass IELTS at seven point five",
            [
                {"subject": "user", "relation": "goal", "object": "ielts 7.5", "ttl": "3m"},
            ],
        ),
        (
            "I want to move to product analytics in a year",
            [
                {"subject": "user", "relation": "goal", "object": "move to product analytics", "ttl": "1y"},
                {"subject": "move to product analytics", "relation": "deadline", "object": "in one year", "ttl": "1y"},
            ],
        ),
        (
            "Since childhood I've dreamed of visiting Lake Baikal",
            [
                {"subject": "user", "relation": "dreams of visiting", "object": "lake baikal", "ttl": "inf"},
                {"subject": "lake baikal", "relation": "dream since", "object": "childhood", "ttl": "inf"},
            ],
        ),
        (
            "I bought bread and milk today, my husband asked for it.",
            [],
        ),
        (
        "I want to learn programming this year. By the way, my cat got sick.",
            [
                {"subject": "user", "relation": "purpose to learn", "object": "programming", "ttl": "3m"},
            ],
        ),
    ],
    "EVENTS": [
        (
            "Last week I was at my cousin's wedding",
            [
                {"subject": "user", "relation": "attended", "object": "cousin's wedding", "ttl": "2w"},
            ],
        ),
        (
            "Last month I spoke at Data Fest",
            [
                {"subject": "user", "relation": "spoke at", "object": "data fest conference", "ttl": "3m"},
            ],
        ),
        (
            "Just got back from the bar, I'm a bit drunk",
            [
                {"subject": "user", "relation": "returned from", "object": "bar", "ttl": "1d"},
                {"subject": "user", "relation": "state", "object": "drunk", "ttl": "1d"},
            ],
        ),
        (
            "Defended my thesis today, now I'm a PhD candidate!",
            [
                {"subject": "user", "relation": "defended", "object": "thesis", "ttl": "2w"},
                {"subject": "user", "relation": "degree", "object": "phd candidate", "ttl": "inf"},
            ],
        ),
        (
            "I have an old camera and a bicycle",
            [],
        ),
        (
            "I went to my cousin's wedding last week, and the exams at the faculty are in two weeks",
            [
                {"subject": "user", "relation": "attended", "object": "cousin's wedding", "ttl": "2w"},
            ],
        ),
    ],
    "HOME": [
        (
            "Renting a one-bedroom on the outskirts, fifth floor",
            [
                {"subject": "user", "relation": "home", "object": "one-bedroom", "ttl": "1y"},
                {"subject": "one-bedroom", "relation": "location", "object": "outskirts", "ttl": "1y"},
                {"subject": "one-bedroom", "relation": "floor", "object": "5", "ttl": "1y"},
            ],
        ),
        (
            "Live in a two-room flat, recently renovated the kitchen",
            [
                {"subject": "user", "relation": "home", "object": "two-room flat", "ttl": "1y"},
                {"subject": "two-room flat", "relation": "renovated", "object": "kitchen", "ttl": "6m"},
            ],
        ),
        (
            "today we discussed politics and news with my son, we didn't see eye to eye",
            [],
        ),
        (
            "I live in a two-room apartment, recently renovated the kitchen, although I'm leaving for Cyprus tomorrow, I won't be able to enjoy the renovation",
            [
                {"subject": "user", "relation": "home", "object": "two-room flat", "ttl": "1y"},
                {"subject": "two-room flat", "relation": "renovated", "object": "kitchen", "ttl": "6m"},
            ],
        ),
    ],
    "IDENTITY": [
        (
            "My name is Alexey, I'm thirty-two",
            [
                {"subject": "user", "relation": "name", "object": "alexey", "ttl": "inf"},
                {"subject": "user", "relation": "age", "object": "32", "ttl": "1y"},
            ],
        ),
        (
            "I'm male, twenty-eight, ethnically Tatar",
            [
                {"subject": "user", "relation": "gender", "object": "male", "ttl": "inf"},
                {"subject": "user", "relation": "age", "object": "28", "ttl": "1y"},
                {"subject": "user", "relation": "ethnicity", "object": "tatar", "ttl": "inf"},
            ],
        ),
        (
            "today was a long meeting and a lot of calls at work",
            [],
        ),
        (
            "my name is Alexey, I'm 32, I work at Yandex and I have a cat",
            [
                {"subject": "user", "relation": "name", "object": "alexey", "ttl": "inf"},
                {"subject": "user", "relation": "age", "object": "32", "ttl": "1y"},
            ],
        ),
    ],
    "ROMANCE": [
        (
            "My partner and I have lived together for two years",
            [
                {"subject": "user", "relation": "has partner", "object": "user's partner", "ttl": "1y"},
                {"subject": "user", "relation": "lives with", "object": "user's partner", "ttl": "1y"},
                {"subject": "user's partner", "relation": "cohabitation", "object": "2 years", "ttl": "1y"},
            ],
        ),
        (
            "I've been dating my girlfriend for a year and a half",
            [
                {"subject": "user", "relation": "has girlfriend", "object": "user's girlfriend", "ttl": "1y"},
                {"subject": "user's girlfriend", "relation": "relationship length", "object": "one and a half years", "ttl": "1y"},
            ],
        ),
        (
            "I watched the match, then went to the bar with my friends",
            [],
        ),
        (
            "I've been dating a guy for a year and a half, I've been in love with him for a long time.",
            [
                {"subject": "user", "relation": "has boyfriend", "object": "user's boyfriend", "ttl": "1y"},
                {"subject": "user's boyfriend", "relation": "relationship length", "object": "one and a half years", "ttl": "1y"},
            ],
        ),
    ],
    "FRIENDS": [
        (
            "Every Friday we meet friends at a bar",
            [
                {"subject": "user", "relation": "meets friends at", "object": "bar", "ttl": "6m"},
                {"subject": "bar", "relation": "with", "object": "friends", "ttl": "6m"},
            ],
        ),
        (
            "My best friend Dima and I have been friends since school",
            [
                {"subject": "user", "relation": "has best friend", "object": "user's best friend", "ttl": "inf"},
                {"subject": "user's best friend", "relation": "name", "object": "dima", "ttl": "inf"},
                {"subject": "user's best friend", "relation": "friends since", "object": "school", "ttl": "inf"},
            ],
        ),
        (
            "going on vacation and choosing a hotel",
            [],
        ),
        (
            "I've been friends with Lera since high school, now I work in a bank, she has her own business",
            [
                {"subject": "user", "relation": "has friend", "object": "user's friend", "ttl": "inf"},
                {"subject": "user's friend", "relation": "name", "object": "lera", "ttl": "inf"},
                {"subject": "lera", "relation": "friends since", "object": "high school", "ttl": "inf"},
                {"subject": "lera", "relation": "has", "object": "business", "ttl": "inf"},
            ],
        ),
    ],
    "HABITS": [
        (
            "I try to sleep before midnight every day",
            [
                {"subject": "user", "relation": "sleep habit", "object": "before midnight", "ttl": "inf"},
            ],
        ),
        (
            "I've smoked for fifteen years",
            [
                {"subject": "user", "relation": "bad habit", "object": "smoking", "ttl": "inf"},
                {"subject": "smoking", "relation": "for", "object": "15 years", "ttl": "inf"},
            ],
        ),
        (
            "I became interested in sports, and recently moved to Kazan",
            [],
        ),
        (
            "I've been smoking for 15 years, and I recently moved to Kazan, and soon my mother will move too",
            [
                {"subject": "user", "relation": "bad habit", "object": "smoking", "ttl": "inf"},
                {"subject": "smoking", "relation": "duration", "object": "15 years", "ttl": "inf"},
            ],
        ),
    ],
    "PREFERENCES": [
        (
            "I don't like sugar in coffee",
            [
                {"subject": "user", "relation": "dislikes", "object": "sweet coffee", "ttl": "inf"},
            ],
        ),
        (
            "I prefer dark UIs and minimal design",
            [
                {"subject": "user", "relation": "prefers", "object": "dark interfaces", "ttl": "6m"},
                {"subject": "user", "relation": "prefers", "object": "minimalism", "ttl": "6m"},
            ],
        ),
        (
            "I was at a conference yesterday and met some new friends",
            [],
        ),
        (
            "I don't like sweets in coffee, and now I live in Kazan, the air is cleaner here",
            [
                {"subject": "user", "relation": "does not like", "object": "sweet coffee", "ttl": "inf"},
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
# Single-pass few-shot (all slots in one pass, slot field in JSON)
# ---------------------------------------------------------------------------

TRIPLET_SINGLE_PASS_BASE: List[Tuple[str, List[dict]]] = [
    (
        "Recently my wife and I went for a walk before picking up our son from school. Glad I even made it after work. I work as an engineer, so I often stay late.",
        [
            {"slot": "FAMILY", "subject": "user", "relation": "married", "object": "user's wife", "ttl": "inf"},
            {"slot": "FAMILY", "subject": "user", "relation": "child", "object": "user's son", "ttl": "inf"},
            {"slot": "FAMILY", "subject": "user's son", "relation": "studies at", "object": "school", "ttl": "1y"},
            {"slot": "WORK", "subject": "user", "relation": "profession", "object": "engineer", "ttl": "1y"},
            {"slot": "WORK", "subject": "user", "relation": "often stays late", "object": "work", "ttl": "1y"},
            {"slot": "SCHEDULE", "subject": "user", "relation": "often stays late", "object": "work", "ttl": "1y"},
        ],
    ),
    (
        "Recently I took my cat Murzik for surgery because he has kidney problems. It only took 2 hours. While waiting, I went to a coffee shop, and they added sugar, but I can’t have it — I have diabetes.",
        [
            {"slot": "PETS", "subject": "user", "relation": "has cat", "object": "user's cat", "ttl": "inf"},
            {"slot": "PETS", "subject": "user's cat", "relation": "name", "object": "Murzik", "ttl": "inf"},
            {"slot": "PETS", "subject": "user's cat", "relation": "ill", "object": "kidney problems", "ttl": "inf"},
            {"slot": "PETS", "subject": "user's cat", "relation": "had", "object": "surgery", "ttl": "1y"},
            {"slot": "PETS", "subject": "surgery", "relation": "duration", "object": "2 hours", "ttl": "1y"},
            {"slot": "FOOD", "subject": "user", "relation": "avoids", "object": "sugar", "ttl": "inf"},
            {"slot": "EVENTS", "subject": "user", "relation": "went to", "object": "coffee shop", "ttl": "1y"},
            {"slot": "EVENTS", "subject": "coffee shop", "relation": "added", "object": "sugar", "ttl": "1y"},
        ],
    ),
    (
        "The weather is great today, sunny and bright.",
        [],
    ),
    (
        "Recently I traveled to Yekaterinburg. I liked the city, but it’s small.",
        [
            {"slot": "TRAVEL", "subject": "user", "relation": "trip", "object": "Yekaterinburg", "ttl": "1y"},
            {"slot": "TRAVEL", "subject": "Yekaterinburg", "relation": "impression", "object": "liked it", "ttl": "1y"},
            {"slot": "TRAVEL", "subject": "Yekaterinburg", "relation": "impression", "object": "small", "ttl": "1y"},
        ],
    ),
    (
        "I’m generally into a healthy lifestyle: I run marathons, practice yoga, and don’t drink alcohol.",
        [
            {"slot": "PREFERENCES", "subject": "user", "relation": "prefers", "object": "healthy lifestyle", "ttl": "6m"},
            {"slot": "SPORTS", "subject": "user", "relation": "runs", "object": "marathons", "ttl": "6m"},
            {"slot": "SPORTS", "subject": "user", "relation": "practices", "object": "yoga", "ttl": "6m"},
            {"slot": "HABITS", "subject": "user", "relation": "does not consume", "object": "alcohol", "ttl": "inf"},
        ],
    ),
    (
        "Just got back from a bar, I’m a bit tipsy.",
        [
            {"slot": "EVENTS", "subject": "user", "relation": "returned from", "object": "bar", "ttl": "1d"},
            {"slot": "EVENTS", "subject": "user", "relation": "state", "object": "tipsy", "ttl": "1d"},
        ],
    ),
    (
        "Caught a cold, sitting at home with a temperature of 37.8. I wanted to go to my dacha in Lebedevka, I’ve been meaning to fix the roof for a while.",
        [
            {"slot": "HEALTH", "subject": "user", "relation": "ill", "object": "cold", "ttl": "10d"},
            {"slot": "HEALTH", "subject": "user", "relation": "temperature", "object": "37.8", "ttl": "3d"},
            {"slot": "HOME", "subject": "user", "relation": "has", "object": "user's dacha", "ttl": "inf"},
            {"slot": "HOME", "subject": "user's dacha", "relation": "location", "object": "Lebedevka", "ttl": "inf"},
            {"slot": "GOALS", "subject": "user", "relation": "wanted", "object": "fix the roof", "ttl": "inf"},
        ],
    ),
    (
        "I’ve always wanted to learn to play the guitar, since childhood.",
        [
            {"slot": "GOALS", "subject": "user", "relation": "lifelong dream", "object": "learn to play the guitar", "ttl": "inf"},
        ],
    ),
    (
        "Before working shifts, I used to go hunting with my father every weekend. Then for almost two years I couldn’t, but this spring we started going again.",
        [
            {"slot": "HOBBIES", "subject": "user", "relation": "used to do", "object": "hunting", "ttl": "inf"},
            {"slot": "HOBBIES", "subject": "user", "relation": "enjoys", "object": "hunting", "ttl": "6m"},
            {"slot": "FAMILY", "subject": "hunting", "relation": "together with", "object": "user's father", "ttl": "6m"},
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
    ("What is my wife's name?", "FAMILY\nWORK\nROMANCE", '{"use_memory": true, "slots": ["FAMILY", "ROMANCE"]}'),
    ("What is quantum mechanics?", "FAMILY\nWORK", '{"use_memory": false, "slots": []}'),
    ("Remind me where I work and how to get there", "WORK\nLOCATION\nVEHICLES", '{"use_memory": true, "slots": ["WORK", "LOCATION", "VEHICLES"]}'),
    ("How are you?", "FAMILY\nHEALTH", '{"use_memory": false, "slots": []}'),
    ("Tell me about my pets", "PETS\nFOOD\nHEALTH", '{"use_memory": true, "slots": ["PETS", "FOOD", "HEALTH"]}'),
    ("Compare my schedule and goals", "SCHEDULE\nGOALS\nWORK\nHABITS", '{"use_memory": true, "slots": ["SCHEDULE", "GOALS", "WORK", "HABITS"]}'),
    ("What TV shows should I watch tonight", "HOBBIES\nPREFERENCES\nEVENTS", '{"use_memory": true, "slots": ["HOBBIES", "PREFERENCES"]}'),
    ("What is two plus two", "EDUCATION\nWORK", '{"use_memory": false, "slots": []}'),
    ("Suggest what to eat today", "FOOD\nHEALTH\nHABITS\nPREFERENCES", '{"use_memory": true, "slots": ["FOOD", "HEALTH", "HABITS", "PREFERENCES"]}'),
    ("I want to start exercising, any advice?", "SPORTS\nHEALTH\nSCHEDULE\nGOALS", '{"use_memory": true, "slots": ["SPORTS", "HEALTH", "SCHEDULE", "GOALS"]}'),
    ("Can I afford a new car?", "FINANCE\nVEHICLES\nWORK", '{"use_memory": true, "slots": ["FINANCE", "VEHICLES", "WORK"]}'),
    ("Remind me about my plans", "GOALS\nSCHEDULE\nTRAVEL\nEVENTS", '{"use_memory": true, "slots": ["GOALS", "SCHEDULE", "TRAVEL", "EVENTS"]}'),
    ("Tell me about my health", "HEALTH\nMENTAL_HEALTH\nHABITS\nSPORTS\nFOOD", '{"use_memory": true, "slots": ["HEALTH", "MENTAL_HEALTH", "HABITS", "SPORTS", "FOOD"]}'),
]


def memory_gate_user_block(question: str, slot_names: List[str], extra: str) -> str:
    slots_text = (
        "\n".join(f"- {name}" for name in slot_names)
        if slot_names
        else "(no slots with saved data)"
    )
    return (
        f"User message:\n{question}\n\n"
        f"Memory slot names:\n{slots_text}\n{extra}"
    )


MEMORY_GATE_FEWSHOT_VECTOR: List[Tuple[str, str, str]] = [
    (
        "What did I say about vacation",
        "TRAVEL\nWORK",
        '{"use_memory": true, "slots": []}',
    ),
    (
        "What is my schedule like",
        "SCHEDULE\nFAMILY",
        '{"use_memory": true, "slots": ["SCHEDULE"]}',
    ),
]


# ---------------------------------------------------------------------------
# Slot update extra few-shots
# ---------------------------------------------------------------------------

SLOT_UPDATE_EXTRA_FEWSHOT: List[Tuple[str, list, str]] = [
    (
        "My daughter is at a different school now",
        [{"id": 1, "value": "daughter: Masha, school #5"}],
        '{"operations":[{"op":"update","id":1,"value":"daughter: Masha, different school"}]}',
    ),
    (
        "Never mind, nothing new",
        [{"id": 1, "value": "car: kia"}],
        '{"operations":[{"op":"nothing"}]}',
    ),
    (
        "Sold the bike, bought a scooter",
        [{"id": 1, "value": "bicycle: city"}],
        '{"operations":[{"op":"delete","id":1},{"op":"add","value":"scooter: commuting"}]}',
    ),
    (
        "My wife took the surname Ivanova",
        [{"id": 1, "value": "wife: Maria Petrova"}],
        '{"operations":[{"op":"update","id":1,"value":"wife: Maria Ivanova"}]}',
    ),
    (
        "My son quit the sports club",
        [{"id": 1, "value": "son: swimming club"}],
        '{"operations":[{"op":"delete","id":1}]}',
    ),
]


# ---------------------------------------------------------------------------
# Context-aware triplet few-shots (slot_context_enabled=True, variant A)
# user turn: current facts + message; assistant: {"triplets":[...], "delete":[...]}
# ---------------------------------------------------------------------------

_CONTEXT_FEWSHOTS: List[Tuple[str, List[str], str, dict]] = [
    (
        "Moved to Syzran with my family",
        ["user | residence | moscow"],
        "HOME",
        {
            "triplets": [
                {"subject": "user", "relation": "residence", "object": "syzran", "ttl": "1y"},
                {"subject": "user", "relation": "former residence", "object": "moscow", "ttl": "1y"},
            ],
            "delete": [
                {"subject": "user", "relation": "residence", "object": "moscow"},
            ],
        },
    ),
    (
        "I don't live in Moscow anymore, I moved out",
        ["user | residence | moscow"],
        "HOME",
        {
            "triplets": [
                {"subject": "user", "relation": "former residence", "object": "moscow", "ttl": "1y"},
            ],
            "delete": [
                {"subject": "user", "relation": "residence", "object": "moscow"},
            ],
        },
    ),
    (
        "I left Yandex, now I work at Sber as a senior analyst",
        [
            "user | works at | yandex",
            "user | role | data analyst",
        ],
        "WORK",
        {
            "triplets": [
                {"subject": "user", "relation": "works at", "object": "sber", "ttl": "1y"},
                {"subject": "user", "relation": "role", "object": "senior analyst", "ttl": "1y"},
                {"subject": "user", "relation": "previous employer", "object": "yandex", "ttl": "1y"},
            ],
            "delete": [
                {"subject": "user", "relation": "works at", "object": "yandex"},
                {"subject": "user", "relation": "role", "object": "data analyst"},
            ],
        },
    ),
    (
        "I quit smoking a month ago",
        [
            "user | smokes | yes",
            "user | cigarettes per day | pack a day",
        ],
        "HABITS",
        {
            "triplets": [
                {"subject": "user", "relation": "quit smoking", "object": "yes", "ttl": "inf"},
            ],
            "delete": [
                {"subject": "user", "relation": "smokes", "object": "yes"},
                {"subject": "user", "relation": "cigarettes per day", "object": "pack a day"},
            ],
        },
    ),
    (
        "Bought a bike for getting around the city",
        ["user | residence | yekaterinburg"],
        "TECH",
        {
            "triplets": [
                {"subject": "user", "relation": "has bicycle", "object": "user's bicycle", "ttl": "1y"},
                {"subject": "user's bicycle", "relation": "type", "object": "city", "ttl": "1y"},
            ],
            "delete": [],
        },
    ),
    (
        "All good, thanks",
        ["user | works at | sber"],
        "WORK",
        {"triplets": [], "delete": []},
    ),
]


def _ctx_assistant_json(data: dict, use_ttl: bool, enable_deletion: bool = True) -> str:
    """Serialize assistant payload for context few-shots."""
    triplets = data.get("triplets", [])
    if not use_ttl:
        triplets = [{k: v for k, v in t.items() if k != "ttl"} for t in triplets]
    if enable_deletion:
        delete = data.get("delete", [])
        return json.dumps({"triplets": triplets, "delete": delete}, ensure_ascii=False)
    return json.dumps({"triplets": triplets}, ensure_ascii=False)


def triplet_context_few_shot_messages(
    user_turn_fn,
    slot_name: str | None = None,
    use_ttl: bool = True,
    enable_deletion: bool = True,
) -> List[Dict[str, Any]]:
    """
    Context-aware extraction few-shots (slot_context_enabled=True).
    enable_deletion=True  → model returns {"triplets":[...], "delete":[...]}.
    enable_deletion=False → only {"triplets":[...]}.

    When slot_name is set, filter examples; otherwise take a shared subset.
    """
    out: List[Dict[str, Any]] = []
    selected = []

    if slot_name:
        for msg, facts, fs_slot, data in _CONTEXT_FEWSHOTS:
            if fs_slot == slot_name:
                selected.append((msg, facts, data))
        # If no examples for this slot — take shared fallback (move + no-op)
        if not selected:
            selected = [
                (_CONTEXT_FEWSHOTS[0][0], _CONTEXT_FEWSHOTS[0][1], _CONTEXT_FEWSHOTS[0][3]),
                (_CONTEXT_FEWSHOTS[4][0], _CONTEXT_FEWSHOTS[4][1], _CONTEXT_FEWSHOTS[4][3]),
                (_CONTEXT_FEWSHOTS[5][0], _CONTEXT_FEWSHOTS[5][1], _CONTEXT_FEWSHOTS[5][3]),
            ]
    else:
        # Single-pass without slot: diverse subset
        selected = [
            (_CONTEXT_FEWSHOTS[0][0], _CONTEXT_FEWSHOTS[0][1], _CONTEXT_FEWSHOTS[0][3]),
            (_CONTEXT_FEWSHOTS[2][0], _CONTEXT_FEWSHOTS[2][1], _CONTEXT_FEWSHOTS[2][3]),
            (_CONTEXT_FEWSHOTS[5][0], _CONTEXT_FEWSHOTS[5][1], _CONTEXT_FEWSHOTS[5][3]),
        ]

    for msg, facts, data in selected:
        out.append({"role": "user", "content": user_turn_fn(msg)})
        out.append({"role": "assistant", "content": _ctx_assistant_json(data, use_ttl, enable_deletion)})
    return out


# ---------------------------------------------------------------------------
# Conflict resolution few-shots
# Request: slot, existing_triplets (list with record_id), new_triplets (indexed)
# Response: {"deactivate":[record_ids], "skip_new":[new_indices]}
# Triplets: lowercase with spaces (no UPPER_CASE_UNDERSCORE)
# ---------------------------------------------------------------------------

def _cr(existing: list, new_triplets: list, answer: str) -> tuple:
    return (json.dumps(existing, ensure_ascii=False),
            json.dumps(new_triplets, ensure_ascii=False),
            answer)


CONFLICT_RESOLUTION_FEWSHOT: List[Tuple[str, str, str]] = [
    _cr(
        existing=[{"record_id": 1, "subject": "user", "relation": "works at", "object": "yandex"},
                  {"record_id": 2, "subject": "user", "relation": "role", "object": "data analyst"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "works at", "object": "sber"}],
        answer='{"deactivate":[1],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 5, "subject": "user", "relation": "lives in", "object": "moscow"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "lives in", "object": "tomsk"}],
        answer='{"deactivate":[5],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 3, "subject": "user", "relation": "marital status", "object": "married"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "marital status", "object": "divorced"}],
        answer='{"deactivate":[3],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 7, "subject": "user", "relation": "has cat", "object": "user's cat"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "has cat", "object": "user's cat"}],
        answer='{"deactivate":[],"skip_new":[0]}',
    ),
    _cr(
        existing=[{"record_id": 2, "subject": "user", "relation": "works at", "object": "yandex"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "role", "object": "team lead"}],
        answer='{"deactivate":[],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 8, "subject": "user", "relation": "car", "object": "kia rio"},
                  {"record_id": 9, "subject": "user", "relation": "has bicycle", "object": "city"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "car", "object": "ford focus"},
                      {"idx": 1, "subject": "user", "relation": "has scooter", "object": "for city"}],
        answer='{"deactivate":[8],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 11, "subject": "user", "relation": "diagnosis", "object": "hypertension"},
                  {"record_id": 12, "subject": "user", "relation": "under care of", "object": "cardiologist"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "diagnosis", "object": "hypertension stage 2"}],
        answer='{"deactivate":[11],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 4, "subject": "user", "relation": "works at", "object": "sber"},
                  {"record_id": 6, "subject": "user", "relation": "role", "object": "analyst"}],
        new_triplets=[{"idx": 0, "subject": "user", "relation": "role", "object": "senior analyst"}],
        answer='{"deactivate":[6],"skip_new":[]}',
    ),
    _cr(
        existing=[{"record_id": 10, "subject": "user", "relation": "has cat", "object": "user's cat"}],
        new_triplets=[{"idx": 0, "subject": "user's cat", "relation": "breed", "object": "maine coon"}],
        answer='{"deactivate":[],"skip_new":[]}',
    ),
]
