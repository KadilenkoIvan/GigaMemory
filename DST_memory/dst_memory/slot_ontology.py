from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class SlotDefinition:
    slot_id: str
    title: str
    description: str


SLOT_DEFINITIONS: List[SlotDefinition] = [
    SlotDefinition(
        slot_id="identity_profile",
        title="Идентичность и базовый профиль",
        description="Имя, возраст, документы, базовые персональные факты пользователя.",
    ),
    SlotDefinition(
        slot_id="family_relationships",
        title="Семья и отношения",
        description="Супруг(а), дети, родители, родственные и семейные связи.",
    ),
    SlotDefinition(
        slot_id="social_circle",
        title="Социальный круг",
        description="Друзья, знакомые, значимые личные контакты вне семьи.",
    ),
    SlotDefinition(
        slot_id="work_education",
        title="Работа и образование",
        description="Профессия, работа, навыки, обучение, учебные задачи.",
    ),
    SlotDefinition(
        slot_id="health_wellbeing",
        title="Здоровье и самочувствие",
        description="Состояние здоровья, симптомы, самочувствие, ограничения.",
    ),
    SlotDefinition(
        slot_id="habits_lifestyle",
        title="Привычки и образ жизни",
        description="Регулярные привычки, режим, стиль повседневной жизни.",
    ),
    SlotDefinition(
        slot_id="sports_activity",
        title="Спорт и активность",
        description="Виды спорта, физическая активность, регулярные тренировки.",
    ),
    SlotDefinition(
        slot_id="hobbies_interests",
        title="Хобби и интересы",
        description="Увлечения, интересы, досуг пользователя.",
    ),
    SlotDefinition(
        slot_id="media_culture_tastes",
        title="Культурные и медиапредпочтения",
        description="Книги, фильмы, авторы, музыка и культурные вкусы.",
    ),
    SlotDefinition(
        slot_id="travel_mobility",
        title="Поездки и мобильность",
        description="Поездки, маршруты, путешествия, планы поездок.",
    ),
    SlotDefinition(
        slot_id="transport_vehicles",
        title="Транспорт и автомобили",
        description="Транспорт, автомобили, владение и выбор транспортных средств.",
    ),
    SlotDefinition(
        slot_id="finance_consumption",
        title="Финансы и потребление",
        description="Бюджет, доходы, траты, накопления, покупки.",
    ),
    SlotDefinition(
        slot_id="home_daily_life",
        title="Быт и домашняя жизнь",
        description="Домашние дела, быт, повседневные семейные ситуации.",
    ),
    SlotDefinition(
        slot_id="goals_plans",
        title="Цели и планы",
        description="Намерения, цели, среднесрочные и долгосрочные планы.",
    ),
]

SLOT_IDS: List[str] = [s.slot_id for s in SLOT_DEFINITIONS]
SLOT_BY_ID: Dict[str, SlotDefinition] = {s.slot_id: s for s in SLOT_DEFINITIONS}

# Short, stable uppercase labels used in prompts (RAGU-style).
# These labels are what the slot-selection LLM should output.
SLOT_LABEL_BY_ID: Dict[str, str] = {
    "identity_profile": "IDENTITY",
    "family_relationships": "FAMILY",
    "social_circle": "SOCIAL",
    "work_education": "WORK",
    "health_wellbeing": "HEALTH",
    "habits_lifestyle": "HABITS",
    "sports_activity": "SPORTS",
    "hobbies_interests": "HOBBIES",
    "media_culture_tastes": "MEDIA",
    "travel_mobility": "TRAVEL",
    "transport_vehicles": "TRANSPORT",
    "finance_consumption": "FINANCE",
    "home_daily_life": "HOME",
    "goals_plans": "GOALS",
}

SLOT_ID_BY_LABEL: Dict[str, str] = {v: k for k, v in SLOT_LABEL_BY_ID.items()}


def slot_catalog_markdown() -> str:
    """
    A compact slot catalog for prompts: one line per slot.
    Uses uppercase prompt labels, keeps descriptions in Russian (domain of dialogs).
    """
    lines: list[str] = []
    for s in SLOT_DEFINITIONS:
        label = SLOT_LABEL_BY_ID.get(s.slot_id, s.slot_id.upper())
        lines.append(f"- {label}: {s.description}")
    return "\n".join(lines)


def slot_descriptions_markdown() -> str:
    return "\n".join(
        f"- {s.slot_id}: {s.description}"
        for s in SLOT_DEFINITIONS
    )

