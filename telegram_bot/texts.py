"""Russian UI texts and human-readable formatting for memory / stats.

The bot interface is always Russian (per product decision). The ru/en choice
only switches the *answer* language the API generates (prompt_language).
"""

from __future__ import annotations

from typing import Any

LANG_LABELS = {"ru": "Русский 🇷🇺", "en": "English 🇬🇧"}

# TTL label → human-readable Russian text (mirrors api.py _TTL_DISPLAY).
TTL_DISPLAY = {
    "inf": "бессрочно",
    "1y": "1 год",
    "6m": "6 месяцев",
    "3m": "3 месяца",
    "1m": "1 месяц",
    "3w": "3 недели",
    "2w": "2 недели",
    "10d": "10 дней",
    "3d": "3 дня",
    "1d": "1 день",
}

TELEGRAM_MAX_LEN = 4096


def greeting(has_language: bool, current_lang: str) -> str:
    base = (
        "👋 <b>GigaMemory — демонстрация долговременной памяти для LLM</b>\n\n"
        "Просто общайтесь со мной обычными сообщениями. Я извлекаю из них факты "
        "о вас (имя, работа, увлечения, питомцы и т.д.), строю из них "
        "<b>граф знаний</b> и учитываю его в следующих ответах — даже когда "
        "история диалога давно ушла за пределы окна контекста модели.\n\n"
        "У каждого пользователя — <b>свой</b> отдельный граф памяти.\n\n"
        "Команды: /info — как это работает · /graph — граф картинкой · "
        "/memory — факты текстом · /stats — статистика · /forget — сброс памяти · "
        "/language — язык ответов."
    )
    if not has_language:
        return base + "\n\n<b>Выберите язык ответов ассистента:</b>"
    return (
        base
        + f"\n\nТекущий язык ответов: <b>{LANG_LABELS.get(current_lang, current_lang)}</b>."
    )


INFO = (
    "ℹ️ <b>Что это за демонстрация</b>\n\n"
    "GigaMemory — это модуль долговременной памяти для языковых моделей. "
    "Вместо того чтобы держать всю историю диалога в контексте (что дорого и "
    "невозможно для длинных диалогов), система хранит знания о пользователе в "
    "виде компактного <b>графа триплетов</b> «субъект — отношение — объект».\n\n"
    "<b>Как обрабатывается сообщение:</b>\n"
    "1. <b>Важность</b> — классификатор решает, есть ли в сообщении факты для запоминания.\n"
    "2. <b>Выбор слотов</b> — к каким категориям относятся факты (личность, работа, "
    "питомцы, предпочтения...).\n"
    "3. <b>Извлечение триплетов</b> — факты превращаются в рёбра графа.\n"
    "4. <b>Дедупликация и конфликты</b> — похожие факты объединяются, устаревшие "
    "заменяются (у фактов есть TTL — срок жизни).\n"
    "5. <b>Ответ</b> — релевантные факты из графа передаются финальной LLM, и она "
    "отвечает с учётом памяти.\n\n"
    "Слот-модель (извлечение фактов) — Qwen3.5 4B через vLLM. "
    "Финальная LLM — через OpenRouter.\n\n"
    "Граф можно посмотреть: /graph (картинка), /graph_html (интерактивный файл), "
    "/memory (текст). Сбросить — /forget."
)


def choose_language_prompt() -> str:
    return "Выберите язык ответов ассистента:"


def language_set(lang: str) -> str:
    return (
        f"✅ Язык ответов: <b>{LANG_LABELS.get(lang, lang)}</b>.\n"
        "Интерфейс бота остаётся на русском — меняется только язык ответов ассистента."
    )


def forget_done() -> str:
    return "🗑️ Память этого диалога очищена. Начнём с чистого листа."


def api_unavailable() -> str:
    return (
        "⚠️ Не удаётся связаться с сервером GigaMemory. "
        "Убедитесь, что API запущен (make serve) и доступен по заданному адресу."
    )


def empty_memory() -> str:
    return (
        "🧠 Память пока пуста. Расскажите о себе — например, как вас зовут, "
        "кем работаете, чем увлекаетесь."
    )


def _ttl_text(ttl: str) -> str:
    return TTL_DISPLAY.get(ttl, ttl)


def format_memory(slots: dict[str, list[dict[str, Any]]]) -> str:
    """Render graph_short slots as a compact human-readable Russian list."""
    if not slots:
        return empty_memory()

    total = sum(len(records) for records in slots.values())
    lines = [f"🧠 <b>Ваша память</b> — слотов: {len(slots)}, фактов: {total}\n"]
    for slot_name in sorted(slots):
        records = slots[slot_name]
        if not records:
            continue
        lines.append(f"📁 <b>{slot_name}</b>")
        for r in records:
            subj = r.get("subject", "?")
            rel = r.get("relation", "?")
            obj = r.get("object", "?")
            ttl = r.get("ttl", "")
            ttl_suffix = f"  <i>(TTL: {_ttl_text(ttl)})</i>" if ttl else ""
            lines.append(f"  • {subj} — {rel} — {obj}{ttl_suffix}")
        lines.append("")

    text = "\n".join(lines).rstrip()
    if len(text) > TELEGRAM_MAX_LEN:
        text = text[: TELEGRAM_MAX_LEN - 60].rstrip() + "\n\n… (список усечён)"
    return text


def format_stats(slots: dict[str, list[dict[str, Any]]]) -> str:
    """Counts derived from graph_short: slots, facts (triplets), entities (nodes)."""
    if not slots:
        return empty_memory()

    facts = 0
    entities: set[str] = set()
    for slot_name, records in slots.items():
        for r in records:
            facts += 1
            subj = str(r.get("subject", "")).strip()
            obj = str(r.get("object", "")).strip()
            if subj:
                entities.add(f"{slot_name}\x00{subj}")
            if obj:
                entities.add(f"{slot_name}\x00{obj}")

    return (
        "📊 <b>Статистика памяти</b>\n\n"
        f"• Слотов: <b>{len(slots)}</b>\n"
        f"• Фактов (триплетов): <b>{facts}</b>\n"
        f"• Сущностей (узлов графа): <b>{len(entities)}</b>"
    )
