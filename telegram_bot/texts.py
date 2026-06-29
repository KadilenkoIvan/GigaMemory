"""Russian UI texts and human-readable formatting for memory / stats.

The bot interface is always Russian (per product decision). The ru/en choice
sets the language the system works in (prompt_language) — memory extraction
and the answer. The memory graph is single-language, so changing it clears the
user's memory.
"""

from __future__ import annotations

import html
from typing import Any

LANG_LABELS = {"ru": "Русский 🇷🇺", "en": "English 🇬🇧 (recommended)"}

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

# Native command menu (the "☰ Menu" button next to the input field).
# (command, description) — description is shown in the menu list.
BOT_COMMANDS = [
    ("start", "Начало и приветствие"),
    ("info", "Как это работает"),
    ("memory", "Факты из памяти (текстом)"),
    ("context", "Что LLM помнит из переписки"),
    ("stats", "Статистика памяти"),
    ("graph", "Граф знаний картинкой"),
    ("graph_html", "Интерактивный граф (файл)"),
    ("language", "Сменить язык ответов"),
    ("forget", "Сбросить память диалога"),
]

# Persistent reply keyboard — buttons shown under the input field.
MENU_KEYBOARD = [
    ["🧠 Память", "💬 Контекст"],
    ["📊 Статистика", "🕸️ Граф"],
    ["🌐 Граф HTML", "🌍 Язык"],
    ["🗑️ Забыть", "ℹ️ О боте"],
]

MENU_HINT = "Готово! Команды всегда под рукой — кнопки ниже 👇"

# Transient status messages shown while a request is processed, then deleted.
WAIT_ANSWER = "⏳ Думаю над ответом…"
WAIT_MEMORY = "🧠 Обновляю память…"


def greeting(has_language: bool, current_lang: str) -> str:
    base = (
        "👋 <b>GigaMemory — демонстрация долговременной памяти для LLM</b>\n\n"
        "Просто общайтесь со мной обычными сообщениями. Я извлекаю из них факты "
        "о вас (имя, работа, увлечения, питомцы и т.д.), строю из них "
        "<b>граф знаний</b> и учитываю его в следующих ответах — даже когда "
        "история диалога давно ушла за пределы окна контекста модели.\n"
        "<b>Модель получает только 5 последних сообщений из контекста, остальную информацию она берёт из системы памяти.</b>\n\n"
        "Команды: /info — как это работает · /graph — граф картинкой · "
        "/memory — факты текстом · /stats — статистика · /forget — сброс памяти · "
        "/language — сменить язык."
    )
    lang_note = (
        "\n\nℹ️ Выбор языка <b>не меняет интерфейс бота</b> (он остаётся на "
        "русском) — он определяет, <b>с каким языком система работает внутри</b>: "
        "на каком языке ассистент формирует ответ."
    )
    if not has_language:
        return base + lang_note + "\n\n<b>Выберите язык:</b>"
    return (
        base
        + lang_note
        + f"\n\nТекущий язык: <b>{LANG_LABELS.get(current_lang, current_lang)}</b>."
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
    "<b>О выборе языка (/language).</b> Он <b>не меняет интерфейс бота</b> — "
    "интерфейс всегда на русском. Выбор определяет, <b>с каким языком система "
    "работает внутри</b>: на этом языке извлекаются и хранятся факты в памяти и "
    "формируется ответ ассистента. Граф памяти одноязычный, поэтому смена языка "
    "при непустой памяти потребует её очистки (бот спросит подтверждение).\n\n"
    "Граф можно посмотреть: /graph (картинка), /graph_html (интерактивный файл), "
    "/memory (текст). Сбросить — /forget."
)


def choose_language_prompt() -> str:
    return (
        "Выберите язык. Он влияет не на интерфейс бота (он остаётся на русском), "
        "а на то, с каким языком система работает внутри: на этом языке "
        "извлекаются и хранятся факты в памяти и формируется ответ ассистента.\n\n"
        "⚠️ Граф памяти одноязычный. Если в памяти уже есть факты, смена языка "
        "потребует её очистки — бот спросит подтверждение."
    )


def language_set(lang: str) -> str:
    return (
        f"✅ Язык: <b>{LANG_LABELS.get(lang, lang)}</b>.\n"
        "Интерфейс бота остаётся на русском — меняется язык, с которым система "
        "работает внутри (язык памяти и ответов)."
    )


def language_unchanged(lang: str) -> str:
    return f"Язык уже <b>{LANG_LABELS.get(lang, lang)}</b> — ничего не изменилось."


def language_change_warning(current: str, new: str) -> str:
    return (
        f"⚠️ Смена языка с <b>{LANG_LABELS.get(current, current)}</b> на "
        f"<b>{LANG_LABELS.get(new, new)}</b>.\n\n"
        "Факты в памяти хранятся на текущем языке, и граф памяти должен быть "
        "одноязычным. Поэтому смена языка <b>очистит вашу память</b> — придётся "
        "рассказать о себе заново.\n\nПродолжить?"
    )


def language_changed_cleared(lang: str) -> str:
    return (
        f"✅ Язык изменён на <b>{LANG_LABELS.get(lang, lang)}</b>, память очищена.\n"
        "Начнём с чистого листа — расскажите о себе."
    )


def language_change_cancelled() -> str:
    return "Отменено. Язык и память не изменились."


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


def format_context(pairs: list[dict[str, str]], limit: int) -> str:
    """Render the recent dialogue turns the final LLM sees directly (raw context)."""
    header = (
        "💬 <b>Контекст диалога</b> — что модель видит напрямую\n\n"
        "Это последние реплики переписки, которые передаются финальной LLM "
        "«как есть». Всё остальное, что модель «помнит», она берёт из системы "
        "памяти (граф фактов) — его показывает /memory.\n"
    )
    if not pairs:
        return (
            header + "\nПока в контексте ничего нет — напишите пару сообщений, и они "
            "появятся здесь."
        )

    out = [header + f"\nВ контексте пар «вопрос-ответ»: {len(pairs)} (лимит: {limit})"]
    for i, p in enumerate(pairs, 1):
        u = html.escape((p.get("user") or "").strip())
        a = html.escape((p.get("assistant") or "").strip())
        out.append(f"<b>{i}.</b> 👤 {u}\n🤖 {a}")

    text = "\n\n".join(out).rstrip()
    if len(text) > TELEGRAM_MAX_LEN:
        text = text[: TELEGRAM_MAX_LEN - 60].rstrip() + "\n\n… (история усечена)"
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
