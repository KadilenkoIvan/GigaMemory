"""GigaMemory Telegram bot — thin client over the REST API.

Each Telegram user gets their own memory graph (dialogue_id = user id). The bot
UI is Russian; the per-user ru/en choice only switches the assistant's answer
language (forwarded to the API as prompt_language).
"""

from __future__ import annotations

import logging
from io import BytesIO

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import texts
from .api_client import GigaMemoryAPIError, GigaMemoryClient
from .config import BotConfig, load_config
from .store import UserStore

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger("gigamemory.bot")

# bot_data keys
_CFG = "config"
_CLIENT = "client"
_STORE = "store"


def _cfg(context: ContextTypes.DEFAULT_TYPE) -> BotConfig:
    return context.application.bot_data[_CFG]


def _client(context: ContextTypes.DEFAULT_TYPE) -> GigaMemoryClient:
    return context.application.bot_data[_CLIENT]


def _store(context: ContextTypes.DEFAULT_TYPE) -> UserStore:
    return context.application.bot_data[_STORE]


def _dialogue_id(update: Update) -> str:
    user = update.effective_user
    return str(user.id) if user else "anonymous"


def _language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(texts.LANG_LABELS["ru"], callback_data="lang:ru"),
                InlineKeyboardButton(texts.LANG_LABELS["en"], callback_data="lang:en"),
            ]
        ]
    )


def _main_keyboard() -> ReplyKeyboardMarkup:
    """Persistent reply keyboard with the main commands as tappable buttons."""
    return ReplyKeyboardMarkup(
        texts.MENU_KEYBOARD, resize_keyboard=True, is_persistent=True
    )


# ── Command handlers ───────────────────────────────────────────────────────
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    store = _store(context)
    uid = _dialogue_id(update)
    has_lang = store.has_language(uid)
    text = texts.greeting(has_lang, store.get_language(uid))
    if not has_lang:
        # First show the language choice (inline); the persistent menu keyboard
        # is sent right after the user picks a language (on_language_choice).
        await update.message.reply_text(
            text, parse_mode=ParseMode.HTML, reply_markup=_language_keyboard()
        )
    else:
        await update.message.reply_text(
            text, parse_mode=ParseMode.HTML, reply_markup=_main_keyboard()
        )


async def cmd_info(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(texts.INFO, parse_mode=ParseMode.HTML)


async def cmd_language(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        texts.choose_language_prompt(), reply_markup=_language_keyboard()
    )


async def on_language_choice(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    query = update.callback_query
    await query.answer()
    _, _, lang = (query.data or "").partition(":")
    if lang not in ("ru", "en"):
        return
    _store(context).set_language(query.from_user.id, lang)
    await query.edit_message_text(texts.language_set(lang), parse_mode=ParseMode.HTML)
    # An inline callback can't carry a reply keyboard, so send the persistent
    # menu keyboard as a follow-up message.
    await query.message.reply_text(texts.MENU_HINT, reply_markup=_main_keyboard())


async def cmd_graph(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = _dialogue_id(update)
    await context.bot.send_chat_action(
        update.effective_chat.id, ChatAction.UPLOAD_PHOTO
    )
    try:
        png = await _client(context).graph_image(uid)
    except GigaMemoryAPIError:
        await update.message.reply_text(texts.api_unavailable())
        return
    bio = BytesIO(png)
    bio.name = "graph.png"
    await update.message.reply_photo(photo=bio, caption="🕸️ Граф знаний")


async def cmd_graph_html(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = _dialogue_id(update)
    await context.bot.send_chat_action(
        update.effective_chat.id, ChatAction.UPLOAD_DOCUMENT
    )
    try:
        html = await _client(context).graph_html(uid)
    except GigaMemoryAPIError:
        await update.message.reply_text(texts.api_unavailable())
        return
    bio = BytesIO(html)
    bio.name = "graph.html"
    await update.message.reply_document(
        document=bio,
        filename="graph.html",
        caption="🌐 Интерактивный граф — откройте файл в браузере.",
    )


async def cmd_memory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = _dialogue_id(update)
    try:
        data = await _client(context).graph_short(uid)
    except GigaMemoryAPIError:
        await update.message.reply_text(texts.api_unavailable())
        return
    await update.message.reply_text(
        texts.format_memory(data.get("slots", {})), parse_mode=ParseMode.HTML
    )


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = _dialogue_id(update)
    try:
        data = await _client(context).graph_short(uid)
    except GigaMemoryAPIError:
        await update.message.reply_text(texts.api_unavailable())
        return
    await update.message.reply_text(
        texts.format_stats(data.get("slots", {})), parse_mode=ParseMode.HTML
    )


async def cmd_forget(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = _dialogue_id(update)
    try:
        await _client(context).forget(uid)
    except GigaMemoryAPIError:
        await update.message.reply_text(texts.api_unavailable())
        return
    await update.message.reply_text(texts.forget_done())


# Reply-keyboard button label → the command handler it should trigger.
_BUTTON_HANDLERS = {
    "🧠 Память": cmd_memory,
    "📊 Статистика": cmd_stats,
    "🕸️ Граф": cmd_graph,
    "🌐 Граф HTML": cmd_graph_html,
    "🌍 Язык": cmd_language,
    "🗑️ Забыть": cmd_forget,
    "ℹ️ О боте": cmd_info,
}


# ── Plain text → conversation ──────────────────────────────────────────────
async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return

    # A tap on a menu button arrives as plain text — route it to its handler
    # instead of treating it as a message for the assistant.
    button_handler = _BUTTON_HANDLERS.get(update.message.text.strip())
    if button_handler is not None:
        await button_handler(update, context)
        return

    uid = _dialogue_id(update)
    cfg = _cfg(context)
    lang = _store(context).get_language(uid)

    await context.bot.send_chat_action(update.effective_chat.id, ChatAction.TYPING)
    try:
        answer = await _client(context).send_message(
            uid,
            update.message.text,
            parallel_write=cfg.parallel_write,
            prompt_language=lang,
        )
    except GigaMemoryAPIError:
        await update.message.reply_text(texts.api_unavailable())
        return

    await update.message.reply_text(answer or "…")


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled error", exc_info=context.error)


async def _post_init(app: Application) -> None:
    """Register the native command menu (the "☰ Menu" button)."""
    await app.bot.set_my_commands(
        [BotCommand(name, desc) for name, desc in texts.BOT_COMMANDS]
    )


def build_application(cfg: BotConfig) -> Application:
    app = Application.builder().token(cfg.token).post_init(_post_init).build()
    app.bot_data[_CFG] = cfg
    app.bot_data[_CLIENT] = GigaMemoryClient(cfg.api_url, timeout=cfg.request_timeout)
    app.bot_data[_STORE] = UserStore(cfg.state_path, cfg.default_language)

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("info", cmd_info))
    app.add_handler(CommandHandler("language", cmd_language))
    app.add_handler(CommandHandler("graph", cmd_graph))
    app.add_handler(CommandHandler("graph_html", cmd_graph_html))
    app.add_handler(CommandHandler("memory", cmd_memory))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("forget", cmd_forget))
    app.add_handler(CallbackQueryHandler(on_language_choice, pattern=r"^lang:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_message))
    app.add_error_handler(on_error)
    return app


def main() -> None:
    cfg = load_config()
    logger.info("Starting GigaMemory bot — api_url=%s", cfg.api_url)
    app = build_application(cfg)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
