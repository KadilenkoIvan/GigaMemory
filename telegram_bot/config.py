"""Bot configuration, loaded from environment (and root .env for local dev)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}


def _load_root_dotenv() -> None:
    """Populate os.environ from the repo-root .env (without overriding real env).

    Dependency-free: the bot may run in Docker where vars are injected directly,
    so this is only a convenience for `python -m telegram_bot` during local dev.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class BotConfig:
    token: str
    api_url: str
    default_language: str
    parallel_write: bool
    request_timeout: float
    state_path: str


def load_config() -> BotConfig:
    _load_root_dotenv()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN is not set. Create a bot via @BotFather and put the "
            "token in .env (TELEGRAM_BOT_TOKEN=...) or the environment."
        )

    lang = os.environ.get("BOT_DEFAULT_LANGUAGE", "ru").strip().lower()
    if lang not in ("ru", "en"):
        lang = "ru"

    return BotConfig(
        token=token,
        api_url=os.environ.get("GIGAMEMORY_API_URL", "http://localhost:8000").rstrip(
            "/"
        ),
        default_language=lang,
        parallel_write=os.environ.get("BOT_PARALLEL_WRITE", "true").strip().lower()
        in _TRUE,
        request_timeout=float(os.environ.get("BOT_REQUEST_TIMEOUT", "180")),
        state_path=os.environ.get("BOT_STATE_PATH", "bot_state/users.json"),
    )
