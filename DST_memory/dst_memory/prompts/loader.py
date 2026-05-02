"""Resolve prompt language → `dst_memory.prompts.<ru|en>` subpackages."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

PromptLang = Literal["ru", "en"]


def normalize_prompt_language(lang: str | None) -> PromptLang:
    if not lang:
        return "ru"
    l = str(lang).strip().lower()
    if l in ("en", "english"):
        return "en"
    return "ru"


@dataclass(frozen=True)
class PromptModules:
    """Handles to per-task prompt builder modules for one UI language."""

    lang: PromptLang
    triplet_messages: ModuleType
    conflict_messages: ModuleType
    deletion_messages: ModuleType
    memory_gate_messages: ModuleType
    slot_select_messages: ModuleType
    slot_update_messages: ModuleType


def load_prompt_modules(lang: str | None) -> PromptModules:
    code = normalize_prompt_language(lang)
    root = f"dst_memory.prompts.{code}"
    return PromptModules(
        lang=code,
        triplet_messages=importlib.import_module(f"{root}.triplet_messages"),
        conflict_messages=importlib.import_module(f"{root}.conflict_messages"),
        deletion_messages=importlib.import_module(f"{root}.deletion_messages"),
        memory_gate_messages=importlib.import_module(f"{root}.memory_gate_messages"),
        slot_select_messages=importlib.import_module(f"{root}.slot_select_messages"),
        slot_update_messages=importlib.import_module(f"{root}.slot_update_messages"),
    )
