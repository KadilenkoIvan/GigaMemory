"""
Нормализация имён слотов: lower case, pymorphy2 (леммы), опционально pyspellchecker (ru).
"""

from __future__ import annotations

import inspect
import logging
import re
from collections import namedtuple
from functools import lru_cache
from typing import Sequence


def _patch_inspect_for_pymorphy2() -> None:
    """
    pymorphy2 uses inspect.getargspec, removed in Python 3.11+.
    Must run before `from pymorphy2 import MorphAnalyzer`.
    """
    if hasattr(inspect, "getargspec"):
        return
    try:
        from inspect import ArgSpec  # type: ignore[attr-defined]
    except ImportError:
        ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])

    def getargspec(func):  # noqa: ANN001
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec  # type: ignore[attr-defined]


_patch_inspect_for_pymorphy2()

logger = logging.getLogger(__name__)

_morph = None
_spell_ru = None
_spell_ru_failed = False


def _get_morph():
    global _morph
    if _morph is None:
        from pymorphy2 import MorphAnalyzer

        _morph = MorphAnalyzer()
    return _morph


def _get_spell_ru():
    global _spell_ru, _spell_ru_failed
    if _spell_ru_failed:
        return None
    if _spell_ru is None:
        try:
            from spellchecker import SpellChecker

            _spell_ru = SpellChecker(language="ru")
        except Exception as e:
            logger.info("Russian SpellChecker unavailable: %s", e)
            _spell_ru_failed = True
            return None
    return _spell_ru


def _normalize_token_ru(token: str) -> str:
    """
    Одно слово: lower, опечатки (если есть ru spellchecker), лемма pymorphy2.
    Если слово «неизвестно» морфологии — пробуем исправление, снова pymorphy2;
    если всё ещё не то — возвращаем последний вариант как есть.
    """
    word = token.strip().lower()
    if not word:
        return ""
    morph = _get_morph()
    if morph.word_is_known(word):
        p = morph.parse(word)
        return p[0].normal_form if p else word

    spell = _get_spell_ru()
    if spell is not None:
        try:
            fixed = spell.correction(word)
        except Exception as e:
            logger.debug("spellchecker.correction failed: %s", e)
            fixed = None
        if fixed and fixed != word:
            if morph.word_is_known(fixed):
                p = morph.parse(fixed)
                return p[0].normal_form if p else fixed
            p2 = morph.parse(fixed)
            if p2:
                return p2[0].normal_form

    p = morph.parse(word)
    if p:
        return p[0].normal_form
    return word


_TOKEN_RE = re.compile(r"[\w\-]+", re.UNICODE)


def normalize_slot_label(text: str) -> str:
    """
    Полное имя слота: lower, разбиение на токены (буквы/цифры/дефис),
    нормализация каждого токена, склейка через пробел.
    """
    if not text or not str(text).strip():
        return ""
    raw = str(text).strip().lower()
    parts: list[str] = []
    for m in _TOKEN_RE.finditer(raw):
        t = _normalize_token_ru(m.group(0))
        if t:
            parts.append(t)
    return " ".join(parts).strip()


@lru_cache(maxsize=4096)
def normalize_slot_label_cached(text: str) -> str:
    return normalize_slot_label(text)


def resolve_slot_key_to_existing(existing_slot_names: Sequence[str], normalized: str) -> str:
    """
    Если нормализованное имя совпадает с нормализацией одного из уже
    существующих ключей — возвращаем тот ключ как в состоянии (без дублей).
    """
    if not normalized:
        return normalized
    if not existing_slot_names:
        return normalized
    for ex in existing_slot_names:
        if normalize_slot_label_cached(ex) == normalized:
            return ex
    return normalized
