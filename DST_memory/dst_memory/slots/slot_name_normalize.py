"""
Нормализация имён слотов: очистка от мусора, pymorphy2 (леммы),
только слова, известные словарю; опционально pyspellchecker (ru).
"""

from __future__ import annotations

import inspect
import logging
import re
from collections import namedtuple
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger(__name__)


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
        ArgSpec = namedtuple("ArgSpec", ["args", "varargs", "keywords", "defaults"])  # type: ignore[no-redef]

    def getargspec(func):  # noqa: ANN001
        spec = inspect.getfullargspec(func)
        return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

    inspect.getargspec = getargspec  # type: ignore[attr-defined]


_patch_inspect_for_pymorphy2()

_morph = None
_spell_ru = None
_spell_ru_failed = False

# Подчёркивания и дефисы → пробел; дальше остаются только буквы и пробелы.
_DASH_UNDERSCORE = re.compile(r"[_\-–—]+")
# Всё, кроме букв (в т.ч. цифры, пунктуация) → пробел.
_NON_LETTER = re.compile(r"[^a-zA-Zа-яА-ЯёЁ]+", re.UNICODE)

# Максимум слов в имени слота (широкие категории); длинные «предложения» обрезаем.
MAX_SLOT_WORDS = 3
# Склейки без пробелов и мусор обычно длиннее.
MAX_TOKEN_LEN = 32

# Стоп-слова: служебная лексика и типичный мусор из «развёрнутых» ответов модели.
RU_STOPWORDS = frozenset(
    {
        "и",
        "в",
        "во",
        "на",
        "по",
        "под",
        "над",
        "при",
        "про",
        "для",
        "без",
        "из",
        "к",
        "ко",
        "о",
        "об",
        "от",
        "до",
        "за",
        "с",
        "со",
        "у",
        "а",
        "но",
        "или",
        "как",
        "что",
        "чтобы",
        "это",
        "этот",
        "эта",
        "эти",
        "тот",
        "та",
        "те",
        "быть",
        "есть",
        "был",
        "была",
        "были",
        "будет",
        "являться",
        "является",
        "часть",
        "весь",
        "вся",
        "всё",
        "все",
        "всего",
        "который",
        "которая",
        "которое",
        "которые",
        "какой",
        "какая",
        "мой",
        "моя",
        "моё",
        "твой",
        "его",
        "её",
        "их",
        "наш",
        "ваш",
        "такой",
        "также",
        "ещё",
        "еще",
        "уже",
        "не",
        "ни",
        "нет",
        "да",
        "ли",
        "бы",
        "же",
        "то",
        "тем",
        "тут",
        "там",
        "где",
        "когда",
        "если",
        "либо",
        "пусть",
    }
)


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


def _sanitize_slot_string(text: str) -> str:
    """Нижний регистр, _ и дефисы → пробел, убрать цифры и пунктуацию, схлопнуть пробелы."""
    t = str(text).strip().lower()
    t = _DASH_UNDERSCORE.sub(" ", t)
    t = _NON_LETTER.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _lemma_if_known(word: str) -> str:
    """
    Возвращает нормальную форму только если слово «есть» в словаре pymorphy2
    (после опционального исправления орфографии). Иначе пустая строка.
    """
    word = word.strip().lower()
    if not word:
        return ""
    if len(word) > MAX_TOKEN_LEN:
        return ""

    morph = _get_morph()

    def lemma_of(w: str) -> str:
        if not morph.word_is_known(w):
            return ""
        p = morph.parse(w)
        return p[0].normal_form if p else ""

    lem = lemma_of(word)
    if lem:
        return lem

    spell = _get_spell_ru()
    if spell is not None:
        try:
            fixed = spell.correction(word)
        except Exception as e:
            logger.debug("spellchecker.correction failed: %s", e)
            fixed = None
        if fixed and fixed != word:
            lem = lemma_of(fixed)
            if lem:
                return lem

    return ""


def normalize_slot_label(text: str) -> str:
    """
    Имя слота: очистка, только известные слова (леммы), без стоп-слов, максимум MAX_SLOT_WORDS слов.
    Склейки без пробелов («едапоиск») и бессмысленные длинные токены отбрасываются.
    """
    if not text or not str(text).strip():
        return ""

    clean = _sanitize_slot_string(text)
    if not clean:
        return ""

    parts: list[str] = []
    seen_lemmas: set[str] = set()

    for raw_tok in clean.split():
        if len(raw_tok) > MAX_TOKEN_LEN:
            continue
        lem = _lemma_if_known(raw_tok)
        if not lem:
            continue
        if lem in RU_STOPWORDS:
            continue
        if lem in seen_lemmas:
            continue
        seen_lemmas.add(lem)
        parts.append(lem)
        if len(parts) >= MAX_SLOT_WORDS:
            break

    return " ".join(parts).strip()


@lru_cache(maxsize=4096)
def normalize_slot_label_cached(text: str) -> str:
    return normalize_slot_label(text)


def resolve_slot_key_to_existing(
    existing_slot_names: Sequence[str], normalized: str
) -> str:
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
