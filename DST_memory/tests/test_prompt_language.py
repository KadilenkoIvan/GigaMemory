"""Проверка, что `prompt_language` в конфиге переключает пакет промптов (ru / en)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Пакет dst_memory при прогоне из подпапки DST_memory
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dst_memory.prompts.loader import load_prompt_modules, normalize_prompt_language


class TestPromptLanguage(unittest.TestCase):
    def test_normalize_prompt_language_variants(self):
        self.assertEqual(normalize_prompt_language(None), "ru")
        self.assertEqual(normalize_prompt_language(""), "ru")
        self.assertEqual(normalize_prompt_language("RU"), "ru")
        self.assertEqual(normalize_prompt_language("en"), "en")
        self.assertEqual(normalize_prompt_language("EN"), "en")
        self.assertEqual(normalize_prompt_language("english"), "en")

    def test_triplet_system_language_distinct(self):
        """
        Системное сообщение для триплетов однозначно русское или английское.
        """
        mods_en = load_prompt_modules("en")
        mods_ru = load_prompt_modules("ru")
        self.assertEqual(mods_en.lang, "en")
        self.assertEqual(mods_ru.lang, "ru")

        msgs_en = mods_en.triplet_messages.build_triplet_messages("test message")
        msgs_ru = mods_ru.triplet_messages.build_triplet_messages("test")
        sys_en = msgs_en[0]["content"]
        sys_ru = msgs_ru[0]["content"]

        self.assertIn("You extract facts from the user's message.", sys_en)
        self.assertNotIn("Ты система извлечения фактов", sys_en)
        self.assertIn("Ты система извлечения фактов из реплики пользователя.", sys_ru)
        self.assertNotIn("You extract facts from the user's message.", sys_ru)

    def test_slot_select_system_language_distinct(self):
        mods_en = load_prompt_modules("en")
        mods_ru = load_prompt_modules("ru")
        sel_en = mods_en.slot_select_messages.build_slot_select_messages("hello")
        sel_ru = mods_ru.slot_select_messages.build_slot_select_messages("привет")
        sys_en = sel_en[0]["content"]
        sys_ru = sel_ru[0]["content"]

        self.assertIn("You are a classifier for long-term user-memory slots.", sys_en)
        self.assertIn("Ты классификатор слотов долговременной памяти пользователя.", sys_ru)
        self.assertNotIn("Ты классификатор", sys_en)
        self.assertNotIn("You are a classifier", sys_ru)

    def test_fewshot_modules_import(self):
        from dst_memory.prompts.en import prompt_fewshots as en_fs
        from dst_memory.prompts.ru import prompt_fewshots as ru_fs

        self.assertGreater(len(en_fs.SLOT_SELECT_FEWSHOT), 0)
        self.assertGreater(len(ru_fs.SLOT_SELECT_FEWSHOT), 0)


if __name__ == "__main__":
    unittest.main()
