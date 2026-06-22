from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..clients.lm_json_schemas import (
    TRIPLET_JSON_SCHEMA,
    TRIPLET_JSON_SCHEMA_WITH_DELETE,
)
from ..clients.serving import GenerationConfig, LocalHFServing
from ..core.models import VALID_TTL_VALUES
from ..prompts.loader import load_prompt_modules
from ..slots.ontology import DEFAULT_USER_SLOTS, SlotOntology

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedTriplet:
    slot: str
    subject: str
    relation: str
    object: str
    ttl: str = "inf"

    def as_line(self) -> str:
        return f"{self.subject} | {self.relation} | {self.object}"


@dataclass(frozen=True)
class DeletionSignal:
    """Сигнал явного удаления факта, выданный моделью или детектором."""

    subject: str
    relation: str
    object: str

    def as_line(self) -> str:
        return f"{self.subject} | {self.relation} | {self.object}"


class TripletExtractionClient:
    def __init__(
        self,
        *,
        use_stub: bool,
        serving: LocalHFServing | None = None,
        ontology: SlotOntology = DEFAULT_USER_SLOTS,
        max_triplets: int = 12,
        max_retries: int = 1,
        ttl_mode: str = "mode2",
        prompt_language: str = "ru",
        parse_retry_temperature: float = 0.65,
        parse_retry_temperature_increment: float = 0.08,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.ontology = ontology
        self.max_triplets = max_triplets
        self.max_retries = max_retries
        self.ttl_mode = ttl_mode
        self.prompt_language = prompt_language
        self.parse_retry_temperature = float(parse_retry_temperature)
        self.parse_retry_temperature_increment = float(
            parse_retry_temperature_increment
        )
        self._prompt_modules = None

        if self.use_stub:
            logger.info("Triplet extractor in STUB mode")
        elif self.serving is None:
            raise ValueError(
                "TripletExtractionClient requires serving when use_stub is False"
            )
        else:
            logger.info(
                "Triplet extractor using LocalHFServing device=%s ttl_mode=%s",
                self.serving.device,
                ttl_mode,
            )

    def extract(self, user_message: str) -> list[ExtractedTriplet]:
        triplets, _ = self._extract_impl(
            user_message, slot_name=None, existing_triplets=None
        )
        return triplets

    def extract_for_slot(
        self, user_message: str, slot_name: str
    ) -> list[ExtractedTriplet]:
        slot = self.ontology.resolve(slot_name)
        if not slot:
            return []
        triplets, _ = self._extract_impl(
            user_message, slot_name=slot, existing_triplets=None
        )
        return triplets

    def extract_with_context(
        self,
        user_message: str,
        existing_triplets: list[str],
    ) -> tuple[list[ExtractedTriplet], list[DeletionSignal]]:
        """
        Извлечение триплетов с передачей контекста текущих фактов (без указания слота).
        Возвращает (новые триплеты, сигналы удаления).
        """
        return self._extract_impl(
            user_message, slot_name=None, existing_triplets=existing_triplets
        )

    def extract_for_slot_with_context(
        self,
        user_message: str,
        slot_name: str,
        existing_triplets: list[str],
        enable_deletion: bool = False,
    ) -> tuple[list[ExtractedTriplet], list[DeletionSignal]]:
        """
        Извлечение триплетов для конкретного слота с передачей контекста.
        enable_deletion=True — включить инструкции про "delete" в промпт
        (только для triplet_deletion_mode="llm_inline").
        Возвращает (новые триплеты, сигналы удаления).
        """
        slot = self.ontology.resolve(slot_name)
        if not slot:
            return [], []
        return self._extract_impl(
            user_message,
            slot_name=slot,
            existing_triplets=existing_triplets,
            enable_deletion=enable_deletion,
        )

    def _extract_impl(
        self,
        user_message: str,
        slot_name: str | None,
        existing_triplets: list[str] | None,
        enable_deletion: bool | None = None,
    ) -> tuple[list[ExtractedTriplet], list[DeletionSignal]]:
        if self.use_stub:
            return [], []

        # enable_deletion=None → автоматически True только при llm_inline режиме.
        # Вызывающий код должен явно передать True для llm_inline, иначе False.
        # Контекст (existing_triplets) и инструкции удаления — независимы.
        if enable_deletion is None:
            enable_deletion = False
        if self._prompt_modules is None:
            self._prompt_modules = load_prompt_modules(self.prompt_language)
        messages = self._prompt_modules.triplet_messages.build_triplet_messages(
            user_message,
            slot_name=slot_name,
            include_slot=slot_name is None,
            ontology_slots=self.ontology.slot_names,
            max_triplets=self.max_triplets,
            ttl_mode=self.ttl_mode,
            existing_triplets=existing_triplets,
            enable_deletion=enable_deletion,
        )

        tries = self.max_retries + 1
        last = ""
        json_schema = None
        if getattr(self.serving, "use_lm_format_enforcer", False):
            json_schema = (
                TRIPLET_JSON_SCHEMA_WITH_DELETE
                if enable_deletion
                else TRIPLET_JSON_SCHEMA
            )
        for attempt in range(1, tries + 1):
            if attempt == 1:
                gen_cfg = GenerationConfig(
                    max_new_tokens=512,
                    do_sample=False,
                    lm_enforcer_json_schema=json_schema,
                )
            else:
                t = min(
                    1.0,
                    self.parse_retry_temperature
                    + self.parse_retry_temperature_increment * float(attempt - 2),
                )
                gen_cfg = GenerationConfig(
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=t,
                    lm_enforcer_json_schema=json_schema,
                )
            last = self.serving.generate_chat(messages, generation_config=gen_cfg)
            slot_label = slot_name if slot_name is not None else "SINGLE_PASS"
            logger.info(
                "Triplet extractor slot=[%s] attempt=%d raw: %s",
                slot_label,
                attempt,
                last[:800],
            )
            parsed = self._parse(
                last, forced_slot=slot_name, with_deletions=enable_deletion
            )
            if parsed is not None:
                return parsed

        logger.warning("Triplet extractor failed to parse JSON, returning empty list")
        return [], []

    def _parse(
        self,
        text: str,
        forced_slot: str | None = None,
        with_deletions: bool = False,
    ) -> tuple[list[ExtractedTriplet], list[DeletionSignal]] | None:
        blob = (text or "").strip()
        if not blob:
            return None
        if blob.startswith("```"):
            lines = blob.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            blob = "\n".join(lines).strip()

        obj: Any
        try:
            obj = json.loads(blob)
        except Exception:
            extracted = self._extract_first_json_object(blob)
            if not extracted:
                return None
            try:
                obj = json.loads(extracted)
            except Exception:
                return None

        if not isinstance(obj, dict) or "triplets" not in obj:
            return None
        items = obj.get("triplets")
        if not isinstance(items, list):
            return None

        out: list[ExtractedTriplet] = []
        for it in items[: self.max_triplets]:
            if not isinstance(it, dict):
                continue
            slot = forced_slot or self.ontology.resolve(str(it.get("slot", "")))
            subj = self._normalize_field(str(it.get("subject", "")))
            rel = self._normalize_field(str(it.get("relation", "")))
            objv = self._normalize_field(str(it.get("object", "")))
            if not slot or not subj or not rel or not objv:
                continue
            ttl = str(it.get("ttl", "inf")).strip().lower()
            if ttl not in VALID_TTL_VALUES:
                ttl = "inf"
            out.append(
                ExtractedTriplet(
                    slot=slot, subject=subj, relation=rel, object=objv, ttl=ttl
                )
            )

        # --- Parse deletion signals (only when with_deletions=True) ---
        deletions: list[DeletionSignal] = []
        if with_deletions:
            delete_items = obj.get("delete", [])
            if isinstance(delete_items, list):
                for d in delete_items:
                    if not isinstance(d, dict):
                        continue
                    ds = self._normalize_field(str(d.get("subject", "")))
                    dr = self._normalize_field(str(d.get("relation", "")))
                    do = self._normalize_field(str(d.get("object", "")))
                    if ds and dr and do:
                        deletions.append(
                            DeletionSignal(subject=ds, relation=dr, object=do)
                        )

        return out, deletions

    @staticmethod
    def _normalize_field(s: str) -> str:
        """
        Normalize a triplet field produced by the LLM.
        Lowercase and single spaces; lemmas match the active prompt pack (Russian or English).
        """
        s = s.strip().lower()
        s = re.sub(r"\s+", " ", s)
        return s

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None
