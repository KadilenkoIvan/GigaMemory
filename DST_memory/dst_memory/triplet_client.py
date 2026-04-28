from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .models import VALID_TTL_VALUES
from .ontology import DEFAULT_USER_SLOTS, SlotOntology
from .serving import GenerationConfig, LocalHFServing
from .triplet_messages import build_triplet_messages

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


class TripletExtractionClient:
    def __init__(
        self,
        *,
        use_stub: bool,
        serving: Optional[LocalHFServing] = None,
        ontology: SlotOntology = DEFAULT_USER_SLOTS,
        max_triplets: int = 12,
        max_retries: int = 1,
        ttl_mode: str = "mode2",
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.ontology = ontology
        self.max_triplets = max_triplets
        self.max_retries = max_retries
        self.ttl_mode = ttl_mode

        if self.use_stub:
            logger.info("Triplet extractor in STUB mode")
        elif self.serving is None:
            raise ValueError("TripletExtractionClient requires serving when use_stub is False")
        else:
            logger.info("Triplet extractor using LocalHFServing device=%s ttl_mode=%s", self.serving.device, ttl_mode)

    def extract(self, user_message: str) -> List[ExtractedTriplet]:
        return self._extract_impl(user_message, slot_name=None)

    def extract_for_slot(self, user_message: str, slot_name: str) -> List[ExtractedTriplet]:
        slot = self.ontology.resolve(slot_name)
        if not slot:
            return []
        return self._extract_impl(user_message, slot_name=slot)

    def _extract_impl(self, user_message: str, slot_name: Optional[str]) -> List[ExtractedTriplet]:
        if self.use_stub:
            return []

        messages = build_triplet_messages(
            user_message,
            slot_name=slot_name,
            include_slot=slot_name is None,
            ontology_slots=self.ontology.slot_names,
            max_triplets=self.max_triplets,
            ttl_mode=self.ttl_mode,
        )

        tries = self.max_retries + 1
        last = ""
        for attempt in range(1, tries + 1):
            last = self.serving.generate_chat(
                messages,
                generation_config=GenerationConfig(max_new_tokens=512, do_sample=False),
            )
            slot_label = slot_name if slot_name is not None else "SINGLE_PASS"
            logger.info("Triplet extractor slot=[%s] attempt=%d raw: %s", slot_label, attempt, last[:800])
            parsed = self._parse(last, forced_slot=slot_name)
            if parsed is not None:
                return parsed

        logger.warning("Triplet extractor failed to parse JSON, returning empty list")
        return []

    def _parse(self, text: str, forced_slot: Optional[str] = None) -> Optional[List[ExtractedTriplet]]:
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

        out: List[ExtractedTriplet] = []
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
            out.append(ExtractedTriplet(slot=slot, subject=subj, relation=rel, object=objv, ttl=ttl))

        return out

    @staticmethod
    def _normalize_field(s: str) -> str:
        """
        Normalize a triplet field produced by the LLM.
        Model outputs lowercase Russian words with spaces; we keep that format
        (lowercase, single spaces) for natural language compatibility with Meno-Lite.
        """
        s = s.strip().lower()
        s = re.sub(r'\s+', ' ', s)
        return s

    @staticmethod
    def _extract_first_json_object(text: str) -> Optional[str]:
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
