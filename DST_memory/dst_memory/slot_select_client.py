from __future__ import annotations

import json
import logging
from typing import Any, List, Optional

from .ontology import DEFAULT_USER_SLOTS, SlotOntology, filter_resolve_slots
from .serving import GenerationConfig, LocalHFServing
from .slot_select_messages import build_slot_select_messages

logger = logging.getLogger(__name__)


class SlotSelectClient:
    def __init__(
        self,
        *,
        use_stub: bool,
        serving: Optional[LocalHFServing] = None,
        ontology: SlotOntology = DEFAULT_USER_SLOTS,
        max_slots: int = 5,
        max_retries: int = 1,
    ):
        self.use_stub = use_stub
        self.serving = serving
        self.ontology = ontology
        self.max_slots = max_slots
        self.max_retries = max_retries

    def select_slots(self, user_message: str) -> List[str]:
        if self.use_stub:
            return []
        messages = build_slot_select_messages(
            user_message,
            ontology_slots=self.ontology.slot_names,
            max_slots=self.max_slots,
        )
        tries = self.max_retries + 1
        for _ in range(tries):
            raw = self.serving.generate_chat(
                messages,
                generation_config=GenerationConfig(max_new_tokens=220, do_sample=False),
            )
            parsed = self._parse(raw)
            if parsed is not None:
                return parsed[: self.max_slots]
        return []

    def _parse(self, text: str) -> Optional[List[str]]:
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
        try:
            obj: Any = json.loads(blob)
        except Exception:
            return None
        if not isinstance(obj, dict):
            return None
        arr = obj.get("slot_assignments")
        if not isinstance(arr, list):
            return None
        names = [str(x) for x in arr if isinstance(x, str)]
        return filter_resolve_slots(names, ontology=self.ontology)

