from typing import Dict, List
import logging

from .models import DialogueMemoryState, FactRecord, MemoryFact

logger = logging.getLogger(__name__)


class DSTManager:
    """
    Slot state manager.
    TODO: replace slot extraction and delete decision with stronger LLM logic.
    """

    def __init__(self):
        self._states: Dict[str, DialogueMemoryState] = {}

    def get_state(self, dialogue_id: str) -> DialogueMemoryState:
        if dialogue_id not in self._states:
            logger.info("Creating new dialogue state dialogue_id=%s", dialogue_id)
            self._states[dialogue_id] = DialogueMemoryState(dialogue_id=dialogue_id)
        return self._states[dialogue_id]

    def extract_slot_value_stub(self, user_text: str) -> List[tuple[str, str]]:
        """
        Very lightweight placeholder extractor.
        TODO: replace with LLM slot extraction with few-shot examples.
        """
        text = user_text.strip()
        if not text:
            return []
        return [("facts", text)]

    def upsert_from_message(self, dialogue_id: str, user_text: str) -> List[MemoryFact]:
        state = self.get_state(dialogue_id)
        state.step += 1
        logger.debug(
            "DST upsert start dialogue_id=%s step=%d text_len=%d",
            dialogue_id,
            state.step,
            len(user_text),
        )
        slot_values = self.extract_slot_value_stub(user_text)

        created: List[MemoryFact] = []
        for slot, value in slot_values:
            if slot not in state.slots:
                state.slots[slot] = []
            rec = FactRecord(
                value=value,
                source_text=user_text,
                created_at_step=state.step,
                updated_at_step=state.step,
                is_active=True,
            )
            state.slots[slot].append(rec)
            logger.info(
                "DST upsert fact dialogue_id=%s slot=%s value_preview=%s",
                dialogue_id,
                slot,
                value[:80],
            )
            created.append(
                MemoryFact(
                    slot=slot,
                    value=value,
                    source_text=user_text,
                    created_at_step=state.step,
                    updated_at_step=state.step,
                    is_active=True,
                )
            )
        return created

    def delete_with_stub_policy(self, dialogue_id: str, user_text: str) -> None:
        """
        TODO: implement LLM-based delete/update policy for contradictory facts.
        Current placeholder intentionally does nothing.
        """
        _ = dialogue_id
        _ = user_text
        logger.debug("DST delete stub invoked dialogue_id=%s", dialogue_id)

    def active_facts(self, dialogue_id: str) -> List[MemoryFact]:
        state = self.get_state(dialogue_id)
        result: List[MemoryFact] = []
        for slot, records in state.slots.items():
            for rec in records:
                if rec.is_active:
                    result.append(
                        MemoryFact(
                            slot=slot,
                            value=rec.value,
                            source_text=rec.source_text,
                            created_at_step=rec.created_at_step,
                            updated_at_step=rec.updated_at_step,
                            is_active=rec.is_active,
                        )
                    )
        return result

    def clear_dialogue(self, dialogue_id: str) -> None:
        logger.info("Clearing DST state dialogue_id=%s", dialogue_id)
        self._states.pop(dialogue_id, None)
