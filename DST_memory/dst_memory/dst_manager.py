from typing import Any, Dict, List
import logging

from .models import DialogueMemoryState, FactRecord, MemoryFact
from .slot_client import SlotDecisionClient

logger = logging.getLogger(__name__)


class DSTManager:
    """
    Slot state manager.
    TODO: replace slot extraction and delete decision with stronger LLM logic.
    """

    def __init__(
        self,
        slot_client: SlotDecisionClient,
        missing_existing_policy: str = "create_new",
    ):
        self._states: Dict[str, DialogueMemoryState] = {}
        self.slot_client = slot_client
        self.missing_existing_policy = missing_existing_policy

    def get_state(self, dialogue_id: str) -> DialogueMemoryState:
        if dialogue_id not in self._states:
            logger.info("Creating new dialogue state dialogue_id=%s", dialogue_id)
            self._states[dialogue_id] = DialogueMemoryState(dialogue_id=dialogue_id)
        return self._states[dialogue_id]

    def upsert_from_message(self, dialogue_id: str, user_text: str) -> List[MemoryFact]:
        state = self.get_state(dialogue_id)
        state.step += 1
        logger.debug(
            "DST upsert start dialogue_id=%s step=%d text_len=%d",
            dialogue_id,
            state.step,
            len(user_text),
        )
        existing_slots = list(state.slots.keys())
        slot_decisions = self.slot_client.decide_slots(
            existing_slots=existing_slots,
            user_message=user_text,
        )
        if not slot_decisions:
            logger.info("No slot decisions dialogue_id=%s step=%d", dialogue_id, state.step)
            return []

        created: List[MemoryFact] = []
        for decision in slot_decisions:
            slot = decision.slot_name
            value = user_text
            if (not decision.create_new) and slot not in state.slots:
                if self.missing_existing_policy == "skip":
                    logger.info(
                        "Skip unknown existing slot dialogue_id=%s slot=%s policy=skip",
                        dialogue_id,
                        slot,
                    )
                    continue
                logger.info(
                    "Unknown existing slot converted to create_new dialogue_id=%s slot=%s",
                    dialogue_id,
                    slot,
                )

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

    def slots_with_messages(self, dialogue_id: str) -> List[Dict[str, Any]]:
        """
        Memory as an ordered list of slots; each slot holds a list of saved
        user messages (full message text per FactRecord, in chronological
        append order for that slot).

        [
          {"slot": "имя слота", "messages": [ {...}, ... ]},
          ...
        ]
        """
        state = self.get_state(dialogue_id)
        result: List[Dict[str, Any]] = []
        for slot, records in state.slots.items():
            messages: List[dict] = []
            for rec in records:
                if not rec.is_active:
                    continue
                messages.append(
                    {
                        "message_text": rec.value,
                        "source_text": rec.source_text,
                        "created_at_step": rec.created_at_step,
                        "updated_at_step": rec.updated_at_step,
                        "is_active": rec.is_active,
                    }
                )
            result.append({"slot": slot, "messages": messages})
        return result

    def clear_dialogue(self, dialogue_id: str) -> None:
        logger.info("Clearing DST state dialogue_id=%s", dialogue_id)
        self._states.pop(dialogue_id, None)
