from dataclasses import asdict
from typing import Dict, List
import logging

from .classifier import ImportanceClassifier
from .config import PipelineConfig
from .dst_manager import DSTManager
from .embedder import TextEmbedder
from .llm_client import FinalLLMClient
from .models import MemoryFact, Message
from .retriever import MemoryRetriever
from .slot_client import SlotDecisionClient
from .vector_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class DSTMemoryPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(
            "Initializing pipeline threshold=%.3f top_k=%d llm_mode=%s gate=%s",
            config.importance_threshold,
            config.retrieval_top_k,
            config.llm_mode,
            config.use_memory_gate,
        )
        self.classifier = ImportanceClassifier(
            model_path=config.importance_model_path,
            threshold=config.importance_threshold,
        )
        slot_client = SlotDecisionClient(
            use_stub=config.slot_use_stub,
            model_path=config.slot_model_path,
            max_slots=config.slot_max_slots_per_message,
            max_retries=1,
        )
        self.dst = DSTManager(slot_client=slot_client)
        self.embedder = TextEmbedder()
        self.store = InMemoryVectorStore()
        self.retriever = MemoryRetriever(store=self.store, embedder=self.embedder)
        self.final_llm = FinalLLMClient(
            mode=config.llm_mode,
            api_url=config.llm_api_url,
            api_key=config.llm_api_key,
            temperature=config.llm_temperature,
        )

    def write_to_memory(self, dialogue_id: str, message: Message) -> Dict:
        logger.info(
            "write_to_memory dialogue_id=%s role=%s content_len=%d",
            dialogue_id,
            message.role,
            len(message.content),
        )
        if message.role != "user":
            return {"saved": False, "reason": "only_user_messages_supported"}

        cls = self.classifier.predict(message.content)
        logger.info(
            "classifier result dialogue_id=%s p_important=%.4f is_important=%s",
            dialogue_id,
            cls["p_important"],
            cls["is_important"],
        )
        if not bool(cls["is_important"]):
            return {"saved": False, "reason": "not_important", "classifier": cls}

        self.dst.delete_with_stub_policy(dialogue_id, message.content)
        new_facts = self.dst.upsert_from_message(dialogue_id, message.content)
        self._index_facts(dialogue_id, new_facts)
        return {
            "saved": True,
            "reason": "important",
            "classifier": cls,
            "new_facts": [asdict(f) for f in new_facts],
        }

    def _index_facts(self, dialogue_id: str, facts: List[MemoryFact]) -> None:
        if not facts:
            logger.debug("index_facts skipped empty facts")
            return
        # Index by slot name only. Message texts are stored in metadata.
        texts = [f.slot for f in facts]
        vectors = self.embedder.encode(texts)
        logger.info("Indexing facts dialogue_id=%s count=%d", dialogue_id, len(facts))
        for fact, vec in zip(facts, vectors):
            payload = {
                "dialogue_id": dialogue_id,
                "slot": fact.slot,
                "slot_name": fact.slot,
                "value": fact.value,
                "message_text": fact.value,
                "source_text": fact.source_text,
                "created_at_step": fact.created_at_step,
                "updated_at_step": fact.updated_at_step,
                "is_active": fact.is_active,
            }
            self.store.add(vec, payload)

    def should_use_memory(self, question: str) -> bool:
        if not self.config.use_memory_gate:
            return True
        markers = [
            "я",
            "мой",
            "моя",
            "моё",
            "меня",
            "у меня",
            "помнишь",
        ]
        lower = question.lower()
        return any(m in lower for m in markers)

    def answer_without_final_llm(self, dialogue_id: str, question: str) -> Dict:
        logger.info("answer_without_final_llm dialogue_id=%s", dialogue_id)
        hits = self.retriever.search(
            dialogue_id=dialogue_id, query=question, top_k=self.config.retrieval_top_k
        )
        memory_slots = self.dst.slots_with_messages(dialogue_id)
        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "use_memory": self.should_use_memory(question),
            "retrieved": hits,
            "memory_slots": memory_slots,
        }

    def answer(self, dialogue_id: str, question: str) -> str:
        logger.info("answer dialogue_id=%s", dialogue_id)
        if not self.should_use_memory(question):
            logger.info("memory gate denied dialogue_id=%s", dialogue_id)
            return self.final_llm.generate(question=question, memory_lines=[])
        hits = self.retriever.search(
            dialogue_id=dialogue_id, query=question, top_k=self.config.retrieval_top_k
        )
        memory_lines = [f"{h.get('slot')}: {h.get('value')}" for h in hits]
        return self.final_llm.generate(question=question, memory_lines=memory_lines)

    def clear_memory(self, dialogue_id: str) -> None:
        logger.info("clear_memory dialogue_id=%s", dialogue_id)
        self.dst.clear_dialogue(dialogue_id)
        self.store.clear_dialogue(dialogue_id)
