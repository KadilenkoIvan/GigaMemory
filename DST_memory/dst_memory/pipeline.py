from dataclasses import asdict
from typing import Dict, List, Tuple
import logging

from .classifier import ImportanceClassifier
from .config import PipelineConfig
from .conflict_client import TripletConflictClient
from .dst_manager import DSTManager
from .embedder import TextEmbedder
from .llm_client import FinalLLMClient
from .memory_gate_client import MemoryGateClient
from .models import MemoryFact, Message
from .retriever import MemoryRetriever
from .serving import LocalHFServing
from .slot_select_client import SlotSelectClient
from .triplet_client import TripletExtractionClient
from .vector_store import InMemoryVectorStore

logger = logging.getLogger(__name__)


class DSTMemoryPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        logger.info(
            "Initializing pipeline threshold=%.3f top_k=%d llm_mode=%s gate=%s "
            "memory_gate_stub=%s memory_context=%s",
            config.importance_threshold,
            config.retrieval_top_k,
            config.llm_mode,
            config.use_memory_gate,
            config.memory_gate_use_stub,
            config.memory_context_source,
        )
        self.classifier = ImportanceClassifier(
            model_path=config.importance_model_path,
            threshold=config.importance_threshold,
        )
        slot_serving = None
        if not config.slot_use_stub:
            slot_serving = LocalHFServing(config.slot_model_path)
        triplet_extractor = TripletExtractionClient(
            use_stub=config.slot_use_stub,
            serving=slot_serving,
            max_triplets=max(6, config.slot_max_slots_per_message * 3),
            max_retries=1,
        )
        slot_selector = SlotSelectClient(
            use_stub=config.slot_use_stub,
            serving=slot_serving,
            max_slots=config.slot_max_slots_per_message,
            max_retries=1,
        )
        conflict_resolver = TripletConflictClient(
            use_stub=config.slot_use_stub,
            serving=slot_serving,
            max_retries=1,
        )
        self.dst = DSTManager(
            triplet_extractor=triplet_extractor,
            slot_selector=slot_selector,
            conflict_resolver=conflict_resolver,
            single_pass_fallback=True,
        )
        gate_stub = config.memory_gate_use_stub or slot_serving is None
        self.memory_gate = MemoryGateClient(
            use_stub=gate_stub,
            serving=slot_serving,
            max_retries=1,
        )
        self.embedder = TextEmbedder()
        self.store = InMemoryVectorStore()
        self.retriever = MemoryRetriever(store=self.store, embedder=self.embedder)
        self.final_llm = FinalLLMClient(
            mode=config.llm_mode,
            api_url=config.llm_api_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            http_referer=config.openrouter_http_referer,
            x_title=config.openrouter_x_title,
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

        new_facts = self.dst.upsert_from_message(dialogue_id, message.content)
        if not new_facts:
            return {
                "saved": False,
                "reason": "no_facts_extracted",
                "classifier": cls,
                "new_facts": [],
            }
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
        # Index by the fact text itself, not only by slot name.
        texts = [f"{f.slot}: {f.value}" for f in facts]
        vectors = self.embedder.encode(texts)
        logger.info("Indexing facts dialogue_id=%s count=%d", dialogue_id, len(facts))
        for fact, vec in zip(facts, vectors):
            payload = {
                "dialogue_id": dialogue_id,
                "slot": fact.slot,
                "slot_name": fact.slot,
                "value": fact.value,
                "subject": fact.subject,
                "relation": fact.relation,
                "object": fact.object,
                "message_text": fact.value,
                "source_text": fact.source_text,
                "created_at_step": fact.created_at_step,
                "updated_at_step": fact.updated_at_step,
                "is_active": fact.is_active,
            }
            self.store.add(vec, payload)

    @staticmethod
    def _lines_from_retriever_hits(hits: List[Dict]) -> List[str]:
        return [f"{h.get('slot')}: {h.get('value')}" for h in hits]

    def _memory_context_for_question(
        self, dialogue_id: str, question: str
    ) -> Tuple[List[str], Dict]:
        """
        Строки памяти для финальной LLM и метаданные шлюза (отладка / jsonl без финальной LLM).
        """
        src = (self.config.memory_context_source or "slots").strip().lower()
        if src not in ("slots", "vector"):
            src = "slots"

        slot_names = self.dst.active_slot_names(dialogue_id)
        meta_base: Dict = {
            "memory_context_source": src,
            "retrieval_top_k": self.config.retrieval_top_k,
        }

        if not slot_names:
            return [], {
                **meta_base,
                "use_memory": False,
                "selected_slots": [],
                "reason": "no_active_slots",
            }

        if src == "vector":
            return self._memory_context_vector(dialogue_id, question, slot_names, meta_base)

        return self._memory_context_slots(dialogue_id, question, slot_names, meta_base)

    def _memory_context_slots(
        self,
        dialogue_id: str,
        question: str,
        slot_names: List[str],
        meta_base: Dict,
    ) -> Tuple[List[str], Dict]:
        if not self.config.use_memory_gate:
            lines = self.dst.memory_lines_for_slots(dialogue_id, slot_names)
            return lines, {
                **meta_base,
                "use_memory": bool(lines),
                "selected_slots": list(slot_names),
                "mode": "gate_disabled_all_slot_records",
            }

        sel = self.memory_gate.select_slots(
            question, slot_names, for_vector_context=False
        )
        if sel.use_memory and sel.slot_names:
            lines = self.dst.memory_lines_for_slots(dialogue_id, sel.slot_names)
            return lines, {
                **meta_base,
                "use_memory": bool(lines),
                "selected_slots": list(sel.slot_names),
                "mode": "llm_gate_slots",
            }

        return [], {
            **meta_base,
            "use_memory": False,
            "selected_slots": [],
            "mode": "llm_gate_rejected_slots",
        }

    def _memory_context_vector(
        self,
        dialogue_id: str,
        question: str,
        slot_names: List[str],
        meta_base: Dict,
    ) -> Tuple[List[str], Dict]:
        if not self.config.use_memory_gate:
            entity_scope = self.dst.entity_scope_for_slots(dialogue_id, slot_names, hops=1)
            hits = self.retriever.search(
                dialogue_id=dialogue_id,
                query=question,
                top_k=self.config.retrieval_top_k,
                allowed_slots=slot_names,
                allowed_entities=entity_scope,
            )
            lines = self._lines_from_retriever_hits(hits)
            return lines, {
                **meta_base,
                "use_memory": bool(lines),
                "selected_slots": list(slot_names),
                "entity_scope": entity_scope,
                "mode": "gate_disabled_vector_topk",
                "retrieved_count": len(hits),
            }

        sel = self.memory_gate.select_slots(
            question, slot_names, for_vector_context=True
        )
        if not sel.use_memory:
            return [], {
                **meta_base,
                "use_memory": False,
                "selected_slots": list(sel.slot_names),
                "mode": "llm_gate_rejected_vector",
            }

        selected_slots = list(sel.slot_names) if sel.slot_names else list(slot_names)
        entity_scope = self.dst.entity_scope_for_slots(dialogue_id, selected_slots, hops=1)
        hits = self.retriever.search(
            dialogue_id=dialogue_id,
            query=question,
            top_k=self.config.retrieval_top_k,
            allowed_slots=selected_slots,
            allowed_entities=entity_scope,
        )
        lines = self._lines_from_retriever_hits(hits)
        return lines, {
            **meta_base,
            "use_memory": bool(lines),
            "selected_slots": selected_slots,
            "entity_scope": entity_scope,
            "mode": "llm_gate_vector_topk",
            "retrieved_count": len(hits),
        }

    def answer_without_final_llm(self, dialogue_id: str, question: str) -> Dict:
        logger.info("answer_without_final_llm dialogue_id=%s", dialogue_id)
        memory_lines, gate_meta = self._memory_context_for_question(dialogue_id, question)
        selected_slots = gate_meta.get("selected_slots") or self.dst.active_slot_names(dialogue_id)
        entity_scope = gate_meta.get("entity_scope") or self.dst.entity_scope_for_slots(
            dialogue_id, selected_slots, hops=1
        )
        hits = self.retriever.search(
            dialogue_id=dialogue_id,
            query=question,
            top_k=self.config.retrieval_top_k,
            allowed_slots=selected_slots,
            allowed_entities=entity_scope,
        )
        memory_slots = self.dst.slots_with_messages(dialogue_id)
        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "use_memory": bool(memory_lines),
            "memory_gate": gate_meta,
            "memory_lines_for_final_llm": memory_lines,
            "retrieved": hits,
            "memory_slots": memory_slots,
        }

    def answer(self, dialogue_id: str, question: str) -> str:
        logger.info("answer dialogue_id=%s", dialogue_id)
        memory_lines, gate_meta = self._memory_context_for_question(dialogue_id, question)
        logger.info(
            "answer memory context lines=%d gate=%s",
            len(memory_lines),
            gate_meta,
        )
        return self.final_llm.generate(question=question, memory_lines=memory_lines)

    def clear_memory(self, dialogue_id: str) -> None:
        logger.info("clear_memory dialogue_id=%s", dialogue_id)
        self.dst.clear_dialogue(dialogue_id)
        self.store.clear_dialogue(dialogue_id)
