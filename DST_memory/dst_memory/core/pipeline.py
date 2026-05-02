from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import logging

from ..clients.classifier import ImportanceClassifier
from .config import PipelineConfig
from ..triplets.conflict_client import TripletConflictClient
from .dst_manager import DSTManager
from ..clients.llm_client import FinalLLMClient
from ..clients.memory_gate_client import MemoryGateClient
from .models import Message
from ..clients.serving import LocalHFServing
from ..slots.slot_select_client import SlotSelectClient
from ..triplets.triplet_client import TripletExtractionClient

logger = logging.getLogger(__name__)


class DSTMemoryPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        ragu_processor: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        config:
            Pipeline configuration.
        ragu_processor:
            Optional ``RaguGraphProcessor`` instance.  When provided:
            - triplets are mirrored to RAGU's KnowledgeGraph on every insert/delete.
            - search and retrieval are done by RAGU LocalSearchEngine.
        """
        self.config = config
        self.ragu_processor = ragu_processor

        if not config.use_ragu:
            raise ValueError("RAGU-only mode: set use_ragu=true in config.")
        if ragu_processor is None:
            raise ValueError("RAGU-only mode: RAGU processor is not initialized.")

        logger.info(
            "Initializing pipeline threshold=%.3f top_k=%d llm_mode=%s gate=%s "
            "memory_gate_stub=%s memory_strategy=%s ragu=%s ttl_mode=%s semantic_dedup=%s "
            "slot_context=%s deletion_mode=%s prompt_language=%s",
            config.importance_threshold,
            config.retrieval_top_k,
            config.llm_mode,
            config.use_memory_gate,
            config.memory_gate_use_stub,
            config.memory_strategy,
            ragu_processor is not None,
            config.ttl_mode,
            config.ttl_semantic_dedup_enabled,
            config.slot_context_enabled,
            config.triplet_deletion_mode,
            getattr(config, "prompt_language", "ru"),
        )
        self.classifier = ImportanceClassifier(
            model_path=config.importance_model_path,
            threshold=config.importance_threshold,
        )
        slot_serving = None
        if not config.slot_use_stub:
            slot_serving = LocalHFServing(
                config.slot_model_path,
                enable_thinking=config.slot_model_enable_thinking,
            )
        triplet_extractor = TripletExtractionClient(
            use_stub=config.slot_use_stub,
            serving=slot_serving,
            max_triplets=max(6, config.slot_max_slots_per_message * 3),
            max_retries=1,
            ttl_mode=config.ttl_mode,
            prompt_language=config.prompt_language,
        )
        slot_selector = SlotSelectClient(
            use_stub=config.slot_use_stub,
            serving=slot_serving,
            max_slots=config.slot_max_slots_per_message,
            max_retries=1,
            prompt_language=config.prompt_language,
        )
        conflict_resolver = TripletConflictClient(
            use_stub=config.slot_use_stub,
            serving=slot_serving,
            max_retries=1,
            allow_multi_relation_same_object=config.conflict_allow_multi_relation_same_object,
            prompt_language=config.prompt_language,
        )

        # --- Deletion components ---
        negation_detector = None
        deletion_client = None

        if config.triplet_deletion_mode == "heuristic":
            from ..triplets.negation_detector import NegationDeletionDetector
            negation_detector = NegationDeletionDetector(
                use_pymorphy=config.deletion_use_pymorphy
            )
            logger.info("NegationDeletionDetector enabled pymorphy=%s", config.deletion_use_pymorphy)

        elif config.triplet_deletion_mode == "llm_separate":
            from ..triplets.deletion_client import TripletDeletionClient
            deletion_client = TripletDeletionClient(
                use_stub=config.slot_use_stub,
                serving=slot_serving,
                max_retries=1,
                prompt_language=config.prompt_language,
            )
            logger.info("TripletDeletionClient (llm_separate) enabled")

        self.dst = DSTManager(
            triplet_extractor=triplet_extractor,
            slot_selector=slot_selector,
            conflict_resolver=conflict_resolver,
            slot_fallback_on_no_slots=config.slot_fallback_on_no_slots,
            triplet_fallback_on_empty=config.triplet_fallback_on_empty,
            ragu_processor=ragu_processor,
            ttl_mode=config.ttl_mode,
            ttl_slot_overrides=config.ttl_slot_overrides,
            semantic_dedup_enabled=config.ttl_semantic_dedup_enabled,
            semantic_dedup_threshold=config.ttl_semantic_dedup_threshold,
            slot_context_enabled=config.slot_context_enabled,
            slot_context_max_facts=config.slot_context_max_facts,
            triplet_deletion_mode=config.triplet_deletion_mode,
            negation_detector=negation_detector,
            deletion_client=deletion_client,
        )
        gate_stub = config.memory_gate_use_stub or slot_serving is None
        self.memory_gate = MemoryGateClient(
            use_stub=gate_stub,
            serving=slot_serving,
            max_retries=1,
            prompt_language=config.prompt_language,
        )
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
            dialogue_id, message.role, len(message.content),
        )
        if message.role != "user":
            return {"saved": False, "reason": "only_user_messages_supported"}

        cls = self.classifier.predict(message.content)
        logger.info(
            "classifier result dialogue_id=%s p_important=%.4f is_important=%s",
            dialogue_id, cls["p_important"], cls["is_important"],
        )
        if not bool(cls["is_important"]):
            return {
                "message": message.content,
                "saved": False,
                "reason": "not_important",
                "classifier": cls,
            }

        new_facts, selected_slots = self.dst.upsert_from_message(dialogue_id, message.content)
        if not new_facts:
            return {
                "slots": selected_slots,
                "message": message.content,
                "saved": False,
                "reason": "no_facts_extracted",
                "classifier": cls,
                "new_facts": [],
            }
        return {
            "slots": selected_slots,
            "message": message.content,
            "saved": True,
            "reason": "important",
            "classifier": cls,
            "new_facts": [asdict(f) for f in new_facts],
        }

    def add_recent_pair(self, dialogue_id: str, user_text: str, assistant_text: str) -> None:
        state = self.dst.get_state(dialogue_id)
        if not user_text.strip() or not assistant_text.strip():
            return
        state.recent_pairs.append({"user": user_text.strip(), "assistant": assistant_text.strip()})
        keep = max(1, int(self.config.recent_history_pairs))
        if len(state.recent_pairs) > keep:
            state.recent_pairs = state.recent_pairs[-keep:]

    def recent_pairs(self, dialogue_id: str) -> List[Dict[str, str]]:
        state = self.dst.get_state(dialogue_id)
        return list(state.recent_pairs[-max(1, int(self.config.recent_history_pairs)):])

    def _memory_context_for_question(
        self, dialogue_id: str, question: str
    ) -> Tuple[Any, Dict]:
        strategy = (self.config.memory_strategy or "relevant_slots_full").strip().lower()
        if strategy not in ("full_graph_json", "relevant_slots_full", "topk_graph_records"):
            strategy = "relevant_slots_full"
        meta_base: Dict[str, Any] = {
            "memory_strategy": strategy,
            "graph_top_k_records": int(self.config.graph_top_k_records),
        }
        slot_names = self.dst.active_slot_names(dialogue_id)

        if strategy == "full_graph_json":
            memory_graph = {
                "dialogue_id": dialogue_id,
                "slots": self.dst.slots_with_messages(dialogue_id),
            }
            has_data = any(s.get("messages") for s in memory_graph["slots"])
            return memory_graph, {
                **meta_base,
                "use_memory": has_data,
                "selected_slots": list(slot_names),
                "mode": "full_graph_json",
            }

        if strategy == "relevant_slots_full":
            if not slot_names:
                return {"slots": []}, {
                    **meta_base,
                    "use_memory": False,
                    "selected_slots": [],
                    "reason": "no_active_slots",
                    "mode": "relevant_slots_full_no_slots",
                }
            if not self.config.use_memory_gate:
                selected = list(slot_names)
                slots = self._slots_payload(dialogue_id, selected)
                return {"slots": slots}, {
                    **meta_base,
                    "use_memory": bool(slots),
                    "selected_slots": selected,
                    "mode": "relevant_slots_full_gate_disabled",
                }
            sel = self.memory_gate.select_slots(question, slot_names, for_vector_context=False)
            selected = list(sel.slot_names) if sel.slot_names else []
            if not sel.use_memory or not selected:
                return {"slots": []}, {
                    **meta_base,
                    "use_memory": False,
                    "selected_slots": selected,
                    "mode": "relevant_slots_full_gate_rejected",
                }
            slots = self._slots_payload(dialogue_id, selected)
            return {"slots": slots}, {
                **meta_base,
                "use_memory": bool(slots),
                "selected_slots": selected,
                "mode": "relevant_slots_full_gate_selected",
            }

        # topk_graph_records
        lines = self.ragu_processor.search_memory(
            query=question,
            slot_names=None,
            top_k=max(1, int(self.config.graph_top_k_records)),
        )
        return {"records": lines}, {
            **meta_base,
            "use_memory": bool(lines),
            "selected_slots": [],
            "retrieved_count": len(lines),
            "mode": "topk_graph_records",
        }

    def _slots_payload(self, dialogue_id: str, selected_slots: List[str]) -> List[Dict[str, Any]]:
        selected = set(selected_slots)
        out = []
        for slot in self.dst.slots_with_messages(dialogue_id):
            name = str(slot.get("slot", ""))
            if name in selected:
                out.append(slot)
        return out

    def answer_without_final_llm(self, dialogue_id: str, question: str) -> Dict:
        logger.info("answer_without_final_llm dialogue_id=%s", dialogue_id)
        memory_context, gate_meta = self._memory_context_for_question(dialogue_id, question)
        selected_slots = gate_meta.get("selected_slots") or self.dst.active_slot_names(dialogue_id)
        retrieved = self.ragu_processor.search_memory(
            query=question,
            slot_names=list(selected_slots) if selected_slots else None,
            top_k=self.config.retrieval_top_k,
        )

        # Build the prompt that WOULD be sent to final LLM (for logging)
        prompt_messages = self.final_llm.build_messages(
            question=question,
            memory_context=memory_context,
            recent_pairs=self.recent_pairs(dialogue_id),
        )
        logger.debug(
            "answer_without_final_llm prompt:\nSYSTEM:\n%s\n\nUSER:\n%s",
            prompt_messages[0]["content"],
            prompt_messages[1]["content"],
        )

        return {
            "dialogue_id": dialogue_id,
            "question": question,
            "use_memory": bool(gate_meta.get("use_memory")),
            "memory_gate": gate_meta,
            "memory_context_for_final_llm": memory_context,
            "final_llm_prompt": prompt_messages,  # full prompt for logs
            "recent_pairs": self.recent_pairs(dialogue_id),
            "retrieved": retrieved,
            "memory_slots": self.dst.slots_with_messages(dialogue_id),
            "expired_facts": self.dst.expired_facts(dialogue_id),
        }

    def answer(self, dialogue_id: str, question: str) -> str:
        logger.info("answer dialogue_id=%s", dialogue_id)
        memory_context, gate_meta = self._memory_context_for_question(dialogue_id, question)
        logger.info(
            "answer memory context ready mode=%s active_slots=%s",
            gate_meta.get("mode"),
            gate_meta.get("selected_slots"),
        )
        logger.debug(
            "answer memory_context full:\n%s",
            __import__("json").dumps(memory_context, ensure_ascii=False, indent=2),
        )
        answer_text = self.final_llm.generate(
            question=question,
            memory_context=memory_context,
            recent_pairs=self.recent_pairs(dialogue_id),
        )
        logger.info("Final LLM answer dialogue_id=%s: %s", dialogue_id, answer_text[:500])
        return answer_text

    def clear_memory(self, dialogue_id: str) -> None:
        logger.info("clear_memory dialogue_id=%s", dialogue_id)
        self.dst.clear_dialogue(dialogue_id)
