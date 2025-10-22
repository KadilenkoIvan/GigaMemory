import os
import sys
import subprocess
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List, Tuple


def _bootstrap_offline_dependencies() -> None:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    current_dir = Path(__file__).resolve().parent  # submit/
    # Support both repository layout (src/submit/...) and final submit root (submit/)
    # Try local libs first; also probe one level up if present
    candidate_dirs = [
        current_dir / "libs",
        current_dir.parent / "libs",
    ]
    libs_dir = next((d for d in candidate_dirs if d.exists() and d.is_dir()), None)
    if not libs_dir:
        return

    archives = list(libs_dir.glob("*.whl")) + list(libs_dir.glob("*.tar.gz"))
    if not archives:
        return

    for pkg in sorted(archives):
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(libs_dir), str(pkg)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception:
            # Продолжить, даже если какой-то пакет не установился
            pass


_bootstrap_offline_dependencies()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from models import Message
from submit_interface import ModelWithMemory

# RAG components (relative imports inside submit package)
from .rag.chunker import chunk_dialogue
from .rag.embedder import Embedder
from .rag.indexer import MemoryIndex

# DST component
from .DST.dst_processor import DSTProcessor, merge_facts


class SubmitModelWithMemory(ModelWithMemory):

    def __init__(self, model_path: str) -> None:
        # In-memory raw message store (original messages)
        self.basic_memory = defaultdict(list)
        # Structured facts store per dialogue: {dialogue_id: {category: [values]}}
        self.facts_memory: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        # Track per-dialogue sequential message index for stable chunk_id
        self._dialogue_msg_counters: Dict[str, int] = defaultdict(int)

        # RAG config
        self._chunk_tokens: int = 500
        self._chunk_overlap: int = 50

        # Model
        self.model_path = model_path
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = LLM(model=self.model_path, trust_remote_code=True)
            self.sampling_params = SamplingParams(temperature=0.0, max_tokens=100, seed=42, truncate_prompt_tokens=131072)
        except Exception as e:
            raise RuntimeError(f"Ошибка загрузки модели {self.model_path}: {str(e)}")

        # RAG runtime: embedder and FAISS index (lazy dim detection)
        # Force embedder to use local path if provided via env or submit/rag/models
        self._embedder = Embedder()
        dim = len(self._embedder.encode(["test"])[0])
        self._index = MemoryIndex(dim=dim)
        
        # Initialize DST processor
        try:
            self._dst_processor = DSTProcessor()
        except Exception as e:
            error_msg = f"Failed to initialize DST processor: {str(e)}"
            print(error_msg)
            sys.exit(121)  # Выход с кодом 121 при ошибке инициализации DST

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        # Filter messages using DST processor and extract structured facts
        # DST checks only user messages, but saves both user and assistant messages from the pair
        filtered_messages = []
        
        # Process messages in pairs (user, assistant)
        i = 0
        while i < len(messages):
            current_msg = messages[i]
            
            # Find user message and its corresponding assistant response
            if current_msg.role == "user":
                # Check with DST if user message should be saved
                should_save = False
                
                if self._dst_processor is not None:
                    try:
                        should_save, _ = self._dst_processor.should_save_message(current_msg)
                        
                        # If message contains important info, extract structured facts
                        if should_save:
                            extracted_facts = self._dst_processor.extract_facts(current_msg)
                            # Merge new facts with existing facts
                            self.facts_memory[dialogue_id] = merge_facts(
                                self.facts_memory[dialogue_id], 
                                extracted_facts
                            )
                    except Exception as e:
                        print(f"Warning: DST processing failed: {str(e)}. Saving message by default.")
                        should_save = True
                else:
                    # If DST processor is not available, save all user messages
                    should_save = True
                
                if should_save:
                    # Save user message
                    filtered_messages.append(current_msg)
                    
                    # Save corresponding assistant message if it exists
                    # if i + 1 < len(messages) and messages[i + 1].role == "assistant":
                    #     filtered_messages.append(messages[i + 1])
                    #     i += 1  # Skip assistant message in next iteration
            
            i += 1
        
        # If no messages to save after filtering, return early
        if not filtered_messages:
            return
            
        # Append to raw memory (only filtered messages)
        self.basic_memory[dialogue_id] += filtered_messages

        # Incrementally add to FAISS index with chunking and metadata
        # entries: List[dict] = []
        # entry_texts: List[str] = []

        # Determine starting sequential message index for this dialogue
        # msg_seq_start = self._dialogue_msg_counters[dialogue_id]

        # for local_idx, msg in enumerate(filtered_messages):
        #     msg_seq_idx = msg_seq_start + local_idx
        #     base_text = f"{msg.role}: {msg.content}"
        #     chunks = chunk_dialogue([base_text], max_tokens=self._chunk_tokens, overlap=self._chunk_overlap)
        #     for ch_idx, ch_text in enumerate(chunks):
        #         entries.append({
        #             "dialogue_id": dialogue_id,
        #             "chunk_id": f"{msg_seq_idx}:{ch_idx}",
        #             "role": msg.role,
        #             "text": ch_text,
        #         })
        #         entry_texts.append(ch_text)

        # # Advance dialogue counter by number of newly saved messages
        # self._dialogue_msg_counters[dialogue_id] += len(filtered_messages)

        # if entry_texts:
        #     embeddings = self._embedder.encode(entry_texts)
        #     self._index.add_entries(embeddings, entries)

    def _search_dialogue_context(self, dialogue_id: str, question: str, k: int = 5) -> List[dict]:
        """Retrieve top-k entries from the index restricted to the given dialogue_id."""
        if not self._index.data:
            return []

        # Get more candidates and filter by dialogue_id
        query_emb = self._embedder.encode([question])
        # Ask for more to allow post-filtering
        raw_k = min(len(self._index.data), max(k * 5, k))
        D, I = self._index.search(query_emb, raw_k)

        candidates: List[Tuple[float, dict]] = []
        for idx, score in zip(I[0], D[0]):
            if 0 <= idx < len(self._index.data):
                entry = self._index.data[idx]
                if entry.get("dialogue_id") == dialogue_id:
                    candidates.append((float(score), entry))

        # Sort by score desc (FAISS returns already sorted, but after filtering enforce order)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in candidates[:k]]

    def extract(self, dialogue_id: str) -> List[Message]:
        # Build system prompt from structured facts
        facts = self.facts_memory.get(dialogue_id, {})
        
        # Format facts into readable text
        facts_text = ""
        if facts:
            facts_text = ""
            for category, values in facts.items():
                facts_text += f"- {category}: {', '.join(values)}\n"
        
        # Also include raw conversation history
        memory = self.basic_memory.get(dialogue_id, [])
        serialized = [asdict(msg) for msg in memory]
        memory_text = "\n".join([f"{m['role']}: {m['content']}" for m in serialized])

        system_memory_prompt = (
            "Твоя задача - ответить на вопрос пользователя. Для этого тебе подается на вход структурированная информация о пользователе и история общения.\n"
            "Пользователь разрешил использовать эту информацию для ответа на вопрос.\n\n"
            f"Информация о пользователе:\n{facts_text}\n"
            f"История диалога:\n{memory_text}"
        )
        return [Message('system', system_memory_prompt)]

    def clear_memory(self, dialogue_id: str) -> None:
        # Clear raw memory and counters
        self.basic_memory[dialogue_id] = []
        self._dialogue_msg_counters[dialogue_id] = 0
        # Clear structured facts
        self.facts_memory[dialogue_id] = {}

        # Remove dialogue entries from the index by rebuilding it
        if not self._index.data:
            return
        remaining = [e for e in self._index.data if e.get("dialogue_id") != dialogue_id]
        # Rebuild in-memory index from remaining entries
        texts = [e.get("text", "") for e in remaining]
        if texts:
            embeddings = self._embedder.encode(texts)
            # Reset index
            dim = len(embeddings[0])
            self._index = MemoryIndex(dim=dim)
            self._index.add_entries(embeddings, remaining)
        else:
            # No entries remain
            dim = len(self._embedder.encode(["test"])[0])
            self._index = MemoryIndex(dim=dim)

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        # 1) Rephrase the question for retrieval (query rewriting)
        # try:
        #     rewritten = self._rewrite_query(dialogue_id, question)
        #     retrieval_query = rewritten if rewritten else question
        # except Exception:
        #     retrieval_query = question

        # 2) Retrieve top-k relevant chunks for this dialogue and rewritten question
        # top_entries = self._search_dialogue_context(dialogue_id, retrieval_query, k=5)
        # context_text = "\n".join([f"{e.get('role')}: {e.get('text')}" for e in top_entries])

        user_promt = "Отвечай КРАТКО без предложений и размышлений, используй ТОЛЬКО ОДНО предложение для ответа.\n"
        context = self.extract(dialogue_id)
        context.append(Message(
            role="user",
            content=user_promt + question
        ))
        # system_prompt = (
        #     "Используй релевантные фрагменты из памяти (RAG) ниже для ответа.\n"
        #     f"Память (top-k):\n{context_text}\n"
        # )

        # context = [Message(role='system', content=system_prompt)]
        # context.append(Message(
        #     role="user",
        #     content=user_promt + question
        # ))
        answer = self._inference(context)
        
        return answer

    def _inference(self, messages: List[Message]) -> str:
        try:
            msg_dicts = [asdict(m) for m in messages]
            input_tensor = self.tokenizer.apply_chat_template(
                msg_dicts,
                add_generation_prompt=True,
            )
            outputs = self.model.generate(prompt_token_ids=input_tensor, sampling_params=self.sampling_params, use_tqdm=False)
            result = outputs[0].outputs[0].text
            return result.strip()

        except Exception as e:
            return f"Ошибка при инференсе локальной модели: {str(e)}"

    def _rewrite_query(self, dialogue_id: str, question: str) -> str:
        """
        Переформулирует исходный вопрос в короткий запрос для поиска по памяти (RAG).
        Сохраняет ключевые сущности, имена, числа. Не добавляет новых фактов.
        """
        instruction = (
            "Переформулируй вопрос в краткий поисковый запрос для извлечения релевантных фрагментов из памяти (RAG). Не используй размышления."
            "Сохраняй имена, числа и факты. Не добавляй новых сведений."
            "Ответ верни одной строкой без пояснений."
        )
        context = [
            Message(role='system', content=instruction),
            Message(role='user', content=question),
        ]
        rewritten = self._inference(context)
        return rewritten.strip()
