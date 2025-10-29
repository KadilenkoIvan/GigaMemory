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
from vllm import LLM, SamplingParams #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

from models import Message
from submit_interface import ModelWithMemory

# RAG components (relative imports inside submit package)
from .rag.chunker import chunk_dialogue
from .rag.embedder import Embedder
from .rag.indexer import MemoryIndex

# DST component
from .DST.dst_processor import DSTProcessor


# def _merge_facts(existing_facts: Dict[str, List[str]], new_facts: Dict[str, List[str]]) -> Dict[str, List[str]]:
#     """
#     Merge new facts into existing facts dictionary without overwriting.
    
#     Args:
#         existing_facts: Current facts dictionary
#         new_facts: New facts to add
        
#     Returns:
#         Merged facts dictionary
#     """
#     merged = existing_facts.copy()
    
#     for category, values in new_facts.items():
#         if category in merged:
#             # Создаём set один раз и добавляем только уникальные
#             existing_set = set(merged[category])
#             new_unique = [v for v in values if v not in existing_set]
#             merged[category].extend(new_unique)
#         else:
#             merged[category] = values.copy()
    
#     return merged


class SubmitModelWithMemory(ModelWithMemory):

    def __init__(self, model_path: str) -> None:
        # In-memory raw message store
        self.basic_memory = defaultdict(list)
        # Structured facts/summarization disabled — DST only decides save/skip
        # self.facts_memory: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
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
        # DST-only filtering. Save full user messages with indices; no summarization/extraction.
        for msg in messages:
            if msg.role != "user":
                continue
            should_save = True if self._dst_processor is None else False
            if self._dst_processor is not None:
                try:
                    last5 = [m.content for m in self.basic_memory.get(dialogue_id, []) if m.role == "user"][-5:]
                    memory_summary = "\n".join(last5)
                    should_save, _ = self._dst_processor.should_save_message(msg, memory_summary)
                except Exception as e:
                    print(f"Warning: DST processing failed: {str(e)}. Saving message by default.")
                    should_save = True
            if should_save:
                self._save_full_message_with_indices(dialogue_id, msg)

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
        # Facts/summarization disabled — rely on raw history only
        facts_text = ""
        
        # Also include raw conversation history
        memory = self.basic_memory.get(dialogue_id, [])
        serialized = [asdict(msg) for msg in memory]
        memory_text = "\n".join([f"{m['role']}: {m['content']}" for m in serialized])

        system_memory_prompt = (
            "Твоя задача - ответить на вопрос пользователя. Для этого тебе подается на вход история общения пользователя с важной информацией о нём.\n"
            "Пользователь разрешил использовать эту информацию для ответа на вопрос. Если в предоставленной информации нет ответа на вопрос, скажи что ты не знаешь.\n\n"
            "Полная информация может находиться в нескольких сообщениях. Используй все сообщения для ответа на вопрос."
            "ВАЖНО: Память упорядочена по времени (от старых к новым). В сообщениях есть два числа, записанных в формате: [сессия : сообщение в сессии]"
            "Если информация противоречит друг другу, используй БОЛЕЕ ПОЗДНИЕ сообщения - они актуальнее и заменяют старые. Наиболее приоритетные сообщения имеют больший номер сессии, и сообщения в сесии.\n\n"
            f"История диалога:\n{memory_text}"
        )
        return [Message('system', system_memory_prompt)]

    def clear_memory(self, dialogue_id: str) -> None:
        # Clear raw memory and counters
        self.basic_memory[dialogue_id] = []
        self._dialogue_msg_counters[dialogue_id] = 0
        # Clear structured facts
        #self.facts_memory[dialogue_id] = {}

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

    def _save_full_message_with_indices(self, dialogue_id: str, msg: Message) -> None:
        """Save the full message, augmenting content with session and message indices."""
        session_id = msg.session_id or "unknown"
        existing = self.basic_memory.get(dialogue_id, [])
        next_idx = sum(1 for m in existing if getattr(m, 'session_id', None) == msg.session_id and m.role == 'user') + 1
        augmented = Message(role=msg.role, content=f"[{session_id}:{next_idx}] {msg.content}", session_id=msg.session_id)
        self.basic_memory[dialogue_id].append(augmented)

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
    
    # def _extract_facts_with_llm(self, message_content: str) -> Dict[str, List[str]]:
    #     """
    #     Extract structured facts from user message using main LLM.
        
    #     Args:
    #         message_content: The content of the user message
            
    #     Returns:
    #         Dictionary with extracted facts in format {category: [values]}
    #     """
    #     system_prompt = """Ты — модуль извлечения структурированной информации из реплик пользователя.
    #                 Твоя задача — вычленить важную информацию о пользователе и представить её в формате "категория: значение".

    #                 Категории информации:
    #                 - интересы: увлечения и интересы пользователя
    #                 - хобби: занятия в свободное время
    #                 - спорт: спортивные увлечения
    #                 - работа: профессия, род занятий
    #                 - место_жительства: город, страна
    #                 - возраст: возраст пользователя
    #                 - привычки: регулярные действия, привычки
    #                 - планы: намерения, будущие действия
    #                 - факты: любые другие важные факты о пользователе
    #                 - имя: имя пользователя
    #                 - семья: информация о семье (жена, муж, дети и т.д.)
    #                 - питомцы: домашние животные
    #                 - образование: учебные заведения, специальность
    #                 - навыки: умения, навыки
    #                 - предпочтения: предпочтения в еде, музыке и т.д.

    #                 Инструкции:
    #                 1. Извлекай ТОЛЬКО явную информацию из реплики.
    #                 2. Формат ответа: каждая категория с новой строки в формате "категория: значение1, значение2".
    #                 3. Если в реплике несколько категорий, выведи все.
    #                 4. Если категория не подходит, используй "факты".
    #                 5. Не добавляй пояснений, только факты в указанном формате.

    #                 Примеры:

    #                 Пользователь: "Я футболист."
    #                 Твой ответ:
    #                 спорт: футбол

    #                 Пользователь: "Классно! Возьму его с собой на баскетбол"
    #                 Твой ответ:
    #                 спорт: баскетбол

    #                 Пользователь: "Завтра пойду на пробежку, а потом встречусь с друзьями"
    #                 Твой ответ:
    #                 спорт: бег
    #                 планы: встреча с друзьями

    #                 Пользователь: "Я в Москву переехал, не думал что тут такие квартиры дорогие"
    #                 Твой ответ:
    #                 место_жительства: Москва

    #                 Пользователь: "У меня есть кот Барсик и собака Лайка"
    #                 Твой ответ:
    #                 питомцы: кот Барсик, собака Лайка
    #                 """
        
    #     user_prompt = f"Реплика пользователя: \"{message_content}\""
        
    #     # Call main LLM for extraction
    #     messages = [
    #         Message(role='system', content=system_prompt),
    #         Message(role='user', content=user_prompt)
    #     ]
        
    #     response = self._inference(messages)
        
    #     # Parse response into facts dictionary
    #     facts = self._parse_facts_response(response)
        
    #     return facts
    
    # def _parse_facts_response(self, response: str) -> Dict[str, List[str]]:
    #     """
    #     Parse LLM response into structured facts dictionary.
        
    #     Args:
    #         response: The LLM's response with facts
            
    #     Returns:
    #         Dictionary with facts in format {category: [values]}
    #     """
    #     facts: Dict[str, List[str]] = {}
        
    #     # Parse line by line
    #     lines = response.strip().split('\n')
    #     for line in lines:
    #         line = line.strip()
    #         if not line or ':' not in line:
    #             continue
            
    #         # Split by first colon
    #         parts = line.split(':', 1)
    #         if len(parts) != 2:
    #             continue
            
    #         category = parts[0].strip().lower()
    #         values_str = parts[1].strip()
            
    #         # Split values by comma
    #         values = [v.strip() for v in values_str.split(',') if v.strip()]
            
    #         if values:
    #             facts[category] = values
        
    #     return facts

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
