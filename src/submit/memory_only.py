import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List
import json

# Ensure `src` is on sys.path when running this file directly
_this_dir = Path(__file__).resolve()
_src_dir = _this_dir.parents[1]  # .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from models import Message, Dialog
from submit_interface import ModelWithMemory

# Import DST processor

from submit.DST.dst_processor import DSTProcessor
# except ImportError:
#     print("Warning: Could not import DSTProcessor. DST functionality will be disabled.")
#     DSTProcessor = None

def _read_utf8(file_path: str) -> List[Dialog]:
    with open(file_path, "r", encoding="utf-8") as file:
        dialogs_data = [json.loads(line) for line in file]
    return [Dialog.from_dict(_) for _ in dialogs_data]


class MemoryOnlyModel(ModelWithMemory):

    def __init__(self) -> None:
        self.basic_memory = defaultdict(list)
        # Structured facts store per dialogue: {dialogue_id: {category: [values]}}
        self.facts_memory: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        # Initialize DST processor if available
        self.dst_processor = None
        if DSTProcessor is not None:
            try:
                self.dst_processor = DSTProcessor()
                print("DST processor initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize DST processor: {str(e)}"
                print(error_msg)
                import sys
                sys.exit(121)  # Выход с кодом 121 при ошибке инициализации DST
        else:
            print("DST processor not found. Exiting with code 121.")
            import sys
            sys.exit(121)  # Выход с кодом 121, если DST процессор не найден

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        # Filter messages using DST processor if available
        # DST checks only user messages (true/false)
        # Fact extraction would use main LLM (but in test mode we skip it)
        filtered_messages = []
        
        for msg in messages:
            # Only process user messages with DST
            if msg.role == "user":
                should_save = False
                reason = ""
                
                if self.dst_processor is not None:
                    try:
                        last5 = [m.content for m in self.basic_memory.get(dialogue_id, []) if m.role == "user"][-5:]
                        memory_summary = "\n".join(last5)
                        should_save, reason = self.dst_processor.should_save_message(msg, memory_summary)
                        if should_save:
                            print(f"DST-✔️: Saving message: '{msg.content}")
                        else:
                            print(f"DST-❌: Skipping message: '{msg.content}")
                    except Exception as e:
                        print(f"Warning: DST processing failed: {str(e)}. Saving message by default.")
                        should_save = True
                else:
                    # If DST processor is not available, save all user messages
                    should_save = True
                    print(f"No DST: Saving user message by default: '{msg.content}'")
                
                if should_save:
                    # Save only user message (no assistant response) with indices
                    session_id = msg.session_id or "unknown"
                    existing = self.basic_memory.get(dialogue_id, [])
                    next_idx = sum(1 for m in existing if getattr(m, 'session_id', None) == msg.session_id and m.role == 'user') + 1
                    augmented = Message(role=msg.role, content=f"[{session_id}:{next_idx}] {msg.content}", session_id=msg.session_id)
                    filtered_messages.append(augmented)
        
        # If no messages to save after filtering, return early
        if not filtered_messages:
            print("No messages to save after DST filtering")
            return
            
        # Append filtered messages to memory
        self.basic_memory[dialogue_id] += filtered_messages
        print(f"Added {len(filtered_messages)} messages to memory for dialogue {dialogue_id}")

    # Optional: helper to keep parity with main model (not used above to preserve batch append)
    def _save_full_message_with_indices(self, dialogue_id: str, msg: Message) -> None:
        session_id = msg.session_id or "unknown"
        existing = self.basic_memory.get(dialogue_id, [])
        next_idx = sum(1 for m in existing if getattr(m, 'session_id', None) == msg.session_id and m.role == 'user') + 1
        augmented = Message(role=msg.role, content=f"[{session_id}:{next_idx}] {msg.content}", session_id=msg.session_id)
        self.basic_memory[dialogue_id].append(augmented)

    def clear_memory(self, dialogue_id: str) -> None:
        self.basic_memory[dialogue_id] = []
        self.facts_memory[dialogue_id] = {}
        print(f"Cleared memory for dialogue {dialogue_id}")

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        # В тестовом режиме модель не вызывается. Возвращаем заглушку.
        num_msgs = len(self.basic_memory.get(dialogue_id, []))
        num_facts = len(self.facts_memory.get(dialogue_id, {}))
        return f"[MEMORY_ONLY] msgs={num_msgs}; facts={num_facts}; q='{question[:200]}'"

    def extract(self, dialogue_id: str) -> List[Message]:
        # Build system prompt from structured facts
        facts = self.facts_memory.get(dialogue_id, {})
        
        # Format facts into readable text
        facts_text = ""
        if facts:
            facts_text = "Структурированная информация о пользователе (извлечена основной LLM):\n"
            for category, values in facts.items():
                facts_text += f"- {category}: {', '.join(values)}\n"
        
        # Also include raw conversation history
        memory = self.basic_memory.get(dialogue_id, [])
        serialized = [asdict(msg) for msg in memory]
        memory_text = "\n".join([f"{m['role']}: {m['content']}" for m in serialized])
        
        system_memory_prompt = (
            "Тест без LLM. Ниже — накопленная структурированная информация и история диалога (после фильтрации DST).\n\n"
            f"{facts_text}\n"
            f"История диалога:\n{memory_text}"
        )
        return [Message(role="system", content=system_memory_prompt)]


def simulate_memory_from_file(input_file: str) -> Dict[str, str]:
    """
    Читает jsonl с диалогами, имитирует запись в память (батчи user->assistant),
    и возвращает для каждого диалога текст системного сообщения с историей.
    """
    dialogs: List[Dialog] = _read_utf8(input_file)
    model = MemoryOnlyModel()
    results: Dict[str, str] = {}

    for dialog in dialogs:
        dialog_messages = dialog.get_messages()
        message_batch: List[Message] = []
        for i, msg in enumerate(dialog_messages):
            message_batch.append(msg)
            if i % 2 == 1:
                model.write_to_memory(message_batch, dialog.id)
                message_batch = []

        system_ctx = model.extract(dialog.id)
        system_text = system_ctx[0].content if system_ctx else ""
        results[dialog.id] = system_text

        model.clear_memory(dialog.id)

    return results

if __name__ == "__main__":
    import argparse
    from submit.rag.chunker import chunk_dialogue
    from submit.rag.embedder import Embedder
    from submit.rag.indexer import MemoryIndex
    from submit.rag.retriever import Retriever

    parser = argparse.ArgumentParser(description="Simulate memory without LLM and optionally build/search RAG index")
    parser.add_argument("input_file", type=str, nargs="?", help="Path to jsonl dialogues file (required for build)")
    parser.add_argument("--max_preview", type=int, default=20000, help="Max characters to print per dialog")
    # RAG options
    parser.add_argument("--build_index", action="store_true", help="Build FAISS index from dialogues")
    parser.add_argument("--index_path", type=str, default="src/submit/rag/index.npy", help="Path to FAISS index file")
    parser.add_argument("--query", type=str, default=None, help="Query to search over built/loaded index")
    parser.add_argument("--k", type=int, default=5, help="Top-k results to retrieve")
    parser.add_argument("--embedder_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--chunk_tokens", type=int, default=500, help="Max tokens (words) per chunk when building index")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Overlap (words) between consecutive chunks")
    args = parser.parse_args()

    # Always: if input_file provided, show memory simulation preview
    if args.input_file:
        results = simulate_memory_from_file(args.input_file)
        with open("memory_only_simulation.txt", "w") as f:
            for did, text in results.items():
                print(did)
                f.write(text)
                preview = text[: args.max_preview]
                print(preview, "..." if len(text) > len(preview) else "")

    # Build index if requested
    if args.build_index:
        if not args.input_file:
            raise SystemExit("--build_index requires input_file")
        dialogs: List[Dialog] = _read_utf8(args.input_file)
        # produce structured entries per message chunk
        entries: List[dict] = []
        for d in dialogs:
            messages = d.get_messages()
            for msg_idx, m in enumerate(messages):
                base_text = f"{m.role}: {m.content}"
                chunks = chunk_dialogue([base_text], max_tokens=args.chunk_tokens, overlap=args.chunk_overlap)
                for ch_idx, ch_text in enumerate(chunks):
                    entries.append({
                        "dialogue_id": d.id,
                        "chunk_id": f"{msg_idx}:{ch_idx}",
                        "role": m.role,
                        "text": ch_text,
                    })

        if not entries:
            raise SystemExit("No chunks produced from dialogues")

        embedder = Embedder(model_name=args.embedder_model)
        embeddings = embedder.encode([e["text"] for e in entries])
        dim = len(embeddings[0])
        index = MemoryIndex(dim=dim, index_path=args.index_path)
        index.add_entries(embeddings, entries)
        index.save()
        print(f"Index saved to {args.index_path} and {args.index_path.rsplit('.', 1)[0]}.json")

    # Search if query provided
    if args.query:
        embedder = Embedder(model_name=args.embedder_model)
        # load index
        # Determine dim from embedder; create dummy index then load
        dim = len(embedder.encode(["test"])[0])
        index = MemoryIndex(dim=dim, index_path=args.index_path)
        index.load()
        if not index.data:
            raise SystemExit(f"Index data not found at {args.index_path}. Build it first with --build_index.")
        retriever = Retriever(index=index, embedder=embedder)
        results = retriever.search(args.query, k=args.k)
        print("Top-" + str(args.k) + " results:")
        import json as _json
        for r in results:
            out = {
                "dialogue_id": r.get("dialogue_id"),
                "chunk_id": r.get("chunk_id"),
                "role": r.get("role"),
                "score": round(r.get("score", 0.0), 4),
                "text": (r.get("text") or "")[:args.max_preview],
            }
            print(_json.dumps(out, ensure_ascii=False))


