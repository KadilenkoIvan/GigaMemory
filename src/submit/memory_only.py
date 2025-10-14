from collections import defaultdict
from dataclasses import asdict
from typing import Dict, List
import json

from models import Message, Dialog
from submit_interface import ModelWithMemory
# avoid Windows cp1251 issues by reading with UTF-8 locally

def _read_utf8(file_path: str) -> List[Dialog]:
    with open(file_path, "r", encoding="utf-8") as file:
        dialogs_data = [json.loads(line) for line in file]
    return [Dialog.from_dict(_) for _ in dialogs_data]


class MemoryOnlyModel(ModelWithMemory):

    def __init__(self) -> None:
        self.basic_memory = defaultdict(list)

    def write_to_memory(self, messages: List[Message], dialogue_id: str) -> None:
        self.basic_memory[dialogue_id] += messages

    def clear_memory(self, dialogue_id: str) -> None:
        self.basic_memory[dialogue_id] = []

    def answer_to_question(self, dialogue_id: str, question: str) -> str:
        # В тестовом режиме модель не вызывается. Возвращаем заглушку.
        num_msgs = len(self.basic_memory.get(dialogue_id, []))
        return f"[MEMORY_ONLY] msgs={num_msgs}; q='{question[:200]}'"

    def extract(self, dialogue_id: str) -> List[Message]:
        memory = self.basic_memory.get(dialogue_id, [])
        serialized = [asdict(msg) for msg in memory]
        memory_text = "\n".join([f"{m['role']}: {m['content']}" for m in serialized])
        system_memory_prompt = (
            "Тест без LLM. Ниже — накопленная история диалога.\n"
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
    from rag.chunker import chunk_dialogue
    from rag.embedder import Embedder
    from rag.indexer import MemoryIndex
    from rag.retriever import Retriever

    parser = argparse.ArgumentParser(description="Simulate memory without LLM and optionally build/search RAG index")
    parser.add_argument("input_file", type=str, nargs="?", help="Path to jsonl dialogues file (required for build)")
    parser.add_argument("--max_preview", type=int, default=500, help="Max characters to print per dialog")
    # RAG options
    parser.add_argument("--build_index", action="store_true", help="Build FAISS index from dialogues")
    parser.add_argument("--index_path", type=str, default="rag/index.faiss", help="Path to FAISS index file")
    parser.add_argument("--query", type=str, default=None, help="Query to search over built/loaded index")
    parser.add_argument("--k", type=int, default=5, help="Top-k results to retrieve")
    parser.add_argument("--embedder_model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    args = parser.parse_args()

    # Always: if input_file provided, show memory simulation preview
    if args.input_file:
        results = simulate_memory_from_file(args.input_file)
        for did, text in results.items():
            print(did)
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
                chunks = chunk_dialogue([base_text], max_tokens=500)
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


