from rag.chunker import chunk_dialogue
from rag.embedder import Embedder
from rag.indexer import MemoryIndex
from rag.retriever import Retriever


def demo():
    # Пример диалога как список строк (user/assistant вперемешку)
    dialogue = [
        "Привет, меня зовут Иван.",
        "Привет, Иван! Чем могу помочь?",
        "У меня есть кот по имени Барсик и собака Лайка.",
        "Запомню это!",
        "Теперь Лайка очень любит мою жену.",
    ]

    chunks = chunk_dialogue(dialogue, max_tokens=500)

    # convert to structured entries
    entries = []
    for i, ch in enumerate(chunks):
        entries.append({
            "dialogue_id": "demo",
            "chunk_id": str(i),
            "role": "mixed",
            "text": ch,
        })

    embedder = Embedder()
    embeddings = embedder.encode([e["text"] for e in entries])

    dim = len(embeddings[0]) if embeddings else 384
    index = MemoryIndex(dim=dim)
    index.add_entries(embeddings, entries)

    retriever = Retriever(index=index, embedder=embedder)
    results = retriever.search("Как зовут моего кота?", k=3)

    for r in results:
        print(r)


if __name__ == "__main__":
    demo()


