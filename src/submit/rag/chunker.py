from typing import List


def _chunk_text_by_tokens(text: str, max_tokens: int, overlap: int) -> List[str]:
    """
    Разбивает один текст на чанки по max_tokens слов с перекрытием overlap.
    Перекрытие задает количество слов, которое повторяется между соседними окнами.
    """
    if not text:
        return [""]

    words = text.split()
    if not words:
        return [text]

    # Страйд — на сколько слов смещаемся вперед
    stride = max(1, max_tokens - max(0, overlap))

    if len(words) <= max_tokens:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += stride

    return chunks


def chunk_dialogue(dialogue: List[str], max_tokens: int = 500, overlap: int = 50) -> List[str]:
    """
    Разбивает диалог на чанки по max_tokens слов (точно по количеству слов),
    используя перекрытие в overlap слов между соседними чанками.
    Если сообщение короче max_tokens — записывается как один чанк.
    Возвращает список строк-чанков.
    """
    chunks: List[str] = []
    if not dialogue:
        return chunks

    for msg in dialogue:
        # Для каждого сообщения применяем нарезку по словам с перекрытием
        msg_chunks = _chunk_text_by_tokens(msg, max_tokens=max_tokens, overlap=overlap)
        chunks.extend(msg_chunks)

    return chunks


