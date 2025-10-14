from typing import List


def _approx_token_len(text: str) -> int:
    # Грубая оценка: 1 токен ≈ 3.5 символа в среднем между EN/RU
    # Без внешних зависимостей; достаточно для разбиения
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def chunk_dialogue(dialogue: List[str], max_tokens: int = 500) -> List[str]:
    """
    Разбивает диалог на чанки по ~max_tokens токенов.
    Если сообщение < max_tokens токенов — отдельный чанк.
    Если больше — делим на части по max_tokens токенов (приближенно по символам).
    Возвращает список строк-чанков.
    """
    chunks: List[str] = []
    if not dialogue:
        return chunks

    # Переводим порог в символы для деления длинных сообщений
    approx_chars_per_token = 3.5
    max_chars = int(max_tokens * approx_chars_per_token)

    for msg in dialogue:
        tok_len = _approx_token_len(msg)
        if tok_len <= max_tokens:
            chunks.append(msg)
            continue

        # Делим по символам на куски ~max_tokens
        for start in range(0, len(msg), max_chars):
            chunks.append(msg[start:start + max_chars])

    return chunks


