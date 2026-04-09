from unittest.mock import AsyncMock

import pytest

from ragu.models.embedder import EmbedderOpenAI


class _FakeClient:
    def __init__(self):
        self.embed_text = AsyncMock()


def test_embedder_openai_init_sets_fields():
    client = _FakeClient()
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=5,
    )

    assert embedder.client is client
    assert embedder.model_name == "text-embedding-3-small"
    assert embedder.dim == 5


@pytest.mark.asyncio
async def test_embedder_openai_forwards_embed_text_call():
    client = _FakeClient()
    client.embed_text.return_value = [0.1, 0.2, 0.3]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=3,
    )

    result = await embedder.embed_text("hello")

    assert result == [0.1, 0.2, 0.3]
    client.embed_text.assert_awaited_once_with(
        model_name="text-embedding-3-small",
        text="hello",
    )


@pytest.mark.asyncio
async def test_embedder_openai_merges_init_kwargs_and_call_kwargs():
    client = _FakeClient()
    client.embed_text.return_value = [0.5, 0.6]
    embedder = EmbedderOpenAI(
        client=client,
        model_name="text-embedding-3-small",
        dim=2,
        timeout=3,
    )

    result = await embedder.embed_text("hi", user="u-1")

    assert result == [0.5, 0.6]
    client.embed_text.assert_awaited_once_with(
        model_name="text-embedding-3-small",
        text="hi",
        timeout=3,
        user="u-1",
    )

