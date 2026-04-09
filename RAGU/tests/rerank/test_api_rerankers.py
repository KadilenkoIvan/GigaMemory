from unittest.mock import AsyncMock

import pytest

from ragu.models.scorer import ScorerOpenAI


class _FakeClient:
    def __init__(self):
        self.score = AsyncMock()


@pytest.mark.asyncio
async def test_scorer_openai_forwards_score_call():
    client = _FakeClient()
    client.score.return_value = [(1, 0.9), (0, 0.3)]
    scorer = ScorerOpenAI(
        client=client,
        model_name="bge-reranker-v2",
    )

    result = await scorer.score("query", ["doc1", "doc2"])

    assert result == [(1, 0.9), (0, 0.3)]
    client.score.assert_awaited_once_with(
        model_name="bge-reranker-v2",
        text_1="query",
        text_2=["doc1", "doc2"],
    )


@pytest.mark.asyncio
async def test_scorer_openai_merges_init_kwargs_and_call_kwargs():
    client = _FakeClient()
    client.score.return_value = [(0, 1.0)]
    scorer = ScorerOpenAI(
        client=client,
        model_name="bge-reranker-v2",
        timeout=5,
    )

    result = await scorer.score("q", ["d"], user="u-1")

    assert result == [(0, 1.0)]
    client.score.assert_awaited_once_with(
        model_name="bge-reranker-v2",
        text_1="q",
        text_2=["d"],
        timeout=5,
        user="u-1",
    )
