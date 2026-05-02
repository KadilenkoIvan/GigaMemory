import pytest

from ragu.models.scorer import Scorer


class DummyScorer(Scorer):
    async def score(self, text_1: str, text_2: list[str], **kwargs) -> list[tuple[int, float]]:
        scores = [(i, float(len(doc))) for i, doc in enumerate(text_2)]
        scores.sort(key=lambda item: item[1], reverse=True)
        top_k = kwargs.get("top_k")
        if top_k is not None:
            scores = scores[:top_k]
        return scores


@pytest.mark.asyncio
async def test_dummy_scorer_orders_by_score_desc():
    scorer = DummyScorer()
    result = await scorer.score("query", ["a", "bbbb", "cc"])
    assert result == [(1, 4.0), (2, 2.0), (0, 1.0)]


@pytest.mark.asyncio
async def test_dummy_scorer_applies_top_k():
    scorer = DummyScorer()
    result = await scorer.score("query", ["a", "bbbb", "cc"], top_k=2)
    assert result == [(1, 4.0), (2, 2.0)]
