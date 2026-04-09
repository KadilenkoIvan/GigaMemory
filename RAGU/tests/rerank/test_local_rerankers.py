import pytest

from ragu.models.scorer import ScorerCrossEncoder


class _ArrayLike:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeCrossEncoder:
    def __init__(self):
        self.predict_calls = []
        self.outputs = []

    def predict(self, batch, show_progress_bar=False):
        self.predict_calls.append((list(batch), show_progress_bar))
        return self.outputs.pop(0)


@pytest.mark.asyncio
async def test_cross_encoder_score_empty_input():
    model = _FakeCrossEncoder()
    scorer = ScorerCrossEncoder(model=model, batch_size=2)

    result = await scorer.score("query", [])
    assert result == []
    assert model.predict_calls == []


@pytest.mark.asyncio
async def test_cross_encoder_score_sorts_and_top_k():
    model = _FakeCrossEncoder()
    model.outputs = [[0.1, 0.9, 0.5]]
    scorer = ScorerCrossEncoder(model=model, batch_size=16)

    result = await scorer.score("q", ["d0", "d1", "d2"], top_k=2)

    assert result == [(1, 0.9), (2, 0.5), (0, 0.1)]
    assert len(model.predict_calls) == 1
    assert model.predict_calls[0][0] == [("q", "d0"), ("q", "d1"), ("q", "d2")]


@pytest.mark.asyncio
async def test_cross_encoder_score_batches_and_handles_tolist_output():
    model = _FakeCrossEncoder()
    model.outputs = [_ArrayLike([0.2, 0.6]), _ArrayLike([0.4])]
    scorer = ScorerCrossEncoder(model=model, batch_size=2)

    result = await scorer.score("q", ["a", "bbb", "cc"], batch_size=2)

    assert len(model.predict_calls) == 2
    assert model.predict_calls[0][0] == [("q", "a"), ("q", "bbb")]
    assert model.predict_calls[1][0] == [("q", "cc")]
    assert result == [(1, 0.6), (2, 0.4), (0, 0.2)]
