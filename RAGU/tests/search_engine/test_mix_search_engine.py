from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.common.prompts.messages import ChatMessages, UserMessage
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.mix_search import MixSearchEngine
from ragu.search_engine.types import GlobalSearchResult, NaiveSearchResult


class DummyEngine(BaseEngine):
    def __init__(self, result=None, error: Exception | None = None):
        super().__init__(llm=SimpleNamespace(chat_completion=AsyncMock()), prompts={})
        self._result = result
        self._error = error

    async def a_search(self, query, *args, **kwargs):
        if self._error is not None:
            raise self._error
        return self._result

    async def a_query(self, query: str):
        return "unused"


@pytest.mark.asyncio
async def test_mix_search_collects_contexts_in_engine_order():
    naive_result = NaiveSearchResult(chunks=[], scores=[], documents_id=["doc-1"])
    global_result = GlobalSearchResult(insights=[{"response": "x", "rating": "3"}])
    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[
            DummyEngine(result=naive_result),
            DummyEngine(result=global_result),
        ],
    )

    result = await engine.a_search("query")

    assert isinstance(result, list)
    assert result == [naive_result, global_result]


@pytest.mark.asyncio
async def test_mix_search_records_partial_failures():
    ok_result = NaiveSearchResult()
    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[
            DummyEngine(result=ok_result),
            DummyEngine(error=ValueError("broken child engine")),
        ],
        allow_partial_failures=True,
    )

    result = await engine.a_search("query")

    assert result == [ok_result, None]


@pytest.mark.asyncio
async def test_mix_search_raises_when_all_engines_fail():
    engine = MixSearchEngine(
        llm=SimpleNamespace(chat_completion=AsyncMock()),
        engines=[
            DummyEngine(error=ValueError("first failure")),
            DummyEngine(error=RuntimeError("second failure")),
        ],
        allow_partial_failures=True,
    )

    with pytest.raises(RuntimeError, match="could not retrieve context"):
        await engine.a_search("query")


@pytest.mark.asyncio
async def test_mix_query_returns_llm_response(monkeypatch):
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value="mix-answer"))
    engine = MixSearchEngine(
        llm=llm,
        engines=[DummyEngine(result=NaiveSearchResult())],
    )
    engine.truncation = lambda s: s
    engine._search_all = AsyncMock(return_value=[NaiveSearchResult()])
    engine._query_all = AsyncMock()

    from ragu.search_engine import mix_search as mix_module
    monkeypatch.setattr(
        mix_module,
        "render",
        lambda messages, **kwargs: [ChatMessages.from_messages([UserMessage(content="prompt")])],
    )
    original_get_prompt = engine.get_prompt
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda prompt_name: (
            SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None)
            if prompt_name == "mix_search"
            else original_get_prompt(prompt_name)
        ),
    )

    result = await engine.a_query("question")
    assert result == "mix-answer"
    engine._search_all.assert_awaited_once_with("question")
    engine._query_all.assert_not_awaited()


@pytest.mark.asyncio
async def test_mix_query_can_ensemble_engine_responses(monkeypatch):
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value="ensemble-answer"))
    engine = MixSearchEngine(
        llm=llm,
        engines=[DummyEngine(result=NaiveSearchResult())],
    )
    engine.truncation = lambda s: s
    engine._search_all = AsyncMock()
    engine._query_all = AsyncMock(return_value=["engine-answer"])

    from ragu.search_engine import mix_search as mix_module
    monkeypatch.setattr(
        mix_module,
        "render",
        lambda messages, **kwargs: [ChatMessages.from_messages([UserMessage(content="prompt")])],
    )
    original_get_prompt = engine.get_prompt
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda prompt_name: (
            SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None)
            if prompt_name == "mix_search"
            else original_get_prompt(prompt_name)
        ),
    )

    result = await engine.a_query("question", ensemble_responses=True)
    assert result == "ensemble-answer"
    engine._query_all.assert_awaited_once_with("question")
    engine._search_all.assert_not_awaited()
