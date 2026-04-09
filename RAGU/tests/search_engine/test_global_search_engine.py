from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ragu.search_engine.global_search import GlobalSearchEngine
from ragu.search_engine.types import GlobalSearchResult


@pytest.mark.asyncio
async def test_global_search_filters_and_sorts_by_rating(monkeypatch, real_kg):
    engine = GlobalSearchEngine(llm=SimpleNamespace(chat_completion=AsyncMock()), knowledge_graph=real_kg)

    monkeypatch.setattr(
        engine,
        "get_meta_responses",
        AsyncMock(
            return_value=[
                {"response": "low", "rating": "1"},
                {"response": "drop", "rating": "0"},
                {"response": "high", "rating": "5"},
            ]
        ),
    )

    result = await engine.a_search("query")
    assert isinstance(result, GlobalSearchResult)
    assert [r["response"] for r in result.insights] == ["high", "low"]


@pytest.mark.asyncio
async def test_global_query_returns_llm_response(monkeypatch, real_kg):
    llm = SimpleNamespace(chat_completion=AsyncMock(return_value="global-answer"))
    engine = GlobalSearchEngine(llm=llm, knowledge_graph=real_kg)
    engine.truncation = lambda s: s
    engine.a_search = AsyncMock(return_value=GlobalSearchResult(insights=[{"response": "x", "rating": "1"}]))

    from ragu.search_engine import global_search as global_module
    monkeypatch.setattr(
        global_module,
        "render",
        lambda messages, **kwargs: [SimpleNamespace(to_openai=lambda: [{"role": "user", "content": "prompt"}])],
    )
    monkeypatch.setattr(
        engine,
        "get_prompt",
        lambda _: SimpleNamespace(messages=[{"role": "user", "content": "{{query}}"}], pydantic_model=None),
    )

    result = await engine.a_query("question")
    assert result == "global-answer"
