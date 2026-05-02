from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from ragu.common.base import RaguGenerativeModule
from ragu.common.prompts.default_models import GlobalSearchContextModel
from ragu.models.llm import LLM
from ragu.search_engine.types import (
    GlobalSearchResult,
    LocalSearchResult,
    MixSearchResult,
    NaiveSearchResult,
)
from ragu.utils.ragu_utils import always_get_an_event_loop


class BaseEngine(RaguGenerativeModule, ABC):
    """
    Base interface for RAGU query/search engines.

    Concrete engines implement retrieval (a_search method) and answer generation
    (a_query method) on top of a knowledge graph.
    """

    def __init__(self, llm: LLM, *args: Any, **kwargs: Any):
        """
        Initialize engine with an LLM client.

        :param client: LLM client.
        """
        super().__init__(*args, **kwargs)
        self.llm = llm

    @abstractmethod
    async def a_search(
        self,
        query,
        *args,
        **kwargs,
    ) -> Any:
        """
        Retrieve context relevant to a query.

        :param query: Input query string.
        :return: Engine-specific retrieval result payload.
        """
        pass

    @abstractmethod
    async def a_query(self, query: str) -> str | BaseModel:
        """
        Execute full query flow and return answer.

        :param query: Input query string.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        pass

    async def query(self, query: str) -> str | BaseModel:
        """
        Synchronous wrapper for ``a_query``.

        :param query: Input query string.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_query(query)
        )

    async def search(
        self,
        query,
        *args,
        **kwargs,
    ) -> NaiveSearchResult | LocalSearchResult | GlobalSearchResult | GlobalSearchContextModel | MixSearchResult:
        """
        Synchronous wrapper for ``a_search``.

        :param query: Input query string.
        :return: Engine-specific retrieval result payload.
        """
        loop = always_get_an_event_loop()
        return loop.run_until_complete(
            self.a_search(query, *args, **kwargs)
        )
