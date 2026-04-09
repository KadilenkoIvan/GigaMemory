import asyncio
from typing import Any, List

from pydantic import BaseModel

from ragu.common.base import RaguGenerativeModule
from ragu.common.global_parameters import Settings
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.types import GlobalSearchResult
from ragu.utils.token_truncation import TokenTruncation

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


class GlobalSearchEngine(BaseEngine, RaguGenerativeModule):
    """
    Executes global retrieval-augmented search (RAG) across the entire knowledge graph.

    Unlike :class:`LocalSearchEngine`, this engine operates at the level of
    *community summaries*, aggregating and ranking high-level semantic clusters
    before generating a global synthesis via the language model.
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        language: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a new `GlobalSearchEngine`.

        :param client: Language model client for generation.
        :param knowledge_graph: Knowledge graph providing access to community-level summaries.
        :param max_context_length: Maximum number of tokens allowed in the truncated context.
        :param tokenizer_backend: Tokenizer backend used for token counting (default: ``"tiktoken"``).
        :param tokenizer_model: Model name for tokenizer calibration (default: ``"gpt-4"``).
        """
        _PROMPTS = ["global_search_context", "global_search"]
        super().__init__(llm=llm, prompts=_PROMPTS, *args, **kwargs)

        self.llm = llm
        self.knowledge_graph = knowledge_graph
        self.language = language if language else Settings.language

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length,
        )

    async def a_search(self, query: str, *args, **kwargs) -> GlobalSearchResult:
        """
        Perform a global semantic search across all communities in the knowledge graph.

        This method retrieves all available community summaries, sends them to the LLM
        for meta-evaluation, filters out low-rated responses, and returns a ranked
        concatenation of the top relevant community insights.

        :param query: The input natural language query.
        :return: Concatenated responses from the top-rated communities.
        """

        communities_ids = await self.knowledge_graph.index.community_summary_kv_storage.all_keys()
        communities = await self.knowledge_graph.index.community_summary_kv_storage.get_by_ids(communities_ids)
        communities = [c for c in communities if c is not None]

        responses = await self.get_meta_responses(query, communities)

        responses = [r for r in responses if int(r.get("rating", 0)) > 0]
        responses = sorted(responses, key=lambda x: int(x.get("rating", 0)), reverse=True)

        return GlobalSearchResult(responses)

    async def get_meta_responses(self, query: str, context: List[str]) -> List[dict]:
        """
        Generate and evaluate meta-responses for each community summary.

        The model receives the full list of community summaries and scores each
        according to relevance to the given query. Only positively rated responses
        are retained.

        :param query: The user query used to assess community relevance.
        :param context: A list of community summary texts to evaluate.
        :return: A list of structured responses with fields such as ``response`` and ``rating``.
        """
        instruction: RAGUInstruction = self.get_prompt("global_search_context")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=context,
            language=self.language,
        )

        meta_responses = await asyncio.gather(*[
            self.llm.chat_completion(
                conversation=rendered.to_openai(),
                output_schema=instruction.pydantic_model or str, # type: ignore
            )
            for rendered in rendered_list
        ]) # type: ignore

        return [r.model_dump() for r in meta_responses if r]

    async def a_query(self, query: str) -> str | BaseModel:
        """
        Execute a full global retrieval-augmented generation query.

        - Retrieves all community-level insights.
        - Generates a final global answer.

        :param query: The natural language query from the user.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        context = await self.a_search(query)
        truncated_context: str = self.truncation(str(context))

        instruction: RAGUInstruction = self.get_prompt("global_search")

        rendered_list: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered = rendered_list[0]
        return await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model or str, # type: ignore
        ) # type: ignore

