# Partially based on https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/
from typing import Any, List
from typing_extensions import override

from pydantic import BaseModel

from ragu.common.global_parameters import Settings
from ragu.graph.knowledge_graph import KnowledgeGraph
from ragu.models.embedder import Embedder
from ragu.models.llm import LLM
from ragu.models.scorer import Scorer
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.search_functional import (
    _find_most_related_edges_from_entities,
    _find_most_related_text_unit_from_entities,
    _find_documents_id,
    _find_most_related_community_from_entities,
    _rerank_items,
)
from ragu.search_engine.types import LocalSearchResult
from ragu.storage import Embedding
from ragu.utils.token_truncation import TokenTruncation

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


class LocalSearchEngine(BaseEngine):
    """
    Performs local retrieval-augmented search (RAG) over a knowledge graph.

    The engine:
      1. Retrieves relevant entities for the query.
      2. Retrieves related items (relations, summary and chunks).
      3. Generates a final response

    Reference
    ---------
    Based on: https://github.com/gusye1234/nano-graphrag/blob/main/nano_graphrag/_op.py#L919
    """

    def __init__(
        self,
        llm: LLM,
        knowledge_graph: KnowledgeGraph,
        embedder: Embedder,
        reranker: Scorer | None = None,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        language: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a `LocalSearchEngine`.

        :param llm: LLM used to generate the final answer.
        :param knowledge_graph: Knowledge graph used for entity and relation retrieval.
        :param embedder: Embedding model used for similarity search.
        :param reranker: Optional reranker used to reorder retrieved context sections.
        :param max_context_length: Max tokens allowed for the final context (after truncation).
        :param tokenizer_backend: Tokenizer backend used for token counting/truncation.
        :param tokenizer_model: Model name used by the tokenizer backend.
        :param language: Default output language (fed into prompt template).
        """
        _PROMPTS_NAMES = ["local_search"]
        super().__init__(llm=llm, prompts=_PROMPTS_NAMES, *args, **kwargs)

        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length,
        )

        self.knowledge_graph = knowledge_graph
        self.embedder = embedder
        self.reranker = reranker
        self.language = language if language else Settings.language

    @override
    async def a_search(self, query: str, top_k: int = 20, *args, **kwargs) -> LocalSearchResult:
        """
        Retrieve local graph context for the given query.

        :param query: Input query string.
        :param top_k: Number of top entities to retrieve from the entity vector DB.
        :return: LocalSearchResult containing entities, relations, summaries, chunks, and document ids.
        """
        embedding = await self.embedder.embed_text(query)
        embedding_hits = await self.knowledge_graph.index.entity_vector_db.query(
            Embedding(embedding),
            top_k=top_k,
        )
        entities = await self.knowledge_graph.index.get_entities([hit.id for hit in embedding_hits])
        entities = [e for e in entities if e is not None]

        relations = await _find_most_related_edges_from_entities(entities, self.knowledge_graph)
        relations = [relation for relation in relations if relation is not None]

        relevant_chunks = await _find_most_related_text_unit_from_entities(entities, self.knowledge_graph)
        relevant_chunks = [chunk for chunk in relevant_chunks if chunk is not None]

        summaries = await _find_most_related_community_from_entities(entities, self.knowledge_graph)
        summaries = [summary for summary in summaries if summary is not None]

        entities = await _rerank_items(
            query,
            entities,
            lambda entity: f"{entity.entity_name}\n{entity.entity_type}\n{entity.description}",
            self.reranker,
        )
        relations = await _rerank_items(
            query,
            relations,
            lambda relation: (
                f"{relation.subject_name}\n{relation.relation_type}\n"
                f"{relation.object_name}\n{relation.description}"
            ),
            self.reranker,
        )
        summaries = await _rerank_items(
            query,
            summaries,
            lambda community_summary: community_summary.summary,
            self.reranker,
        )
        relevant_chunks = await _rerank_items(
            query,
            relevant_chunks,
            lambda chunk: chunk.content,
            self.reranker,
        )

        documents_id = await _find_documents_id(entities)

        return LocalSearchResult(
            entities=entities,
            relations=relations,
            summaries=summaries,
            chunks=relevant_chunks,
            documents_id=documents_id,
        )

    @override
    async def a_query(
            self,
            query: str,
            top_k: int = 20,
            use_summary: bool=False,
            use_chunks: bool=False
    ) -> str | BaseModel:
        """
        Execute a local RAG query.

        :param query: User query in natural language.
        :param top_k: Number of entities to retrieve into context.
        :param use_summary: Whether to use summary or not.
        :param use_chunks: Whether to use chunks or not.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        context: LocalSearchResult = await self.a_search(query, top_k)

        if not use_summary:
            context.summaries = []
        if not use_chunks:
            context.chunks = []

        truncated_context: str = self.truncation(str(context))
        instruction: RAGUInstruction = self.get_prompt("local_search")

        rendered_conversations: List[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
        )
        rendered: ChatMessages = rendered_conversations[0]
        return await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model or str, # type: ignore
        ) # type: ignore
