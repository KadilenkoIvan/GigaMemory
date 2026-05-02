import asyncio
from typing import Any, List
from typing_extensions import override

from pydantic import BaseModel

from ragu.common.global_parameters import Settings
from ragu.models.llm import LLM
from ragu.search_engine.base_engine import BaseEngine
from ragu.search_engine.types import MixSearchResult
from ragu.utils.token_truncation import TokenTruncation

from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render


class MixSearchEngine(BaseEngine):
    """
    Performs ensemble retrieval-augmented search over multiple engines.

    The engine supports two synthesis modes:
      1. Retrieve raw contexts from each child engine and combine them into one final answer.
      2. Retrieve a full answer from each child engine and combine those answers into one final answer.

    Child engines are executed in the order provided at construction time.
    """

    def __init__(
        self,
        llm: LLM,
        engines: List[BaseEngine],
        allow_partial_failures: bool = True,
        max_context_length: int = 30_000,
        tokenizer_backend: str = "tiktoken",
        tokenizer_model: str = "gpt-4",
        language: str | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        """
        Initialize a `MixSearchEngine`.

        :param llm: LLM used to generate the final synthesized answer.
        :param engines: Ordered list of child engines used for retrieval or answer ensembling.
        :param allow_partial_failures: Whether to tolerate failures from individual child engines.
                                       Failed engines yield ``None`` in the ordered result list.
        :param max_context_length: Max tokens allowed for the synthesized context after truncation.
        :param tokenizer_backend: Tokenizer backend used for token truncation.
        :param tokenizer_model: Model name used by the tokenizer backend.
        :param language: Default output language.
        """
        prompts = ["mix_search_context", "mix_search"]
        super().__init__(llm=llm, prompts=prompts, *args, **kwargs)

        self.engines = engines
        if not self.engines:
            raise ValueError("MixSearchEngine requires at least one child engine")

        self.allow_partial_failures = allow_partial_failures
        self.language = language if language else Settings.language
        self.truncation = TokenTruncation(
            tokenizer_model,
            tokenizer_backend,
            max_context_length,
        )

    async def _search_all(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any | None]:
        """
        Execute ``a_search`` on each child engine.

        :param query: Input query string.
        :return: Ordered list of per-engine search contexts. Failed engines return ``None`` when
                 ``allow_partial_failures=True``.
        """
        tasks = [
            engine.a_search(query, *args, **kwargs)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts: list[Any | None] = []
        for result in results:
            if isinstance(result, Exception):
                if not self.allow_partial_failures:
                    raise result
                contexts.append(None)
                continue
            contexts.append(result)

        if not any(context is not None for context in contexts):
            raise RuntimeError("MixSearchEngine could not retrieve context from any child engine")

        return contexts

    async def _query_all(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> list[Any | None]:
        """
        Execute ``a_query`` on each child engine.

        :param query: Input query string.
        :return: Ordered list of per-engine answers. Failed engines return ``None`` when
                 ``allow_partial_failures=True``.
        """
        tasks = [
            engine.a_query(query, *args, **kwargs)
            for engine in self.engines
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        contexts: list[Any | None] = []
        for result in results:
            if isinstance(result, Exception):
                if not self.allow_partial_failures:
                    raise result
                contexts.append(None)
                continue
            contexts.append(result)

        if not any(context is not None for context in contexts):
            raise RuntimeError("MixSearchEngine could not retrieve context from any child engine")

        return contexts

    @override
    async def a_search(self, query: str, *args: Any, **kwargs: Any) -> MixSearchResult:
        """
        Retrieve raw contexts from all child engines.

        :param query: Input query string.
        :return: Ordered list of per-engine search contexts matching the engine order passed to
                 the constructor. Failed engines are represented as ``None`` when partial failures
                 are enabled.
        """
        return await self._search_all(query, *args, **kwargs)

    @override
    async def a_query(
        self,
        query: str,
        *args: Any,
        ensemble_responses: bool = False,
        **kwargs: Any,
    ) -> str | BaseModel:
        """
        Execute an ensemble query across child engines.

        When ``ensemble_responses=False``, this method retrieves raw contexts from each child
        engine via ``a_search`` and synthesizes one final answer from the combined contexts.

        When ``ensemble_responses=True``, this method first retrieves a full answer from each child
        engine via ``a_query`` and then synthesizes one final answer from those per-engine answers.

        :param query: Input query string.
        :param ensemble_responses: Whether to ensemble child-engine answers instead of child-engine
                                   search contexts.
        :return: Generated answer as a string or Pydantic model when a response schema is set.
        """
        results = await (
            self._query_all(query, *args, **kwargs)
            if ensemble_responses
            else self._search_all(query, *args, **kwargs)
        )
        section_label = "Response" if ensemble_responses else "Context"
        context_instruction: RAGUInstruction = self.get_prompt("mix_search_context")
        rendered_context_list: list[ChatMessages] = render(
            context_instruction.messages,
            payload={"entries": results},
            section_label=section_label,
        )
        rendered_context = rendered_context_list[0]
        formatted_context = rendered_context.messages[0].content
        if not formatted_context:
            raise RuntimeError("MixSearchEngine could not build synthesis input from child engines")

        truncated_context = self.truncation(formatted_context)

        instruction: RAGUInstruction = self.get_prompt("mix_search")
        rendered_list: list[ChatMessages] = render(
            instruction.messages,
            query=query,
            context=truncated_context,
            language=self.language,
            ensemble_responses=ensemble_responses,
            section_label=section_label.lower(),
        )
        rendered = rendered_list[0]

        return await self.llm.chat_completion(
            conversation=rendered.to_openai(),
            output_schema=instruction.pydantic_model or str,  # type: ignore[arg-type]
        )  # type: ignore[return-value]
