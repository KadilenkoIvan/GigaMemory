from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Sequence, TypeVar
from tqdm.asyncio import tqdm_asyncio
from typing_extensions import override

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel

from ragu.common.logger import logger
from ragu.models.openai import CachedAsyncOpenAI


T = TypeVar('T', BaseModel, str)


class LLM(ABC):
    """LLM interface to support various backends (openai, transformers etc.)."""

    @abstractmethod
    async def chat_completion(
        self,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        """
        Returns one chat completion response.

        :param conversation: OpenAI-format conversation messages.
        :param output_schema: ``str`` for plain text or ``BaseModel`` subclass
            for structured output.
        :param kwargs: Backend-specific options (for example temperature).
        :returns: Response value as plain text or validated schema instance.
        """

    async def batch_chat_completion(
        self,
        conversations: list[list[ChatCompletionMessageParam]],
        output_schema: type[T] = str,
        desc: str | None = None,
        **kwargs: Any,
    ) -> Sequence[T]:
        """
        Runs multiple :meth:`chat_completion` calls concurrently.

        :param conversations: List of conversation message lists.
        :param output_schema: Output schema applied to each call.
        :param desc: Optional tqdm progress description.
        :param kwargs: Extra kwargs forwarded to each call.
        :returns: Responses in the same order as input conversations.
        """
        logger.debug(f'Calling batch_chat_completion with size {len(conversations)}')
        return await tqdm_asyncio.gather(*[ # type: ignore
            self.chat_completion(
                conversation=conversation,
                output_schema=output_schema,
                **kwargs,
            )
            for conversation in conversations
        ], desc=desc)
        


class LLMOpenAI(LLM):
    """Adapts :class:`CachedAsyncOpenAI` to the :class:`LLM` interface."""
    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        **kwargs: Any,
    ):
        """
        Adapts :class:`CachedAsyncOpenAI` to the :class:`LLM` interface.

        :param client: OpenAI-compatible backend client.
        :param model_name: Model identifier passed to backend calls.
        :param kwargs: Default kwargs merged into each ``chat_completion`` call.
        
        Example:
        ```
        llm = LLMOpenAI(
            client=CachedAsyncOpenAI(),
            model_name='gpt-5',
        )
        ```
        """
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
    
    @override
    async def chat_completion(
        self,
        conversation: list[ChatCompletionMessageParam],
        output_schema: type[T] = str,
        **kwargs: Any,
    ) -> T:
        """
        Forwards chat completion request.

        :param conversation: OpenAI-format conversation messages.
        :param output_schema: ``str`` or ``BaseModel`` subclass.
        :param kwargs: Per-call kwargs merged with constructor kwargs.
            Per-call values override constructor defaults on conflicts.
        :returns: Response value as plain text or validated schema instance.
        """
        return await self.client.chat_completion(
            model_name=self.model_name,
            conversation=conversation,
            output_schema=output_schema,
            **(self.kwargs | kwargs),
        )
