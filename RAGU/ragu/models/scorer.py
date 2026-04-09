from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar
from typing_extensions import override
from tqdm.asyncio import tqdm_asyncio

from pydantic import BaseModel
if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from ragu.common.logger import logger
from ragu.common.batch_generator import BatchGenerator
from ragu.models.openai import CachedAsyncOpenAI


T = TypeVar('T', BaseModel, str)


class Scorer(ABC):
    """Scorer interface to support various backends (openai, transformers etc.)."""

    @abstractmethod
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Scores ``text_1`` against candidate texts in ``text_2`` list.

        :param text_1: Source or query text.
        :param text_2: Candidate texts to score against ``text_1``.
        :param kwargs: Backend-specific optional arguments.
        :returns: List of ``(index, score)`` tuples sorted by score descending.
        """

    async def batch_score(
        self,
        texts: list[tuple[str, list[str]]],
        desc: str | None = None,
        **kwargs: Any,
    ) -> list[list[tuple[int, float]]]:
        """
        Runs multiple `score` calls concurrently.

        :param texts: List of ``(text_1, text_2_list)`` pairs.
        :param desc: Optional tqdm progress description.
        :param kwargs: Extra kwargs 
        :returns: One scored result list per input item.
        """
        logger.debug(f'Calling batch_score with size {len(texts)}')
        return await tqdm_asyncio.gather(*[ # type: ignore
            self.score(
                text_1=text_1,
                text_2=text_2,
                **kwargs,
            )
            for text_1, text_2 in texts
        ], desc=desc)


class ScorerOpenAI(Scorer):
    """
    Adapts :class:`CachedAsyncOpenAI` to the :class:`Scorer` interface.

    :param client: OpenAI-compatible backend client.
    :param model_name: Model identifier passed to backend score calls.
    :param dim: Reserved value currently stored but not used by this class.
    :param kwargs: Default kwargs merged into each ``score`` call.
    
    Example:
    ```
    scorer = ScorerOpenAI(
        client=CachedAsyncOpenAI(),
        model_name='my-model',
    )
    ```
    """
    
    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        **kwargs: Any,
    ):
        self.client = client
        self.model_name = model_name
        self.kwargs = kwargs
    
    @override
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Forwards score request to backend using a fixed model name.

        :param text_1: Source or query text.
        :param text_2: Candidate texts to score.
        :param kwargs: Per-call kwargs merged with constructor kwargs.
            Per-call values override constructor defaults on conflicts.
        :returns: List of ``(index, score)`` tuples sorted by score descending.
        """
        return await self.client.score(
            model_name=self.model_name,
            text_1=text_1,
            text_2=text_2,
            **(self.kwargs | kwargs),
        )

class ScorerCrossEncoder(Scorer):
    """
    Scorer (reranker) based on Sentence Transformers ``CrossEncoder``.

    :param model: CrossEncoder-compatible model instance.
    :param batch_size: Default batch size for inference.
    """

    def __init__(self, model: CrossEncoder, batch_size: int = 16):
        self.model = model
        self.batch_size = batch_size
    
    @override
    async def score(
        self,
        text_1: str,
        text_2: list[str],
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> list[tuple[int, float]]:
        """
        Scores candidates using batched CrossEncoder inference.

        :param text_1: Source or query text.
        :param text_2: Candidate texts to score.
        :param batch_size: Optional per-call batch size override.
        :param kwargs: Reserved for interface compatibility.
        :returns: ``(index, score)`` tuples where index maps to original
            ``text_2`` position, sorted by score descending.
        """
        pairs = [(text_1, doc) for doc in text_2]
        batch_generator = BatchGenerator(pairs, batch_size=batch_size or self.batch_size)
        
        scores_list: list[float] = []
        for batch in batch_generator.get_batches():
            batch_scores = self.model.predict(batch, show_progress_bar=False) # type: ignore
            scores_list.extend(batch_scores.tolist() if hasattr(batch_scores, 'tolist') else list(batch_scores)) # type: ignore

        indexed_scores = [(i, score) for i, score in enumerate(scores_list)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores
