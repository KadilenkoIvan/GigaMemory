from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any

from jinja2 import Template

from ragu.chunker.types import Chunk
from ragu.graph.types import Entity, Relation


@dataclass
class LocalSearchResult:
    """
    Structured retrieval payload returned by local graph search.
    """

    entities: list[Entity] = field(default_factory=list[Entity])
    relations: list[Relation] = field(default_factory=list[Relation])
    summaries: list[Any] = field(default_factory=list[Any])
    chunks: list[Chunk] = field(default_factory=list[Chunk])
    documents_id: list[str] = field(default_factory=list[str])

    def __str__(self) -> str:
        """
        Render search context into prompt-friendly text.

        :return: Human-readable context string.
        """
        _template: Template = Template(dedent(
            """
            **Entities**
            Entity, entity type, entity description
            {%- for e in entities %}
            {{ e.entity_name }}, {{ e.entity_type }}, {{ e.description }}
            {%- endfor %}
            
            **Relations**
            Subject, relation type, object, relation description, rank
            {%- for r in relations %}
            {{ r.subject_name }}, {{ r.relation_type }}, {{ r.object_name }} - {{ r.description }}, {{ r.rank }}
            {%- endfor %}
            
            {%- if summaries %}
            **Summary**
            {%- for s in summaries %}
            {{ s.summary }}
            {%- endfor %}
            {% endif %}
            
            {%- if chunks %}
            **Chunks**
            {%- for c in chunks %}
            {{ c.content }}
            {%- endfor %}
            {% endif %}
            """
            )
        )
        return _template.render(
            entities=self.entities,
            relations=self.relations,
            summaries=self.summaries,
            chunks=self.chunks,
        )


@dataclass
class GlobalSearchResult:
    """
    Aggregated global-search insights with relevance ratings.
    """

    insights: list[Any] = field(default_factory=list[Any])

    def __str__(self) -> str:
        """
        Render insights into prompt-friendly text.

        :return: Human-readable insights string.
        """
        _template: Template = Template(dedent(
            """
            {%- for insight in insights %}
            {{ loop.index}}. Insight: {{ insight.response }}, rating: {{ insight.rating }}
            {%- endfor %}
            """)
        )
        return _template.render(insights=self.insights)


@dataclass
class NaiveSearchResult:
    """
    Retrieval payload for vector-only (naive) search.
    """

    chunks: list[Chunk] = field(default_factory=list[Chunk])
    scores: list[float] = field(default_factory=list[float])
    documents_id: list[str] = field(default_factory=list[str])

    def __str__(self) -> str:
        """
        Render retrieved chunks and scores into prompt-friendly text.

        :return: Human-readable chunk listing.
        """
        _template: Template = Template(dedent(
            """
            **Retrieved Chunks**
            {%- for chunk, score in zip(chunks, scores) %}
            [{{ loop.index }}] (score: {{ "%.3f"|format(score) }})
            {{ chunk.content }}
            {%- endfor %}
            """)
        )
        return _template.render(
            chunks=self.chunks,
            scores=self.scores,
            zip=zip,
        )
    
MixSearchResult = list[NaiveSearchResult | LocalSearchResult | GlobalSearchResult]
