from __future__ import annotations

from typing import Any, List, Tuple, Optional, cast

from pydantic import BaseModel

from ragu.common.logger import logger
from ragu.chunker.types import Chunk
from ragu.common.global_parameters import Settings
from ragu.common.prompts.default_models import ArtifactsModel
from ragu.common.prompts.prompt_storage import RAGUInstruction
from ragu.common.prompts.messages import ChatMessages, render
from ragu.graph.types import Entity, Relation
from ragu.models.llm import LLM
from ragu.triplet.base_artifact_extractor import BaseArtifactExtractor
from ragu.triplet.types import NEREL_ENTITY_TYPES, NEREL_RELATION_TYPES


class ArtifactsExtractorLLM(BaseArtifactExtractor):
    """
    Extracts entities and relations from text chunks using LLM.

    Pipeline:
      1. Render the `artifact_extraction` instruction in batch mode over chunk texts.
      2. Call the LLM to produce structured artifacts for each chunk.
      3. Optionally render and run `artifact_validation` to refine extracted artifacts.
      4. Convert model outputs into Entity/Relation objects, preserving source chunk ids.
    """

    def __init__(
        self,
        llm: LLM,
        do_validation: bool = False,
        language: str | None = None,
        entity_types: Optional[List[str]] = NEREL_ENTITY_TYPES,
        relation_types: Optional[List[str]] = NEREL_RELATION_TYPES,
    ):
        """
        Initialize a new :class:`ArtifactsExtractorLLM`.

        :param client: Language model client for generation and validation.
        :param model_name: Model name to use for generation and validation.
        :param do_validation: Whether to perform additional LLM-based validation of artifacts.
        :param language: Output text language.
        :param entity_types: List of entity types to guide extraction prompts.
        :param relation_types: List of relation types to guide extraction prompts.
        """
        _PROMPTS = ["artifact_extraction", "artifact_validation"]
        super().__init__(prompts=_PROMPTS)

        self.llm = llm
        self.do_validation = do_validation
        self.language = language if language else Settings.language
        self.entity_types = ", ".join(entity_types) if entity_types else None
        self.relation_types = ", ".join(relation_types) if relation_types else None

    async def extract(
        self,
        chunks: List[Chunk],
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[List[Entity], List[Relation]]:
        """
        Extract entities and relations from a collection of chunks.

        Steps:
          1) Batch-render the extraction prompt with `context=<chunk_texts>`,
          2) Generate structured artifacts per chunk,
          3) Optionally validate artifacts against the original context,
          4) Convert artifacts into Entity/Relation objects.

        :param chunks: Iterable of Chunk objects.
        :return: (entities, relations) extracted from all chunks.
        """

        entities_result: List[Entity] = []
        relations_result: List[Relation] = []

        context: List[str] = [chunk.content for chunk in chunks]

        extraction_instruction: RAGUInstruction = self.get_prompt("artifact_extraction")
        assert extraction_instruction.pydantic_model is ArtifactsModel
        extraction_conversations: List[ChatMessages] = render(
            extraction_instruction.messages,
            context=context,
            language=self.language,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
        )

        result_list = await self.llm.batch_chat_completion( # type: ignore
            [c.to_openai() for c in extraction_conversations],
            output_schema=extraction_instruction.pydantic_model or str, # type: ignore
            desc="Extracting a knowledge graph from chunks",
        )
        result_list = cast(list[ArtifactsModel], result_list)

        for artifacts, chunk in zip(result_list, chunks):
            logger.debug(
                f'Got {len(artifacts.entities)} entities'
                f' and {len(artifacts.relations)} relations for chunk'
            )
        
        if self.do_validation:
            validation_instruction: RAGUInstruction = self.get_prompt("artifact_validation")
            assert validation_instruction.pydantic_model is ArtifactsModel

            validation_conversations: List[ChatMessages] = render(
                validation_instruction.messages,
                artifacts=result_list,
                context=context,
                entity_types=self.entity_types,
                relation_types=self.relation_types,
                language=self.language,
            )
            
            result_list = await self.llm.batch_chat_completion( # type: ignore
                [c.to_openai() for c in validation_conversations],
                output_schema=validation_instruction.pydantic_model or str, # type: ignore
                desc="Validation of extracted artifacts",
            )
            result_list = cast(list[ArtifactsModel], result_list)


            for artifacts, chunk in zip(result_list, chunks):
                logger.debug(
                    f'After validation got {len(artifacts.entities)} entities'
                    f' and {len(artifacts.relations)} relations for chunk'
                )

        for artifacts, chunk in zip(result_list, chunks):

            current_chunk_entities: List[Entity] = []

            for entity_model in artifacts.entities:
                entity = Entity(
                    entity_name=entity_model.entity_name,
                    entity_type=entity_model.entity_type or "UNKNOWN",
                    description=entity_model.description,
                    source_chunk_id=[chunk.id],
                    documents_id=[],
                    clusters=[],
                )
                current_chunk_entities.append(entity)

            entities_result.extend(current_chunk_entities)

            # Parse relations
            for relation in artifacts.relations:
                subject_name = relation.source_entity
                object_name = relation.target_entity
                if not (subject_name and object_name):
                    continue
                subject_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == subject_name),
                    None,
                )
                object_entity = next(
                    (e for e in current_chunk_entities if e.entity_name == object_name),
                    None,
                )

                if subject_entity and object_entity:
                    relation = Relation(
                        subject_name=subject_name,
                        object_name=object_name,
                        subject_id=subject_entity.id,
                        object_id=object_entity.id,
                        relation_type=relation.relation_type or "UNKNOWN",
                        description=relation.description,
                        relation_strength=float(relation.relationship_strength), # type: ignore
                        source_chunk_id=[chunk.id],
                    )
                    relations_result.append(relation)

        return entities_result, relations_result
