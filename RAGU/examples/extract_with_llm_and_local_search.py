import asyncio

from ragu import (
    ArtifactsExtractorLLM,
    BuilderArguments,
    KnowledgeGraph,
    LocalSearchEngine,
    Settings,
    SimpleChunker,
)
from ragu.embedder import OpenAIEmbedder
from ragu.models import OpenAIClient
from ragu.utils.ragu_utils import read_text_from_files


EMBEDDER_MODEL_NAME = "..."
LLM_MODEL_NAME = "..."
BASE_URL = "..."
API_KEY = "..."


async def main():
    # Configure working directory and language
    Settings.storage_folder = "ragu_working_dir/example_knowledge_graph"
    Settings.language = "russian"

    # Load documents
    docs = read_text_from_files("examples/data/ru")

    # Initialize chunker
    chunker = SimpleChunker(max_chunk_size=1000)

    # Set up LLM client
    client = OpenAIClient(
        model_name=LLM_MODEL_NAME,
        base_url=BASE_URL,
        api_token=API_KEY,
        max_requests_per_second=1,
        max_requests_per_minute=60,
        cache_flush_every=10,
    )

    # Set up artifact extractor
    artifact_extractor = ArtifactsExtractorLLM(llm=client, do_validation=False)

    # Set up embedder
    embedder = OpenAIEmbedder(
        model_name=EMBEDDER_MODEL_NAME,
        base_url=BASE_URL,
        api_token=API_KEY,
        dim=3072,
        max_requests_per_second=1,
        max_requests_per_minute=60,
        use_cache=True,
    )

    # Configure graph builder
    builder_settings = BuilderArguments(
        use_llm_summarization=True,
        vectorize_chunks=True,
    )

    # Build knowledge graph
    knowledge_graph = KnowledgeGraph(
        llm=client,
        embedder=embedder,
        chunker=chunker,
        artifact_extractor=artifact_extractor,
        builder_settings=builder_settings,
    )
    await knowledge_graph.build_from_docs(docs)

    # Set up search engine
    search_engine = LocalSearchEngine(
        client,
        knowledge_graph,
        embedder,
        tokenizer_model="gpt-4o-mini",
    )

    # Run local search
    questions = [
        "Кто написал гимн Норвегии?",
        "Шум, издаваемый ЭТИМИ ПАУКООБРАЗНЫМИ, слышен за пять километров. Отсюда и их название.",
        "Как переводится роман 'Ка́мо гряде́ши, Го́споди?'"
    ]

    for question in questions:
        print(await search_engine.a_query(question))

if __name__ == "__main__":
    asyncio.run(main())
