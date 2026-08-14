import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from src.config import LangfuseSettings
from src.schemas.retrieval import ChunkSearchResponse, ChunkSearchResult
from src.services.langfuse.client import LangfuseTracer
from src.services.rag import RagService

disabled_tracer = LangfuseTracer(LangfuseSettings())


async def collect_stream(service: RagService) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    async for event in service.answer_stream(
        query="How do transformers work?",
        top_k=1,
    ):
        events.append(event)

    return events


def test_answer_retrieves_context_and_generates_grounded_response() -> None:
    chunk = ChunkSearchResult(
        chunk_id="1234.5678v1-0",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        chunk_index=0,
        chunk_text="Transformers use attention to process token relationships.",
        chunk_word_count=8,
        section_title="Introduction",
        title="Test paper",
        abstract="Test abstract",
        authors=["Ada Lovelace"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
        score=0.03,
    )
    hybrid_search_service = Mock()
    hybrid_search_service.search = AsyncMock(
        return_value=ChunkSearchResponse(
            total=1,
            results=[chunk],
        )
    )
    prompt_builder = Mock()
    prompt_builder.build.return_value = "Grounded prompt"
    ollama_client = Mock()
    ollama_client.generate = AsyncMock(return_value="Generated answer")
    service = RagService(
        hybrid_search_service=hybrid_search_service,
        prompt_builder=prompt_builder,
        ollama_client=ollama_client,
        tracer=disabled_tracer,
    )

    response = asyncio.run(
        service.answer(
            query="How do transformers work?",
            top_k=1,
        )
    )

    hybrid_search_service.search.assert_awaited_once_with(
        query="How do transformers work?",
        size=1,
    )
    prompt_builder.build.assert_called_once_with(
        query="How do transformers work?",
        chunks=[chunk],
    )
    ollama_client.generate.assert_awaited_once_with("Grounded prompt")
    assert response.answer == "Generated answer"
    assert response.sources == ["https://arxiv.org/pdf/1234.5678"]
    assert response.chunks_used == 1
    assert response.search_mode == "hybrid"


def test_answer_does_not_call_ollama_without_retrieved_context() -> None:
    hybrid_search_service = Mock()
    hybrid_search_service.search = AsyncMock(
        return_value=ChunkSearchResponse(
            total=0,
            results=[],
        )
    )
    prompt_builder = Mock()
    ollama_client = Mock()
    ollama_client.generate = AsyncMock()
    service = RagService(
        hybrid_search_service=hybrid_search_service,
        prompt_builder=prompt_builder,
        ollama_client=ollama_client,
        tracer=disabled_tracer,
    )

    response = asyncio.run(service.answer("What is quantum attention?"))

    assert response.answer == "I could not find relevant research papers for this question."
    assert response.sources == []
    assert response.chunks_used == 0
    prompt_builder.build.assert_not_called()
    ollama_client.generate.assert_not_awaited()


def test_answer_stream_returns_metadata_chunks_and_complete_answer() -> None:
    chunk = ChunkSearchResult(
        chunk_id="1234.5678v1-0",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        chunk_index=0,
        chunk_text="Transformers use attention to process token relationships.",
        chunk_word_count=8,
        section_title="Introduction",
        title="Test paper",
        abstract="Test abstract",
        authors=["Ada Lovelace"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
        score=0.03,
    )
    hybrid_search_service = Mock()
    hybrid_search_service.search = AsyncMock(return_value=ChunkSearchResponse(total=1, results=[chunk]))
    prompt_builder = Mock()
    prompt_builder.build.return_value = "Grounded prompt"
    ollama_client = Mock()

    async def generate_stream(_: str):
        yield "Generated "
        yield "answer"

    ollama_client.generate_stream = generate_stream
    service = RagService(
        hybrid_search_service=hybrid_search_service,
        prompt_builder=prompt_builder,
        ollama_client=ollama_client,
        tracer=disabled_tracer,
    )

    events = asyncio.run(collect_stream(service))

    assert events == [
        {
            "sources": ["https://arxiv.org/pdf/1234.5678"],
            "chunks_used": 1,
            "search_mode": "hybrid",
        },
        {"chunk": "Generated "},
        {"chunk": "answer"},
        {"answer": "Generated answer", "done": True},
    ]
