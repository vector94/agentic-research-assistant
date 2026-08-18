import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from src.config import LangfuseSettings
from src.schemas.retrieval import ChunkSearchResponse, ChunkSearchResult
from src.services.agents import (
    AgentContext,
    AgenticRagService,
    AgentPromptBuilder,
    DocumentGrade,
    GuardrailResult,
)
from src.services.langfuse.client import LangfuseTracer

disabled_tracer = LangfuseTracer(LangfuseSettings())


def make_chunk(chunk_id: str, text: str) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        chunk_index=0,
        chunk_text=text,
        chunk_word_count=len(text.split()),
        section_title="Introduction",
        title="Test paper",
        abstract="Test abstract",
        authors=["Ada Lovelace"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
        score=0.03,
    )


def make_service(
    search_service: Mock,
    ollama_client: Mock,
    answer_prompt_builder: Mock | None = None,
) -> AgenticRagService:
    ollama_client.model = "test-model"
    context = AgentContext(
        search_service=search_service,
        ollama_client=ollama_client,
        prompt_builder=AgentPromptBuilder(),
        tracer=disabled_tracer,
        answer_prompt_builder=answer_prompt_builder or Mock(),
    )
    return AgenticRagService(context)


def test_answer_stops_after_guardrail_for_out_of_scope_question() -> None:
    search_service = Mock()
    search_service.search = AsyncMock()
    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock(
        return_value=GuardrailResult(score=10, reason="Not a research question"),
    )
    ollama_client.generate = AsyncMock()
    service = make_service(search_service, ollama_client)

    response = asyncio.run(service.answer("What is the weather?"))

    search_service.search.assert_not_awaited()
    ollama_client.generate.assert_not_awaited()
    assert response.sources == []
    assert response.chunks_used == 0
    assert response.retrieval_attempts == 0
    assert "outside that scope" in response.answer
    assert response.reasoning_steps


def test_answer_generates_from_relevant_retrieved_chunks() -> None:
    chunk = make_chunk("chunk-1", "Transformers use attention.")
    search_service = Mock()
    search_service.search = AsyncMock(
        return_value=ChunkSearchResponse(total=1, results=[chunk]),
    )
    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock(
        side_effect=[
            GuardrailResult(score=90, reason="Research question"),
            DocumentGrade(is_relevant=True, reason="The excerpt answers it"),
        ],
    )
    ollama_client.generate = AsyncMock(return_value="Attention relates every token to other tokens.")
    answer_prompt_builder = Mock()
    answer_prompt_builder.build.return_value = "Answer prompt"
    service = make_service(search_service, ollama_client, answer_prompt_builder)

    response = asyncio.run(service.answer("How do transformers work?", top_k=1))

    search_service.search.assert_awaited_once_with(
        query="How do transformers work?",
        size=1,
    )
    assert response.answer == "Attention relates every token to other tokens."
    assert response.sources == [chunk.pdf_url]
    assert response.chunks_used == 1
    assert response.retrieval_attempts == 1


def test_answer_rewrites_query_and_retries_retrieval() -> None:
    first_chunk = make_chunk("chunk-1", "A weakly related excerpt.")
    second_chunk = make_chunk("chunk-2", "A directly relevant transformer excerpt.")
    search_service = Mock()
    search_service.search = AsyncMock(
        side_effect=[
            ChunkSearchResponse(total=1, results=[first_chunk]),
            ChunkSearchResponse(total=1, results=[second_chunk]),
        ],
    )
    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock(
        side_effect=[
            GuardrailResult(score=90, reason="Research question"),
            DocumentGrade(is_relevant=False, reason="The first excerpt is too broad"),
            DocumentGrade(is_relevant=True, reason="The second excerpt answers it"),
        ],
    )
    ollama_client.generate = AsyncMock(
        side_effect=[
            "transformer self-attention token relationships",
            "Grounded answer after retry",
        ],
    )
    answer_prompt_builder = Mock()
    answer_prompt_builder.build.return_value = "Answer prompt"
    service = make_service(search_service, ollama_client, answer_prompt_builder)

    response = asyncio.run(service.answer("How do transformers work?", top_k=2))

    assert search_service.search.await_args_list[0].kwargs == {
        "query": "How do transformers work?",
        "size": 2,
    }
    assert search_service.search.await_args_list[1].kwargs == {
        "query": "transformer self-attention token relationships",
        "size": 2,
    }
    assert response.answer == "Grounded answer after retry"
    assert response.retrieval_attempts == 2
    assert response.reasoning_steps


def test_answer_returns_fallback_after_retrieval_attempts_are_exhausted() -> None:
    search_service = Mock()
    search_service.search = AsyncMock(
        return_value=ChunkSearchResponse(total=0, results=[]),
    )
    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock(
        return_value=GuardrailResult(score=90, reason="Research question"),
    )
    ollama_client.generate = AsyncMock(return_value="more specific research query")
    service = make_service(search_service, ollama_client)

    response = asyncio.run(service.answer("What is a missing research topic?"))

    assert search_service.search.await_count == 2
    ollama_client.generate.assert_awaited_once()
    assert response.answer == "I could not find relevant research papers for this question."
    assert response.sources == []
    assert response.chunks_used == 0
    assert response.retrieval_attempts == 2
