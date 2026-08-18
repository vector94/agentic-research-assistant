import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch

from src.routers.rag import ask_agentic_question, ask_question, stream_answer
from src.schemas.rag import AgenticRagResponse, RagRequest, RagResponse


def make_response(query: str) -> RagResponse:
    return RagResponse(
        query=query,
        answer="Cached answer",
        sources=["https://arxiv.org/pdf/1234.5678"],
        chunks_used=3,
    )


async def collect_stream_events(request: RagRequest) -> list[dict[str, object]]:
    response = await stream_answer(request)
    events: list[dict[str, object]] = []

    async for part in response.body_iterator:
        if isinstance(part, bytes):
            part = part.decode()

        event_json = part.removeprefix("data: ").strip()
        events.append(json.loads(event_json))

    return events


def test_ask_question_returns_cached_response() -> None:
    request = RagRequest(query="How do transformers work?", top_k=3)
    cached_response = make_response(request.query)
    service = AsyncMock()
    cache = AsyncMock()
    cache.get.return_value = cached_response

    with (
        patch("src.routers.rag.make_rag_service", return_value=service),
        patch("src.routers.rag.make_cache_client", return_value=cache),
    ):
        response = asyncio.run(ask_question(request))

    assert response == cached_response
    cache.get.assert_awaited_once_with(request)
    service.answer.assert_not_awaited()
    cache.set.assert_not_awaited()
    service.close.assert_awaited_once()
    cache.close.assert_awaited_once()


def test_ask_question_caches_generated_response() -> None:
    request = RagRequest(query="How does retrieval work?", top_k=2)
    generated_response = make_response(request.query)
    service = AsyncMock()
    service.answer.return_value = generated_response
    cache = AsyncMock()
    cache.get.return_value = None

    with (
        patch("src.routers.rag.make_rag_service", return_value=service),
        patch("src.routers.rag.make_cache_client", return_value=cache),
    ):
        response = asyncio.run(ask_question(request))

    assert response == generated_response
    service.answer.assert_awaited_once_with(
        query=request.query,
        top_k=request.top_k,
    )
    cache.set.assert_awaited_once_with(request, generated_response)
    service.close.assert_awaited_once()
    cache.close.assert_awaited_once()


def test_ask_agentic_question_runs_agent_workflow() -> None:
    request = RagRequest(query="How does retrieval work?", top_k=2)
    generated_response = AgenticRagResponse(
        query=request.query,
        answer="Agentic answer",
        sources=["https://arxiv.org/pdf/1234.5678"],
        chunks_used=1,
        reasoning_steps=["Retrieved and graded relevant chunks"],
        retrieval_attempts=1,
    )
    service = AsyncMock()
    service.answer.return_value = generated_response

    with patch("src.routers.rag.make_agentic_rag_service", return_value=service):
        response = asyncio.run(ask_agentic_question(request))

    assert response == generated_response
    service.answer.assert_awaited_once_with(
        query=request.query,
        top_k=request.top_k,
    )
    service.close.assert_awaited_once()


def test_stream_answer_replays_cached_response() -> None:
    request = RagRequest(query="How do transformers work?", top_k=3)
    cached_response = make_response(request.query)
    service = Mock()
    service.answer_stream = Mock()
    service.close = AsyncMock()
    cache = AsyncMock()
    cache.get.return_value = cached_response

    with (
        patch("src.routers.rag.make_rag_service", return_value=service),
        patch("src.routers.rag.make_cache_client", return_value=cache),
    ):
        events = asyncio.run(collect_stream_events(request))

    assert events == [
        {
            "sources": cached_response.sources,
            "chunks_used": cached_response.chunks_used,
            "search_mode": "hybrid",
        },
        {"chunk": cached_response.answer},
        {"answer": cached_response.answer, "done": True},
    ]
    service.answer_stream.assert_not_called()
    cache.set.assert_not_awaited()
    service.close.assert_awaited_once()
    cache.close.assert_awaited_once()


def test_stream_answer_caches_completed_response() -> None:
    request = RagRequest(query="How does retrieval work?", top_k=2)
    service = Mock()
    service.close = AsyncMock()

    async def generated_events(query: str, top_k: int):
        yield {
            "sources": ["https://arxiv.org/pdf/1234.5678"],
            "chunks_used": top_k,
            "search_mode": "hybrid",
        }
        yield {"chunk": "Generated answer"}
        yield {"answer": "Generated answer", "done": True}

    service.answer_stream = Mock(side_effect=generated_events)
    cache = AsyncMock()
    cache.get.return_value = None

    with (
        patch("src.routers.rag.make_rag_service", return_value=service),
        patch("src.routers.rag.make_cache_client", return_value=cache),
    ):
        events = asyncio.run(collect_stream_events(request))

    assert events[-1] == {"answer": "Generated answer", "done": True}
    service.answer_stream.assert_called_once_with(
        query=request.query,
        top_k=request.top_k,
    )
    cache.set.assert_awaited_once_with(
        request,
        RagResponse(
            query=request.query,
            answer="Generated answer",
            sources=["https://arxiv.org/pdf/1234.5678"],
            chunks_used=request.top_k,
        ),
    )
    service.close.assert_awaited_once()
    cache.close.assert_awaited_once()
