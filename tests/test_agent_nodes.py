import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import HumanMessage

from src.services.agents import AgentContext, AgentState, GuardrailResult
from src.services.agents.nodes import run_guardrail, run_retrieval


@pytest.mark.parametrize(
    ("score", "expected_route"),
    [
        (60, "retrieve"),
        (59, "out_of_scope"),
    ],
)
def test_guardrail_routes_using_configured_threshold(
    score: int,
    expected_route: str,
) -> None:
    prompt_builder = Mock()
    prompt_builder.build_guardrail.return_value = "Guardrail prompt"

    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock(
        return_value=GuardrailResult(
            score=score,
            reason="Test reason",
        )
    )

    tracer = Mock()
    tracer.span.return_value = nullcontext()

    context = AgentContext(
        search_service=Mock(),
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
        tracer=tracer,
        guardrail_threshold=60,
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="How do transformers work?")],
        original_query="How do transformers work?",
        rewritten_query=None,
        retrieval_attempts=0,
        retrieved_chunks=[],
        guardrail_result=None,
        document_grade=None,
        routing_decision=None,
        reasoning_steps=[],
    )

    result = asyncio.run(run_guardrail(state, runtime))

    assert result["routing_decision"] == expected_route
    prompt_builder.build_guardrail.assert_called_once_with(state["original_query"])
    ollama_client.generate_structured.assert_awaited_once_with(
        prompt="Guardrail prompt",
        response_model=GuardrailResult,
    )


def test_retrieval_uses_rewritten_query_and_updates_state() -> None:
    chunks = [Mock(), Mock()]
    search_service = Mock()
    search_service.search = AsyncMock(
        return_value=SimpleNamespace(results=chunks),
    )

    tracer = Mock()
    tracer.span.return_value = nullcontext()

    context = AgentContext(
        search_service=search_service,
        ollama_client=Mock(),
        prompt_builder=Mock(),
        tracer=tracer,
        top_k=3,
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="Original question")],
        original_query="Original question",
        rewritten_query="Improved search query",
        retrieval_attempts=1,
        retrieved_chunks=[],
        guardrail_result=None,
        document_grade=None,
        routing_decision="retrieve",
        reasoning_steps=[],
    )

    result = asyncio.run(run_retrieval(state, runtime))

    search_service.search.assert_awaited_once_with(
        query="Improved search query",
        size=3,
    )
    assert result["retrieval_attempts"] == 2
    assert result["retrieved_chunks"] == chunks
