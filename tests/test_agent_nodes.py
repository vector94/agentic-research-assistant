import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.services.agents import AgentContext, AgentState, DocumentGrade, GuardrailResult
from src.services.agents.nodes import (
    run_answer_generation,
    run_document_grading,
    run_guardrail,
    run_query_rewriting,
    run_retrieval,
)


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
        answer_prompt_builder=Mock(),
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
        answer_prompt_builder=Mock(),
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


@pytest.mark.parametrize(
    ("is_relevant", "attempt", "expected_route"),
    [
        (True, 1, "generate_answer"),
        (False, 1, "rewrite_query"),
        (False, 2, "generate_answer"),
    ],
)
def test_document_grading_routes_using_relevance_and_attempt_limit(
    is_relevant: bool,
    attempt: int,
    expected_route: str,
) -> None:
    chunks = [Mock()]
    prompt_builder = Mock()
    prompt_builder.build_document_grade.return_value = "Grading prompt"

    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock(
        return_value=DocumentGrade(
            is_relevant=is_relevant,
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
        answer_prompt_builder=Mock(),
        max_retrieval_attempts=2,
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="Original question")],
        original_query="Original question",
        rewritten_query=None,
        retrieval_attempts=attempt,
        retrieved_chunks=chunks,
        guardrail_result=None,
        document_grade=None,
        routing_decision=None,
        reasoning_steps=[],
    )

    result = asyncio.run(run_document_grading(state, runtime))

    assert result["routing_decision"] == expected_route
    assert result["document_grade"].is_relevant is is_relevant


def test_document_grading_skips_model_when_no_chunks_are_found() -> None:
    ollama_client = Mock()
    ollama_client.generate_structured = AsyncMock()

    tracer = Mock()
    tracer.span.return_value = nullcontext()

    context = AgentContext(
        search_service=Mock(),
        ollama_client=ollama_client,
        prompt_builder=Mock(),
        tracer=tracer,
        answer_prompt_builder=Mock(),
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="Original question")],
        original_query="Original question",
        rewritten_query=None,
        retrieval_attempts=1,
        retrieved_chunks=[],
        guardrail_result=None,
        document_grade=None,
        routing_decision=None,
        reasoning_steps=[],
    )

    result = asyncio.run(run_document_grading(state, runtime))

    ollama_client.generate_structured.assert_not_awaited()
    assert result["routing_decision"] == "rewrite_query"


def test_query_rewriting_uses_original_query_and_updates_state() -> None:
    prompt_builder = Mock()
    prompt_builder.build_query_rewrite.return_value = "Rewrite prompt"

    ollama_client = Mock()
    ollama_client.model = "test-model"
    ollama_client.generate = AsyncMock(
        return_value="  Improved transformer retrieval question  ",
    )

    tracer = Mock()
    tracer.generation.return_value = nullcontext()

    context = AgentContext(
        search_service=Mock(),
        ollama_client=ollama_client,
        prompt_builder=prompt_builder,
        tracer=tracer,
        answer_prompt_builder=Mock(),
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="Original question")],
        original_query="Original question",
        rewritten_query="Previous rewrite",
        retrieval_attempts=1,
        retrieved_chunks=[],
        guardrail_result=None,
        document_grade=None,
        routing_decision="rewrite_query",
        reasoning_steps=[],
    )

    result = asyncio.run(run_query_rewriting(state, runtime))

    prompt_builder.build_query_rewrite.assert_called_once_with("Original question")
    ollama_client.generate.assert_awaited_once_with("Rewrite prompt")
    assert result["rewritten_query"] == "Improved transformer retrieval question"
    assert result["routing_decision"] == "retrieve"


def test_answer_generation_uses_only_graded_chunks() -> None:
    chunks = [Mock(), Mock()]
    answer_prompt_builder = Mock()
    answer_prompt_builder.build.return_value = "Answer prompt"

    ollama_client = Mock()
    ollama_client.model = "test-model"
    ollama_client.generate = AsyncMock(return_value="  Grounded answer  ")

    tracer = Mock()
    tracer.generation.return_value = nullcontext()

    context = AgentContext(
        search_service=Mock(),
        ollama_client=ollama_client,
        prompt_builder=Mock(),
        tracer=tracer,
        answer_prompt_builder=answer_prompt_builder,
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="Original question")],
        original_query="Original question",
        rewritten_query="Improved question",
        retrieval_attempts=1,
        retrieved_chunks=chunks,
        guardrail_result=None,
        document_grade=DocumentGrade(
            is_relevant=True,
            reason="The chunks answer the question",
        ),
        routing_decision="generate_answer",
        reasoning_steps=[],
    )

    result = asyncio.run(run_answer_generation(state, runtime))

    answer_prompt_builder.build.assert_called_once_with(
        query="Original question",
        chunks=chunks,
    )
    ollama_client.generate.assert_awaited_once_with("Answer prompt")
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "Grounded answer"


def test_answer_generation_skips_model_for_irrelevant_chunks() -> None:
    answer_prompt_builder = Mock()
    ollama_client = Mock()
    ollama_client.generate = AsyncMock()

    context = AgentContext(
        search_service=Mock(),
        ollama_client=ollama_client,
        prompt_builder=Mock(),
        tracer=Mock(),
        answer_prompt_builder=answer_prompt_builder,
    )
    runtime = SimpleNamespace(context=context)
    state = AgentState(
        messages=[HumanMessage(content="Original question")],
        original_query="Original question",
        rewritten_query="Improved question",
        retrieval_attempts=2,
        retrieved_chunks=[Mock()],
        guardrail_result=None,
        document_grade=DocumentGrade(
            is_relevant=False,
            reason="The chunks do not answer the question",
        ),
        routing_decision="generate_answer",
        reasoning_steps=[],
    )

    result = asyncio.run(run_answer_generation(state, runtime))

    answer_prompt_builder.build.assert_not_called()
    ollama_client.generate.assert_not_awaited()
    assert "could not find relevant" in result["messages"][0].content
