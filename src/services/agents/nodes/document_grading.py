from langgraph.runtime import Runtime

from src.exceptions import OllamaError
from src.services.agents.context import AgentContext
from src.services.agents.models import DocumentGrade
from src.services.agents.state import AgentRoute, AgentState

_MIN_RELEVANT_CHUNK_CHARS = 200


def _fallback_grade(chunks: list) -> DocumentGrade:
    """A cheap heuristic used only when the LLM grading call is unavailable."""

    total_chars = sum(len(chunk.chunk_text) for chunk in chunks)
    return DocumentGrade(
        is_relevant=total_chars >= _MIN_RELEVANT_CHUNK_CHARS,
        reason="Grading unavailable; used a content-length heuristic instead.",
    )


async def run_document_grading(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, object]:
    """Decide whether the retrieved chunks can answer the question."""

    context = runtime.context
    query = state["original_query"]
    chunks = state["retrieved_chunks"]
    attempt = state["retrieval_attempts"]

    with context.tracer.span(
        name="document_grading",
        input_data={"query": query, "chunk_count": len(chunks)},
        metadata={"attempt": attempt},
    ) as observation:
        if chunks:
            prompt = context.prompt_builder.build_document_grade(
                query=query,
                chunks=chunks,
            )
            try:
                grade = await context.ollama_client.generate_structured(
                    prompt=prompt,
                    response_model=DocumentGrade,
                )
            except OllamaError:
                grade = _fallback_grade(chunks)
        else:
            grade = DocumentGrade(
                is_relevant=False,
                reason="No paper chunks were retrieved",
            )

        route: AgentRoute
        if grade.is_relevant or attempt >= context.max_retrieval_attempts:
            route = "generate_answer"
        else:
            route = "rewrite_query"

        context.tracer.update(
            observation,
            {
                "is_relevant": grade.is_relevant,
                "reason": grade.reason,
                "route": route,
            },
        )

    relevance = "relevant" if grade.is_relevant else "not relevant"

    return {
        "document_grade": grade,
        "routing_decision": route,
        "reasoning_steps": [
            f"Retrieved chunks are {relevance}: {grade.reason}",
        ],
    }
