from langgraph.runtime import Runtime

from src.services.agents.context import AgentContext
from src.services.agents.state import AgentState


async def run_retrieval(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, object]:
    """Retrieve paper chunks for the current search query."""

    context = runtime.context
    query = state["rewritten_query"] or state["original_query"]
    attempt = state["retrieval_attempts"] + 1

    with context.tracer.span(
        name="retrieval",
        input_data={"query": query, "top_k": context.top_k},
        metadata={"attempt": attempt},
    ) as observation:
        response = await context.search_service.search(
            query=query,
            size=context.top_k,
        )
        chunks = response.results
        context.tracer.update(
            observation,
            {"chunks_found": len(chunks)},
        )

    return {
        "retrieval_attempts": attempt,
        "retrieved_chunks": chunks,
        "reasoning_steps": [
            f"Retrieved {len(chunks)} chunks on attempt {attempt}",
        ],
    }
