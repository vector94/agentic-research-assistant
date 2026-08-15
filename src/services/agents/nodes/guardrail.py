from langgraph.runtime import Runtime

from src.services.agents.context import AgentContext
from src.services.agents.models import GuardrailResult
from src.services.agents.state import AgentRoute, AgentState


async def run_guardrail(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, object]:
    """Check whether the original question belongs to the research domain."""

    context = runtime.context
    query = state["original_query"]
    prompt = context.prompt_builder.build_guardrail(query)

    with context.tracer.span(
        name="guardrail",
        input_data={"query": query},
        metadata={"threshold": context.guardrail_threshold},
    ) as observation:
        result = await context.ollama_client.generate_structured(
            prompt=prompt,
            response_model=GuardrailResult,
        )
        route: AgentRoute = "retrieve" if result.score >= context.guardrail_threshold else "out_of_scope"
        context.tracer.update(
            observation,
            {
                "score": result.score,
                "reason": result.reason,
                "route": route,
            },
        )

    return {
        "guardrail_result": result,
        "routing_decision": route,
        "reasoning_steps": [f"Scope score {result.score}: {result.reason}"],
    }
