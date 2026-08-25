from langgraph.runtime import Runtime

from src.exceptions import OllamaError
from src.services.agents.context import AgentContext
from src.services.agents.state import AgentState

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "am", "be", "been", "being",
    "what", "how", "why", "when", "where", "who", "which", "does", "do", "did",
    "can", "could", "would", "should", "of", "in", "on", "for", "to", "about",
    "and", "or", "please", "tell", "me", "explain",
}


def _fallback_rewrite(query: str) -> str:
    """A cheap keyword-expansion fallback used only when the LLM is unavailable."""

    keywords = [word for word in query.split() if word.strip(".,?!").lower() not in _STOPWORDS]
    return " ".join(keywords).strip() or query


async def run_query_rewriting(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, object]:
    """Rewrite the original question for another retrieval attempt."""

    context = runtime.context
    original_query = state["original_query"]
    prompt = context.prompt_builder.build_query_rewrite(original_query)

    with context.tracer.generation(
        name="query_rewriting",
        model=context.ollama_client.model,
        input_data=prompt,
        metadata={"completed_attempts": state["retrieval_attempts"]},
    ) as observation:
        try:
            rewritten_query = (await context.ollama_client.generate(prompt)).strip()
            if not rewritten_query:
                raise ValueError("Ollama returned an empty rewritten query")
        except (OllamaError, ValueError):
            rewritten_query = _fallback_rewrite(original_query)

        context.tracer.update(observation, rewritten_query)

    return {
        "rewritten_query": rewritten_query,
        "routing_decision": "retrieve",
        "reasoning_steps": [
            f"Rewrote the search query as: {rewritten_query}",
        ],
    }
