from langchain_core.messages import AIMessage

from src.services.agents.state import AgentState


def run_out_of_scope(_: AgentState) -> dict[str, object]:
    """Return a short explanation when a question is outside the project scope."""

    answer = (
        "I can help with questions about computer science, artificial intelligence, "
        "and machine learning research papers. This question appears to be outside "
        "that scope. Try asking about a paper, model, method, or research finding."
    )

    return {
        "messages": [AIMessage(content=answer)],
        "reasoning_steps": ["Stopped because the question is outside the research scope"],
    }
