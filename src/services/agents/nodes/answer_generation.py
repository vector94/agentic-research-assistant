from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.exceptions import OllamaError
from src.services.agents.context import AgentContext
from src.services.agents.state import AgentState

_UNAVAILABLE_ANSWER = (
    "I found relevant research papers, but couldn't generate an answer right now "
    "due to a service issue. Please try again shortly."
)


async def run_answer_generation(
    state: AgentState,
    runtime: Runtime[AgentContext],
) -> dict[str, object]:
    """Generate a grounded answer from chunks that passed document grading."""

    grade = state["document_grade"]

    if grade is None:
        raise ValueError("Documents must be graded before generating an answer")

    chunks = state["retrieved_chunks"]

    if not grade.is_relevant or not chunks:
        return {
            "messages": [
                AIMessage(
                    content="I could not find relevant research papers for this question.",
                )
            ],
            "reasoning_steps": [
                "Stopped because no retrieved chunks could answer the question",
            ],
        }

    context = runtime.context
    prompt = context.answer_prompt_builder.build(
        query=state["original_query"],
        chunks=chunks,
    )

    with context.tracer.generation(
        name="answer_generation",
        model=context.ollama_client.model,
        input_data=prompt,
        metadata={"chunk_count": len(chunks)},
    ) as observation:
        try:
            answer = (await context.ollama_client.generate(prompt)).strip()
            if not answer:
                raise ValueError("Ollama returned an empty answer")
            reasoning_step = f"Generated an answer from {len(chunks)} retrieved chunks"
        except (OllamaError, ValueError):
            answer = _UNAVAILABLE_ANSWER
            reasoning_step = "Answer generation was unavailable; returned a fallback message"

        context.tracer.update(observation, answer)

    return {
        "messages": [AIMessage(content=answer)],
        "reasoning_steps": [reasoning_step],
    }
