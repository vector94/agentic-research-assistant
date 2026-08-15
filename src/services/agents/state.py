from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.schemas.retrieval import ChunkSearchResult
from src.services.agents.models import DocumentGrade, GuardrailResult

AgentRoute = Literal[
    "retrieve",
    "out_of_scope",
    "generate_answer",
    "rewrite_query",
]


class AgentState(TypedDict):
    """Data shared by the nodes in the agent workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
    original_query: str
    rewritten_query: str | None
    retrieval_attempts: int
    retrieved_chunks: list[ChunkSearchResult]
    guardrail_result: GuardrailResult | None
    document_grade: DocumentGrade | None
    routing_decision: AgentRoute | None
    reasoning_steps: Annotated[list[str], add]
