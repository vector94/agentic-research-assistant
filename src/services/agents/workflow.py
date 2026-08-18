from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.services.agents.context import AgentContext
from src.services.agents.nodes import (
    run_answer_generation,
    run_document_grading,
    run_guardrail,
    run_out_of_scope,
    run_query_rewriting,
    run_retrieval,
)
from src.services.agents.state import AgentState

GuardrailRoute = Literal["retrieve", "out_of_scope"]
GradingRoute = Literal["generate_answer", "rewrite_query"]
AgentWorkflow = CompiledStateGraph[AgentState, AgentContext, AgentState, AgentState]


def _route_after_guardrail(state: AgentState) -> GuardrailRoute:
    route = state["routing_decision"]

    if route == "retrieve":
        return "retrieve"

    if route == "out_of_scope":
        return "out_of_scope"

    raise ValueError("Guardrail did not select a valid route")


def _route_after_grading(state: AgentState) -> GradingRoute:
    route = state["routing_decision"]

    if route == "generate_answer":
        return "generate_answer"

    if route == "rewrite_query":
        return "rewrite_query"

    raise ValueError("Document grading did not select a valid route")


def build_agent_workflow() -> AgentWorkflow:
    """Build the research assistant's retrieval and answer workflow."""

    workflow = StateGraph(AgentState, context_schema=AgentContext)

    workflow.add_node("guardrail", run_guardrail)
    workflow.add_node("out_of_scope", run_out_of_scope)
    workflow.add_node("retrieve", run_retrieval)
    workflow.add_node("grade_documents", run_document_grading)
    workflow.add_node("rewrite_query", run_query_rewriting)
    workflow.add_node("generate_answer", run_answer_generation)

    workflow.add_edge(START, "guardrail")
    workflow.add_conditional_edges(
        "guardrail",
        _route_after_guardrail,
        {
            "retrieve": "retrieve",
            "out_of_scope": "out_of_scope",
        },
    )
    workflow.add_edge("out_of_scope", END)

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        _route_after_grading,
        {
            "generate_answer": "generate_answer",
            "rewrite_query": "rewrite_query",
        },
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)

    return workflow.compile(name="agentic_research_assistant")
