from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from src.schemas.rag import AgenticRagResponse
from src.schemas.retrieval import ChunkSearchResult
from src.services.agents.context import AgentContext
from src.services.agents.state import AgentState
from src.services.agents.workflow import AgentWorkflow, build_agent_workflow


class AgenticRagService:
    """Run the LangGraph research workflow and shape its final API response."""

    def __init__(
        self,
        context: AgentContext,
        workflow: AgentWorkflow | None = None,
    ) -> None:
        self.context = context
        self.workflow = workflow or build_agent_workflow()

    async def answer(
        self,
        query: str,
        top_k: int = 3,
    ) -> AgenticRagResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("Top K must be greater than zero")

        state = AgentState(
            messages=[HumanMessage(content=query)],
            original_query=query,
            rewritten_query=None,
            retrieval_attempts=0,
            retrieved_chunks=[],
            guardrail_result=None,
            document_grade=None,
            routing_decision=None,
            reasoning_steps=[],
        )
        runtime_context = replace(self.context, top_k=top_k)

        with self.context.tracer.span(
            name="agentic_rag_request",
            input_data={"query": query, "top_k": top_k},
        ) as observation:
            result = await self.workflow.ainvoke(
                state,
                context=runtime_context,
            )
            response = self._build_response(query, result)
            self.context.tracer.update(observation, response.model_dump())

        return response

    @staticmethod
    def _build_response(
        query: str,
        result: AgentState,
    ) -> AgenticRagResponse:
        messages = result["messages"]

        if not messages or not isinstance(messages[-1], AIMessage):
            raise ValueError("Agent workflow did not produce an answer")

        answer = messages[-1].content

        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Agent workflow produced an invalid answer")

        grade = result["document_grade"]
        chunks: list[ChunkSearchResult] = []

        if grade is not None and grade.is_relevant:
            chunks = result["retrieved_chunks"]

        sources = list(dict.fromkeys(chunk.pdf_url for chunk in chunks))

        return AgenticRagResponse(
            query=query,
            answer=answer,
            sources=sources,
            chunks_used=len(chunks),
            reasoning_steps=result["reasoning_steps"],
            retrieval_attempts=result["retrieval_attempts"],
        )

    async def close(self) -> None:
        await self.context.search_service.close()
        await self.context.ollama_client.close()
        self.context.tracer.flush()
