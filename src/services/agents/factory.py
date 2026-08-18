from src.services.agents.context import AgentContext
from src.services.agents.prompts import AgentPromptBuilder
from src.services.agents.service import AgenticRagService
from src.services.hybrid_search_factory import make_hybrid_search_service
from src.services.langfuse.factory import make_langfuse_tracer
from src.services.ollama.factory import make_ollama_client
from src.services.ollama.prompts import RagPromptBuilder


def make_agentic_rag_service() -> AgenticRagService:
    context = AgentContext(
        search_service=make_hybrid_search_service(),
        ollama_client=make_ollama_client(),
        prompt_builder=AgentPromptBuilder(),
        tracer=make_langfuse_tracer(),
        answer_prompt_builder=RagPromptBuilder(),
    )
    return AgenticRagService(context)
