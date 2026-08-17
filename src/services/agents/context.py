from dataclasses import dataclass

from src.services.agents.prompts import AgentPromptBuilder
from src.services.hybrid_search import HybridSearchService
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.ollama.prompts import RagPromptBuilder


@dataclass(frozen=True)
class AgentContext:
    """Dependencies and settings shared by the agent nodes."""

    search_service: HybridSearchService
    ollama_client: OllamaClient
    prompt_builder: AgentPromptBuilder
    tracer: LangfuseTracer
    answer_prompt_builder: RagPromptBuilder
    top_k: int = 3
    max_retrieval_attempts: int = 2
    guardrail_threshold: int = 60
