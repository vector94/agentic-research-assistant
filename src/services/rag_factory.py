from src.services.hybrid_search_factory import make_hybrid_search_service
from src.services.ollama.factory import make_ollama_client
from src.services.ollama.prompts import RagPromptBuilder
from src.services.rag import RagService


def make_rag_service() -> RagService:
    return RagService(
        hybrid_search_service=make_hybrid_search_service(),
        prompt_builder=RagPromptBuilder(),
        ollama_client=make_ollama_client(),
    )
