from collections.abc import AsyncIterator

from src.schemas.rag import RagResponse
from src.services.hybrid_search import HybridSearchService
from src.services.ollama.client import OllamaClient
from src.services.ollama.prompts import RagPromptBuilder


class RagService:
    def __init__(
        self,
        hybrid_search_service: HybridSearchService,
        prompt_builder: RagPromptBuilder,
        ollama_client: OllamaClient,
    ) -> None:
        self.hybrid_search_service = hybrid_search_service
        self.prompt_builder = prompt_builder
        self.ollama_client = ollama_client

    async def answer(
        self,
        query: str,
        top_k: int = 3,
    ) -> RagResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("Top K must be greater than zero")

        search_response = await self.hybrid_search_service.search(
            query=query,
            size=top_k,
        )

        if not search_response.results:
            return RagResponse(
                query=query,
                answer="I could not find relevant research papers for this question.",
                sources=[],
                chunks_used=0,
            )

        prompt = self.prompt_builder.build(
            query=query,
            chunks=search_response.results,
        )
        answer = await self.ollama_client.generate(prompt)

        sources: list[str] = []
        for result in search_response.results:
            if result.pdf_url not in sources:
                sources.append(result.pdf_url)

        return RagResponse(
            query=query,
            answer=answer,
            sources=sources,
            chunks_used=len(search_response.results),
        )

    async def answer_stream(
        self,
        query: str,
        top_k: int = 3,
    ) -> AsyncIterator[dict[str, object]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("Top K must be greater than zero")

        search_response = await self.hybrid_search_service.search(
            query=query,
            size=top_k,
        )

        if not search_response.results:
            answer = "I could not find relevant research papers for this question."
            yield {
                "answer": answer,
                "sources": [],
                "chunks_used": 0,
                "search_mode": "hybrid",
                "done": True,
            }
            return

        sources: list[str] = []
        for result in search_response.results:
            if result.pdf_url not in sources:
                sources.append(result.pdf_url)

        yield {
            "sources": sources,
            "chunks_used": len(search_response.results),
            "search_mode": "hybrid",
        }

        prompt = self.prompt_builder.build(
            query=query,
            chunks=search_response.results,
        )
        answer_parts: list[str] = []

        async for text in self.ollama_client.generate_stream(prompt):
            answer_parts.append(text)
            yield {"chunk": text}

        yield {
            "answer": "".join(answer_parts),
            "done": True,
        }

    async def close(self) -> None:
        await self.hybrid_search_service.close()
        await self.ollama_client.close()
