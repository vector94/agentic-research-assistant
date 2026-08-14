from collections.abc import AsyncIterator

from src.schemas.rag import RagResponse
from src.services.hybrid_search import HybridSearchService
from src.services.langfuse.client import LangfuseTracer
from src.services.ollama.client import OllamaClient
from src.services.ollama.prompts import RagPromptBuilder


class RagService:
    def __init__(
        self,
        hybrid_search_service: HybridSearchService,
        prompt_builder: RagPromptBuilder,
        ollama_client: OllamaClient,
        tracer: LangfuseTracer,
    ) -> None:
        self.hybrid_search_service = hybrid_search_service
        self.prompt_builder = prompt_builder
        self.ollama_client = ollama_client
        self.tracer = tracer

    async def answer(
        self,
        query: str,
        top_k: int = 3,
    ) -> RagResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("Top K must be greater than zero")

        with self.tracer.span(
            name="rag_request",
            input_data={"query": query, "top_k": top_k},
            metadata={"stream": False},
        ) as request_span:
            with self.tracer.span(
                name="retrieval",
                input_data={"query": query, "top_k": top_k},
            ) as retrieval_span:
                search_response = await self.hybrid_search_service.search(
                    query=query,
                    size=top_k,
                )
                self.tracer.update(
                    retrieval_span,
                    {"chunks_found": len(search_response.results)},
                )

            if not search_response.results:
                response = RagResponse(
                    query=query,
                    answer="I could not find relevant research papers for this question.",
                    sources=[],
                    chunks_used=0,
                )
                self.tracer.update(request_span, response.model_dump())
                return response

            with self.tracer.span(
                name="prompt_construction",
                input_data={"chunk_count": len(search_response.results)},
            ) as prompt_span:
                prompt = self.prompt_builder.build(
                    query=query,
                    chunks=search_response.results,
                )
                self.tracer.update(
                    prompt_span,
                    {"prompt_length": len(prompt)},
                )

            with self.tracer.generation(
                name="ollama_generation",
                model=self.ollama_client.model,
                input_data=prompt,
            ) as generation:
                answer = await self.ollama_client.generate(prompt)
                self.tracer.update(generation, answer)

            sources: list[str] = []
            for result in search_response.results:
                if result.pdf_url not in sources:
                    sources.append(result.pdf_url)

            response = RagResponse(
                query=query,
                answer=answer,
                sources=sources,
                chunks_used=len(search_response.results),
            )
            self.tracer.update(request_span, response.model_dump())
            return response

    async def answer_stream(
        self,
        query: str,
        top_k: int = 3,
    ) -> AsyncIterator[dict[str, object]]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("Top K must be greater than zero")

        with self.tracer.span(
            name="rag_request",
            input_data={"query": query, "top_k": top_k},
            metadata={"stream": True},
        ) as request_span:
            with self.tracer.span(
                name="retrieval",
                input_data={"query": query, "top_k": top_k},
            ) as retrieval_span:
                search_response = await self.hybrid_search_service.search(
                    query=query,
                    size=top_k,
                )
                self.tracer.update(
                    retrieval_span,
                    {"chunks_found": len(search_response.results)},
                )

            if not search_response.results:
                answer = "I could not find relevant research papers for this question."
                response = RagResponse(
                    query=query,
                    answer=answer,
                    sources=[],
                    chunks_used=0,
                )
                self.tracer.update(request_span, response.model_dump())
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

            with self.tracer.span(
                name="prompt_construction",
                input_data={"chunk_count": len(search_response.results)},
            ) as prompt_span:
                prompt = self.prompt_builder.build(
                    query=query,
                    chunks=search_response.results,
                )
                self.tracer.update(
                    prompt_span,
                    {"prompt_length": len(prompt)},
                )

            answer_parts: list[str] = []
            with self.tracer.generation(
                name="ollama_generation",
                model=self.ollama_client.model,
                input_data=prompt,
            ) as generation:
                async for text in self.ollama_client.generate_stream(prompt):
                    answer_parts.append(text)
                    yield {"chunk": text}

                answer = "".join(answer_parts)
                self.tracer.update(generation, answer)

            response = RagResponse(
                query=query,
                answer=answer,
                sources=sources,
                chunks_used=len(search_response.results),
            )
            self.tracer.update(request_span, response.model_dump())
            yield {
                "answer": answer,
                "done": True,
            }

    async def close(self) -> None:
        await self.hybrid_search_service.close()
        await self.ollama_client.close()
        self.tracer.flush()
