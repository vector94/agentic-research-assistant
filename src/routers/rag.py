import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.rag import RagRequest, RagResponse
from src.services.cache.factory import make_cache_client
from src.services.rag_factory import make_rag_service

router = APIRouter(prefix="/api/v1", tags=["rag"])
logger = logging.getLogger(__name__)


@router.post("/ask", response_model=RagResponse)
async def ask_question(request: RagRequest) -> RagResponse:
    service = make_rag_service()
    cache = make_cache_client()

    try:
        try:
            cached_response = await cache.get(request)
        except Exception:
            logger.exception("Failed to read the RAG response cache")
            cached_response = None

        if cached_response is not None:
            logger.info("RAG response cache hit")
            return cached_response

        response = await service.answer(
            query=request.query,
            top_k=request.top_k,
        )

        try:
            await cache.set(request, response)
        except Exception:
            logger.exception("Failed to store the RAG response cache")

        return response
    finally:
        await service.close()
        await cache.close()


@router.post("/stream")
async def stream_answer(request: RagRequest) -> StreamingResponse:
    service = make_rag_service()
    cache = make_cache_client()

    async def generate_events() -> AsyncIterator[str]:
        sources: list[str] = []
        chunks_used = 0

        try:
            try:
                cached_response = await cache.get(request)
            except Exception:
                logger.exception("Failed to read the streaming RAG response cache")
                cached_response = None

            if cached_response is not None:
                logger.info("Streaming RAG response cache hit")
                metadata = {
                    "sources": cached_response.sources,
                    "chunks_used": cached_response.chunks_used,
                    "search_mode": cached_response.search_mode,
                }
                yield f"data: {json.dumps(metadata)}\n\n"
                yield f"data: {json.dumps({'chunk': cached_response.answer})}\n\n"
                completion = {
                    "answer": cached_response.answer,
                    "done": True,
                }
                yield f"data: {json.dumps(completion)}\n\n"
                return

            async for event in service.answer_stream(
                query=request.query,
                top_k=request.top_k,
            ):
                event_sources = event.get("sources")
                if isinstance(event_sources, list):
                    sources = event_sources

                event_chunks_used = event.get("chunks_used")
                if isinstance(event_chunks_used, int):
                    chunks_used = event_chunks_used

                answer = event.get("answer")
                if event.get("done") is True and isinstance(answer, str):
                    response = RagResponse(
                        query=request.query,
                        answer=answer,
                        sources=sources,
                        chunks_used=chunks_used,
                    )

                    try:
                        await cache.set(request, response)
                    except Exception:
                        logger.exception("Failed to store the streaming RAG response cache")

                yield f"data: {json.dumps(event)}\n\n"
        finally:
            await service.close()
            await cache.close()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
