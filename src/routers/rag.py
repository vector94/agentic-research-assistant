import json
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.schemas.rag import RagRequest, RagResponse
from src.services.rag_factory import make_rag_service

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.post("/ask", response_model=RagResponse)
async def ask_question(request: RagRequest) -> RagResponse:
    service = make_rag_service()

    try:
        return await service.answer(
            query=request.query,
            top_k=request.top_k,
        )
    finally:
        await service.close()


@router.post("/stream")
async def stream_answer(request: RagRequest) -> StreamingResponse:
    service = make_rag_service()

    async def generate_events() -> AsyncIterator[str]:
        try:
            async for event in service.answer_stream(
                query=request.query,
                top_k=request.top_k,
            ):
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            await service.close()

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )
