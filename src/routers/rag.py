from fastapi import APIRouter

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
