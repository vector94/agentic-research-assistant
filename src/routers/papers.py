from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.repositories.paper import PaperRepository
from src.services.arxiv import ArxivClient
from src.services.ingestion import PaperIngestionService

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])


@router.post("/ingest")
async def ingest_papers(
    query: str = "cat:cs.AI",
    max_results: int = Query(default=5, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    arxiv_client = ArxivClient()
    paper_repository = PaperRepository(session)
    service = PaperIngestionService(arxiv_client, paper_repository)

    result = await service.ingest(query=query, max_results=max_results)
    return result
