from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.exceptions import ArxivApiError
from src.repositories.paper import PaperRepository
from src.services.arxiv import ArxivClient
from src.services.indexing import PaperIndexingService
from src.services.ingestion import PaperIngestionService
from src.services.opensearch.factory import make_opensearch_client
from src.services.paper_indexer import PaperIndexer
from src.services.pdf_downloader import PdfDownloader
from src.services.pdf_parser.factory import make_pdf_parser_service

router = APIRouter(prefix="/api/v1/papers", tags=["papers"])


@router.post("/ingest")
async def ingest_papers(
    query: str = "cat:cs.AI",
    max_results: int = Query(default=5, ge=1, le=20),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    arxiv_client = ArxivClient()
    paper_repository = PaperRepository(session)
    pdf_downloader = PdfDownloader()
    pdf_parser = make_pdf_parser_service()
    service = PaperIngestionService(arxiv_client, paper_repository, pdf_downloader, pdf_parser)

    try:
        return await service.ingest(
            query=query,
            max_results=max_results,
        )
    except ArxivApiError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(error),
        ) from error


@router.post("/index")
async def index_papers(
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    paper_repository = PaperRepository(session)
    opensearch_client = make_opensearch_client()
    paper_indexer = PaperIndexer(opensearch_client)

    service = PaperIndexingService(
        paper_repository=paper_repository,
        paper_indexer=paper_indexer,
    )

    return await service.index_processed_papers(limit=limit)
