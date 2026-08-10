from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_session
from src.exceptions import ArxivApiError
from src.repositories.paper import PaperRepository
from src.schemas.retrieval import ChunkSearchResponse
from src.schemas.search import PaperSearchResponse
from src.services.arxiv import ArxivClient
from src.services.hybrid_indexing_factory import make_hybrid_indexing_service
from src.services.hybrid_search_factory import make_hybrid_search_service
from src.services.indexing import PaperIndexingService
from src.services.ingestion import PaperIngestionService
from src.services.opensearch.factory import make_opensearch_client
from src.services.paper_indexer import PaperIndexer
from src.services.pdf_downloader import PdfDownloader
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.search import PaperSearchService
from src.services.vector_search_factory import make_vector_search_service

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


@router.post("/index/chunks")
async def index_paper_chunks(
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> dict[str, int]:
    paper_repository = PaperRepository(session)
    papers = await paper_repository.list_processed(limit=limit)
    service = make_hybrid_indexing_service()

    try:
        return await service.index_papers(papers)
    finally:
        await service.close()


@router.get(
    "/search",
    response_model=PaperSearchResponse,
)
async def search_papers(
    query: str = Query(min_length=2),
    size: int = Query(default=10, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    categories: list[str] | None = Query(default=None),
    latest: bool = Query(default=False),
    fuzzy: bool = Query(default=False),
) -> PaperSearchResponse:
    opensearch_client = make_opensearch_client()
    service = PaperSearchService(opensearch_client)

    return await service.search(
        query=query,
        size=size,
        from_=offset,
        categories=categories,
        latest=latest,
        fuzzy=fuzzy,
    )


@router.get(
    "/search/vector",
    response_model=ChunkSearchResponse,
)
async def search_paper_chunks_vector(
    query: str = Query(min_length=2),
    size: int = Query(default=10, ge=1, le=50),
) -> ChunkSearchResponse:
    service = make_vector_search_service()

    try:
        return await service.search(
            query=query,
            size=size,
        )
    finally:
        await service.close()


@router.get(
    "/search/hybrid",
    response_model=ChunkSearchResponse,
)
async def search_paper_chunks_hybrid(
    query: str = Query(min_length=2),
    size: int = Query(default=10, ge=1, le=50),
) -> ChunkSearchResponse:
    service = make_hybrid_search_service()

    try:
        return await service.search(
            query=query,
            size=size,
        )
    finally:
        await service.close()
