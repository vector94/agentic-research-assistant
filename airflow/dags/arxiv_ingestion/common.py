import logging
from functools import lru_cache

from src.services.arxiv import ArxivClient
from src.services.opensearch.client import OpenSearchClient
from src.services.opensearch.factory import make_chunk_opensearch_client
from src.services.pdf_downloader import PdfDownloader
from src.services.pdf_parser.factory import make_pdf_parser_service
from src.services.pdf_parser.parser import PdfParserService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cached_services() -> tuple[
    ArxivClient,
    PdfParserService,
    PdfDownloader,
    OpenSearchClient,
]:
    """Create the services shared by an Airflow task process."""
    logger.info("Initializing arXiv ingestion services")

    services = (
        ArxivClient(),
        make_pdf_parser_service(),
        PdfDownloader(),
        make_chunk_opensearch_client(),
    )

    logger.info("arXiv ingestion services are ready")
    return services
