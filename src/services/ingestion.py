import asyncio
import logging

from src.repositories.paper import PaperRepository
from src.services.arxiv import ArxivClient
from src.services.pdf_downloader import PdfDownloader
from src.services.pdf_parser.parser import PdfParserService

logger = logging.getLogger(__name__)


class PaperIngestionService:
    def __init__(
        self,
        arxiv_client: ArxivClient,
        paper_repository: PaperRepository,
        pdf_downloader: PdfDownloader,
        pdf_parser: PdfParserService,
    ) -> None:
        self.arxiv_client = arxiv_client
        self.paper_repository = paper_repository
        self.pdf_downloader = pdf_downloader
        self.pdf_parser = pdf_parser

    async def ingest(self, query: str, max_results: int = 5) -> dict[str, int]:
        papers = await asyncio.to_thread(self.arxiv_client.search, query=query, max_results=max_results)

        stored = 0
        skipped = 0
        processed = 0
        failed = 0

        for paper in papers:
            database_paper = await self.paper_repository.get_by_arxiv_id(paper.arxiv_id)
            if database_paper is None:
                database_paper = await self.paper_repository.add(paper)
                stored += 1
            else:
                skipped += 1

            if database_paper.pdf_processed:
                continue

            try:
                pdf_path = await self.pdf_downloader.download_pdf(paper.arxiv_id, paper.pdf_url)
                content = await self.pdf_parser.parse(pdf_path)

                await self.paper_repository.update_pdf_content(database_paper, content)
                processed += 1

            except Exception:
                logger.exception(
                    "Failed to process PDF for paper %s",
                    paper.arxiv_id,
                )
                failed += 1

        return {"fetched": len(papers), "stored": stored, "skipped": skipped, "processed": processed, "failed": failed}
