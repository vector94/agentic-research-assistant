import asyncio
import logging

from src.repositories.paper import PaperRepository
from src.services.paper_indexer import PaperIndexer

logger = logging.getLogger(__name__)


class PaperIndexingService:
    def __init__(self, paper_repository: PaperRepository, paper_indexer: PaperIndexer) -> None:
        self.paper_repository = paper_repository
        self.paper_indexer = paper_indexer

    async def index_processed_papers(self, limit: int = 100) -> dict[str, int]:
        papers = await self.paper_repository.list_processed(limit=limit)

        indexed = 0
        failed = 0

        for paper in papers:
            try:
                success = await asyncio.to_thread(
                    self.paper_indexer.index_paper,
                    paper,
                )

                if success:
                    indexed += 1
                else:
                    failed += 1
            except Exception:
                logger.exception(
                    "Failed to index paper %s",
                    paper.arxiv_id,
                )
                failed += 1

        return {"found": len(papers), "indexed": indexed, "failed": failed}
