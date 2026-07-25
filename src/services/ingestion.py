import asyncio

from src.repositories.paper import PaperRepository
from src.services.arxiv import ArxivClient


class PaperIngestionService:
    def __init__(self, arxiv_client: ArxivClient, paper_repository: PaperRepository) -> None:
        self.arxiv_client = arxiv_client
        self.paper_repository = paper_repository

    async def ingest(self, query: str, max_results: int = 5) -> dict[str, int]:
        papers = await asyncio.to_thread(self.arxiv_client.search, query=query, max_results=max_results)

        stored = 0
        skipped = 0

        for paper in papers:
            existing_paper = await self.paper_repository.get_by_arxiv_id(paper.arxiv_id)
            if existing_paper is None:
                await self.paper_repository.add(paper)
                stored += 1
            else:
                skipped += 1

        return {
            "fetched": len(papers),
            "stored": stored,
            "skipped": skipped,
        }
