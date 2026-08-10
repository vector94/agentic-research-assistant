import asyncio

from src.schemas.search import PaperSearchResponse, PaperSearchResult
from src.services.opensearch.client import OpenSearchClient


class PaperSearchService:
    def __init__(self, opensearch_client: OpenSearchClient) -> None:
        self.opensearch_client = opensearch_client

    async def search(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = False,
        fuzzy: bool = False,
    ) -> PaperSearchResponse:
        response = await asyncio.to_thread(
            self.opensearch_client.search_papers,
            query=query,
            size=size,
            from_=from_,
            categories=categories,
            latest=latest,
            fuzzy=fuzzy,
        )

        hits = response["hits"]

        results = [
            PaperSearchResult(
                **hit["_source"],
                score=hit["_score"],
                highlights=hit.get("highlight", {}),
            )
            for hit in hits["hits"]
        ]

        return PaperSearchResponse(
            total=hits["total"]["value"],
            results=results,
        )
