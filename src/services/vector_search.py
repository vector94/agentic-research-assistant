import asyncio

from src.schemas.retrieval import ChunkSearchResponse, ChunkSearchResult
from src.services.embeddings.client import EmbeddingClient
from src.services.opensearch.client import OpenSearchClient


class VectorSearchService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        opensearch_client: OpenSearchClient,
    ) -> None:
        self.embedding_client = embedding_client
        self.opensearch_client = opensearch_client

    async def search(
        self,
        query: str,
        size: int = 10,
    ) -> ChunkSearchResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        query_embedding = await self.embedding_client.embed_query(query)
        response = await asyncio.to_thread(
            self.opensearch_client.search_chunks_vector,
            query_embedding,
            size,
        )

        hits = response["hits"]
        results = [
            ChunkSearchResult(
                **hit["_source"],
                score=hit["_score"],
            )
            for hit in hits["hits"]
        ]

        return ChunkSearchResponse(
            total=hits["total"]["value"],
            results=results,
        )

    async def close(self) -> None:
        await self.embedding_client.close()
