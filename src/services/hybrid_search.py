import asyncio
from typing import Any

from src.schemas.retrieval import ChunkSearchResponse, ChunkSearchResult
from src.services.embeddings.client import EmbeddingClient
from src.services.opensearch.client import OpenSearchClient


class HybridSearchService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        opensearch_client: OpenSearchClient,
        rrf_rank_constant: int = 60,
    ) -> None:
        if rrf_rank_constant <= 0:
            raise ValueError("RRF rank constant must be greater than zero")

        self.embedding_client = embedding_client
        self.opensearch_client = opensearch_client
        self.rrf_rank_constant = rrf_rank_constant

    async def search(
        self,
        query: str,
        size: int = 10,
    ) -> ChunkSearchResponse:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        candidate_size = max(size * 2, 10)
        bm25_task = asyncio.create_task(
            asyncio.to_thread(
                self.opensearch_client.search_chunks_bm25,
                query,
                candidate_size,
            )
        )

        query_embedding = await self.embedding_client.embed_query(query)
        vector_task = asyncio.create_task(
            asyncio.to_thread(
                self.opensearch_client.search_chunks_vector,
                query_embedding,
                candidate_size,
            )
        )

        bm25_response, vector_response = await asyncio.gather(
            bm25_task,
            vector_task,
        )
        results = self._fuse_results(
            bm25_response=bm25_response,
            vector_response=vector_response,
            size=size,
        )

        return ChunkSearchResponse(
            total=len(results),
            results=results,
        )

    def _fuse_results(
        self,
        bm25_response: dict[str, Any],
        vector_response: dict[str, Any],
        size: int,
    ) -> list[ChunkSearchResult]:
        scores: dict[str, float] = {}
        sources: dict[str, dict[str, Any]] = {}

        for response in (bm25_response, vector_response):
            for rank, hit in enumerate(response["hits"]["hits"], start=1):
                source = hit["_source"]
                chunk_id = source["chunk_id"]
                sources[chunk_id] = source
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.rrf_rank_constant + rank)

        ranked_chunk_ids = sorted(
            scores,
            key=scores.get,
            reverse=True,
        )[:size]

        return [
            ChunkSearchResult(
                **sources[chunk_id],
                score=scores[chunk_id],
            )
            for chunk_id in ranked_chunk_ids
        ]

    async def close(self) -> None:
        await self.embedding_client.close()
