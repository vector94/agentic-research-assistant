import asyncio
from unittest.mock import AsyncMock

from src.schemas.rag import RagRequest, RagResponse
from src.services.cache.client import CacheClient


def test_cache_stores_and_restores_rag_response() -> None:
    redis_client = AsyncMock()
    request = RagRequest(query="How do transformers work?", top_k=3)
    response = RagResponse(
        query=request.query,
        answer="Transformers use attention.",
        sources=["https://arxiv.org/pdf/1234.5678"],
        chunks_used=3,
    )
    redis_client.get.return_value = response.model_dump_json()
    cache = CacheClient(redis_client=redis_client, ttl_seconds=3600)

    asyncio.run(cache.set(request, response))
    cached_response = asyncio.run(cache.get(request))

    cache_key = cache._make_key(request)
    redis_client.set.assert_awaited_once_with(
        cache_key,
        response.model_dump_json(),
        ex=3600,
    )
    redis_client.get.assert_awaited_once_with(cache_key)
    assert cached_response == response
