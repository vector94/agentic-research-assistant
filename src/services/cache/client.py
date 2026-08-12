import hashlib

from redis.asyncio import Redis

from src.schemas.rag import RagRequest, RagResponse


class CacheClient:
    def __init__(
        self,
        redis_client: Redis,
        ttl_seconds: int,
    ) -> None:
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _make_key(request: RagRequest) -> str:
        request_json = request.model_dump_json()
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        return f"rag_response:{request_hash}"

    async def get(self, request: RagRequest) -> RagResponse | None:
        cached_json = await self.redis_client.get(self._make_key(request))

        if cached_json is None:
            return None

        return RagResponse.model_validate_json(cached_json)

    async def set(
        self,
        request: RagRequest,
        response: RagResponse,
    ) -> None:
        await self.redis_client.set(
            self._make_key(request),
            response.model_dump_json(),
            ex=self.ttl_seconds,
        )

    async def close(self) -> None:
        await self.redis_client.aclose()
