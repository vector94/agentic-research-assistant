from redis.asyncio import Redis

from src.config import Settings, get_settings
from src.services.cache.client import CacheClient


def make_cache_client(settings: Settings | None = None) -> CacheClient:
    if settings is None:
        settings = get_settings()

    redis_client = Redis.from_url(
        settings.redis.url,
        decode_responses=True,
    )

    return CacheClient(
        redis_client=redis_client,
        ttl_seconds=settings.redis.cache_ttl_seconds,
    )
