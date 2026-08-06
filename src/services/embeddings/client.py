from typing import Protocol


class EmbeddingClient(Protocol):
    async def embed_passages(
        self,
        texts: list[str],
    ) -> list[list[float]]: ...

    async def embed_query(
        self,
        query: str,
    ) -> list[float]: ...

    async def close(self) -> None: ...
