import httpx


class JinaEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        model: str = "jina-embeddings-v3",
        dimensions: int = 1024,
        base_url: str = "https://api.jina.ai/v1",
    ) -> None:
        if not api_key:
            raise ValueError("Jina API key is required")

        self.model = model
        self.dimensions = dimensions

        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    async def embed_passages(
        self,
        texts: list[str],
        batch_size: int = 100,
    ) -> list[list[float]]:
        if batch_size <= 0:
            raise ValueError("Batch size must be greater than zero")

        if not texts:
            return []

        embeddings: list[list[float]] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            response = await self.client.post(
                "embeddings",
                json={
                    "model": self.model,
                    "task": "retrieval.passage",
                    "dimensions": self.dimensions,
                    "input": batch,
                },
            )
            response.raise_for_status()

            response_data = response.json()
            batch_embeddings = [item["embedding"] for item in response_data["data"]]
            embeddings.extend(batch_embeddings)

        return embeddings

    async def embed_query(
        self,
        query: str,
    ) -> list[float]:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        response = await self.client.post(
            "embeddings",
            json={
                "model": self.model,
                "task": "retrieval.query",
                "dimensions": self.dimensions,
                "input": [query],
            },
        )
        response.raise_for_status()

        response_data = response.json()
        return response_data["data"][0]["embedding"]

    async def close(self) -> None:
        await self.client.aclose()
