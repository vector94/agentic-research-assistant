import json
from collections.abc import AsyncIterator
from typing import TypeVar

import httpx
from pydantic import BaseModel

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
        )

    async def generate(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = await self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()

        return response.json()["response"]

    async def generate_structured(
        self,
        prompt: str,
        response_model: type[ResponseModelT],
    ) -> ResponseModelT:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        response = await self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": response_model.model_json_schema(),
                "options": {"temperature": 0},
            },
        )
        response.raise_for_status()

        return response_model.model_validate_json(response.json()["response"])

    async def generate_stream(self, prompt: str) -> AsyncIterator[str]:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        async with self.client.stream(
            "POST",
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                data = json.loads(line)
                text = data.get("response", "")

                if text:
                    yield text

    async def close(self) -> None:
        await self.client.aclose()
