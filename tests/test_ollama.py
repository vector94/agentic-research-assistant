import asyncio
from unittest.mock import AsyncMock, Mock

from src.services.agents import GuardrailResult
from src.services.ollama.client import OllamaClient


def test_generate_structured_sends_schema_and_validates_response() -> None:
    response = Mock()
    response.json.return_value = {
        "response": '{"score": 85, "reason": "Research question"}',
    }

    client = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:1b",
    )
    client.client.post = AsyncMock(return_value=response)

    try:
        result = asyncio.run(
            client.generate_structured(
                prompt="Score this question",
                response_model=GuardrailResult,
            )
        )
    finally:
        asyncio.run(client.close())

    client.client.post.assert_awaited_once_with(
        "/api/generate",
        json={
            "model": "llama3.2:1b",
            "prompt": "Score this question",
            "stream": False,
            "format": GuardrailResult.model_json_schema(),
            "options": {"temperature": 0},
        },
    )
    response.raise_for_status.assert_called_once_with()
    assert result == GuardrailResult(
        score=85,
        reason="Research question",
    )
