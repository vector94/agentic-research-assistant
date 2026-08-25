import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from src.exceptions import OllamaConnectionError, OllamaTimeoutError
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


def test_generate_raises_ollama_timeout_error_on_timeout() -> None:
    client = OllamaClient(base_url="http://localhost:11434", model="llama3.2:1b")
    client.client.post = AsyncMock(side_effect=httpx.ReadTimeout("timed out"))

    try:
        with pytest.raises(OllamaTimeoutError):
            asyncio.run(client.generate("A prompt"))
    finally:
        asyncio.run(client.close())


def test_generate_raises_ollama_connection_error_on_connect_failure() -> None:
    client = OllamaClient(base_url="http://localhost:11434", model="llama3.2:1b")
    client.client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))

    try:
        with pytest.raises(OllamaConnectionError):
            asyncio.run(client.generate("A prompt"))
    finally:
        asyncio.run(client.close())


def test_generate_raises_ollama_connection_error_on_bad_status() -> None:
    request = httpx.Request("POST", "http://localhost:11434/api/generate")
    response = httpx.Response(500, request=request)

    client = OllamaClient(base_url="http://localhost:11434", model="llama3.2:1b")
    client.client.post = AsyncMock(return_value=response)

    try:
        with pytest.raises(OllamaConnectionError):
            asyncio.run(client.generate("A prompt"))
    finally:
        asyncio.run(client.close())


def test_health_check_returns_false_on_connection_failure() -> None:
    client = OllamaClient(base_url="http://localhost:11434", model="llama3.2:1b")
    client.client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))

    try:
        result = asyncio.run(client.health_check())
    finally:
        asyncio.run(client.close())

    assert result is False
