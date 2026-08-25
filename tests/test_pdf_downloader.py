import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from src.services.pdf_downloader import PdfDownloader


class _FakeStreamResponse:
    def __init__(
        self,
        *,
        content_type: str = "application/pdf",
        chunks: list[bytes] | None = None,
        status_error: Exception | None = None,
    ) -> None:
        self.headers = {"Content-Type": content_type}
        self._chunks = chunks if chunks is not None else [b"%PDF-1.4 fake pdf content"]
        self._status_error = status_error

    async def __aenter__(self) -> "_FakeStreamResponse":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient; `stream_responses` is consumed one per call."""

    def __init__(self, stream_responses: list) -> None:
        self._responses = list(stream_responses)

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    def stream(self, method: str, url: str):
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _patched_client(stream_responses: list):
    return patch(
        "src.services.pdf_downloader.httpx.AsyncClient",
        return_value=_FakeAsyncClient(stream_responses),
    )


def test_download_pdf_reuses_a_valid_cached_file(tmp_path: Path) -> None:
    downloader = PdfDownloader(cache_dir=tmp_path)
    cached = tmp_path / "1234.5678.pdf"
    cached.write_bytes(b"%PDF-1.4 already downloaded")

    with _patched_client([]):
        result = asyncio.run(downloader.download_pdf("1234.5678", "https://example.com/paper.pdf"))

    assert result == cached
    assert cached.read_bytes() == b"%PDF-1.4 already downloaded"


def test_download_pdf_redownloads_a_corrupted_cached_file(tmp_path: Path) -> None:
    downloader = PdfDownloader(cache_dir=tmp_path)
    cached = tmp_path / "1234.5678.pdf"
    cached.write_bytes(b"not a real pdf")

    with _patched_client([_FakeStreamResponse()]):
        result = asyncio.run(downloader.download_pdf("1234.5678", "https://example.com/paper.pdf"))

    assert result == cached
    assert cached.read_bytes().startswith(b"%PDF-")


def test_download_pdf_retries_transient_failures_then_succeeds(tmp_path: Path) -> None:
    downloader = PdfDownloader(cache_dir=tmp_path, max_retries=3, retry_delay_seconds=0)

    responses = [
        httpx.ConnectError("connection refused"),
        _FakeStreamResponse(),
    ]

    with _patched_client(responses):
        result = asyncio.run(downloader.download_pdf("1234.5678", "https://example.com/paper.pdf"))

    assert result.exists()
    assert result.read_bytes().startswith(b"%PDF-")


def test_download_pdf_raises_after_exhausting_retries(tmp_path: Path) -> None:
    downloader = PdfDownloader(cache_dir=tmp_path, max_retries=2, retry_delay_seconds=0)
    responses = [httpx.ConnectError("down"), httpx.ConnectError("still down")]

    with _patched_client(responses), pytest.raises(httpx.ConnectError):
        asyncio.run(downloader.download_pdf("1234.5678", "https://example.com/paper.pdf"))

    assert not (tmp_path / "1234.5678.pdf").exists()


def test_download_pdf_does_not_retry_on_wrong_content_type(tmp_path: Path) -> None:
    downloader = PdfDownloader(cache_dir=tmp_path, max_retries=3, retry_delay_seconds=0)
    responses = [_FakeStreamResponse(content_type="text/html")]

    with _patched_client(responses), pytest.raises(ValueError, match="does not point to a PDF"):
        asyncio.run(downloader.download_pdf("1234.5678", "https://example.com/paper.pdf"))

    assert not (tmp_path / "1234.5678.pdf").exists()


def test_download_pdf_does_not_retry_when_file_exceeds_max_size(tmp_path: Path) -> None:
    downloader = PdfDownloader(cache_dir=tmp_path, max_size_mb=0, max_retries=3, retry_delay_seconds=0)
    responses = [_FakeStreamResponse(chunks=[b"x" * 10])]

    with _patched_client(responses), pytest.raises(ValueError, match="exceeds the maximum allowed size"):
        asyncio.run(downloader.download_pdf("1234.5678", "https://example.com/paper.pdf"))

    assert not (tmp_path / "1234.5678.pdf").exists()
