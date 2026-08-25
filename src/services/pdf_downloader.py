import asyncio
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"


class PdfDownloader:
    def __init__(
        self,
        cache_dir: Path = Path("data/arxiv_pdfs"),
        max_size_mb: int = 20,
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ) -> None:
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _is_valid_cached_pdf(self, pdf_path: Path) -> bool:
        if not pdf_path.exists() or pdf_path.stat().st_size == 0:
            return False

        with pdf_path.open("rb") as f:
            return f.read(len(_PDF_MAGIC)) == _PDF_MAGIC

    async def download_pdf(self, arxiv_id: str, pdf_url: str) -> Path:
        safe_arxiv_id = arxiv_id.replace("/", "_")
        pdf_path = self.cache_dir / f"{safe_arxiv_id}.pdf"

        if await asyncio.to_thread(self._is_valid_cached_pdf, pdf_path):
            return pdf_path

        if pdf_path.exists():
            logger.warning(
                "Discarding corrupted cached PDF for %s, re-downloading",
                arxiv_id,
            )
            await asyncio.to_thread(pdf_path.unlink)

        last_error: httpx.HTTPError | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                await self._download_once(pdf_path, pdf_url)
                return pdf_path
            except httpx.HTTPError as error:
                await asyncio.to_thread(pdf_path.unlink, missing_ok=True)
                last_error = error

                if attempt < self.max_retries:
                    logger.warning(
                        "PDF download attempt %s/%s failed for %s: %s",
                        attempt,
                        self.max_retries,
                        arxiv_id,
                        error,
                    )
                    await asyncio.sleep(self.retry_delay_seconds * attempt)
            except ValueError:
                await asyncio.to_thread(pdf_path.unlink, missing_ok=True)
                raise

        assert last_error is not None  # noqa: S101 - loop above always sets this before falling through
        raise last_error

    async def _download_once(self, pdf_path: Path, pdf_url: str) -> None:
        timeout = httpx.Timeout(30.0)

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            async with client.stream("GET", pdf_url) as response:
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                if "application/pdf" not in content_type:
                    raise ValueError(f"URL does not point to a PDF file. Content-Type: {content_type}")

                chunks: list[bytes] = []
                downloaded_size = 0

                async for chunk in response.aiter_bytes():
                    downloaded_size += len(chunk)

                    if downloaded_size > self.max_size_bytes:
                        raise ValueError(f"PDF file exceeds the maximum allowed size of {self.max_size_bytes} bytes.")

                    chunks.append(chunk)

            pdf_content = b"".join(chunks)
            await asyncio.to_thread(pdf_path.write_bytes, pdf_content)
