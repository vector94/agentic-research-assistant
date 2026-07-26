import asyncio
from pathlib import Path

import httpx


class PdfDownloader:
    def __init__(self, cache_dir: Path = Path("data/arxiv_pdfs"), max_size_mb: int = 20) -> None:
        self.cache_dir = cache_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def download_pdf(self, arxiv_id: str, pdf_url: str) -> Path:
        safe_arxiv_id = arxiv_id.replace("/", "_")
        pdf_path = self.cache_dir / f"{safe_arxiv_id}.pdf"

        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return pdf_path

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

            return pdf_path
