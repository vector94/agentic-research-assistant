import asyncio
from pathlib import Path

from src.schemas.pdf import PdfContent
from src.services.pdf_parser.docling import DoclingParser


class PdfParserService:
    def __init__(
        self,
        max_pages: int,
        max_file_size_mb: int,
        do_ocr: bool = False,
        do_table_structure: bool = True,
    ) -> None:
        self.parser = DoclingParser(
            max_pages=max_pages,
            max_file_size_mb=max_file_size_mb,
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )

    async def parse(self, pdf_path: Path) -> PdfContent:
        return await asyncio.to_thread(
            self.parser.parse,
            pdf_path,
        )
