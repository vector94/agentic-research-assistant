from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from src.exceptions import PdfParsingError, PdfValidationError
from src.schemas.pdf import PaperSection, ParserType, PdfContent


class DoclingParser:
    def __init__(
        self, max_pages: int = 30, max_file_size_mb: int = 20, do_ocr: bool = False, do_table_structure: bool = True
    ) -> None:
        pipeline_options = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
        )

        self.converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        self.max_pages = max_pages
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024

    def _validate_pdf(self, pdf_path: Path) -> None:
        if not pdf_path.exists() or not pdf_path.is_file():
            raise PdfValidationError(f"PDF file does not exist: {pdf_path}")

        file_size = pdf_path.stat().st_size
        if file_size == 0:
            raise PdfValidationError(f"PDF file is empty: {pdf_path}")

        if file_size > self.max_file_size_bytes:
            raise PdfValidationError(
                f"PDF file exceeds the maximum allowed size of {self.max_file_size_bytes / (1024 * 1024)} MB: {pdf_path}"
            )

        with pdf_path.open("rb") as file:
            if not file.read(5).startswith(b"%PDF-"):
                raise PdfValidationError(f"File does not have a valid PDF header: {pdf_path}")

        document = pdfium.PdfDocument(str(pdf_path))

        try:
            page_count = len(document)
        finally:
            document.close()

        if page_count > self.max_pages:
            raise PdfValidationError(
                f"PDF file exceeds the maximum allowed number of pages ({self.max_pages}): {pdf_path}"
            )

    def parse(self, pdf_path: Path) -> PdfContent:
        self._validate_pdf(pdf_path)

        try:
            result = self.converter.convert(pdf_path)
        except Exception as error:
            raise PdfParsingError(f"Failed to parse PDF: {pdf_path.name}") from error

        document = result.document

        sections: list[PaperSection] = []
        current_title = "Content"
        current_content: list[str] = []

        for element in document.texts:
            label = getattr(element, "label", None)
            text = getattr(element, "text", "").strip()

            if not text:
                continue

            if label in {"title", "section_header"}:
                if current_content:
                    sections.append(
                        PaperSection(
                            title=current_title,
                            content="\n".join(current_content),
                        )
                    )

                current_content = []
                current_title = text

            else:
                current_content.append(text)

        if current_content:
            sections.append(
                PaperSection(
                    title=current_title,
                    content="\n".join(current_content),
                )
            )

        return PdfContent(
            raw_text=document.export_to_text(),
            sections=sections,
            parser_used=ParserType.DOCLING,
            metadata={"source": "docling"},
        )
