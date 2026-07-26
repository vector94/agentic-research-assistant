class PdfProcessingError(Exception):
    """Base error for PDF processing failures."""


class PdfValidationError(PdfProcessingError):
    """Raised when a PDF fails validation."""


class PdfParsingError(PdfProcessingError):
    """Raised when a valid PDF cannot be parsed."""


class ArxivApiError(Exception):
    """Raised when the arXiv API request fails."""
