class PdfProcessingError(Exception):
    """Base error for PDF processing failures."""


class PdfValidationError(PdfProcessingError):
    """Raised when a PDF fails validation."""


class PdfParsingError(PdfProcessingError):
    """Raised when a valid PDF cannot be parsed."""


class ArxivApiError(Exception):
    """Raised when the arXiv API request fails."""


class OllamaError(Exception):
    """Base error for Ollama request failures."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached or returns an error."""


class OllamaTimeoutError(OllamaError):
    """Raised when an Ollama request times out."""
