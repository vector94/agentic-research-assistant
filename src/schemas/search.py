from datetime import datetime

from pydantic import BaseModel, Field


class PaperSearchResult(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: datetime
    pdf_url: str
    score: float
    highlights: dict[str, list[str]] = Field(default_factory=dict)


class PaperSearchResponse(BaseModel):
    total: int
    results: list[PaperSearchResult]
