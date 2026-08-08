from datetime import datetime

from pydantic import BaseModel


class ChunkSearchResult(BaseModel):
    chunk_id: str
    arxiv_id: str
    paper_id: str
    chunk_index: int
    chunk_text: str
    chunk_word_count: int
    section_title: str | None
    title: str
    abstract: str
    authors: list[str]
    categories: list[str]
    published_at: datetime
    pdf_url: str
    score: float


class ChunkSearchResponse(BaseModel):
    total: int
    results: list[ChunkSearchResult]
