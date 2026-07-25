from datetime import datetime

from pydantic import BaseModel


class ArxivPaper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    published_at: datetime
    pdf_url: str
