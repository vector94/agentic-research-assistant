from typing import Literal

from pydantic import BaseModel


class RagResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks_used: int
    search_mode: Literal["hybrid"] = "hybrid"
