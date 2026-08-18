from typing import Literal

from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    query: str = Field(min_length=2)
    top_k: int = Field(default=3, ge=1, le=10)


class RagResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks_used: int
    search_mode: Literal["hybrid"] = "hybrid"


class AgenticRagResponse(RagResponse):
    reasoning_steps: list[str]
    retrieval_attempts: int
