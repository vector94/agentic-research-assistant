from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Assessment of whether a question belongs to the research domain."""

    score: int = Field(ge=0, le=100)
    reason: str = Field(min_length=1)


class DocumentGrade(BaseModel):
    """Assessment of whether retrieved chunks can answer a question."""

    is_relevant: bool
    reason: str = Field(min_length=1)
