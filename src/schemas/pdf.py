from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class ParserType(StrEnum):
    PYPDF = "pypdf"
    DOCLING = "docling"


class PaperSection(BaseModel):
    title: str
    content: str
    level: int = 1


class PaperFigure(BaseModel):
    caption: str
    id: str


class PaperTable(BaseModel):
    caption: str
    id: str


class PdfContent(BaseModel):
    raw_text: str
    sections: list[PaperSection] = Field(default_factory=list)
    figures: list[PaperFigure] = Field(default_factory=list)
    tables: list[PaperTable] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    parser_used: ParserType
    parser_metadata: dict[str, Any] = Field(default_factory=dict)
