from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    chunk_index: int
    word_count: int
    overlap_with_previous: int
    overlap_with_next: int
    section_title: str | None = None


class TextChunk(BaseModel):
    text: str
    arxiv_id: str
    paper_id: str
    metadata: ChunkMetadata
