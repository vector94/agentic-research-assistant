from src.schemas.indexing.models import ChunkMetadata, TextChunk
from src.schemas.pdf import PaperSection


class TextChunker:
    def __init__(
        self,
        chunk_size: int = 600,
        overlap_size: int = 100,
        min_chunk_size: int = 100,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero")

        if overlap_size < 0:
            raise ValueError("Overlap size cannot be negative")

        if overlap_size >= chunk_size:
            raise ValueError("Overlap size must be less than chunk size")

        if min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be greater than zero")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size

    def chunk_paper(
        self,
        title: str,
        abstract: str,
        raw_text: str,
        arxiv_id: str,
        paper_id: str,
        sections: list[PaperSection] | None = None,
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []

        if sections:
            for section in sections:
                section_parts = [
                    title,
                    abstract,
                    f"Section: {section.title}",
                    section.content,
                ]

                section_text = "\n\n".join(part for part in section_parts if part)

                section_chunks = self.chunk_text(
                    text=section_text,
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                )

                for chunk in section_chunks:
                    chunk.metadata.chunk_index = len(chunks)
                    chunk.metadata.section_title = section.title
                    chunks.append(chunk)

            if chunks:
                return chunks

        document_parts = [title, abstract, raw_text]
        full_text = "\n\n".join(part for part in document_parts if part)

        return self.chunk_text(
            text=full_text,
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

    def chunk_text(
        self,
        text: str,
        arxiv_id: str,
        paper_id: str,
    ) -> list[TextChunk]:
        words = text.split()

        if not words:
            return []

        if len(words) < self.min_chunk_size:
            return [
                self._create_chunk(
                    words=words,
                    chunk_index=0,
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                    overlap_with_previous=0,
                    overlap_with_next=0,
                )
            ]

        chunks: list[TextChunk] = []
        step_size = self.chunk_size - self.overlap_size
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]

            chunks.append(
                self._create_chunk(
                    words=chunk_words,
                    chunk_index=len(chunks),
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                    overlap_with_previous=self.overlap_size if start > 0 else 0,
                    overlap_with_next=self.overlap_size if end < len(words) else 0,
                )
            )

            if end == len(words):
                break

            start += step_size

        return chunks

    @staticmethod
    def _create_chunk(
        words: list[str],
        chunk_index: int,
        arxiv_id: str,
        paper_id: str,
        overlap_with_previous: int,
        overlap_with_next: int,
    ) -> TextChunk:
        return TextChunk(
            text=" ".join(words),
            arxiv_id=arxiv_id,
            paper_id=paper_id,
            metadata=ChunkMetadata(
                chunk_index=chunk_index,
                word_count=len(words),
                overlap_with_previous=overlap_with_previous,
                overlap_with_next=overlap_with_next,
            ),
        )
