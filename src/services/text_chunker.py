from src.schemas.indexing.models import ChunkMetadata, TextChunk
from src.schemas.pdf import PaperSection

_METADATA_TITLE_INDICATORS = {
    "content",
    "header",
    "authors",
    "author",
    "affiliation",
    "email",
    "arxiv",
    "preprint",
    "submitted",
    "received",
    "accepted",
}

_METADATA_CONTENT_PATTERNS = (
    "@",
    "arxiv:",
    "university",
    "institute",
    "department",
    "college",
    "gmail.com",
    "edu",
    "ac.uk",
    "preprint",
)

_ABSTRACT_DUPLICATE_OVERLAP_RATIO = 0.8
_SMALL_SECTION_WORD_THRESHOLD = 100
_MERGE_INTO_PREVIOUS_WORD_THRESHOLD = 200


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
        if sections:
            section_chunks = self._chunk_by_sections(
                title=title,
                abstract=abstract,
                sections=sections,
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )

            if section_chunks:
                return section_chunks

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

    def _chunk_by_sections(
        self,
        title: str,
        abstract: str,
        sections: list[PaperSection],
        arxiv_id: str,
        paper_id: str,
    ) -> list[TextChunk]:
        filtered_sections = self._filter_sections(sections, abstract)

        if not filtered_sections:
            return []

        chunks: list[TextChunk] = []
        pending: list[PaperSection] = []
        header_word_count = len(title.split()) + len(abstract.split())

        def flush_pending() -> None:
            if not pending:
                return

            combined_title = self._combined_section_title(pending)
            combined_content = "\n\n".join(f"Section: {section.title}\n\n{section.content}" for section in pending)

            if chunks and len(combined_content.split()) + header_word_count < _MERGE_INTO_PREVIOUS_WORD_THRESHOLD:
                self._merge_into_previous_chunk(chunks, combined_title, combined_content)
            else:
                self._append_section_chunks(
                    chunks=chunks,
                    title=title,
                    abstract=abstract,
                    section_title=combined_title,
                    section_content=combined_content,
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                )

            pending.clear()

        word_counts = [len(section.content.split()) for section in filtered_sections]

        for index, section in enumerate(filtered_sections):
            if word_counts[index] < _SMALL_SECTION_WORD_THRESHOLD:
                pending.append(section)

                is_last = index == len(filtered_sections) - 1
                next_is_large = not is_last and word_counts[index + 1] >= _SMALL_SECTION_WORD_THRESHOLD

                if is_last or next_is_large:
                    flush_pending()

                continue

            flush_pending()
            self._append_section_chunks(
                chunks=chunks,
                title=title,
                abstract=abstract,
                section_title=section.title,
                section_content=section.content,
                arxiv_id=arxiv_id,
                paper_id=paper_id,
            )

        flush_pending()

        return chunks

    def _append_section_chunks(
        self,
        chunks: list[TextChunk],
        title: str,
        abstract: str,
        section_title: str,
        section_content: str,
        arxiv_id: str,
        paper_id: str,
    ) -> None:
        header_parts = [title, abstract, f"Section: {section_title}"]
        header = "\n\n".join(part for part in header_parts if part)

        content_chunks = self.chunk_text(
            text=section_content,
            arxiv_id=arxiv_id,
            paper_id=paper_id,
        )

        for content_chunk in content_chunks:
            full_text = f"{header}\n\n{content_chunk.text}" if header else content_chunk.text

            chunks.append(
                TextChunk(
                    text=full_text,
                    arxiv_id=arxiv_id,
                    paper_id=paper_id,
                    metadata=ChunkMetadata(
                        chunk_index=len(chunks),
                        word_count=len(full_text.split()),
                        overlap_with_previous=content_chunk.metadata.overlap_with_previous,
                        overlap_with_next=content_chunk.metadata.overlap_with_next,
                        section_title=section_title,
                    ),
                )
            )

    @staticmethod
    def _merge_into_previous_chunk(
        chunks: list[TextChunk],
        combined_title: str,
        combined_content: str,
    ) -> None:
        previous = chunks[-1]
        merged_text = f"{previous.text}\n\n{combined_content}"

        chunks[-1] = TextChunk(
            text=merged_text,
            arxiv_id=previous.arxiv_id,
            paper_id=previous.paper_id,
            metadata=ChunkMetadata(
                chunk_index=previous.metadata.chunk_index,
                word_count=len(merged_text.split()),
                overlap_with_previous=previous.metadata.overlap_with_previous,
                overlap_with_next=0,
                section_title=f"{previous.metadata.section_title} + {combined_title}",
            ),
        )

    @staticmethod
    def _combined_section_title(sections: list[PaperSection]) -> str:
        titles = [section.title for section in sections]
        combined = " + ".join(titles[:3])

        if len(titles) > 3:
            combined += f" + {len(titles) - 3} more"

        return combined

    def _filter_sections(
        self,
        sections: list[PaperSection],
        abstract: str,
    ) -> list[PaperSection]:
        abstract_words = set(abstract.lower().split())
        abstract_lower = abstract.lower().strip()
        filtered: list[PaperSection] = []

        for section in sections:
            content = section.content.strip()

            if not content:
                continue

            if self._is_metadata_title(section.title):
                continue

            if self._is_duplicate_abstract(content, abstract_lower, abstract_words):
                continue

            if len(content.split()) < 20 and self._is_metadata_content(content):
                continue

            filtered.append(section)

        return filtered

    @staticmethod
    def _is_metadata_title(title: str) -> bool:
        title_lower = title.lower().strip()

        if title_lower in _METADATA_TITLE_INDICATORS or len(title_lower) < 5:
            return True

        return any(indicator in title_lower and len(title_lower) < 20 for indicator in _METADATA_TITLE_INDICATORS)

    @staticmethod
    def _is_duplicate_abstract(
        content: str,
        abstract_lower: str,
        abstract_words: set[str],
    ) -> bool:
        content_lower = content.lower()

        if abstract_lower and (abstract_lower in content_lower or content_lower in abstract_lower):
            return True

        if len(abstract_words) <= 10:
            return False

        content_words = set(content_lower.split())
        overlap = len(abstract_words & content_words)

        return overlap / len(abstract_words) > _ABSTRACT_DUPLICATE_OVERLAP_RATIO

    @staticmethod
    def _is_metadata_content(content: str) -> bool:
        content_lower = content.lower()
        matches = sum(1 for pattern in _METADATA_CONTENT_PATTERNS if pattern in content_lower)

        return matches >= 2

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
