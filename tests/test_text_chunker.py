from src.schemas.pdf import PaperSection
from src.services.text_chunker import TextChunker


def test_chunk_text_creates_overlapping_chunks_with_metadata() -> None:
    chunker = TextChunker(chunk_size=5, overlap_size=2, min_chunk_size=2)
    text = "one two three four five six seven eight"

    chunks = chunker.chunk_text(
        text=text,
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
    )

    assert [chunk.text for chunk in chunks] == [
        "one two three four five",
        "four five six seven eight",
    ]
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.metadata.word_count for chunk in chunks] == [5, 5]
    assert chunks[0].metadata.overlap_with_previous == 0
    assert chunks[0].metadata.overlap_with_next == 2
    assert chunks[1].metadata.overlap_with_previous == 2
    assert chunks[1].metadata.overlap_with_next == 0
    assert all(chunk.arxiv_id == "1234.5678v1" for chunk in chunks)
    assert all(chunk.paper_id == "paper-1" for chunk in chunks)


def test_chunk_text_returns_empty_list_for_empty_text() -> None:
    chunker = TextChunker()

    assert chunker.chunk_text("", "1234.5678v1", "paper-1") == []


def test_chunk_paper_preserves_section_order_and_metadata() -> None:
    chunker = TextChunker(chunk_size=100, overlap_size=10, min_chunk_size=1)
    sections = [
        PaperSection(title="Introduction", content="Introduction content."),
        PaperSection(title="Methods", content="Methods content."),
    ]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract="Test abstract.",
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert len(chunks) == 2
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.metadata.section_title for chunk in chunks] == [
        "Introduction",
        "Methods",
    ]
    assert "Introduction content." in chunks[0].text
    assert "Methods content." in chunks[1].text
    assert all("Fallback text" not in chunk.text for chunk in chunks)
