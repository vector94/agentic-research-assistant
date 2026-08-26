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


def test_chunk_paper_repeats_the_header_on_every_chunk_of_a_large_section() -> None:
    chunker = TextChunker(chunk_size=20, overlap_size=5, min_chunk_size=1)
    sections = [PaperSection(title="Method", content=" ".join(f"word{i}" for i in range(50)))]

    chunks = chunker.chunk_paper(
        title="My Paper Title",
        abstract="My paper abstract.",
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("My Paper Title\n\nMy paper abstract.\n\nSection: Method\n\n")


def test_chunk_paper_keeps_a_large_section_as_its_own_chunk() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    sections = [
        PaperSection(title="Introduction", content=" ".join(["word"] * 150)),
        PaperSection(title="Methods", content=" ".join(["term"] * 150)),
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
    assert [chunk.metadata.section_title for chunk in chunks] == ["Introduction", "Methods"]
    assert all("Fallback text" not in chunk.text for chunk in chunks)


def test_chunk_paper_merges_small_adjacent_sections_into_one_chunk() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
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

    assert len(chunks) == 1
    assert chunks[0].metadata.section_title == "Introduction + Methods"
    assert "Introduction content." in chunks[0].text
    assert "Methods content." in chunks[0].text


def test_chunk_paper_merges_a_trailing_small_section_into_the_previous_chunk() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    sections = [
        PaperSection(title="Related Work", content="Related work content."),
        PaperSection(title="Methods", content=" ".join(["term"] * 150)),
        PaperSection(title="Limitations", content="Limitations content."),
    ]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract="Test abstract.",
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert [chunk.metadata.section_title for chunk in chunks] == [
        "Related Work",
        "Methods + Limitations",
    ]
    assert "Limitations content." in chunks[-1].text


def test_chunk_paper_handles_multiple_separate_merge_cycles() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    sections = [
        PaperSection(title="Intro", content="Intro content."),
        PaperSection(title="Methods", content=" ".join(["term"] * 150)),
        PaperSection(title="Results", content="Results content."),
        PaperSection(title="Discussion", content=" ".join(["word"] * 150)),
        PaperSection(title="Conclusion", content="Conclusion content."),
    ]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract="Test abstract.",
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert [chunk.metadata.section_title for chunk in chunks] == [
        "Intro",
        "Methods + Results",
        "Discussion + Conclusion",
    ]
    assert [chunk.metadata.chunk_index for chunk in chunks] == [0, 1, 2]
    assert "Results content." in chunks[1].text
    assert "Conclusion content." in chunks[2].text


def test_chunk_paper_keeps_a_large_combined_tail_as_its_own_chunk() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    sections = [
        PaperSection(title="Methods", content=" ".join(["term"] * 150)),
        PaperSection(title="Limitations", content=" ".join(["word"] * 99)),
        PaperSection(title="Ethics", content=" ".join(["word"] * 99)),
    ]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract="Test abstract.",
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert [chunk.metadata.section_title for chunk in chunks] == [
        "Methods",
        "Limitations + Ethics",
    ]


def test_chunk_paper_skips_sections_with_metadata_titles() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    sections = [
        PaperSection(title="Authors", content="Ada Lovelace, Charles Babbage"),
        PaperSection(title="Introduction", content=" ".join(["word"] * 150)),
    ]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract="Test abstract.",
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert [chunk.metadata.section_title for chunk in chunks] == ["Introduction"]


def test_chunk_paper_skips_sections_that_duplicate_the_abstract() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    abstract = " ".join([f"word{i}" for i in range(30)])
    sections = [
        PaperSection(title="Overview", content=abstract),
        PaperSection(title="Introduction", content=" ".join(["term"] * 150)),
    ]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract=abstract,
        raw_text="Fallback text that should not be used.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert [chunk.metadata.section_title for chunk in chunks] == ["Introduction"]


def test_chunk_paper_falls_back_to_full_text_when_all_sections_are_filtered() -> None:
    chunker = TextChunker(chunk_size=1000, overlap_size=10, min_chunk_size=1)
    sections = [PaperSection(title="Authors", content="Ada Lovelace")]

    chunks = chunker.chunk_paper(
        title="Test paper",
        abstract="Test abstract.",
        raw_text="This is the fallback body text for the paper.",
        arxiv_id="1234.5678v1",
        paper_id="paper-1",
        sections=sections,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata.section_title is None
    assert "fallback body text" in chunks[0].text
