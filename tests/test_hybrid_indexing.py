import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from src.models import Paper
from src.schemas.indexing.models import ChunkMetadata, TextChunk
from src.services.hybrid_indexing import HybridIndexingService


def test_index_paper_chunks_embeds_and_indexes_documents() -> None:
    paper = Paper(
        id=uuid4(),
        arxiv_id="1234.5678v1",
        title="Test paper",
        abstract="Test abstract",
        authors=["Ada Lovelace"],
        categories=["cs.AI"],
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
        raw_text="Parsed paper content",
        sections=[],
    )
    chunk = TextChunk(
        text="A meaningful paper chunk",
        arxiv_id=paper.arxiv_id,
        paper_id=str(paper.id),
        metadata=ChunkMetadata(
            chunk_index=0,
            word_count=4,
            overlap_with_previous=0,
            overlap_with_next=0,
            section_title="Introduction",
        ),
    )

    chunker = Mock()
    chunker.chunk_paper.return_value = [chunk]

    embedding_client = Mock()
    embedding_client.embed_passages = AsyncMock(return_value=[[0.1, 0.2]])

    opensearch_client = Mock()
    opensearch_client.bulk_index_chunks.return_value = {
        "indexed": 1,
        "failed": 0,
    }

    service = HybridIndexingService(
        chunker=chunker,
        embedding_client=embedding_client,
        opensearch_client=opensearch_client,
        embedding_model="jina-embeddings-v3",
    )

    result = asyncio.run(service.index_paper(paper))

    assert result == {
        "chunks_created": 1,
        "embeddings_generated": 1,
        "indexed": 1,
        "failed": 0,
    }
    embedding_client.embed_passages.assert_awaited_once_with([chunk.text])

    indexed_documents = opensearch_client.bulk_index_chunks.call_args.args[0]
    assert indexed_documents == [
        {
            "chunk_id": "1234.5678v1-0",
            "arxiv_id": "1234.5678v1",
            "paper_id": str(paper.id),
            "chunk_index": 0,
            "chunk_text": "A meaningful paper chunk",
            "chunk_word_count": 4,
            "section_title": "Introduction",
            "title": "Test paper",
            "abstract": "Test abstract",
            "authors": ["Ada Lovelace"],
            "categories": ["cs.AI"],
            "published_at": "2026-01-01T00:00:00+00:00",
            "pdf_url": "https://arxiv.org/pdf/1234.5678",
            "embedding_model": "jina-embeddings-v3",
            "embedding": [0.1, 0.2],
        }
    ]
