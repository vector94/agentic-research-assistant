import asyncio
import logging

from src.models import Paper
from src.schemas.pdf import PaperSection
from src.services.embeddings.client import EmbeddingClient
from src.services.opensearch.client import OpenSearchClient
from src.services.text_chunker import TextChunker

logger = logging.getLogger(__name__)


class HybridIndexingService:
    def __init__(
        self,
        chunker: TextChunker,
        embedding_client: EmbeddingClient,
        opensearch_client: OpenSearchClient,
        embedding_model: str,
    ) -> None:
        self.chunker = chunker
        self.embedding_client = embedding_client
        self.opensearch_client = opensearch_client
        self.embedding_model = embedding_model

    async def index_paper(
        self,
        paper: Paper,
    ) -> dict[str, int]:
        sections = [PaperSection.model_validate(section) for section in paper.sections or []]

        chunks = self.chunker.chunk_paper(
            title=paper.title,
            abstract=paper.abstract,
            raw_text=paper.raw_text or "",
            arxiv_id=paper.arxiv_id,
            paper_id=str(paper.id),
            sections=sections,
        )

        if not chunks:
            return {
                "chunks_created": 0,
                "embeddings_generated": 0,
                "indexed": 0,
                "failed": 0,
            }

        embeddings = await self.embedding_client.embed_passages([chunk.text for chunk in chunks])

        if len(embeddings) != len(chunks):
            raise ValueError("The number of embeddings does not match the number of chunks")

        documents = []

        for chunk, embedding in zip(chunks, embeddings):
            documents.append(
                {
                    "chunk_id": f"{chunk.arxiv_id}-{chunk.metadata.chunk_index}",
                    "arxiv_id": chunk.arxiv_id,
                    "paper_id": chunk.paper_id,
                    "chunk_index": chunk.metadata.chunk_index,
                    "chunk_text": chunk.text,
                    "chunk_word_count": chunk.metadata.word_count,
                    "section_title": chunk.metadata.section_title,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": paper.authors,
                    "categories": paper.categories,
                    "published_at": paper.published_at.isoformat(),
                    "pdf_url": paper.pdf_url,
                    "embedding_model": self.embedding_model,
                    "embedding": embedding,
                }
            )

        result = await asyncio.to_thread(
            self.opensearch_client.bulk_index_chunks,
            documents,
        )

        return {
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
            "indexed": result["indexed"],
            "failed": result["failed"],
        }

    async def index_papers(
        self,
        papers: list[Paper],
    ) -> dict[str, int]:
        totals = {
            "papers_found": len(papers),
            "papers_indexed": 0,
            "papers_skipped": 0,
            "papers_failed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "chunks_indexed": 0,
            "chunks_failed": 0,
        }

        for paper in papers:
            try:
                result = await self.index_paper(paper)
            except Exception:
                logger.exception(
                    "Failed to index chunks for paper %s",
                    paper.arxiv_id,
                )
                totals["papers_failed"] += 1
                continue

            totals["chunks_created"] += result["chunks_created"]
            totals["embeddings_generated"] += result["embeddings_generated"]
            totals["chunks_indexed"] += result["indexed"]
            totals["chunks_failed"] += result["failed"]

            if result["chunks_created"] == 0:
                totals["papers_skipped"] += 1
            elif result["failed"] > 0:
                totals["papers_failed"] += 1
            else:
                totals["papers_indexed"] += 1

        return totals

    async def close(self) -> None:
        await self.embedding_client.close()
