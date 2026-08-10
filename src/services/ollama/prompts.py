from src.schemas.retrieval import ChunkSearchResult


class RagPromptBuilder:
    def build(
        self,
        query: str,
        chunks: list[ChunkSearchResult],
    ) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        context_parts: list[str] = []

        for position, chunk in enumerate(chunks, start=1):
            context_parts.append(f"[Source {position}: arXiv:{chunk.arxiv_id}]\n{chunk.chunk_text}")

        context = "\n\n".join(context_parts)

        return (
            "You are a research assistant. Answer the question using only the "
            "provided excerpts from research papers. If the excerpts do not contain "
            "enough information, say that clearly. Cite claims using the arXiv "
            "identifier shown in each source, for example [arXiv:1234.5678v1].\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer:"
        )
