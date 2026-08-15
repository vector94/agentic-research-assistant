from src.schemas.retrieval import ChunkSearchResult


class AgentPromptBuilder:
    def build_guardrail(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return (
            "Score how closely this question relates to academic research in "
            "computer science, artificial intelligence, or machine learning. "
            "Use a score from 0 to 100 and briefly explain the score. Questions "
            "about research papers, models, methods, or technical findings should "
            "score highly. Greetings, basic calculations, and unrelated subjects "
            "should score poorly.\n\n"
            f"Question:\n{query}\n\n"
            "Return valid JSON only in this form:\n"
            '{"score": 0, "reason": "brief explanation"}'
        )

    def build_document_grade(
        self,
        query: str,
        chunks: list[ChunkSearchResult],
    ) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        if not chunks:
            raise ValueError("At least one chunk is required")

        context = self._format_chunks(chunks)

        return (
            "Decide whether the research excerpts contain relevant information "
            "that can help answer the question. Briefly explain the decision.\n\n"
            f"Excerpts:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Return valid JSON only in this form:\n"
            '{"is_relevant": true, "reason": "brief explanation"}'
        )

    def build_query_rewrite(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return (
            "Rewrite this research question so a search engine can find more "
            "relevant academic paper passages. Preserve the original meaning and "
            "use specific technical terms. Return only the rewritten question.\n\n"
            f"Question:\n{query}"
        )

    @staticmethod
    def _format_chunks(chunks: list[ChunkSearchResult]) -> str:
        parts: list[str] = []

        for position, chunk in enumerate(chunks, start=1):
            parts.append(f"Excerpt {position}, arXiv {chunk.arxiv_id}:\n{chunk.chunk_text}")

        return "\n\n".join(parts)
