from src.schemas.retrieval import ChunkSearchResult


class AgentPromptBuilder:
    def build_guardrail(self, query: str) -> str:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        return (
            "You are a guardrail evaluator assessing whether a user query is "
            "within the scope of academic research papers from arXiv in Computer "
            "Science, AI, and Machine Learning.\n\n"
            f"User Query: {query}\n\n"
            "Evaluate whether this query is:\n"
            "- About CS/AI/ML research topics such as neural networks, algorithms, "
            "models, architectures, or techniques\n"
            "- A question that requires academic paper knowledge to answer\n"
            "- Within the domain of Computer Science research\n\n"
            "Assign a relevance score from 0 to 100:\n"
            "- 80-100: Clearly about CS/AI/ML research, such as 'What are "
            "transformer architectures?' or 'How does BERT work?'\n"
            "- 60-79: Potentially research-related but unclear, such as 'Tell me "
            "about attention mechanisms'\n"
            "- 40-59: Borderline or ambiguous, such as 'What is machine learning?'\n"
            "- 0-39: Not about research papers, such as 'What is a dog?', 'Hello', "
            "or 'What is 2+2?'\n\n"
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
