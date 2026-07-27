from src.models import Paper
from src.services.opensearch.client import OpenSearchClient


class PaperIndexer:
    def __init__(self, opensearch_client: OpenSearchClient) -> None:
        self.opensearch_client = opensearch_client

    def index_paper(self, paper: Paper) -> bool:
        document = {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "raw_text": paper.raw_text or "",
            "authors": paper.authors,
            "categories": paper.categories,
            "published_at": paper.published_at.isoformat(),
            "pdf_url": paper.pdf_url,
        }

        return self.opensearch_client.index_paper(document)
