import arxiv

from src.schemas.paper import ArxivPaper


class ArxivClient:
    def __init__(self) -> None:
        self.client = arxiv.Client(
            page_size=10,
            delay_seconds=3.0,
            num_retries=3,
        )

    def search(self, query: str, max_results: int = 5) -> list[arxiv.Result]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        return [self._to_paper(result) for result in self.client.results(search)]

    @staticmethod
    def _to_paper(result: arxiv.Result) -> ArxivPaper:
        return ArxivPaper(
            arxiv_id=result.get_short_id(),
            title=" ".join(result.title.split()),
            abstract=" ".join(result.summary.split()),
            authors=[author.name for author in result.authors],
            categories=result.categories,
            published_at=result.published,
            pdf_url=result.pdf_url,
        )
