from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

from src.schemas.paper import ArxivPaper
from src.services.arxiv import ArxivClient


def test_search_returns_papers() -> None:
    result = SimpleNamespace(
        get_short_id=lambda: "1234.5678v1",
        title="  Test\npaper  ",
        summary="  Test\nabstract  ",
        authors=[SimpleNamespace(name="Ada Lovelace")],
        categories=["cs.AI"],
        published=datetime(2026, 1, 1, tzinfo=UTC),
        pdf_url="https://arxiv.org/pdf/1234.5678",
    )

    client = ArxivClient()
    client.client.results = Mock(return_value=iter([result]))

    papers = client.search("cat:cs.AI", max_results=1)

    assert papers == [
        ArxivPaper(
            arxiv_id="1234.5678v1",
            title="Test paper",
            abstract="Test abstract",
            authors=["Ada Lovelace"],
            categories=["cs.AI"],
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            pdf_url="https://arxiv.org/pdf/1234.5678",
        )
    ]
    client.client.results.assert_called_once()
