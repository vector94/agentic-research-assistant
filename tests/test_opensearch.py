from unittest.mock import Mock

import pytest
from opensearchpy import RequestError

from src.services.opensearch.client import OpenSearchClient


def test_create_index_creates_when_missing() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.indices.exists.return_value = False

    created = client.create_index({"mappings": {}})

    assert created is True
    client.client.indices.create.assert_called_once_with(
        index="research-paper-chunks",
        body={"mappings": {}},
    )


def test_create_index_skips_when_already_exists() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.indices.exists.return_value = True

    created = client.create_index({"mappings": {}})

    assert created is False
    client.client.indices.create.assert_not_called()


def test_create_index_treats_concurrent_creation_as_a_no_op() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.indices.exists.return_value = False
    client.client.indices.create.side_effect = RequestError(
        400,
        "resource_already_exists_exception",
        {"error": {"reason": "index already exists"}},
    )

    created = client.create_index({"mappings": {}})

    assert created is False


def test_create_index_reraises_unrelated_request_errors() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.indices.exists.return_value = False
    client.client.indices.create.side_effect = RequestError(
        400,
        "mapper_parsing_exception",
        {"error": {"reason": "bad mapping"}},
    )

    with pytest.raises(RequestError):
        client.create_index({"mappings": {}})


def test_search_chunks_vector_adds_category_filter() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    client.search_chunks_vector(
        query_embedding=[0.1, 0.2],
        categories=["cs.AI", "cs.LG"],
    )

    body = client.client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == [
        {
            "terms": {
                "categories": ["cs.AI", "cs.LG"],
            }
        }
    ]
    assert "knn" in body["query"]["bool"]["must"][0]


def test_search_chunks_vector_skips_filter_without_categories() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    client.search_chunks_vector(query_embedding=[0.1, 0.2])

    body = client.client.search.call_args.kwargs["body"]
    assert "knn" in body["query"]
    assert "bool" not in body["query"]


def test_search_chunks_bm25_adds_category_filter() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    client.search_chunks_bm25(
        query="attention mechanisms",
        categories=["cs.AI"],
    )

    body = client.client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == [
        {
            "terms": {
                "categories": ["cs.AI"],
            }
        }
    ]
    assert "multi_match" in body["query"]["bool"]["must"][0]


def test_delete_paper_chunks_deletes_matching_arxiv_id() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-paper-chunks",
    )
    client.client = Mock()
    client.client.delete_by_query.return_value = {"deleted": 3}

    deleted = client.delete_paper_chunks("1234.5678v1")

    assert deleted is True
    client.client.delete_by_query.assert_called_once_with(
        index="research-paper-chunks",
        body={"query": {"term": {"arxiv_id": "1234.5678v1"}}},
        refresh=True,
    )


def test_search_papers_builds_multifield_query() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-papers",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    result = client.search_papers(
        query="3D geometry",
        size=5,
        from_=10,
    )

    assert result == {"hits": {"hits": []}}
    client.client.search.assert_called_once_with(
        index="research-papers",
        body={
            "from": 10,
            "size": 5,
            "query": {
                "multi_match": {
                    "query": "3D geometry",
                    "fields": [
                        "title^3",
                        "abstract^2",
                        "raw_text",
                    ],
                }
            },
            "_source": {
                "excludes": ["raw_text"],
            },
            "highlight": {
                "pre_tags": ["<mark>"],
                "post_tags": ["</mark>"],
                "fields": {
                    "title": {},
                    "abstract": {
                        "fragment_size": 200,
                        "number_of_fragments": 1,
                    },
                    "raw_text": {
                        "fragment_size": 200,
                        "number_of_fragments": 1,
                    },
                },
            },
        },
    )


def test_search_papers_adds_category_filter() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-papers",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    client.search_papers(
        query="language models",
        categories=["cs.AI", "cs.LG"],
    )

    body = client.client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["filter"] == [
        {
            "terms": {
                "categories": ["cs.AI", "cs.LG"],
            }
        }
    ]


def test_search_papers_can_sort_by_latest() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-papers",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    client.search_papers(
        query="language models",
        latest=True,
    )

    body = client.client.search.call_args.kwargs["body"]
    assert body["sort"] == [
        {
            "published_at": {
                "order": "desc",
            }
        }
    ]
    assert body["track_scores"] is True


def test_search_papers_can_enable_fuzziness() -> None:
    client = OpenSearchClient(
        host="http://localhost:9200",
        index_name="research-papers",
    )
    client.client = Mock()
    client.client.search.return_value = {"hits": {"hits": []}}

    client.search_papers(
        query="langauge models",
        fuzzy=True,
    )

    body = client.client.search.call_args.kwargs["body"]
    assert body["query"]["multi_match"]["fuzziness"] == "AUTO"
