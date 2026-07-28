from unittest.mock import Mock

from src.services.opensearch.client import OpenSearchClient


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
