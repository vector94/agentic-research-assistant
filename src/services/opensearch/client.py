from typing import Any

from opensearchpy import OpenSearch


class OpenSearchClient:
    def __init__(self, host: str, index_name: str) -> None:
        self.index_name = index_name
        self.client = OpenSearch(
            hosts=[host],
            use_ssl=False,
            verify_certs=False,
        )

    def health_check(self) -> bool:
        try:
            health = self.client.cluster.health()
            return health["status"] in {"green", "yellow"}
        except Exception:
            return False

    def create_index(self, mapping: dict[str, Any]) -> bool:
        if self.client.indices.exists(index=self.index_name):
            return False

        self.client.indices.create(
            index=self.index_name,
            body=mapping,
        )
        return True

    def index_paper(self, paper: dict[str, Any]) -> bool:
        response = self.client.index(
            index=self.index_name,
            id=paper["arxiv_id"],
            body=paper,
            refresh=True,
        )
        return response["result"] in {"created", "updated"}

    def search_papers(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        categories: list[str] | None = None,
        latest: bool = False,
        fuzzy: bool = False,
    ) -> dict[str, Any]:
        multi_match: dict[str, Any] = {
            "query": query,
            "fields": [
                "title^3",
                "abstract^2",
                "raw_text",
            ],
        }

        if fuzzy:
            multi_match["fuzziness"] = "AUTO"

        search_query: dict[str, Any] = {
            "multi_match": multi_match,
        }

        if categories:
            search_query = {
                "bool": {
                    "must": [search_query],
                    "filter": [
                        {
                            "terms": {
                                "categories": categories,
                            }
                        }
                    ],
                }
            }

        body = {
            "from": from_,
            "size": size,
            "query": search_query,
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
        }

        if latest:
            body["sort"] = [
                {
                    "published_at": {
                        "order": "desc",
                    }
                }
            ]
            body["track_scores"] = True

        return self.client.search(
            index=self.index_name,
            body=body,
        )
