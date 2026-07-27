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

    def create_index(self, mapping: dict[str, any]) -> bool:
        if self.client.indices.exists(index=self.index_name):
            return False

        self.client.indices.create(index=self.index_name, body=mapping)
        return True

    def index_paper(self, paper: dict[str, any]) -> bool:
        response = self.client.index(index=self.index_name, id=paper["arxiv_id"], body=paper, refresh=True)

        return response["result"] in {"created", "updated"}
