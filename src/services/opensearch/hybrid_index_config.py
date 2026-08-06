from typing import Any


def make_chunk_index_mapping(
    embedding_dimensions: int,
) -> dict[str, Any]:
    if embedding_dimensions <= 0:
        raise ValueError("Embedding dimensions must be greater than zero")

    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index.knn": True,
            "analysis": {
                "analyzer": {
                    "english_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "stop", "snowball"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "arxiv_id": {"type": "keyword"},
                "paper_id": {"type": "keyword"},
                "chunk_index": {"type": "integer"},
                "chunk_text": {
                    "type": "text",
                    "analyzer": "english_text",
                },
                "chunk_word_count": {"type": "integer"},
                "section_title": {"type": "keyword"},
                "title": {
                    "type": "text",
                    "analyzer": "english_text",
                },
                "abstract": {
                    "type": "text",
                    "analyzer": "english_text",
                },
                "authors": {"type": "keyword"},
                "categories": {"type": "keyword"},
                "published_at": {"type": "date"},
                "pdf_url": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": embedding_dimensions,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16,
                        },
                    },
                },
            },
        },
    }
