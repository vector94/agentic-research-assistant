PAPER_INDEX_MAPPING = {
    "settings": {
        "analysis": {
            "analyzer": {
                "english_text": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"],
                }
            }
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "arxiv_id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "english_text",
            },
            "abstract": {
                "type": "text",
                "analyzer": "english_text",
            },
            "raw_text": {
                "type": "text",
                "analyzer": "english_text",
            },
            "authors": {"type": "keyword"},
            "categories": {"type": "keyword"},
            "published_at": {"type": "date"},
            "pdf_url": {"type": "keyword"},
        },
    },
}
