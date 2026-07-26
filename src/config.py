from functools import lru_cache
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class PdfParserSettings(BaseModel):
    max_pages: int = 30
    max_file_size_mb: int = 20
    do_ocr: bool = False
    do_table_structure: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    app_name: str = "Agentic Research Assistant API"
    app_version: str = "0.1.0"
    environment: Literal["development", "testing", "production"] = "development"
    debug: bool = False

    database_url: str
    opensearch_url: str = "http://127.0.0.1:9200"
    ollama_url: str = "http://127.0.0.1:11434"

    pdf_parser: PdfParserSettings = PdfParserSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()
