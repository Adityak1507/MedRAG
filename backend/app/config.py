from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/medrag"

    # Embeddings: the dimension must match the model, it sizes the pgvector column
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking and retrieval
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 4

    # Generation
    llm_provider: str = "auto"  # auto | openai | anthropic | cohere | none
    temperature: float = 0.1
    max_tokens: int = 500
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    cohere_api_key: str = ""
    cohere_model: str = "command-r"

    # Uploads
    max_upload_mb: int = 25

    cors_origins: list[str] = ["http://localhost:5173"]

    # Load the embedding model at startup instead of on the first request
    preload_models: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
