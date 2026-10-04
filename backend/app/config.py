from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, read from environment variables or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/medrag"

    # Embeddings: the dimension must match the model, it sizes the pgvector column
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking and retrieval
    chunk_size: int = 512
    chunk_overlap: int = 128
    top_k: int = 4
    # Questions whose best-matching passage scores below this cosine similarity are refused without
    # calling the LLM. 0 disables the check. See eval/README.md for how the default was chosen.
    min_similarity: float = 0.35

    # Generation
    # Comma-separated fallback order: if one provider errors, the next is tried. "none" disables generation.
    llm_providers: str = "gemini,groq"
    temperature: float = 0.1
    max_tokens: int = 500
    # Kept low so a failing provider hands over to the next one quickly
    llm_timeout: float = 30
    llm_max_retries: int = 1
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Uploads
    max_upload_mb: int = 25
    # OCR for PDF pages without a text layer (scanned documents); needs the tesseract binary
    ocr_enabled: bool = True
    ocr_language: str = "eng"  # tesseract language code(s), e.g. "eng+fra"
    ocr_dpi: int = 300

    cors_origins: list[str] = ["http://localhost:5173"]

    # Accounts. With registration off, create users with: python -m app.cli create-user EMAIL
    allow_registration: bool = True
    session_ttl_hours: int = 24 * 7
    # Set true when the app is served over HTTPS so the session cookie is never sent in clear text
    cookie_secure: bool = False
    # Failed logins allowed per email and client address in the window before answering 429
    login_max_failures: int = 5
    login_window_minutes: int = 15

    # Load the embedding model at startup instead of on the first request
    preload_models: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
