"""FastAPI dependencies that turn model-loading problems into 503s instead of 500s."""

from fastapi import HTTPException, status

from app.rag.embeddings import Embedder, get_embedder
from app.rag.llm import LLM, get_llm


def embedder_dep() -> Embedder:
    try:
        return get_embedder()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Embedding model unavailable: {exc}")


def llm_dep() -> LLM | None:
    try:
        return get_llm()
    except Exception as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"LLM unavailable: {exc}")
