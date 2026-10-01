from functools import lru_cache
from typing import Protocol

from app.config import get_settings


class Embedder(Protocol):
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one L2-normalised vector per text."""
        ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(model_name, device=device)
        self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.tolist()


@lru_cache
def get_embedder() -> Embedder:
    settings = get_settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    if embedder.dimension != settings.embedding_dim:
        raise RuntimeError(
            f"Embedding model {settings.embedding_model} produces {embedder.dimension}-d vectors "
            f"but EMBEDDING_DIM is {settings.embedding_dim}. Set EMBEDDING_DIM to match "
            "(existing chunks must be re-indexed)."
        )
    return embedder
