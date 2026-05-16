"""
Protocol for embedding service.

This module defines the interface for text embedding services.
Infrastructure layer provides concrete implementations using ML models.
"""
from typing import Protocol


class EmbeddingServiceProtocol(Protocol):
    """
    Protocol for text embedding service using ML models (e.g., sentence-transformers).

    Model is lazy-loaded on first use. Batch embedding is significantly faster than
    calling embed_single() in a loop — use embed_batch() when processing multiple texts.
    """

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text. Raises ValueError for empty/whitespace text."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts. Returns list of same length as input."""
        ...

    def get_embedding_dimension(self) -> int:
        """Return embedding vector dimension (384 for paraphrase-multilingual-MiniLM-L12-v2)."""
        ...

    def similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Returns value in [-1.0, 1.0]; typically 0.0–1.0 for text.
        Zero-norm vectors return 0.0 (no ZeroDivisionError).
        """
        ...
