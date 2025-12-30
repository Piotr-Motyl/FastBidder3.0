"""
Protocol for embedding service.

This module defines the interface for text embedding services.
Infrastructure layer provides concrete implementations using ML models.
"""
from typing import Protocol


class EmbeddingServiceProtocol(Protocol):
    """
    Protocol for text embedding service.

    Generates vector embeddings from text using ML models (e.g., sentence-transformers).
    Used for semantic similarity matching in HVAC descriptions.

    Implementations must support:
    - Single text embedding (for on-the-fly working description embedding)
    - Batch embedding (for efficient reference file pre-embedding)
    - GPU acceleration when available with CPU fallback
    - Lazy loading (model loaded on first use, not at initialization)

    Example:
        >>> service = EmbeddingService()  # Implementation from infrastructure
        >>> embedding = service.embed_single("Zawór kulowy DN50 PN16")
        >>> len(embedding)
        384
        >>> embeddings = service.embed_batch(["Zawór DN50", "Rura PVC 160"])
        >>> len(embeddings)
        2
    """

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding vector for single text.

        The model is lazy-loaded on first call, not at initialization.
        Empty or whitespace-only text should raise ValueError.

        Args:
            text: Input text to embed (will be trimmed).

        Returns:
            Embedding vector as list of floats (typically 384 dimensions).

        Raises:
            ValueError: If text is empty or whitespace-only after trimming.

        Example:
            >>> embedding = service.embed_single("Zawór kulowy DN50")
            >>> isinstance(embedding, list)
            True
            >>> all(isinstance(x, float) for x in embedding)
            True
        """
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Batch processing is more efficient than calling embed_single() in a loop.
        Empty list returns empty list. Individual empty strings are skipped.

        Args:
            texts: List of input texts to embed.

        Returns:
            List of embedding vectors. Length equals len(texts).

        Example:
            >>> texts = ["Zawór DN50", "Rura PVC 160", "Klapka zwrotna"]
            >>> embeddings = service.embed_batch(texts)
            >>> len(embeddings) == len(texts)
            True
            >>> all(len(emb) == service.get_embedding_dimension() for emb in embeddings)
            True
        """
        ...

    def get_embedding_dimension(self) -> int:
        """
        Return dimension of embedding vectors.

        For paraphrase-multilingual-MiniLM-L12-v2 model, this is 384.

        Returns:
            Embedding dimension as integer (e.g., 384).

        Example:
            >>> service.get_embedding_dimension()
            384
        """
        ...
