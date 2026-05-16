"""
Concrete implementation of EmbeddingServiceProtocol.

Uses sentence-transformers library for embedding generation with support for:
- Lazy model loading (loaded on first use, not at initialization)
- GPU acceleration with automatic CPU fallback
- Batch processing for efficiency
- Multilingual support (Polish and English)
"""
from __future__ import annotations

import logging
import threading
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    sentence-transformers embedding service. Lazy-loads model on first use (~420MB, cached).

    Default model: paraphrase-multilingual-MiniLM-L12-v2 (384-dim, Polish + English).
    GPU auto-detected; falls back to CPU.
    """

    # Default model: multilingual, 384-dim, ~420MB, supports Polish & English
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None) -> None:
        """Model not loaded here — lazy-loads on first embed call (~2-5s delay on first use)."""
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: SentenceTransformer | None = None
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load and cache model. First access takes ~2-5s; subsequent accesses instant."""
        if self._model is None:
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            logger.info("This may take 2-5 seconds on first use (model download + loading)")

            # Import here to avoid loading at module import time
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)

            # Log device info (GPU or CPU)
            device = self._model.device
            logger.info(f"Model loaded successfully. Device: {device}")

            # Log embedding dimension for verification
            dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {dimension}")

        return self._model

    def embed_single(self, text: str) -> list[float]:
        """Embed single text. Raises ValueError for empty/whitespace input."""
        # Trim whitespace
        text = text.strip()

        # Validate non-empty
        if not text:
            raise ValueError("Cannot embed empty text")

        # convert_to_numpy=True returns np.ndarray; cast keeps the type-checker happy
        # because SentenceTransformer.encode() has a Union return type.
        embedding = np.asarray(self.model.encode(text, convert_to_numpy=True))
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Embed multiple texts in one batch (~6× faster than N embed_single() calls)."""
        # Handle empty input
        if not texts:
            return []

        # Trim all texts
        texts = [t.strip() for t in texts]

        # show_progress_bar: only for large batches (>100 items)
        embeddings = np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 100,
            )
        )
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """Return embedding dimension (384 for the default multilingual MiniLM model)."""
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError(
                f"Model '{self.model_name}' did not expose an embedding dimension"
            )
        return dimension

    def similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.

        Zero-norm vectors return 0.0 to avoid ZeroDivisionError.

        Args:
            embedding_a: First embedding vector.
            embedding_b: Second embedding vector.

        Returns:
            Cosine similarity in [-1.0, 1.0]; typically 0.0–1.0 for text.
        """
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class EmbeddingServiceSingleton:
    """
    Thread-safe singleton for EmbeddingService.

    Critical for Celery --pool=solo workers: without this, each task would reload the
    ~420MB model (30-60s penalty per task). Uses double-checked locking.
    """

    _instance: EmbeddingService | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> EmbeddingService:
        """Return singleton instance. model_name only applies on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = EmbeddingService(model_name=model_name)
                    logger.info("EmbeddingService singleton instance created")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (testing only). Next get_instance() reloads the model."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None
                logger.warning("EmbeddingService singleton instance reset (testing only)")
