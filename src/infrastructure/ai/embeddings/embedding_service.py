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
    Embedding service using sentence-transformers.

    Implements EmbeddingServiceProtocol from domain layer.
    Uses paraphrase-multilingual-MiniLM-L12-v2 model by default.

    Key features:
    - Lazy-loads model on first use to avoid startup delay (not in __init__)
    - Supports GPU acceleration when available (automatic detection)
    - Batch processing with configurable batch size
    - Deterministic embeddings (same text = same vector)
    - ~420MB model download on first use (cached afterwards)

    Performance targets:
    - embed_single(): < 100ms on CPU, < 10ms on GPU
    - embed_batch(100): < 10s on CPU, < 1s on GPU

    Example:
        >>> service = EmbeddingService()  # Model NOT loaded yet
        >>> embedding = service.embed_single("Zawór kulowy DN50")  # NOW model loads
        >>> len(embedding)
        384
        >>> embeddings = service.embed_batch(["Text 1", "Text 2"])
        >>> len(embeddings)
        2
    """

    # Default model: multilingual, 384-dim, ~420MB, supports Polish & English
    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialize embedding service.

        Model is NOT loaded here - it will be lazy-loaded on first use.
        This avoids ~2-5 second startup delay when service is created but not used.

        Args:
            model_name: Model identifier from sentence-transformers.
                Defaults to paraphrase-multilingual-MiniLM-L12-v2.

        Example:
            >>> service = EmbeddingService()  # Fast, no model loading
            >>> service = EmbeddingService("all-MiniLM-L6-v2")  # Custom model
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model: SentenceTransformer | None = None
        logger.info(f"EmbeddingService initialized with model: {self.model_name}")

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load and cache the sentence-transformers model.

        Model is loaded on first access, not at __init__.
        This avoids startup delay when service is created but not immediately used.

        GPU detection is automatic:
        - If CUDA available: uses GPU
        - Otherwise: uses CPU

        Returns:
            Loaded SentenceTransformer model instance.

        Example:
            >>> service = EmbeddingService()
            >>> model = service.model  # First access: loads model (~2-5s)
            >>> model = service.model  # Second access: returns cached model (instant)
        """
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
        """
        Generate embedding vector for single text.

        Text is trimmed before embedding. Empty or whitespace-only text raises ValueError.
        Model is lazy-loaded on first call.

        Args:
            text: Input text to embed.

        Returns:
            Embedding vector as list of floats (384 dimensions for default model).

        Raises:
            ValueError: If text is empty or whitespace-only after trimming.

        Example:
            >>> service = EmbeddingService()
            >>> embedding = service.embed_single("Zawór kulowy DN50 PN16 mosiądz")
            >>> len(embedding)
            384
            >>> isinstance(embedding[0], float)
            True
        """
        # Trim whitespace
        text = text.strip()

        # Validate non-empty
        if not text:
            raise ValueError("Cannot embed empty text")

        # Generate embedding
        # convert_to_numpy=True returns numpy array (we convert to list)
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Convert numpy array to Python list for JSON serialization
        return embedding.tolist()

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[list[float]]:
        """
        Generate embeddings for multiple texts (batch processing).

        Batch processing is significantly faster than calling embed_single() in a loop:
        - 100 texts: ~10s batch vs ~60s individual (6x speedup)
        - Shares GPU memory allocation across batch

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process at once.
                Smaller = less memory, larger = faster.
                Default: 32 (good balance for 4GB GPU).

        Returns:
            List of embedding vectors. Length equals len(texts).
            Empty input returns empty list.

        Example:
            >>> service = EmbeddingService()
            >>> texts = ["Zawór DN50", "Rura PVC 160", "Klapka zwrotna DN100"]
            >>> embeddings = service.embed_batch(texts)
            >>> len(embeddings) == len(texts)
            True
            >>> all(len(emb) == 384 for emb in embeddings)
            True
        """
        # Handle empty input
        if not texts:
            return []

        # Trim all texts
        texts = [t.strip() for t in texts]

        # Encode batch
        # show_progress_bar: only for large batches (>100 items)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
        )

        # Convert numpy array to list of lists
        return embeddings.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Return dimension of embedding vectors.

        For default model (paraphrase-multilingual-MiniLM-L12-v2): 384
        For other models: varies (e.g., all-MiniLM-L6-v2 is also 384)

        Returns:
            Embedding dimension as integer.

        Example:
            >>> service = EmbeddingService()
            >>> service.get_embedding_dimension()
            384
        """
        return self.model.get_sentence_embedding_dimension()

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
    Thread-safe Singleton wrapper for EmbeddingService.

    Ensures only one EmbeddingService (and therefore one loaded ML model) exists
    throughout the application/worker lifecycle. Critical for Celery workers running
    with --pool=solo: without this, every task creates a fresh EmbeddingService and
    re-loads the ~420MB sentence-transformers model (~30-60s per task).

    Mirrors the ChromaClientSingleton pattern from chroma_client.py.

    Example:
        >>> service = EmbeddingServiceSingleton.get_instance()
        >>> embedding = service.embed_single("Zawór kulowy DN50")
        >>>
        >>> # Returns same instance (model already loaded)
        >>> same = EmbeddingServiceSingleton.get_instance()
        >>> assert service is same
        >>>
        >>> # Reset for testing only
        >>> EmbeddingServiceSingleton.reset_instance()
    """

    _instance: EmbeddingService | None = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> EmbeddingService:
        """
        Get or create the singleton EmbeddingService instance.

        Thread-safe lazy initialization using double-checked locking.
        First call creates instance; subsequent calls return existing instance.
        Model is still lazy-loaded on first embed call (not here).

        Args:
            model_name: Optional model override. Only used on first call.

        Returns:
            EmbeddingService singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = EmbeddingService(model_name=model_name)
                    logger.info("EmbeddingService singleton instance created")
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (for testing purposes only).

        Releases the loaded model so the next get_instance() call creates a fresh one.
        WARNING: Do not call in production — causes full model reload on next use.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance = None
                logger.warning("EmbeddingService singleton instance reset (testing only)")
