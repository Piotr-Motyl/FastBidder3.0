"""
ChromaDB client for vector storage and similarity search.

Provides embedded vector database for storing and querying HVAC description embeddings.
Uses persistent storage that survives application restarts.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class ChromaClient:
    """
    ChromaDB persistent client. Persist dir: CHROMA_PERSIST_DIR env or ./data/chroma_db.
    Distance metric: cosine. Telemetry disabled.

    In production use ChromaClientSingleton.get_instance() to avoid resource leaks.
    Direct instantiation is fine in tests (pass a temp dir).
    """

    # Default collection name for reference descriptions
    COLLECTION_NAME = "reference_descriptions"

    # Default persist directory if not set in env
    # NOTE: Must match CHROMA_PERSIST_DIR in .env for consistency
    DEFAULT_PERSIST_DIR = "./data/chroma_db"

    def __init__(self, persist_directory: str | None = None) -> None:
        """Persist dir priority: param > CHROMA_PERSIST_DIR env > DEFAULT_PERSIST_DIR. Created if missing."""
        # Determine persist directory (priority: param > env > default)
        if persist_directory is None:
            persist_directory = os.getenv(
                "CHROMA_PERSIST_DIR", self.DEFAULT_PERSIST_DIR
            )

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB persistent client
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        logger.info(f"ChromaDB initialized at: {self.persist_directory}")
        logger.info(f"Default collection name: {self.COLLECTION_NAME}")

    def get_or_create_collection(
        self,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> chromadb.Collection:
        """Get or create collection. Default metadata: {"hnsw:space": "cosine"}."""
        collection_name = name or self.COLLECTION_NAME

        # Default metadata: cosine distance for embeddings
        if metadata is None:
            metadata = {"hnsw:space": "cosine"}

        collection = self._client.get_or_create_collection(
            name=collection_name, metadata=metadata
        )

        logger.debug(f"Got or created collection: {collection_name}")
        return collection

    def delete_collection(self, name: str | None = None) -> None:
        """Delete collection. Logs warning if not found, does not raise."""
        collection_name = name or self.COLLECTION_NAME

        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            # ChromaDB 0.4.0 doesn't have NotFoundError - catch all exceptions
            logger.warning(f"Failed to delete collection {collection_name}: {e}")

    def health_check(self) -> bool:
        """Return True if ChromaDB heartbeat succeeds, False otherwise (never raises)."""
        try:
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False

    def get_collection_stats(self, name: str | None = None) -> dict[str, Any]:
        """Return {"name": ..., "count": ...}. On error: adds "error" key and count=0."""
        collection_name = name or self.COLLECTION_NAME

        try:
            collection = self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
            }
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {e}")
            return {"name": collection_name, "count": 0, "error": str(e)}

    def list_collections(self) -> list[str]:
        """List all collection names. Returns [] on error."""
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def reset(self) -> None:
        """Clear system cache to release file handles. Use in test teardown to prevent SQLite lock on Windows."""
        try:
            # Clear client to release file handles
            self._client.clear_system_cache()
            logger.debug("ChromaDB client reset successfully")
        except Exception as e:
            logger.warning(f"Failed to reset ChromaDB client: {e}")


class ChromaClientSingleton:
    """Thread-safe singleton for ChromaClient. Prevents multiple PersistentClient connections."""

    # Singleton instance (None until first get_instance() call)
    _instance: ChromaClient | None = None

    # Thread lock for thread-safe initialization
    _lock = threading.Lock()

    @classmethod
    def get_instance(
        cls,
        persist_directory: str | None = None,
    ) -> ChromaClient:
        """
        Get or create the singleton ChromaClient instance.

        Thread-safe lazy initialization using double-checked locking pattern.
        First call creates instance, subsequent calls return existing instance.

        Args:
            persist_directory: Optional path for data persistence.
                Only used on first call when instance is created.
                Ignored on subsequent calls (existing instance is returned).

        Returns:
            ChromaClient singleton instance.

        Example:
            >>> # First call - creates instance
            >>> client1 = ChromaClientSingleton.get_instance()
            >>>
            >>> # Second call - returns same instance
            >>> client2 = ChromaClientSingleton.get_instance()
            >>> assert client1 is client2
        """
        # First check (without lock) - fast path for existing instance
        if cls._instance is None:
            # Acquire lock for thread-safe initialization
            with cls._lock:
                # Second check (with lock) - ensure only one thread creates instance
                if cls._instance is None:
                    cls._instance = ChromaClient(persist_directory=persist_directory)
                    logger.info(
                        f"ChromaClient singleton instance created at: {cls._instance.persist_directory}"
                    )

        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """
        Drop the cached singleton so the next get_instance() builds a fresh
        ChromaClient. Used by both test teardown and the Celery matching task
        to pick up reference files indexed by the API process between jobs.
        """
        with cls._lock:
            if cls._instance is not None:
                # Reset client to release resources before clearing instance
                cls._instance.reset()
                cls._instance = None
                logger.debug("ChromaClient singleton instance reset")
