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
from chromadb.errors import NotFoundError

logger = logging.getLogger(__name__)


class ChromaClient:
    """
    ChromaDB client wrapper for vector storage operations.

    Provides simplified interface for managing vector collections.
    Uses persistent storage (data survives restarts).

    Key features:
    - Persistent storage at configurable path (from env or default)
    - Cosine distance metric (optimal for sentence-transformers)
    - Health checking for operational status
    - Collection statistics (count, metadata)

    Configuration:
    - CHROMA_PERSIST_DIR env variable or default: ./data/chroma
    - Distance metric: cosine (hnsw:space)
    - Telemetry: disabled (anonymized_telemetry=False)

    WARNING:
    - Do NOT instantiate ChromaClient directly in production code
    - Use ChromaClientSingleton.get_instance() instead to avoid resource leaks
    - Direct instantiation is only for testing with temporary directories

    Example:
        >>> # Production code - use Singleton
        >>> client = ChromaClientSingleton.get_instance()
        >>> collection = client.get_or_create_collection()
        >>> client.health_check()
        True
        >>>
        >>> # Test code - direct instantiation with temp directory is OK
        >>> test_client = ChromaClient(persist_directory="/tmp/test_chroma")
    """

    # Default collection name for reference descriptions
    COLLECTION_NAME = "reference_descriptions"

    # Default persist directory if not set in env
    DEFAULT_PERSIST_DIR = "./data/chroma"

    def __init__(self, persist_directory: str | None = None) -> None:
        """
        Initialize ChromaDB client with persistent storage.

        Persist directory is determined by (in order of priority):
        1. persist_directory parameter
        2. CHROMA_PERSIST_DIR environment variable
        3. DEFAULT_PERSIST_DIR constant (./data/chroma)

        Directory is created if it doesn't exist.

        Args:
            persist_directory: Optional path for data persistence.
                If None, uses CHROMA_PERSIST_DIR env or default.

        Example:
            >>> client = ChromaClient()  # Uses env or default
            >>> client = ChromaClient("./custom/path")  # Custom path
        """
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
        """
        Get existing collection or create new one.

        If collection exists, returns it. If not, creates with specified metadata.
        Default metadata uses cosine distance (optimal for sentence-transformers).

        Args:
            name: Collection name. Defaults to COLLECTION_NAME.
            metadata: Collection metadata. Defaults to {"hnsw:space": "cosine"}.

        Returns:
            ChromaDB Collection object for adding/querying embeddings.

        Example:
            >>> collection = client.get_or_create_collection()
            >>> collection.count()
            0
            >>> collection = client.get_or_create_collection("custom_collection")
        """
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
        """
        Delete a collection and all its data.

        If collection doesn't exist, logs warning but doesn't raise error.

        Args:
            name: Collection name. Defaults to COLLECTION_NAME.

        Example:
            >>> client.delete_collection("old_collection")
            >>> client.delete_collection()  # Deletes default collection
        """
        collection_name = name or self.COLLECTION_NAME

        try:
            self._client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except NotFoundError:
            logger.warning(f"Collection not found: {collection_name}")

    def health_check(self) -> bool:
        """
        Check if ChromaDB is operational.

        Uses heartbeat method to verify client is responsive.
        Returns False instead of raising exception on failure.

        Returns:
            True if healthy and responsive, False otherwise.

        Example:
            >>> if client.health_check():
            ...     print("ChromaDB is healthy")
            ... else:
            ...     print("ChromaDB is down")
        """
        try:
            self._client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return False

    def get_collection_stats(self, name: str | None = None) -> dict[str, Any]:
        """
        Get statistics for a collection.

        Returns count and basic metadata for specified collection.
        If collection doesn't exist, returns count=0 with error.

        Args:
            name: Collection name. Defaults to COLLECTION_NAME.

        Returns:
            Dict with keys:
            - name: Collection name
            - count: Number of items in collection
            - error: Error message if collection doesn't exist (optional)

        Example:
            >>> stats = client.get_collection_stats()
            >>> print(f"Collection has {stats['count']} items")
            >>> stats = client.get_collection_stats("non_existent")
            >>> "error" in stats
            True
        """
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
        """
        List all collection names in the database.

        Useful for debugging and understanding what's stored.

        Returns:
            List of collection names.

        Example:
            >>> collections = client.list_collections()
            >>> print(collections)
            ['reference_descriptions', 'test_collection']
        """
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def reset(self) -> None:
        """
        Reset and close ChromaDB client.

        Useful for cleanup in tests, especially on Windows where
        SQLite database may remain locked.

        Example:
            >>> client.reset()  # Close connections
        """
        try:
            # Clear client to release file handles
            self._client.clear_system_cache()
            logger.debug("ChromaDB client reset successfully")
        except Exception as e:
            logger.warning(f"Failed to reset ChromaDB client: {e}")


class ChromaClientSingleton:
    """
    Thread-safe Singleton wrapper for ChromaClient.

    Ensures only one ChromaDB client instance exists throughout application lifecycle.
    Prevents resource leaks and conflicts from multiple PersistentClient connections.

    Uses double-checked locking pattern for thread-safe lazy initialization.

    Key features:
    - Thread-safe initialization with double-checked locking
    - Single persistent connection to ChromaDB
    - Test-friendly reset_instance() method for cleanup
    - Forwards all arguments to underlying ChromaClient

    Example:
        >>> # Get singleton instance (creates on first call)
        >>> client = ChromaClientSingleton.get_instance()
        >>> collection = client.get_or_create_collection()
        >>>
        >>> # Subsequent calls return same instance
        >>> same_client = ChromaClientSingleton.get_instance()
        >>> assert client is same_client
        >>>
        >>> # Reset for testing (use with caution)
        >>> ChromaClientSingleton.reset_instance()
    """

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
        Reset the singleton instance (for testing purposes).

        WARNING: Use with extreme caution in production code.
        This method is intended for test teardown/setup.

        Acquires lock to ensure thread-safe reset.
        Resets client before clearing instance to release resources.

        Example:
            >>> # In test teardown
            >>> ChromaClientSingleton.reset_instance()
            >>> assert ChromaClientSingleton._instance is None
        """
        with cls._lock:
            if cls._instance is not None:
                # Reset client to release resources before clearing instance
                cls._instance.reset()
                cls._instance = None
                logger.warning("ChromaClient singleton instance reset (testing only)")
