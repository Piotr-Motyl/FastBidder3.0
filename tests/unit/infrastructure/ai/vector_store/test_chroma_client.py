"""
Tests for ChromaClient.

Uses real ChromaDB with temporary storage for fast, isolated tests.
No mocks needed - ChromaDB is embedded and fast enough for unit tests.
"""
import pytest
import tempfile
from pathlib import Path

from src.infrastructure.ai.vector_store.chroma_client import ChromaClient


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_chroma_dir():
    """
    Temporary directory for ChromaDB storage.

    Creates a temp directory that's automatically cleaned up after test.
    Each test gets a fresh, isolated ChromaDB instance.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def chroma_client(temp_chroma_dir, request):
    """
    ChromaClient with temporary storage.

    Provides a clean ChromaDB instance for each test.
    Data is automatically cleaned up after test completion.
    """
    client = ChromaClient(persist_directory=temp_chroma_dir)

    # Add finalizer to reset client before temp dir cleanup (Windows fix)
    def cleanup():
        client.reset()

    request.addfinalizer(cleanup)
    return client


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================

def test_client_initialization_creates_directory(temp_chroma_dir):
    """Test that ChromaClient creates persist directory if it doesn't exist."""
    # Arrange
    persist_path = Path(temp_chroma_dir) / "subdir" / "chroma"

    # Act
    client = ChromaClient(persist_directory=str(persist_path))

    # Assert
    assert persist_path.exists()
    assert client.persist_directory == persist_path


def test_health_check_returns_true(chroma_client):
    """Test that health check passes for healthy client."""
    # Act
    result = chroma_client.health_check()

    # Assert
    assert result is True


def test_get_or_create_collection_creates_new(chroma_client):
    """Test creating a new collection."""
    # Act
    collection = chroma_client.get_or_create_collection("test_collection")

    # Assert
    assert collection is not None
    assert collection.count() == 0
    assert collection.name == "test_collection"


def test_get_or_create_collection_uses_default_name(chroma_client):
    """Test that default collection name is used when none provided."""
    # Act
    collection = chroma_client.get_or_create_collection()

    # Assert
    assert collection.name == ChromaClient.COLLECTION_NAME
    assert collection.name == "reference_descriptions"


def test_get_or_create_collection_returns_existing(chroma_client):
    """Test that calling get_or_create twice returns same collection."""
    # Arrange
    collection1 = chroma_client.get_or_create_collection("test_collection")
    collection1.add(
        ids=["doc1"],
        embeddings=[[0.1] * 384],
        documents=["Test document"],
    )

    # Act
    collection2 = chroma_client.get_or_create_collection("test_collection")

    # Assert
    assert collection2.count() == 1  # Same collection with data
    assert collection1.name == collection2.name


def test_get_or_create_collection_uses_cosine_distance(chroma_client):
    """Test that default metadata uses cosine distance."""
    # Act
    collection = chroma_client.get_or_create_collection()

    # Assert
    metadata = collection.metadata
    assert metadata is not None
    assert metadata.get("hnsw:space") == "cosine"


def test_collection_stats_returns_count(chroma_client):
    """Test getting statistics for a collection."""
    # Arrange
    collection = chroma_client.get_or_create_collection("test_collection")
    collection.add(
        ids=["doc1", "doc2"],
        embeddings=[[0.1] * 384, [0.2] * 384],
        documents=["Doc 1", "Doc 2"],
    )

    # Act
    stats = chroma_client.get_collection_stats("test_collection")

    # Assert
    assert stats["name"] == "test_collection"
    assert stats["count"] == 2
    assert "error" not in stats


def test_collection_stats_for_default_collection(chroma_client):
    """Test getting stats for default collection."""
    # Arrange
    chroma_client.get_or_create_collection()  # Create default collection

    # Act
    stats = chroma_client.get_collection_stats()

    # Assert
    assert stats["name"] == ChromaClient.COLLECTION_NAME
    assert stats["count"] == 0


def test_delete_collection_removes_collection(chroma_client):
    """Test deleting a collection."""
    # Arrange
    chroma_client.get_or_create_collection("to_delete")

    # Act
    chroma_client.delete_collection("to_delete")

    # Assert
    stats = chroma_client.get_collection_stats("to_delete")
    assert "error" in stats  # Collection doesn't exist


def test_delete_collection_uses_default_name(chroma_client):
    """Test that delete uses default collection name when none provided."""
    # Arrange
    chroma_client.get_or_create_collection()  # Create default

    # Act
    chroma_client.delete_collection()  # Delete default

    # Assert
    stats = chroma_client.get_collection_stats()
    assert "error" in stats  # Default collection doesn't exist


def test_list_collections_returns_names(chroma_client):
    """Test listing all collections."""
    # Arrange
    chroma_client.get_or_create_collection("collection1")
    chroma_client.get_or_create_collection("collection2")

    # Act
    collections = chroma_client.list_collections()

    # Assert
    assert len(collections) == 2
    assert "collection1" in collections
    assert "collection2" in collections


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================

def test_collection_stats_for_non_existent_collection(chroma_client):
    """Test getting stats for collection that doesn't exist."""
    # Act
    stats = chroma_client.get_collection_stats("non_existent")

    # Assert
    assert stats["name"] == "non_existent"
    assert stats["count"] == 0
    assert "error" in stats
    assert isinstance(stats["error"], str)


def test_delete_non_existent_collection_does_not_crash(chroma_client):
    """Test that deleting non-existent collection doesn't raise error."""
    # Act & Assert (should not raise)
    chroma_client.delete_collection("non_existent")


def test_list_collections_empty_database(chroma_client):
    """Test listing collections when database is empty."""
    # Act
    collections = chroma_client.list_collections()

    # Assert
    assert collections == []


def test_custom_metadata_preserved(chroma_client):
    """Test that custom collection metadata is preserved."""
    # Arrange
    custom_metadata = {"hnsw:space": "l2", "custom_key": "custom_value"}

    # Act
    collection = chroma_client.get_or_create_collection(
        "custom_collection", metadata=custom_metadata
    )

    # Assert
    assert collection.metadata["hnsw:space"] == "l2"
    assert collection.metadata["custom_key"] == "custom_value"


# ============================================================================
# PERSISTENCE TESTS
# ============================================================================

def test_data_persists_across_client_instances(temp_chroma_dir):
    """Test that data persists when creating new client with same directory."""
    # Arrange - Create client and add data
    client1 = ChromaClient(persist_directory=temp_chroma_dir)
    collection1 = client1.get_or_create_collection("persistent")
    collection1.add(
        ids=["doc1"],
        embeddings=[[0.1] * 384],
        documents=["Persistent doc"],
    )

    # Act - Create new client with same directory
    client2 = ChromaClient(persist_directory=temp_chroma_dir)
    collection2 = client2.get_or_create_collection("persistent")

    # Assert - Data is still there
    assert collection2.count() == 1


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

def test_uses_env_variable_for_persist_dir(monkeypatch, temp_chroma_dir):
    """Test that CHROMA_PERSIST_DIR env variable is used."""
    # Arrange
    env_path = Path(temp_chroma_dir) / "from_env"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(env_path))

    # Act
    client = ChromaClient()

    # Assert
    assert client.persist_directory == env_path
    assert env_path.exists()


def test_parameter_overrides_env_variable(monkeypatch, temp_chroma_dir):
    """Test that constructor parameter overrides env variable."""
    # Arrange
    env_path = Path(temp_chroma_dir) / "from_env"
    param_path = Path(temp_chroma_dir) / "from_param"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(env_path))

    # Act
    client = ChromaClient(persist_directory=str(param_path))

    # Assert
    assert client.persist_directory == param_path
    assert param_path.exists()
    assert not env_path.exists()  # Env path was not used


def test_uses_default_when_no_env_and_no_param(temp_chroma_dir, monkeypatch):
    """Test that default path is used when no env and no parameter."""
    # Arrange
    monkeypatch.delenv("CHROMA_PERSIST_DIR", raising=False)
    # Change to temp dir so default path is created there
    import os

    original_cwd = os.getcwd()
    os.chdir(temp_chroma_dir)

    try:
        # Act
        client = ChromaClient()

        # Assert
        # When cwd is temp_chroma_dir, default "./data/chroma" becomes temp_chroma_dir/data/chroma
        expected_path = Path(temp_chroma_dir) / "data" / "chroma"
        # Normalize paths for comparison (resolve() converts to absolute)
        assert client.persist_directory.resolve() == expected_path.resolve()
        assert expected_path.exists()
    finally:
        os.chdir(original_cwd)


# ============================================================================
# SINGLETON PATTERN TESTS (CRITICAL-1 FIX)
# ============================================================================

@pytest.fixture
def reset_singleton_after_test():
    """
    Reset ChromaClientSingleton after each Singleton test.

    Ensures each Singleton test starts with clean singleton state.
    Use this fixture explicitly in Singleton tests.
    """
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    # Reset before test (in case previous test didn't clean up)
    ChromaClientSingleton.reset_instance()

    yield  # Test runs here

    # Cleanup after test
    ChromaClientSingleton.reset_instance()


def test_singleton_returns_same_instance_on_multiple_calls(
    temp_chroma_dir, reset_singleton_after_test
):
    """
    Test CRITICAL-1 fix: Singleton returns same instance on multiple calls.

    Verifies:
    - First get_instance() creates instance
    - Second get_instance() returns same instance (not new one)
    - Instance identity is preserved
    """
    # Arrange & Act
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    client1 = ChromaClientSingleton.get_instance(persist_directory=temp_chroma_dir)
    client2 = ChromaClientSingleton.get_instance(persist_directory=temp_chroma_dir)

    # Assert
    assert client1 is client2  # Same object reference
    assert id(client1) == id(client2)  # Same memory address


def test_singleton_ignores_persist_directory_on_subsequent_calls(
    temp_chroma_dir, reset_singleton_after_test
):
    """
    Test CRITICAL-1 fix: Singleton ignores persist_directory on subsequent calls.

    Verifies:
    - First call uses provided persist_directory
    - Second call with different persist_directory returns same instance
    - Persist directory is set only on first call
    """
    # Arrange
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    first_dir = Path(temp_chroma_dir) / "first"
    second_dir = Path(temp_chroma_dir) / "second"

    # Act
    client1 = ChromaClientSingleton.get_instance(persist_directory=str(first_dir))
    client2 = ChromaClientSingleton.get_instance(
        persist_directory=str(second_dir)
    )  # Should be ignored

    # Assert
    assert client1 is client2  # Same instance
    assert client1.persist_directory == first_dir  # First directory used
    assert first_dir.exists()  # First directory created
    assert not second_dir.exists()  # Second directory NOT created


def test_singleton_reset_instance_clears_singleton(
    temp_chroma_dir, reset_singleton_after_test
):
    """
    Test CRITICAL-1 fix: reset_instance() properly clears singleton.

    Verifies:
    - reset_instance() clears internal _instance
    - After reset, get_instance() creates new instance
    - New instance is different from old instance
    """
    # Arrange
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    client1 = ChromaClientSingleton.get_instance(persist_directory=temp_chroma_dir)

    # Act
    ChromaClientSingleton.reset_instance()
    client2 = ChromaClientSingleton.get_instance(persist_directory=temp_chroma_dir)

    # Assert
    assert client1 is not client2  # Different instances
    assert id(client1) != id(client2)  # Different memory addresses


def test_singleton_thread_safety(temp_chroma_dir, reset_singleton_after_test):
    """
    Test CRITICAL-1 fix: Singleton is thread-safe with concurrent access.

    Verifies:
    - Multiple threads calling get_instance() concurrently
    - All threads receive the same instance
    - No race conditions create multiple instances
    """
    # Arrange
    import threading
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    instances = []
    num_threads = 10

    def get_instance_in_thread():
        """Get singleton instance and append to shared list."""
        instance = ChromaClientSingleton.get_instance(
            persist_directory=temp_chroma_dir
        )
        instances.append(instance)

    # Act - Create and start threads
    threads = [
        threading.Thread(target=get_instance_in_thread) for _ in range(num_threads)
    ]
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Assert - All threads got the same instance
    assert len(instances) == num_threads
    first_instance = instances[0]
    for instance in instances:
        assert instance is first_instance  # All same object reference
        assert id(instance) == id(first_instance)  # All same memory address


def test_singleton_reset_instance_calls_client_reset(
    temp_chroma_dir, reset_singleton_after_test
):
    """
    Test CRITICAL-1 fix: reset_instance() calls client.reset() before clearing.

    Verifies:
    - reset_instance() properly cleans up ChromaDB resources
    - Prevents resource leaks on Windows (SQLite file locks)
    """
    # Arrange
    from unittest.mock import patch
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    client = ChromaClientSingleton.get_instance(persist_directory=temp_chroma_dir)

    # Act - Patch the reset method to verify it's called
    with patch.object(client, "reset", wraps=client.reset) as mock_reset:
        ChromaClientSingleton.reset_instance()

        # Assert
        mock_reset.assert_called_once()


def test_singleton_with_default_persist_directory(reset_singleton_after_test):
    """
    Test CRITICAL-1 fix: Singleton works with default persist_directory.

    Verifies:
    - get_instance() without arguments uses default directory
    - Default directory is created by ChromaClient
    """
    # Arrange & Act
    from src.infrastructure.ai.vector_store.chroma_client import (
        ChromaClientSingleton,
    )

    client = ChromaClientSingleton.get_instance()  # No persist_directory argument

    # Assert
    assert client is not None
    assert client.persist_directory.exists()


# Summary: 6 new Singleton tests for CRITICAL-1 fix
# - test_singleton_returns_same_instance_on_multiple_calls
# - test_singleton_ignores_persist_directory_on_subsequent_calls
# - test_singleton_reset_instance_clears_singleton
# - test_singleton_thread_safety
# - test_singleton_reset_instance_calls_client_reset
# - test_singleton_with_default_persist_directory
