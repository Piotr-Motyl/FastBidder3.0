"""
Tests for SemanticRetriever.

Uses mocked EmbeddingService (fake embeddings) and real ChromaDB (temp storage).
Tests retrieval workflow, filtering, score normalization, and error handling.
"""
import pytest
import tempfile
from unittest.mock import Mock
from uuid import uuid4

from src.domain.hvac.services.semantic_retriever import RetrievalResult
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient
from src.infrastructure.ai.retrieval.semantic_retriever import SemanticRetriever


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def temp_chroma_dir():
    """Temporary directory for ChromaDB storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def chroma_client(temp_chroma_dir, request):
    """ChromaDB client with temporary storage."""
    client = ChromaClient(persist_directory=temp_chroma_dir)

    def cleanup():
        client.reset()

    request.addfinalizer(cleanup)
    return client


@pytest.fixture
def mock_embedding_service():
    """
    Mocked EmbeddingService that returns fake embeddings.

    Returns 384-dimensional vectors filled with 0.1, 0.2, etc.
    """
    mock = Mock()

    # Mock embed_single to return fake 384-dim embedding
    def fake_embed_single(text, **kwargs):
        # Return unique embedding based on text hash
        # This ensures different texts get different embeddings
        hash_val = hash(text) % 100 / 100.0
        return [hash_val] * 384

    mock.embed_single.side_effect = fake_embed_single
    return mock


@pytest.fixture
def semantic_retriever(mock_embedding_service, chroma_client):
    """SemanticRetriever with mocked embeddings and real ChromaDB."""
    return SemanticRetriever(mock_embedding_service, chroma_client)


@pytest.fixture
def populated_chroma(chroma_client):
    """ChromaDB populated with sample reference descriptions."""
    collection = chroma_client.get_or_create_collection()

    # Add sample descriptions
    file_id = uuid4()
    ids = [
        f"{file_id}_1",
        f"{file_id}_2",
        f"{file_id}_3",
        f"{file_id}_4",
        f"{file_id}_5",
    ]
    documents = [
        "Zawór kulowy DN50 PN16 mosiężny",
        "Zawór kulowy DN100 PN16 mosiężny",
        "Zawór motylkowy DN50 PN10 stalowy",
        "Rura stalowa DN50 SDR11",
        "Kolano 90° DN50 mosiężne",
    ]
    embeddings = [
        [0.1] * 384,
        [0.2] * 384,
        [0.3] * 384,
        [0.4] * 384,
        [0.5] * 384,
    ]
    metadatas = [
        {"dn": "50", "pn": "16", "material": "brass", "valve_type": "ball_valve"},
        {"dn": "100", "pn": "16", "material": "brass", "valve_type": "ball_valve"},
        {"dn": "50", "pn": "10", "material": "steel", "valve_type": "butterfly_valve"},
        {"dn": "50", "material": "steel"},
        {"dn": "50", "material": "brass"},
    ]

    collection.add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )

    return chroma_client


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================


def test_retrieve_returns_results(semantic_retriever, populated_chroma):
    """Test basic retrieval returns RetrievalResult objects."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve("Zawór kulowy DN50", top_k=5)

    # Assert
    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    assert all(r.description_id for r in results)
    assert all(r.reference_text for r in results)
    assert all(0.0 <= r.similarity_score <= 1.0 for r in results)


def test_retrieve_respects_top_k(semantic_retriever, populated_chroma):
    """Test that top_k limits number of results."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve("test query", top_k=3)

    # Assert
    assert len(results) <= 3


def test_retrieve_results_sorted_by_similarity(semantic_retriever, populated_chroma):
    """Test that results are sorted by similarity (descending)."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve("test query", top_k=5)

    # Assert
    similarities = [r.similarity_score for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_retrieve_similarity_scores_normalized(semantic_retriever, populated_chroma):
    """Test that similarity scores are normalized to 0.0-1.0 range."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve("test query", top_k=5)

    # Assert
    for result in results:
        assert 0.0 <= result.similarity_score <= 1.0


def test_retrieve_parses_description_id(semantic_retriever, populated_chroma):
    """Test that RetrievalResult parses file_id and source_row_number."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve("test query", top_k=1)

    # Assert
    assert len(results) > 0
    result = results[0]
    assert result.file_id is not None
    assert result.source_row_number is not None


# ============================================================================
# FILTERING TESTS
# ============================================================================


def test_retrieve_with_single_filter(semantic_retriever, populated_chroma):
    """Test retrieval with single metadata filter."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve(
        query_text="Zawór", filters={"dn": "50"}, top_k=10
    )

    # Assert
    assert len(results) > 0
    # All results should have DN50
    assert all(r.metadata.get("dn") == "50" for r in results)


def test_retrieve_with_multiple_filters(semantic_retriever, populated_chroma):
    """Test retrieval with multiple metadata filters (AND logic)."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve(
        query_text="Zawór",
        filters={"dn": "50", "material": "brass"},
        top_k=10,
    )

    # Assert
    assert len(results) > 0
    # All results should match both filters
    for result in results:
        assert result.metadata.get("dn") == "50"
        assert result.metadata.get("material") == "brass"


def test_retrieve_filters_skip_none_values(semantic_retriever, populated_chroma):
    """Test that None values in filters are skipped."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve(
        query_text="Zawór",
        filters={"dn": "50", "pn": None, "material": None},
        top_k=10,
    )

    # Assert
    assert len(results) > 0
    # Only DN filter should be applied
    assert all(r.metadata.get("dn") == "50" for r in results)


def test_retrieve_without_filters(semantic_retriever, populated_chroma):
    """Test retrieval without filters returns all results."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve("test", top_k=10)

    # Assert
    assert len(results) > 0
    # No filtering - can have any metadata


def test_retrieve_no_results_with_strict_filters(semantic_retriever, populated_chroma):
    """Test that strict filters can return no results."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act - Filter that matches nothing
    results = semantic_retriever.retrieve(
        query_text="test",
        filters={"dn": "999", "material": "nonexistent"},
        top_k=10,
    )

    # Assert
    assert len(results) == 0


# ============================================================================
# BUILD WHERE CLAUSE TESTS
# ============================================================================


def test_build_where_clause_single_filter(semantic_retriever):
    """Test building where clause for single filter."""
    # Arrange
    filters = {"dn": "50"}

    # Act
    where = semantic_retriever._build_where_clause(filters)

    # Assert
    assert where == {"dn": {"$eq": "50"}}


def test_build_where_clause_multiple_filters(semantic_retriever):
    """Test building where clause for multiple filters with AND."""
    # Arrange
    filters = {"dn": "50", "pn": "16"}

    # Act
    where = semantic_retriever._build_where_clause(filters)

    # Assert
    assert where == {"$and": [{"dn": {"$eq": "50"}}, {"pn": {"$eq": "16"}}]}


def test_build_where_clause_skips_none_values(semantic_retriever):
    """Test that None values are skipped."""
    # Arrange
    filters = {"dn": "50", "pn": None, "material": "brass"}

    # Act
    where = semantic_retriever._build_where_clause(filters)

    # Assert
    # Should only include dn and material
    assert where == {"$and": [{"dn": {"$eq": "50"}}, {"material": {"$eq": "brass"}}]}


def test_build_where_clause_all_none_returns_none(semantic_retriever):
    """Test that all None values returns None."""
    # Arrange
    filters = {"dn": None, "pn": None, "material": None}

    # Act
    where = semantic_retriever._build_where_clause(filters)

    # Assert
    assert where is None


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_retrieve_empty_query_text_raises_error(semantic_retriever):
    """Test that empty query_text raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="query_text cannot be empty"):
        semantic_retriever.retrieve("", top_k=10)

    with pytest.raises(ValueError, match="query_text cannot be empty"):
        semantic_retriever.retrieve("   ", top_k=10)


def test_retrieve_invalid_top_k_raises_error(semantic_retriever):
    """Test that top_k < 1 raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        semantic_retriever.retrieve("test", top_k=0)

    with pytest.raises(ValueError, match="top_k must be >= 1"):
        semantic_retriever.retrieve("test", top_k=-5)


def test_retrieve_embedding_error_raises_runtime_error(semantic_retriever):
    """Test that embedding error raises RuntimeError."""
    # Arrange - Mock embedding service to raise error
    semantic_retriever.embedding_service.embed_single.side_effect = Exception(
        "Embedding failed"
    )

    # Act & Assert
    with pytest.raises(RuntimeError, match="Failed to generate embedding"):
        semantic_retriever.retrieve("test", top_k=10)


# ============================================================================
# EDGE CASES
# ============================================================================


def test_retrieve_empty_database_returns_empty_list(semantic_retriever, chroma_client):
    """Test retrieval from empty database returns empty list."""
    # Arrange - Empty database
    semantic_retriever.chroma_client = chroma_client

    # Act
    results = semantic_retriever.retrieve("test query", top_k=10)

    # Assert
    assert results == []


def test_retrieve_top_k_larger_than_database(semantic_retriever, populated_chroma):
    """Test that top_k larger than database size returns all results."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act - Ask for more results than exist
    results = semantic_retriever.retrieve("test", top_k=1000)

    # Assert - Should return all available (5 in populated_chroma)
    assert len(results) <= 5


def test_retrieve_result_metadata_preserved(semantic_retriever, populated_chroma):
    """Test that metadata from ChromaDB is preserved in results."""
    # Arrange
    semantic_retriever.chroma_client = populated_chroma

    # Act
    results = semantic_retriever.retrieve(
        "Zawór kulowy", filters={"dn": "50", "pn": "16"}, top_k=1
    )

    # Assert
    assert len(results) > 0
    result = results[0]
    assert result.metadata is not None
    assert "dn" in result.metadata
    assert "pn" in result.metadata
    assert result.metadata["dn"] == "50"
    assert result.metadata["pn"] == "16"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_workflow_with_real_components(mock_embedding_service, chroma_client):
    """Test full workflow: index → retrieve → verify results."""
    # Arrange - Create retriever
    retriever = SemanticRetriever(mock_embedding_service, chroma_client)

    # Add sample data to ChromaDB
    collection = chroma_client.get_or_create_collection()
    file_id = uuid4()

    collection.add(
        ids=[f"{file_id}_1", f"{file_id}_2", f"{file_id}_3"],
        embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
        documents=[
            "Zawór kulowy DN50 PN16",
            "Zawór kulowy DN100 PN10",
            "Rura PVC DN50",
        ],
        metadatas=[
            {"dn": "50", "pn": "16", "valve_type": "ball_valve"},
            {"dn": "100", "pn": "10", "valve_type": "ball_valve"},
            {"dn": "50", "material": "pvc"},
        ],
    )

    # Act - Retrieve with filters
    results = retriever.retrieve(
        query_text="Zawór kulowy DN50",
        filters={"dn": "50"},
        top_k=10,
    )

    # Assert
    assert len(results) == 2  # Should match 2 out of 3 (DN50 only)
    assert all(r.metadata["dn"] == "50" for r in results)


def test_custom_collection_name(mock_embedding_service, chroma_client):
    """Test using custom collection name."""
    # Arrange - Create retriever with custom collection
    retriever = SemanticRetriever(
        mock_embedding_service, chroma_client, collection_name="custom_collection"
    )

    # Add data to custom collection
    collection = chroma_client.get_or_create_collection("custom_collection")
    file_id = uuid4()
    collection.add(
        ids=[f"{file_id}_1"],
        embeddings=[[0.1] * 384],
        documents=["Test document"],
        metadatas=[{"dn": "50"}],
    )

    # Act
    results = retriever.retrieve("Test", top_k=10)

    # Assert
    assert len(results) == 1
    assert results[0].reference_text == "Test document"
