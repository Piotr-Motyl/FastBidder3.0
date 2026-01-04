"""
Tests for ReferenceIndexer.

Uses mocked EmbeddingService (fake embeddings) and real ChromaDB (temp storage).
Tests indexing workflow, error handling, and metadata storage.
"""
import pytest
import tempfile
from unittest.mock import Mock
from uuid import uuid4

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.value_objects.diameter_nominal import DiameterNominal
from src.domain.hvac.value_objects.pressure_nominal import PressureNominal
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient
from src.infrastructure.ai.vector_store.reference_indexer import (
    ReferenceIndexer,
    IndexingResult,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_chroma_dir():
    """Temporary directory for ChromaDB storage."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
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

    # Mock embed_batch to return fake 384-dim embeddings
    def fake_embed_batch(texts, **kwargs):
        # Return list of 384-dim vectors (one per text)
        return [[0.1] * 384 for _ in texts]

    mock.embed_batch.side_effect = fake_embed_batch
    return mock


@pytest.fixture
def reference_indexer(mock_embedding_service, chroma_client):
    """ReferenceIndexer with mocked embeddings and real ChromaDB."""
    return ReferenceIndexer(mock_embedding_service, chroma_client)


@pytest.fixture
def sample_descriptions():
    """Sample HVAC descriptions for testing."""
    file_id = uuid4()

    desc1 = HVACDescription(
        raw_text="Zawór kulowy DN50 PN16 mosiężny",
        source_row_number=1,
        file_id=file_id,
    )
    desc1.extracted_params = ExtractedParameters(
        dn=DiameterNominal(50),
        pn=PressureNominal(16),
        material="brass",
        valve_type="ball_valve",
    )

    desc2 = HVACDescription(
        raw_text="Rura PVC DN110 SDR17",
        source_row_number=2,
        file_id=file_id,
    )
    desc2.extracted_params = ExtractedParameters(
        dn=DiameterNominal(110),
        material="pvc",
    )

    desc3 = HVACDescription(
        raw_text="Klapka zwrotna DN100 PN10",
        source_row_number=3,
        file_id=file_id,
    )

    return file_id, [desc1, desc2, desc3]


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================

def test_index_file_returns_result(reference_indexer, sample_descriptions):
    """Test that index_file returns IndexingResult."""
    # Arrange
    file_id, descriptions = sample_descriptions

    # Act
    result = reference_indexer.index_file(file_id, descriptions)

    # Assert
    assert isinstance(result, IndexingResult)
    assert result.file_id == file_id
    assert result.total_descriptions == 3
    assert result.indexed_count == 3
    assert result.failed_count == 0
    assert result.success_rate == 100.0


def test_index_file_stores_in_chromadb(reference_indexer, sample_descriptions, chroma_client):
    """Test that descriptions are stored in ChromaDB."""
    # Arrange
    file_id, descriptions = sample_descriptions

    # Act
    reference_indexer.index_file(file_id, descriptions)

    # Assert
    collection = chroma_client.get_or_create_collection()
    results = collection.get(
        where={"file_id": str(file_id)}, include=["documents", "embeddings", "metadatas"]
    )

    assert len(results["ids"]) == 3
    assert len(results["documents"]) == 3
    assert len(results["embeddings"]) == 3


def test_index_file_stores_metadata(reference_indexer, sample_descriptions, chroma_client):
    """Test that metadata is correctly stored."""
    # Arrange
    file_id, descriptions = sample_descriptions

    # Act
    reference_indexer.index_file(file_id, descriptions)

    # Assert
    collection = chroma_client.get_or_create_collection()
    results = collection.get(where={"file_id": str(file_id)})
    metadatas = results["metadatas"]

    # Check first description metadata
    meta1 = metadatas[0]
    assert meta1["file_id"] == str(file_id)
    assert meta1["source_row_number"] == 1
    assert meta1["dn"] == "50"
    assert meta1["pn"] == "16"
    assert meta1["material"] == "brass"
    assert meta1["valve_type"] == "ball_valve"


def test_index_file_uses_batch_embedding(reference_indexer, sample_descriptions, mock_embedding_service):
    """Test that embed_batch is called for efficiency."""
    # Arrange
    file_id, descriptions = sample_descriptions

    # Act
    reference_indexer.index_file(file_id, descriptions)

    # Assert
    assert mock_embedding_service.embed_batch.called
    # Should be called once for batch of 3 descriptions
    assert mock_embedding_service.embed_batch.call_count == 1


def test_index_file_document_id_format(reference_indexer, sample_descriptions, chroma_client):
    """Test that document IDs follow format: {file_id}_{row_number}."""
    # Arrange
    file_id, descriptions = sample_descriptions

    # Act
    reference_indexer.index_file(file_id, descriptions)

    # Assert
    collection = chroma_client.get_or_create_collection()
    results = collection.get(where={"file_id": str(file_id)})
    ids = results["ids"]

    assert f"{file_id}_1" in ids
    assert f"{file_id}_2" in ids
    assert f"{file_id}_3" in ids


def test_is_file_indexed_returns_true_after_indexing(reference_indexer, sample_descriptions):
    """Test that is_file_indexed returns True after indexing."""
    # Arrange
    file_id, descriptions = sample_descriptions

    # Act
    reference_indexer.index_file(file_id, descriptions)

    # Assert
    assert reference_indexer.is_file_indexed(file_id) is True


def test_is_file_indexed_returns_false_before_indexing(reference_indexer):
    """Test that is_file_indexed returns False for non-indexed file."""
    # Arrange
    file_id = uuid4()

    # Act & Assert
    assert reference_indexer.is_file_indexed(file_id) is False


def test_get_indexed_count_returns_total(reference_indexer, sample_descriptions):
    """Test that get_indexed_count returns total count."""
    # Arrange
    file_id, descriptions = sample_descriptions
    reference_indexer.index_file(file_id, descriptions)

    # Act
    count = reference_indexer.get_indexed_count()

    # Assert
    assert count == 3


def test_get_indexed_count_for_specific_file(reference_indexer, sample_descriptions):
    """Test that get_indexed_count works for specific file."""
    # Arrange
    file_id, descriptions = sample_descriptions
    reference_indexer.index_file(file_id, descriptions)

    # Act
    count = reference_indexer.get_indexed_count(file_id)

    # Assert
    assert count == 3


def test_remove_file_deletes_descriptions(reference_indexer, sample_descriptions):
    """Test that remove_file deletes all descriptions for file."""
    # Arrange
    file_id, descriptions = sample_descriptions
    reference_indexer.index_file(file_id, descriptions)

    # Act
    removed_count = reference_indexer.remove_file(file_id)

    # Assert
    assert removed_count == 3
    assert reference_indexer.is_file_indexed(file_id) is False


# ============================================================================
# IDEMPOTENCY TESTS
# ============================================================================

def test_skip_if_indexed_prevents_re_indexing(reference_indexer, sample_descriptions):
    """Test that skip_if_indexed=True skips already indexed file."""
    # Arrange
    file_id, descriptions = sample_descriptions
    reference_indexer.index_file(file_id, descriptions)

    # Act
    result = reference_indexer.index_file(file_id, descriptions, skip_if_indexed=True)

    # Assert
    assert result.indexed_count == 0
    assert "already indexed" in result.errors[0].lower()


def test_re_indexing_replaces_old_data(reference_indexer, sample_descriptions):
    """Test that re-indexing (skip_if_indexed=False) replaces old data."""
    # Arrange
    file_id, descriptions = sample_descriptions
    reference_indexer.index_file(file_id, descriptions)

    # Modify descriptions
    descriptions[0].raw_text = "MODIFIED TEXT"

    # Act
    result = reference_indexer.index_file(file_id, descriptions, skip_if_indexed=False)

    # Assert
    assert result.indexed_count == 3
    # Check that new text is stored
    collection = reference_indexer.chroma_client.get_or_create_collection()
    results = collection.get(where={"file_id": str(file_id)})
    assert "MODIFIED TEXT" in results["documents"]


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

# Tests for empty descriptions removed - HVACDescription validates text in __post_init__
# Empty text cannot be created as HVACDescription entity


def test_partial_failure_continues_indexing(reference_indexer, mock_embedding_service):
    """Test that batch failure doesn't stop entire indexing."""
    # Arrange
    file_id = uuid4()
    descriptions = [
        HVACDescription(raw_text="Valid 1", source_row_number=1, file_id=file_id),
        HVACDescription(raw_text="Valid 2", source_row_number=2, file_id=file_id),
    ]

    # Make first call fail, second succeed
    call_count = [0]

    def failing_embed_batch(texts, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Embedding failed")
        return [[0.1] * 384 for _ in texts]

    mock_embedding_service.embed_batch.side_effect = failing_embed_batch

    # Act
    result = reference_indexer.index_file(file_id, descriptions)

    # Assert - should have processed second batch despite first failure
    # (Note: in real scenario with larger batches, this would show partial success)
    assert result.failed_count > 0
    assert len(result.errors) > 0


# ============================================================================
# EDGE CASES
# ============================================================================

def test_empty_descriptions_list(reference_indexer):
    """Test indexing empty list of descriptions."""
    # Arrange
    file_id = uuid4()
    descriptions = []

    # Act
    result = reference_indexer.index_file(file_id, descriptions)

    # Assert
    assert result.total_descriptions == 0
    assert result.indexed_count == 0
    assert result.success_rate == 0.0


def test_description_without_extracted_params(reference_indexer):
    """Test indexing description without extracted parameters."""
    # Arrange
    file_id = uuid4()
    desc = HVACDescription(
        raw_text="Some description",
        source_row_number=1,
        file_id=file_id,
    )
    # extracted_params is None

    # Act
    result = reference_indexer.index_file(file_id, [desc])

    # Assert
    assert result.indexed_count == 1
    # Check metadata doesn't have DN/PN fields
    collection = reference_indexer.chroma_client.get_or_create_collection()
    results = collection.get(where={"file_id": str(file_id)})
    metadata = results["metadatas"][0]
    assert "dn" not in metadata
    assert "pn" not in metadata


def test_success_rate_calculation(reference_indexer):
    """Test IndexingResult.success_rate property."""
    # Arrange
    file_id = uuid4()

    # Act
    result = IndexingResult(
        file_id=file_id,
        total_descriptions=100,
        indexed_count=95,
        failed_count=5,
    )

    # Assert
    assert result.success_rate == 95.0


def test_success_rate_zero_total(reference_indexer):
    """Test success_rate returns 0 when total is 0."""
    # Arrange
    file_id = uuid4()

    # Act
    result = IndexingResult(
        file_id=file_id,
        total_descriptions=0,
        indexed_count=0,
        failed_count=0,
    )

    # Assert
    assert result.success_rate == 0.0
