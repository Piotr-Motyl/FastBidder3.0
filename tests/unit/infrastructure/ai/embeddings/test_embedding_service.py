"""
Tests for EmbeddingService.

Uses mocks to avoid loading real ML model (~420MB).
Tests cover lazy loading, batch processing, error handling, and edge cases.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.infrastructure.ai.embeddings.embedding_service import EmbeddingService


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_sentence_transformer():
    """
    Mock SentenceTransformer model.

    Returns deterministic embeddings for testing without loading real model.
    """
    mock_model = MagicMock()

    # Mock encode() to return numpy array with 384 dimensions
    mock_model.encode.return_value = np.random.rand(384).astype(np.float32)

    # Mock embedding dimension getter
    mock_model.get_sentence_embedding_dimension.return_value = 384

    # Mock device (CPU or CUDA)
    mock_model.device = "cpu"

    return mock_model


@pytest.fixture
def embedding_service(mock_sentence_transformer):
    """
    EmbeddingService with mocked SentenceTransformer.

    Model is pre-loaded (forced) to avoid lazy loading in tests.
    """
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ):
        service = EmbeddingService()
        # Force model loading to avoid lazy loading checks in every test
        _ = service.model
        return service


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================

def test_embed_single_returns_list_of_floats(embedding_service):
    """Test that embed_single returns list of floats with correct dimension."""
    # Act
    result = embedding_service.embed_single("Zawór kulowy DN50")

    # Assert
    assert isinstance(result, list)
    assert len(result) == 384
    assert all(isinstance(x, float) for x in result)


def test_embed_single_trims_whitespace(embedding_service, mock_sentence_transformer):
    """Test that embed_single trims whitespace from input."""
    # Arrange
    text_with_whitespace = "  Zawór kulowy DN50  "

    # Act
    embedding_service.embed_single(text_with_whitespace)

    # Assert
    # Check that encode was called with trimmed text
    call_args = mock_sentence_transformer.encode.call_args
    called_text = call_args[0][0]
    assert called_text == "Zawór kulowy DN50"


def test_embed_batch_returns_correct_count(embedding_service, mock_sentence_transformer):
    """Test that embed_batch returns embeddings for all inputs."""
    # Arrange
    mock_sentence_transformer.encode.return_value = np.random.rand(3, 384).astype(np.float32)
    texts = ["Text 1", "Text 2", "Text 3"]

    # Act
    result = embedding_service.embed_batch(texts)

    # Assert
    assert len(result) == 3
    assert all(len(emb) == 384 for emb in result)
    assert all(isinstance(emb, list) for emb in result)
    assert all(all(isinstance(x, float) for x in emb) for emb in result)


def test_embed_batch_uses_correct_batch_size(embedding_service, mock_sentence_transformer):
    """Test that embed_batch passes batch_size parameter correctly."""
    # Arrange
    mock_sentence_transformer.encode.return_value = np.random.rand(5, 384).astype(np.float32)
    texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
    custom_batch_size = 16

    # Act
    embedding_service.embed_batch(texts, batch_size=custom_batch_size)

    # Assert
    call_kwargs = mock_sentence_transformer.encode.call_args.kwargs
    assert call_kwargs["batch_size"] == custom_batch_size


def test_embed_batch_shows_progress_for_large_batches(embedding_service, mock_sentence_transformer):
    """Test that embed_batch enables progress bar for >100 items."""
    # Arrange
    num_texts = 150
    mock_sentence_transformer.encode.return_value = np.random.rand(num_texts, 384).astype(np.float32)
    texts = [f"Text {i}" for i in range(num_texts)]

    # Act
    embedding_service.embed_batch(texts)

    # Assert
    call_kwargs = mock_sentence_transformer.encode.call_args.kwargs
    assert call_kwargs["show_progress_bar"] is True


def test_embed_batch_no_progress_for_small_batches(embedding_service, mock_sentence_transformer):
    """Test that embed_batch disables progress bar for <=100 items."""
    # Arrange
    mock_sentence_transformer.encode.return_value = np.random.rand(10, 384).astype(np.float32)
    texts = [f"Text {i}" for i in range(10)]

    # Act
    embedding_service.embed_batch(texts)

    # Assert
    call_kwargs = mock_sentence_transformer.encode.call_args.kwargs
    assert call_kwargs["show_progress_bar"] is False


def test_get_embedding_dimension_returns_384(embedding_service):
    """Test that embedding dimension is 384 for default model."""
    # Act
    dimension = embedding_service.get_embedding_dimension()

    # Assert
    assert dimension == 384


def test_lazy_loading_model_not_loaded_at_init(mock_sentence_transformer):
    """Test that model is NOT loaded at initialization."""
    # Arrange & Act
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ) as mock_class:
        service = EmbeddingService()

        # Assert
        mock_class.assert_not_called()  # Model not loaded yet


def test_lazy_loading_model_loaded_on_first_access(mock_sentence_transformer):
    """Test that model is loaded on first property access."""
    # Arrange
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ) as mock_class:
        service = EmbeddingService()

        # Act
        _ = service.model  # First access

        # Assert
        mock_class.assert_called_once()  # Now loaded


def test_lazy_loading_model_cached_after_first_load(mock_sentence_transformer):
    """Test that model is cached and not reloaded on subsequent access."""
    # Arrange
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ) as mock_class:
        service = EmbeddingService()

        # Act
        _ = service.model  # First access
        _ = service.model  # Second access
        _ = service.model  # Third access

        # Assert
        mock_class.assert_called_once()  # Still only one call


def test_custom_model_name_passed_to_constructor(mock_sentence_transformer):
    """Test that custom model name is passed to SentenceTransformer."""
    # Arrange
    custom_model = "all-MiniLM-L6-v2"

    # Act
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ) as mock_class:
        service = EmbeddingService(model_name=custom_model)
        _ = service.model  # Trigger loading

        # Assert
        mock_class.assert_called_once_with(custom_model)


def test_default_model_name_used_when_none_provided(mock_sentence_transformer):
    """Test that default model is used when no model_name provided."""
    # Act
    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=mock_sentence_transformer,
    ) as mock_class:
        service = EmbeddingService()
        _ = service.model  # Trigger loading

        # Assert
        expected_model = "paraphrase-multilingual-MiniLM-L12-v2"
        mock_class.assert_called_once_with(expected_model)


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================

def test_embed_single_empty_text_raises_error(embedding_service):
    """Test that empty text raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        embedding_service.embed_single("")


def test_embed_single_whitespace_only_raises_error(embedding_service):
    """Test that whitespace-only text raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        embedding_service.embed_single("   ")


def test_embed_single_whitespace_with_tabs_raises_error(embedding_service):
    """Test that whitespace with tabs and newlines raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        embedding_service.embed_single("\t\n  \n\t")


def test_embed_batch_empty_list_returns_empty(embedding_service):
    """Test that empty input list returns empty output."""
    # Act
    result = embedding_service.embed_batch([])

    # Assert
    assert result == []


def test_embed_batch_trims_all_texts(embedding_service, mock_sentence_transformer):
    """Test that embed_batch trims whitespace from all texts."""
    # Arrange
    mock_sentence_transformer.encode.return_value = np.random.rand(2, 384).astype(np.float32)
    texts = ["  Text 1  ", "\tText 2\n"]

    # Act
    embedding_service.embed_batch(texts)

    # Assert
    call_args = mock_sentence_transformer.encode.call_args
    called_texts = call_args[0][0]
    assert called_texts == ["Text 1", "Text 2"]


def test_embed_single_very_long_text_works(embedding_service, mock_sentence_transformer):
    """Test that very long text is handled correctly."""
    # Arrange
    long_text = "Zawór kulowy " * 1000  # Very long text

    # Act
    result = embedding_service.embed_single(long_text)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 384


def test_embed_single_unicode_polish_characters(embedding_service, mock_sentence_transformer):
    """Test that Polish unicode characters are handled correctly."""
    # Arrange
    polish_text = "Zawór kulowy DN50 PN16 mosiądz ręczny"

    # Act
    result = embedding_service.embed_single(polish_text)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 384
    # Verify encode was called with Polish text
    call_args = mock_sentence_transformer.encode.call_args
    assert call_args[0][0] == polish_text


def test_embed_single_special_characters(embedding_service, mock_sentence_transformer):
    """Test that special characters (symbols, numbers) are handled."""
    # Arrange
    text_with_symbols = "DN50 PN16 @ 230V ~ 50Hz [OK]"

    # Act
    result = embedding_service.embed_single(text_with_symbols)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 384


# ============================================================================
# INTEGRATION-LIKE TESTS (still with mocks)
# ============================================================================

def test_full_workflow_single_then_batch(embedding_service, mock_sentence_transformer):
    """Test realistic workflow: single embedding then batch."""
    # Arrange
    mock_sentence_transformer.encode.side_effect = [
        np.random.rand(384).astype(np.float32),  # For single
        np.random.rand(3, 384).astype(np.float32),  # For batch
    ]

    # Act
    single_result = embedding_service.embed_single("Working description")
    batch_result = embedding_service.embed_batch(["Ref 1", "Ref 2", "Ref 3"])

    # Assert
    assert len(single_result) == 384
    assert len(batch_result) == 3
    assert all(len(emb) == 384 for emb in batch_result)


def test_embeddings_are_deterministic_structure(embedding_service, mock_sentence_transformer):
    """Test that same text structure returns consistent embedding structure."""
    # Arrange
    text = "Zawór kulowy DN50"

    # Mock to return same array each time
    fixed_embedding = np.array([0.1, 0.2, 0.3] + [0.0] * 381, dtype=np.float32)
    mock_sentence_transformer.encode.return_value = fixed_embedding

    # Act
    result1 = embedding_service.embed_single(text)
    result2 = embedding_service.embed_single(text)

    # Assert
    assert result1 == result2  # Same structure
    assert len(result1) == 384
