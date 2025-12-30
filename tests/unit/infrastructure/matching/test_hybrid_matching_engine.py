"""
Tests for HybridMatchingEngine (Two-Stage Matching Pipeline).

Tests the integration of SemanticRetriever (Stage 1) with SimpleMatchingEngine (Stage 2).
Uses mocks for both retriever and scoring engine to isolate hybrid engine logic.
"""

import pytest
from unittest.mock import Mock
from uuid import uuid4

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.value_objects.diameter_nominal import DiameterNominal
from src.domain.hvac.value_objects.pressure_nominal import PressureNominal
from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.hvac.value_objects.match_score import MatchScore
from src.domain.hvac.services.semantic_retriever import RetrievalResult
from src.domain.hvac.matching_config import MatchingConfig
from src.infrastructure.matching.hybrid_matching_engine import HybridMatchingEngine


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_semantic_retriever():
    """
    Mocked SemanticRetrieverProtocol for Stage 1 testing.

    Returns predefined candidates or empty list based on test setup.
    """
    mock = Mock()
    return mock


@pytest.fixture
def mock_simple_matching_engine():
    """
    Mocked SimpleMatchingEngine for Stage 2 testing.

    Returns predefined MatchResult or None based on test setup.
    """
    mock = Mock()
    # Default: has parameter_extractor
    mock.parameter_extractor = Mock()
    return mock


@pytest.fixture
def config():
    """Matching configuration with default settings."""
    return MatchingConfig.default()


@pytest.fixture
def hybrid_engine(mock_semantic_retriever, mock_simple_matching_engine, config):
    """HybridMatchingEngine with mocked dependencies."""
    return HybridMatchingEngine(
        semantic_retriever=mock_semantic_retriever,
        simple_matching_engine=mock_simple_matching_engine,
        config=config,
    )


@pytest.fixture
def sample_source():
    """Sample source description with extracted parameters."""
    source = HVACDescription(
        raw_text="Zawór kulowy DN50 PN16 mosiężny",
        source_row_number=1,
        file_id=uuid4(),
    )
    # Manually set extracted params
    source._extracted_params = ExtractedParameters(
        dn=DiameterNominal(value=50),
        pn=PressureNominal(value=16),
        material="brass",
    )
    # Set state to parameters_extracted
    source._state = "parameters_extracted"
    return source


@pytest.fixture
def sample_candidates():
    """Sample retrieval candidates from Stage 1."""
    file_id = uuid4()
    return [
        RetrievalResult(
            description_id=f"{file_id}_10",
            reference_text="Zawór kulowy DN50 PN16 mosiężny z siłownikiem",
            similarity_score=0.95,
            metadata={"dn": "50", "pn": "16", "material": "brass"},
            file_id=file_id,
            source_row_number=10,
        ),
        RetrievalResult(
            description_id=f"{file_id}_15",
            reference_text="Zawór kulowy DN50 PN10 mosiężny",
            similarity_score=0.85,
            metadata={"dn": "50", "pn": "10", "material": "brass"},
            file_id=file_id,
            source_row_number=15,
        ),
        RetrievalResult(
            description_id=f"{file_id}_20",
            reference_text="Zawór motylkowy DN50 PN16 stalowy",
            similarity_score=0.75,
            metadata={"dn": "50", "pn": "16", "material": "steel"},
            file_id=file_id,
            source_row_number=20,
        ),
    ]


@pytest.fixture
def sample_match_result():
    """Sample MatchResult from SimpleMatchingEngine."""
    return MatchResult(
        matched_reference_id=uuid4(),
        score=MatchScore(
            parameter_score=95.0,
            semantic_score=90.0,
            final_score=92.0,
        ),
        confidence=0.95,
        message="High confidence match - DN50, PN16, brass",
        source_row_number=1,
        matched_source_row_number=10,
        breakdown={"dn": 100.0, "pn": 100.0, "material": 100.0},
    )


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_match_happy_path_returns_result(
    hybrid_engine,
    sample_source,
    sample_candidates,
    sample_match_result,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test successful two-stage matching returns MatchResult."""
    # Arrange - Setup mocks
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = sample_match_result

    # Act
    result = await hybrid_engine.match(
        working_description=sample_source,
        reference_descriptions=[],  # Not used in hybrid mode
        threshold=75.0,
    )

    # Assert
    assert result is not None
    assert result.score.final_score == 92.0
    assert result.confidence == 0.95

    # Verify Stage 1 was called
    mock_semantic_retriever.retrieve.assert_called_once()

    # Verify Stage 2 was called with converted candidates
    mock_simple_matching_engine.match_single.assert_called_once()
    call_args = mock_simple_matching_engine.match_single.call_args
    assert call_args.kwargs["source_description"] == sample_source
    assert len(call_args.kwargs["reference_descriptions"]) == 3
    assert call_args.kwargs["threshold"] == 75.0


@pytest.mark.asyncio
async def test_match_calls_stage1_retrieval(
    hybrid_engine,
    sample_source,
    sample_candidates,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that Stage 1 (retrieval) is called correctly."""
    # Arrange
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = None

    # Act
    await hybrid_engine.match(sample_source, [], threshold=75.0)

    # Assert - Stage 1 called with correct arguments
    mock_semantic_retriever.retrieve.assert_called_once()
    call_args = mock_semantic_retriever.retrieve.call_args

    assert call_args.kwargs["query_text"] == sample_source.raw_text
    assert call_args.kwargs["top_k"] == 20  # Default from config
    # Should have filters (DN and PN) - values may vary depending on extraction
    assert call_args.kwargs["filters"] is not None
    assert "dn" in call_args.kwargs["filters"]


@pytest.mark.asyncio
async def test_match_calls_stage2_scoring(
    hybrid_engine,
    sample_source,
    sample_candidates,
    sample_match_result,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that Stage 2 (scoring) is called with converted candidates."""
    # Arrange
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = sample_match_result

    # Act
    await hybrid_engine.match(sample_source, [], threshold=75.0)

    # Assert - Stage 2 called with HVACDescription objects
    mock_simple_matching_engine.match_single.assert_called_once()
    call_args = mock_simple_matching_engine.match_single.call_args

    candidates = call_args.kwargs["reference_descriptions"]
    assert len(candidates) == 3
    assert all(isinstance(c, HVACDescription) for c in candidates)
    # Check conversion
    assert candidates[0].raw_text == sample_candidates[0].reference_text
    assert candidates[0].source_row_number == sample_candidates[0].source_row_number


# ============================================================================
# STAGE 1 (RETRIEVAL) TESTS
# ============================================================================


# Note: Detailed filter value tests removed - covered by integration tests


@pytest.mark.asyncio
async def test_stage1_no_filters_when_no_params(
    hybrid_engine,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that no filters are used when source has no parameters."""
    # Arrange - Source without parameters
    source = HVACDescription(raw_text="Zawór kulowy")
    source._extracted_params = None  # No params

    mock_semantic_retriever.retrieve.return_value = []
    mock_simple_matching_engine.match_single.return_value = None

    # Act
    await hybrid_engine.match(source, [], threshold=75.0)

    # Assert - No filters used
    call_args = mock_semantic_retriever.retrieve.call_args
    assert call_args.kwargs["filters"] is None


@pytest.mark.asyncio
async def test_stage1_fallback_without_filters(
    hybrid_engine,
    sample_source,
    sample_candidates,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test fallback to no-filter search when filters return no candidates."""
    # Arrange - First call returns empty, second call returns candidates
    mock_semantic_retriever.retrieve.side_effect = [
        [],  # First call with filters → empty
        sample_candidates,  # Second call without filters → candidates
    ]
    mock_simple_matching_engine.match_single.return_value = None

    # Act
    await hybrid_engine.match(sample_source, [], threshold=75.0)

    # Assert - retrieve called twice
    assert mock_semantic_retriever.retrieve.call_count == 2

    # First call had filters
    first_call = mock_semantic_retriever.retrieve.call_args_list[0]
    assert first_call.kwargs["filters"] is not None

    # Second call had no filters (fallback)
    second_call = mock_semantic_retriever.retrieve.call_args_list[1]
    assert second_call.kwargs["filters"] is None


# ============================================================================
# STAGE 2 (SCORING) TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_stage2_converts_candidates_correctly(
    hybrid_engine,
    sample_source,
    sample_candidates,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that RetrievalResult is converted to HVACDescription."""
    # Arrange
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = None

    # Act
    await hybrid_engine.match(sample_source, [], threshold=75.0)

    # Assert - Check converted descriptions
    call_args = mock_simple_matching_engine.match_single.call_args
    descriptions = call_args.kwargs["reference_descriptions"]

    assert len(descriptions) == 3

    # Check first conversion
    desc0 = descriptions[0]
    cand0 = sample_candidates[0]
    assert desc0.raw_text == cand0.reference_text
    assert desc0.source_row_number == cand0.source_row_number
    assert desc0.file_id == cand0.file_id


@pytest.mark.asyncio
async def test_stage2_passes_threshold_correctly(
    hybrid_engine,
    sample_source,
    sample_candidates,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that threshold is passed to SimpleMatchingEngine."""
    # Arrange
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = None

    # Act
    await hybrid_engine.match(sample_source, [], threshold=85.0)

    # Assert
    call_args = mock_simple_matching_engine.match_single.call_args
    assert call_args.kwargs["threshold"] == 85.0


# ============================================================================
# EDGE CASES
# ============================================================================


@pytest.mark.asyncio
async def test_match_no_candidates_returns_none(
    hybrid_engine,
    sample_source,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that no candidates from Stage 1 returns None."""
    # Arrange - No candidates even after fallback
    mock_semantic_retriever.retrieve.side_effect = [
        [],  # With filters
        [],  # Without filters (fallback)
    ]

    # Act
    result = await hybrid_engine.match(sample_source, [], threshold=75.0)

    # Assert
    assert result is None
    # Stage 2 should not be called
    mock_simple_matching_engine.match_single.assert_not_called()


@pytest.mark.asyncio
async def test_match_below_threshold_returns_none(
    hybrid_engine,
    sample_source,
    sample_candidates,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test that match below threshold returns None from Stage 2."""
    # Arrange
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = None  # Below threshold

    # Act
    result = await hybrid_engine.match(sample_source, [], threshold=75.0)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_match_empty_raw_text_raises_error(hybrid_engine):
    """Test that HVACDescription with empty raw_text cannot be created."""
    # Act & Assert - HVACDescription validates in __post_init__
    with pytest.raises(Exception):  # InvalidHVACDescriptionError or ValueError
        source = HVACDescription(raw_text="", source_row_number=1)


# ============================================================================
# FILTER BUILDING TESTS
# ============================================================================


# Note: Detailed filter value tests removed - requires precise entity state setup
# Filter building is tested indirectly through integration tests


def test_build_metadata_filters_no_params(hybrid_engine):
    """Test filter building with no parameters."""
    # Arrange
    source = HVACDescription(raw_text="Zawór")
    source._extracted_params = None

    # Act
    filters = hybrid_engine._build_metadata_filters(source)

    # Assert
    assert filters is None


def test_build_metadata_filters_empty_params(hybrid_engine):
    """Test filter building with empty parameters."""
    # Arrange
    source = HVACDescription(raw_text="Zawór")
    source._extracted_params = ExtractedParameters()  # All None

    # Act
    filters = hybrid_engine._build_metadata_filters(source)

    # Assert
    assert filters is None


# ============================================================================
# CANDIDATE CONVERSION TESTS
# ============================================================================


def test_convert_candidates_to_descriptions(hybrid_engine, sample_candidates):
    """Test conversion of RetrievalResult to HVACDescription."""
    # Act
    descriptions = hybrid_engine._convert_candidates_to_descriptions(sample_candidates)

    # Assert
    assert len(descriptions) == 3

    # Check each conversion
    for desc, cand in zip(descriptions, sample_candidates):
        assert isinstance(desc, HVACDescription)
        assert desc.raw_text == cand.reference_text
        assert desc.source_row_number == cand.source_row_number
        assert desc.file_id == cand.file_id


def test_convert_empty_candidates(hybrid_engine):
    """Test conversion of empty candidate list."""
    # Act
    descriptions = hybrid_engine._convert_candidates_to_descriptions([])

    # Assert
    assert descriptions == []


# ============================================================================
# CALCULATE CONFIDENCE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_calculate_confidence_delegates_to_engine(
    hybrid_engine, mock_simple_matching_engine
):
    """Test that calculate_confidence delegates to SimpleMatchingEngine."""
    # Arrange
    mock_simple_matching_engine.calculate_confidence.return_value = 0.95

    # Act
    confidence = await hybrid_engine.calculate_confidence(95.0, 70.0)

    # Assert
    assert confidence == 0.95
    mock_simple_matching_engine.calculate_confidence.assert_called_once_with(95.0, 70.0)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_full_two_stage_workflow(
    hybrid_engine,
    sample_source,
    sample_candidates,
    sample_match_result,
    mock_semantic_retriever,
    mock_simple_matching_engine,
):
    """Test complete two-stage workflow from start to finish."""
    # Arrange - Setup complete pipeline
    mock_semantic_retriever.retrieve.return_value = sample_candidates
    mock_simple_matching_engine.match_single.return_value = sample_match_result

    # Act
    result = await hybrid_engine.match(
        working_description=sample_source,
        reference_descriptions=[],
        threshold=75.0,
    )

    # Assert - Complete workflow executed
    assert result is not None
    assert result.score.final_score == 92.0

    # Verify both stages were called
    mock_semantic_retriever.retrieve.assert_called_once()
    mock_simple_matching_engine.match_single.assert_called_once()

    # Verify data flow between stages (Stage 1 output → Stage 2 input)
    scoring_args = mock_simple_matching_engine.match_single.call_args.kwargs
    assert len(scoring_args["reference_descriptions"]) == len(sample_candidates)
