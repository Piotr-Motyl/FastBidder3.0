"""
Tests for MatchResult Value Object.
Covers: match result creation, validation, serialization, confidence checks.
"""

import pytest
from pydantic import ValidationError
from uuid import UUID, uuid4

from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.hvac.value_objects.match_score import MatchScore


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_uuid():
    """Fixture for a sample UUID."""
    return UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")


@pytest.fixture
def high_confidence_result(sample_uuid):
    """Fixture for high confidence match result."""
    return MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=100.0,
        semantic_score=92.0,
        confidence=0.95,
        message="High confidence match - exact DN, PN, and valve type",
        breakdown={
            "parameter_matches": {"DN": True, "PN": True, "type": True},
            "semantic_similarity": 0.92,
            "score_gap_to_second": 15.3,
        },
        threshold=75.0,
    )


@pytest.fixture
def low_confidence_result(sample_uuid):
    """Fixture for low confidence match result."""
    return MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=60.0,
        semantic_score=75.0,
        confidence=0.65,
        message="Moderate confidence - partial parameter match",
        breakdown={
            "parameter_matches": {"DN": True, "PN": False, "type": True},
            "semantic_similarity": 0.75,
            "score_gap_to_second": 5.2,
        },
        threshold=75.0,
    )


# ============================================================================
# HAPPY PATH TESTS - create() factory method
# ============================================================================


def test_create_with_all_parameters(sample_uuid):
    """Test creating MatchResult with all parameters."""
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=100.0,
        semantic_score=92.0,
        confidence=0.95,
        message="Test match",
        breakdown={"test": "data"},
        threshold=75.0,
    )

    assert result.matched_reference_id == sample_uuid
    assert result.score.parameter_score == 100.0
    assert result.score.semantic_score == 92.0
    assert result.score.final_score == pytest.approx(95.2, rel=1e-5)
    assert result.confidence == 0.95
    assert result.message == "Test match"
    assert result.breakdown == {"test": "data"}


def test_create_with_default_threshold(sample_uuid):
    """Test create() uses default threshold (75.0) when not specified."""
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={},
    )
    assert result.score.threshold == 75.0


def test_create_automatically_creates_match_score(sample_uuid):
    """Test that create() automatically creates nested MatchScore."""
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=100.0,
        semantic_score=90.0,
        confidence=0.90,
        message="Test",
        breakdown={},
    )

    assert isinstance(result.score, MatchScore)
    assert result.score.parameter_score == 100.0
    assert result.score.semantic_score == 90.0


# ============================================================================
# HAPPY PATH TESTS - is_high_confidence()
# ============================================================================


def test_is_high_confidence_true(high_confidence_result):
    """Test is_high_confidence() returns True for high confidence matches."""
    # Default threshold is 0.8, confidence is 0.95
    assert high_confidence_result.is_high_confidence() is True


def test_is_high_confidence_false(low_confidence_result):
    """Test is_high_confidence() returns False for low confidence matches."""
    # Default threshold is 0.8, confidence is 0.65
    assert low_confidence_result.is_high_confidence() is False


def test_is_high_confidence_with_custom_threshold(high_confidence_result):
    """Test is_high_confidence() with custom threshold."""
    # confidence is 0.95
    assert high_confidence_result.is_high_confidence(threshold=0.90) is True
    assert high_confidence_result.is_high_confidence(threshold=0.98) is False


def test_is_high_confidence_exactly_at_threshold():
    """Test is_high_confidence() when confidence equals threshold."""
    result = MatchResult.create(
        matched_item_id=uuid4(),
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.80,
        message="Test",
        breakdown={},
    )
    # Confidence exactly at threshold should return True (>=)
    assert result.is_high_confidence(threshold=0.80) is True


# ============================================================================
# HAPPY PATH TESTS - to_dict()
# ============================================================================


def test_to_dict_structure(high_confidence_result):
    """Test to_dict() returns correct structure."""
    result_dict = high_confidence_result.to_dict()

    assert isinstance(result_dict, dict)
    assert "matched_item_id" in result_dict
    assert "score" in result_dict
    assert "confidence" in result_dict
    assert "message" in result_dict
    assert "breakdown" in result_dict


def test_to_dict_uuid_converted_to_string(sample_uuid):
    """Test that UUID is converted to string in to_dict()."""
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={},
    )
    result_dict = result.to_dict()

    assert result_dict["matched_item_id"] == str(sample_uuid)
    assert result_dict["matched_item_id"] == "3fa85f64-5717-4562-b3fc-2c963f66afa6"


def test_to_dict_score_is_nested_dict(high_confidence_result):
    """Test that score in to_dict() is a nested dictionary."""
    result_dict = high_confidence_result.to_dict()

    assert isinstance(result_dict["score"], dict)
    assert "parameter_score" in result_dict["score"]
    assert "semantic_score" in result_dict["score"]
    assert "final_score" in result_dict["score"]
    assert "threshold" in result_dict["score"]


def test_to_dict_breakdown_preserved(sample_uuid):
    """Test that breakdown dict is preserved in to_dict()."""
    breakdown_data = {
        "parameter_matches": {"DN": True, "PN": False},
        "semantic_similarity": 0.85,
        "extra_info": "test",
    }

    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown=breakdown_data,
    )

    result_dict = result.to_dict()
    assert result_dict["breakdown"] == breakdown_data


# ============================================================================
# EDGE CASES & ERROR HANDLING - Validation
# ============================================================================


def test_validation_confidence_above_range(sample_uuid):
    """Test that confidence > 1.0 raises ValidationError."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        MatchResult.create(
            matched_item_id=sample_uuid,
            parameter_score=80.0,
            semantic_score=85.0,
            confidence=1.5,  # Invalid!
            message="Test",
            breakdown={},
        )


def test_validation_confidence_below_range(sample_uuid):
    """Test that confidence < 0.0 raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        MatchResult.create(
            matched_item_id=sample_uuid,
            parameter_score=80.0,
            semantic_score=85.0,
            confidence=-0.1,  # Invalid!
            message="Test",
            breakdown={},
        )


def test_validation_message_empty_string(sample_uuid):
    """Test that empty message raises ValidationError."""
    with pytest.raises(ValidationError, match="at least 1 character"):
        MatchResult.create(
            matched_item_id=sample_uuid,
            parameter_score=80.0,
            semantic_score=85.0,
            confidence=0.90,
            message="",  # Invalid - empty!
            breakdown={},
        )


def test_validation_message_too_long(sample_uuid):
    """Test that message longer than 500 chars raises ValidationError."""
    long_message = "x" * 501  # 501 characters
    with pytest.raises(ValidationError, match="at most 500 characters"):
        MatchResult.create(
            matched_item_id=sample_uuid,
            parameter_score=80.0,
            semantic_score=85.0,
            confidence=0.90,
            message=long_message,
            breakdown={},
        )


def test_validation_message_exactly_500_chars(sample_uuid):
    """Test that message with exactly 500 chars is valid."""
    message_500 = "x" * 500
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message=message_500,
        breakdown={},
    )
    assert len(result.message) == 500


# ============================================================================
# IMMUTABILITY TESTS
# ============================================================================


def test_immutability_cannot_modify_confidence(high_confidence_result):
    """Test that MatchResult is immutable (frozen Pydantic model)."""
    with pytest.raises(ValidationError, match="Instance is frozen"):
        high_confidence_result.confidence = 0.50  # type: ignore


def test_immutability_cannot_modify_message(high_confidence_result):
    """Test that message cannot be modified."""
    with pytest.raises(ValidationError, match="Instance is frozen"):
        high_confidence_result.message = "New message"  # type: ignore


# ============================================================================
# EQUALITY TESTS
# ============================================================================


def test_equality_same_values(sample_uuid):
    """Test that two MatchResult instances with same values are equal."""
    result1 = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={"key": "value"},
    )
    result2 = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={"key": "value"},
    )
    assert result1 == result2


def test_inequality_different_uuid():
    """Test that MatchResult instances with different UUIDs are not equal."""
    uuid1 = uuid4()
    uuid2 = uuid4()

    result1 = MatchResult.create(
        matched_item_id=uuid1,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={},
    )
    result2 = MatchResult.create(
        matched_item_id=uuid2,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={},
    )
    assert result1 != result2


# ============================================================================
# INTEGRATION TESTS - MatchScore nested validation
# ============================================================================


def test_nested_match_score_validation_propagates(sample_uuid):
    """Test that MatchScore validation errors propagate to MatchResult."""
    # Trying to create with invalid parameter_score (> 100)
    with pytest.raises(ValidationError):
        MatchResult.create(
            matched_item_id=sample_uuid,
            parameter_score=150.0,  # Invalid!
            semantic_score=85.0,
            confidence=0.90,
            message="Test",
            breakdown={},
        )


def test_breakdown_can_be_empty_dict(sample_uuid):
    """Test that breakdown can be an empty dict."""
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown={},  # Empty is valid
    )
    assert result.breakdown == {}


def test_breakdown_can_contain_nested_structures(sample_uuid):
    """Test that breakdown can contain complex nested structures."""
    complex_breakdown = {
        "parameter_matches": {"DN": True, "PN": False, "material": True},
        "scores": {"dn_score": 100, "pn_score": 0},
        "metadata": {"extraction_time_ms": 123, "model_version": "v1.2.3"},
    }

    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=80.0,
        semantic_score=85.0,
        confidence=0.90,
        message="Test",
        breakdown=complex_breakdown,
    )
    assert result.breakdown == complex_breakdown
