"""
Tests for HVACDescription Entity.
Covers: creation, validation, state transitions, business methods, serialization.
"""

import pytest
from datetime import datetime
from decimal import Decimal
from uuid import UUID, uuid4

from src.domain.hvac.entities.hvac_description import (
    HVACDescription,
    HVACDescriptionState,
)
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.value_objects.match_score import MatchScore
from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.hvac.services.concrete_parameter_extractor import (
    ConcreteParameterExtractor,
)
from src.domain.shared.exceptions import InvalidHVACDescriptionError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_uuid():
    """Fixture for a sample UUID."""
    return UUID("550e8400-e29b-41d4-a716-446655440000")


@pytest.fixture
def basic_description():
    """Fixture for basic HVACDescription."""
    return HVACDescription(raw_text="Zawór kulowy DN50 PN16")


@pytest.fixture
def description_with_metadata(sample_uuid):
    """Fixture for HVACDescription with metadata."""
    return HVACDescription(
        raw_text="Zawór kulowy DN50 PN16 mosiężny",
        source_row_number=5,
        file_id=sample_uuid,
    )


@pytest.fixture
def extractor():
    """Fixture for ConcreteParameterExtractor."""
    return ConcreteParameterExtractor()


# ============================================================================
# TESTS - Creation and Validation
# ============================================================================


def test_create_basic_description():
    """Test creating basic HVACDescription."""
    desc = HVACDescription(raw_text="Zawór DN50")

    assert desc.raw_text == "Zawór DN50"
    assert desc.state == HVACDescriptionState.CREATED
    assert desc.extracted_params is None
    assert desc.match_score is None
    assert desc.matched_price is None
    assert isinstance(desc.id, UUID)
    assert isinstance(desc.created_at, datetime)
    assert isinstance(desc.updated_at, datetime)


def test_create_with_metadata(sample_uuid):
    """Test creating HVACDescription with metadata."""
    desc = HVACDescription(
        raw_text="Zawór DN50",
        source_row_number=10,
        file_id=sample_uuid,
    )

    assert desc.raw_text == "Zawór DN50"
    assert desc.source_row_number == 10
    assert desc.file_id == sample_uuid


def test_from_excel_row_factory(sample_uuid):
    """Test from_excel_row() factory method."""
    desc = HVACDescription.from_excel_row(
        raw_text="Zawór kulowy DN50 PN16",
        source_row_number=5,
        file_id=sample_uuid,
    )

    assert desc.raw_text == "Zawór kulowy DN50 PN16"
    assert desc.source_row_number == 5
    assert desc.file_id == sample_uuid
    assert desc.state == HVACDescriptionState.CREATED


def test_text_normalization():
    """Test that raw_text is normalized (whitespace removed)."""
    desc = HVACDescription(raw_text="  Zawór  DN50  PN16  ")
    assert desc.raw_text == "Zawór DN50 PN16"


def test_text_normalization_with_tabs_and_newlines():
    """Test normalization removes tabs and newlines."""
    desc = HVACDescription(raw_text="Zawór\tDN50\nPN16")
    assert desc.raw_text == "Zawór DN50 PN16"


def test_validation_error_too_short():
    """Test that text shorter than MIN_TEXT_LENGTH raises error."""
    with pytest.raises(InvalidHVACDescriptionError, match="at least 3 characters"):
        HVACDescription(raw_text="DN")


def test_validation_error_empty_string():
    """Test that empty string raises error."""
    with pytest.raises(InvalidHVACDescriptionError, match="at least 3 characters"):
        HVACDescription(raw_text="")


def test_validation_error_only_whitespace():
    """Test that whitespace-only string raises error."""
    with pytest.raises(InvalidHVACDescriptionError, match="at least 3 characters"):
        HVACDescription(raw_text="   ")


def test_validation_error_not_string():
    """Test that non-string raw_text raises error."""
    with pytest.raises(InvalidHVACDescriptionError, match="must be string"):
        HVACDescription(raw_text=123)  # type: ignore


# ============================================================================
# TESTS - is_valid()
# ============================================================================


def test_is_valid_returns_true(basic_description):
    """Test is_valid() returns True for valid description."""
    assert basic_description.is_valid() is True


def test_is_valid_returns_false_for_invalid_row_number():
    """Test is_valid() returns False when source_row_number is 0 or negative."""
    desc = HVACDescription(raw_text="Zawór DN50", source_row_number=0)
    assert desc.is_valid() is False

    desc2 = HVACDescription(raw_text="Zawór DN50", source_row_number=-1)
    assert desc2.is_valid() is False


def test_is_valid_returns_true_for_valid_row_number():
    """Test is_valid() returns True when source_row_number is positive."""
    desc = HVACDescription(raw_text="Zawór DN50", source_row_number=5)
    assert desc.is_valid() is True


def test_is_valid_returns_true_when_row_number_is_none():
    """Test is_valid() returns True when source_row_number is None."""
    desc = HVACDescription(raw_text="Zawór DN50", source_row_number=None)
    assert desc.is_valid() is True


# ============================================================================
# TESTS - has_parameters() and has_critical_parameters()
# ============================================================================


def test_has_parameters_false_when_none(basic_description):
    """Test has_parameters() returns False when extracted_params is None."""
    assert basic_description.has_parameters() is False


def test_has_parameters_true_when_params_exist(basic_description):
    """Test has_parameters() returns True when params exist."""
    basic_description.extracted_params = ExtractedParameters(
        dn=50, confidence_scores={"dn": 1.0}
    )
    assert basic_description.has_parameters() is True


def test_has_critical_parameters_false_when_none(basic_description):
    """Test has_critical_parameters() returns False when no params."""
    assert basic_description.has_critical_parameters() is False


def test_has_critical_parameters_true_when_dn_and_pn_exist(basic_description):
    """Test has_critical_parameters() returns True when DN and PN exist."""
    basic_description.extracted_params = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    assert basic_description.has_critical_parameters() is True


def test_has_critical_parameters_false_when_only_dn(basic_description):
    """Test has_critical_parameters() returns False when only DN exists."""
    basic_description.extracted_params = ExtractedParameters(
        dn=50, confidence_scores={"dn": 1.0}
    )
    assert basic_description.has_critical_parameters() is False


def test_has_critical_parameters_false_when_only_pn(basic_description):
    """Test has_critical_parameters() returns False when only PN exists."""
    basic_description.extracted_params = ExtractedParameters(
        pn=16, confidence_scores={"pn": 1.0}
    )
    assert basic_description.has_critical_parameters() is False


# ============================================================================
# TESTS - extract_parameters() State Transition
# ============================================================================


def test_extract_parameters_success(basic_description, extractor):
    """Test extract_parameters() extracts params and transitions state."""
    assert basic_description.state == HVACDescriptionState.CREATED

    basic_description.extract_parameters(extractor)

    assert basic_description.state == HVACDescriptionState.PARAMETERS_EXTRACTED
    assert basic_description.extracted_params is not None
    assert basic_description.extracted_params.dn == 50
    assert basic_description.extracted_params.pn == 16


def test_extract_parameters_updates_timestamp(basic_description, extractor):
    """Test extract_parameters() updates updated_at timestamp."""
    old_timestamp = basic_description.updated_at

    basic_description.extract_parameters(extractor)

    assert basic_description.updated_at >= old_timestamp


def test_extract_parameters_with_none_extractor_raises_error(basic_description):
    """Test extract_parameters() raises error when extractor is None."""
    with pytest.raises(InvalidHVACDescriptionError, match="extractor cannot be None"):
        basic_description.extract_parameters(None)  # type: ignore


def test_extract_parameters_works_with_empty_results():
    """Test extract_parameters() works even when no params found."""
    desc = HVACDescription(raw_text="Random text without params")
    extractor = ConcreteParameterExtractor()

    desc.extract_parameters(extractor)

    assert desc.state == HVACDescriptionState.PARAMETERS_EXTRACTED
    assert desc.extracted_params is not None
    # Parameters should be empty (all None)
    assert desc.extracted_params.dn is None
    assert desc.extracted_params.pn is None


# ============================================================================
# TESTS - apply_match_result() State Transition
# ============================================================================


def test_apply_match_result_success(basic_description, extractor, sample_uuid):
    """Test apply_match_result() applies result and transitions state."""
    # First extract parameters
    basic_description.extract_parameters(extractor)

    # Create match result
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=100.0,
        semantic_score=90.0,
        confidence=0.95,
        message="High confidence match",
        breakdown={},
    )

    basic_description.apply_match_result(result)

    assert basic_description.state == HVACDescriptionState.MATCHED
    assert basic_description.match_score is not None
    assert basic_description.match_score.parameter_score == 100.0


def test_apply_match_result_updates_timestamp(basic_description, extractor, sample_uuid):
    """Test apply_match_result() updates updated_at timestamp."""
    basic_description.extract_parameters(extractor)
    old_timestamp = basic_description.updated_at

    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=100.0,
        semantic_score=90.0,
        confidence=0.95,
        message="Test",
        breakdown={},
    )

    basic_description.apply_match_result(result)

    assert basic_description.updated_at >= old_timestamp


def test_apply_match_result_with_none_raises_error(basic_description, extractor):
    """Test apply_match_result() raises error when result is None."""
    basic_description.extract_parameters(extractor)

    with pytest.raises(InvalidHVACDescriptionError, match="result cannot be None"):
        basic_description.apply_match_result(None)  # type: ignore


def test_apply_match_result_with_invalid_type_raises_error(basic_description, extractor):
    """Test apply_match_result() raises error for invalid type."""
    basic_description.extract_parameters(extractor)

    with pytest.raises(InvalidHVACDescriptionError, match="must be MatchResult"):
        basic_description.apply_match_result("not a match result")  # type: ignore


def test_apply_match_result_enforces_state_machine(basic_description, sample_uuid):
    """Test apply_match_result() enforces state must be PARAMETERS_EXTRACTED."""
    result = MatchResult.create(
        matched_item_id=sample_uuid,
        parameter_score=100.0,
        semantic_score=90.0,
        confidence=0.95,
        message="Test",
        breakdown={},
    )

    # Try to apply match without extracting parameters first
    with pytest.raises(
        InvalidHVACDescriptionError, match="Must be in PARAMETERS_EXTRACTED state"
    ):
        basic_description.apply_match_result(result)


# ============================================================================
# TESTS - merge_with_price() State Transition
# ============================================================================


def test_merge_with_price_success(basic_description):
    """Test merge_with_price() merges price and transitions state."""
    # Setup: transition to MATCHED state first
    basic_description.state = HVACDescriptionState.MATCHED

    score = MatchScore.create(100.0, 90.0, 75.0)
    price = Decimal("250.00")

    basic_description.merge_with_price(price, "Zawór DN50 PN16", score)

    assert basic_description.state == HVACDescriptionState.PRICED
    assert basic_description.matched_price == Decimal("250.00")
    assert basic_description.matched_description == "Zawór DN50 PN16"
    assert basic_description.match_score == score


def test_merge_with_price_negative_price_raises_error(basic_description):
    """Test merge_with_price() raises error for negative price."""
    basic_description.state = HVACDescriptionState.MATCHED
    score = MatchScore.create(100.0, 90.0, 75.0)

    with pytest.raises(InvalidHVACDescriptionError, match="cannot be negative"):
        basic_description.merge_with_price(Decimal("-10.00"), "Test", score)


def test_merge_with_price_invalid_match_score_raises_error(basic_description):
    """Test merge_with_price() raises error for invalid match_score."""
    basic_description.state = HVACDescriptionState.MATCHED

    with pytest.raises(InvalidHVACDescriptionError, match="must be MatchScore"):
        basic_description.merge_with_price(Decimal("250.00"), "Test", "not a score")  # type: ignore


# ============================================================================
# TESTS - get_match_report()
# ============================================================================


def test_get_match_report_returns_none_when_not_matched(basic_description):
    """Test get_match_report() returns None when not matched."""
    assert basic_description.get_match_report() is None


def test_get_match_report_returns_formatted_string(basic_description):
    """Test get_match_report() returns formatted report."""
    # Setup: set to MATCHED state
    basic_description.state = HVACDescriptionState.MATCHED
    basic_description.match_score = MatchScore.create(100.0, 92.0, 75.0)
    basic_description.matched_description = "Zawór kulowy DN50 PN16"
    basic_description.extracted_params = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    report = basic_description.get_match_report()

    assert "Matched: Zawór kulowy DN50 PN16" in report
    assert "Score:" in report
    assert "DN: 50" in report
    assert "PN: 16" in report


def test_get_match_report_includes_price_when_priced(basic_description):
    """Test get_match_report() includes price in PRICED state."""
    basic_description.state = HVACDescriptionState.PRICED
    basic_description.match_score = MatchScore.create(100.0, 92.0, 75.0)
    basic_description.matched_description = "Zawór DN50"
    basic_description.matched_price = Decimal("350.00")

    report = basic_description.get_match_report()

    assert "Price: 350.00 PLN" in report


# ============================================================================
# TESTS - Serialization (to_dict, from_dict)
# ============================================================================


def test_to_dict_structure(basic_description):
    """Test to_dict() returns correct structure."""
    data = basic_description.to_dict()

    assert isinstance(data, dict)
    assert "id" in data
    assert "raw_text" in data
    assert "state" in data
    assert "created_at" in data
    assert "updated_at" in data
    assert "extracted_params" in data
    assert "match_score" in data


def test_to_dict_serializes_values(basic_description, sample_uuid):
    """Test to_dict() serializes all values correctly."""
    basic_description.file_id = sample_uuid
    basic_description.source_row_number = 5

    data = basic_description.to_dict()

    assert data["raw_text"] == "Zawór kulowy DN50 PN16"
    assert data["state"] == "created"
    assert data["source_row_number"] == 5
    assert data["file_id"] == str(sample_uuid)
    assert isinstance(data["id"], str)  # UUID as string
    assert isinstance(data["created_at"], str)  # ISO format


def test_from_dict_reconstructs_entity():
    """Test from_dict() reconstructs entity from dictionary."""
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "raw_text": "Zawór DN50",
        "state": "created",
        "created_at": "2024-01-15T10:30:00",
        "updated_at": "2024-01-15T10:30:00",
        "source_row_number": 5,
        "file_id": None,
        "extracted_params": None,
        "match_score": None,
        "matched_price": None,
        "matched_description": None,
    }

    desc = HVACDescription.from_dict(data)

    assert desc.raw_text == "Zawór DN50"
    assert desc.source_row_number == 5
    assert desc.state == HVACDescriptionState.CREATED
    assert str(desc.id) == "550e8400-e29b-41d4-a716-446655440000"


def test_from_dict_reconstructs_nested_objects():
    """Test from_dict() reconstructs nested Value Objects."""
    data = {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "raw_text": "Zawór DN50",
        "state": "parameters_extracted",
        "created_at": "2024-01-15T10:30:00",
        "updated_at": "2024-01-15T10:30:00",
        "source_row_number": None,
        "file_id": None,
        "extracted_params": {
            "dn": 50,
            "pn": 16,
            "valve_type": None,
            "material": None,
            "drive_type": None,
            "voltage": None,
            "manufacturer": None,
            "confidence_scores": {"dn": 1.0, "pn": 1.0},
        },
        "match_score": {
            "parameter_score": 100.0,
            "semantic_score": 90.0,
            "final_score": 94.0,
            "threshold": 75.0,
        },
        "matched_price": "250.50",
        "matched_description": "Zawór kulowy DN50 PN16",
    }

    desc = HVACDescription.from_dict(data)

    assert desc.extracted_params is not None
    assert desc.extracted_params.dn == 50
    assert desc.extracted_params.pn == 16
    assert desc.match_score is not None
    assert desc.match_score.parameter_score == 100.0
    assert desc.matched_price == Decimal("250.50")


def test_serialization_roundtrip(description_with_metadata, extractor):
    """Test that to_dict() and from_dict() are inverse operations."""
    # Add some data
    description_with_metadata.extract_parameters(extractor)

    # Serialize
    data = description_with_metadata.to_dict()

    # Deserialize
    reconstructed = HVACDescription.from_dict(data)

    # Compare
    assert reconstructed.raw_text == description_with_metadata.raw_text
    assert reconstructed.state == description_with_metadata.state
    assert reconstructed.source_row_number == description_with_metadata.source_row_number
    assert reconstructed.file_id == description_with_metadata.file_id
    assert reconstructed.extracted_params.dn == description_with_metadata.extracted_params.dn


# ============================================================================
# TESTS - __repr__ and __str__
# ============================================================================


def test_repr_contains_id_and_state(basic_description):
    """Test __repr__() contains essential info."""
    repr_str = repr(basic_description)

    assert "HVACDescription" in repr_str
    assert str(basic_description.id) in repr_str
    assert "created" in repr_str


def test_str_contains_text_and_state(basic_description):
    """Test __str__() returns readable format."""
    str_repr = str(basic_description)

    assert "Zawór kulowy DN50 PN16" in str_repr
    assert "created" in str_repr


# ============================================================================
# INTEGRATION TESTS - Full Workflow
# ============================================================================


def test_full_workflow_created_to_priced(extractor, sample_uuid):
    """Integration test: CREATED -> PARAMETERS_EXTRACTED -> MATCHED -> PRICED."""
    # Step 1: Create description
    desc = HVACDescription.from_excel_row(
        raw_text="Zawór kulowy DN50 PN16 mosiężny napęd elektryczny 230V KSB",
        source_row_number=10,
        file_id=sample_uuid,
    )
    assert desc.state == HVACDescriptionState.CREATED

    # Step 2: Extract parameters
    desc.extract_parameters(extractor)
    assert desc.state == HVACDescriptionState.PARAMETERS_EXTRACTED
    assert desc.has_critical_parameters() is True

    # Step 3: Apply match result
    result = MatchResult.create(
        matched_item_id=uuid4(),
        parameter_score=100.0,
        semantic_score=92.0,
        confidence=0.95,
        message="High confidence match",
        breakdown={},
    )
    desc.apply_match_result(result)
    assert desc.state == HVACDescriptionState.MATCHED

    # Step 4: Merge with price
    desc.merge_with_price(
        price=Decimal("450.00"),
        matched_description="Zawór kulowy DN50 PN16 KSB",
        match_score=result.score,
    )
    assert desc.state == HVACDescriptionState.PRICED
    assert desc.matched_price == Decimal("450.00")

    # Step 5: Get report
    report = desc.get_match_report()
    assert report is not None
    assert "450.00" in report
    assert "DN: 50" in report
