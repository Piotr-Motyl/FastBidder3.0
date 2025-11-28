"""
Tests for ExtractedParameters Value Object.
Covers: parameter extraction, confidence scoring, serialization, validation.
"""

import pytest

from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def empty_params():
    """Fixture for empty ExtractedParameters (no parameters extracted)."""
    return ExtractedParameters()


@pytest.fixture
def partial_params():
    """Fixture for ExtractedParameters with some parameters."""
    return ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="kulowy",
        confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 0.9},
    )


@pytest.fixture
def full_params():
    """Fixture for ExtractedParameters with all parameters."""
    return ExtractedParameters(
        dn=100,
        pn=25,
        valve_type="zwrotny",
        material="mosiądz",
        drive_type="elektryczny",
        voltage="230V",
        manufacturer="KSB",
        confidence_scores={
            "dn": 1.0,
            "pn": 1.0,
            "valve_type": 0.95,
            "material": 0.9,
            "drive_type": 0.85,
            "voltage": 1.0,
            "manufacturer": 0.8,
        },
    )


# ============================================================================
# HAPPY PATH TESTS - Object Creation
# ============================================================================


def test_create_empty_params():
    """Test creating ExtractedParameters with no parameters."""
    params = ExtractedParameters()
    assert params.dn is None
    assert params.pn is None
    assert params.valve_type is None
    assert params.material is None
    assert params.drive_type is None
    assert params.voltage is None
    assert params.manufacturer is None
    assert params.confidence_scores == {}


def test_create_partial_params():
    """Test creating ExtractedParameters with some parameters."""
    params = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type is None
    assert params.confidence_scores == {"dn": 1.0, "pn": 1.0}


def test_create_full_params(full_params):
    """Test creating ExtractedParameters with all parameters."""
    assert full_params.dn == 100
    assert full_params.pn == 25
    assert full_params.valve_type == "zwrotny"
    assert full_params.material == "mosiądz"
    assert full_params.drive_type == "elektryczny"
    assert full_params.voltage == "230V"
    assert full_params.manufacturer == "KSB"
    assert len(full_params.confidence_scores) == 7


# ============================================================================
# HAPPY PATH TESTS - has_parameters()
# ============================================================================


def test_has_parameters_empty(empty_params):
    """Test has_parameters() returns False for empty parameters."""
    assert empty_params.has_parameters() is False


def test_has_parameters_with_dn_only():
    """Test has_parameters() returns True when only DN is set."""
    params = ExtractedParameters(dn=50)
    assert params.has_parameters() is True


def test_has_parameters_with_multiple(partial_params):
    """Test has_parameters() returns True with multiple parameters."""
    assert partial_params.has_parameters() is True


def test_has_parameters_manufacturer_only():
    """Test has_parameters() returns False when only manufacturer is set."""
    # Business rule: manufacturer alone is not sufficient
    params = ExtractedParameters(manufacturer="KSB")
    assert params.has_parameters() is False


# ============================================================================
# HAPPY PATH TESTS - has_critical_parameters()
# ============================================================================


def test_has_critical_parameters_empty(empty_params):
    """Test has_critical_parameters() returns False for empty parameters."""
    assert empty_params.has_critical_parameters() is False


def test_has_critical_parameters_dn_only():
    """Test has_critical_parameters() returns True when DN is set."""
    params = ExtractedParameters(dn=50)
    assert params.has_critical_parameters() is True


def test_has_critical_parameters_pn_only():
    """Test has_critical_parameters() returns True when PN is set."""
    params = ExtractedParameters(pn=16)
    assert params.has_critical_parameters() is True


def test_has_critical_parameters_both():
    """Test has_critical_parameters() returns True when both DN and PN are set."""
    params = ExtractedParameters(dn=50, pn=16)
    assert params.has_critical_parameters() is True


def test_has_critical_parameters_no_dn_pn():
    """Test has_critical_parameters() returns False without DN or PN."""
    params = ExtractedParameters(valve_type="kulowy", material="mosiądz")
    assert params.has_critical_parameters() is False


# ============================================================================
# HAPPY PATH TESTS - get_confidence()
# ============================================================================


def test_get_confidence_existing_parameter():
    """Test get_confidence() for parameter with score."""
    params = ExtractedParameters(
        dn=50, confidence_scores={"dn": 1.0, "valve_type": 0.9}
    )
    assert params.get_confidence("dn") == 1.0
    assert params.get_confidence("valve_type") == 0.9


def test_get_confidence_missing_parameter():
    """Test get_confidence() returns 0.0 for missing parameter."""
    params = ExtractedParameters(dn=50, confidence_scores={"dn": 1.0})
    assert params.get_confidence("pn") == 0.0
    assert params.get_confidence("material") == 0.0


def test_get_confidence_empty_scores(empty_params):
    """Test get_confidence() returns 0.0 when no scores exist."""
    assert empty_params.get_confidence("dn") == 0.0


# ============================================================================
# HAPPY PATH TESTS - get_average_confidence()
# ============================================================================


def test_get_average_confidence_empty(empty_params):
    """Test get_average_confidence() returns 0.0 for empty scores."""
    assert empty_params.get_average_confidence() == 0.0


def test_get_average_confidence_single_score():
    """Test get_average_confidence() with single score."""
    params = ExtractedParameters(dn=50, confidence_scores={"dn": 1.0})
    assert params.get_average_confidence() == 1.0


def test_get_average_confidence_multiple_scores():
    """Test get_average_confidence() with multiple scores."""
    params = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 0.8}
    )
    # Average: (1.0 + 0.8) / 2 = 0.9
    assert params.get_average_confidence() == 0.9


def test_get_average_confidence_full(full_params):
    """Test get_average_confidence() with all parameters."""
    # Average: (1.0 + 1.0 + 0.95 + 0.9 + 0.85 + 1.0 + 0.8) / 7 = 0.9285714285714286
    expected = sum(full_params.confidence_scores.values()) / len(
        full_params.confidence_scores
    )
    assert full_params.get_average_confidence() == pytest.approx(expected, rel=1e-5)


# ============================================================================
# HAPPY PATH TESTS - is_empty()
# ============================================================================


def test_is_empty_true(empty_params):
    """Test is_empty() returns True for empty parameters."""
    assert empty_params.is_empty() is True


def test_is_empty_false_with_dn():
    """Test is_empty() returns False when DN is set."""
    params = ExtractedParameters(dn=50)
    assert params.is_empty() is False


def test_is_empty_false_with_any_parameter(partial_params):
    """Test is_empty() returns False when any parameter is set."""
    assert partial_params.is_empty() is False


def test_is_empty_false_manufacturer_only():
    """Test is_empty() returns False even with only manufacturer."""
    params = ExtractedParameters(manufacturer="KSB")
    assert params.is_empty() is False


# ============================================================================
# HAPPY PATH TESTS - to_dict()
# ============================================================================


def test_to_dict_empty(empty_params):
    """Test to_dict() for empty parameters."""
    result = empty_params.to_dict()
    assert result["dn"] is None
    assert result["pn"] is None
    assert result["confidence_scores"] == {}
    assert result["has_parameters"] is False
    assert result["has_critical_parameters"] is False
    assert result["average_confidence"] == 0.0


def test_to_dict_partial(partial_params):
    """Test to_dict() for partial parameters."""
    result = partial_params.to_dict()
    assert result["dn"] == 50
    assert result["pn"] == 16
    assert result["valve_type"] == "kulowy"
    assert result["material"] is None
    assert result["confidence_scores"] == {"dn": 1.0, "pn": 1.0, "valve_type": 0.9}
    assert result["has_parameters"] is True
    assert result["has_critical_parameters"] is True
    assert result["average_confidence"] == pytest.approx(0.9666666, rel=1e-5)


def test_to_dict_includes_computed_fields():
    """Test to_dict() includes computed fields."""
    params = ExtractedParameters(dn=50, confidence_scores={"dn": 1.0})
    result = params.to_dict()
    # Verify computed fields are present
    assert "has_parameters" in result
    assert "has_critical_parameters" in result
    assert "average_confidence" in result


# ============================================================================
# HAPPY PATH TESTS - from_dict()
# ============================================================================


def test_from_dict_empty():
    """Test from_dict() with empty dictionary."""
    data = {}
    params = ExtractedParameters.from_dict(data)
    assert params.dn is None
    assert params.pn is None
    assert params.confidence_scores == {}


def test_from_dict_partial():
    """Test from_dict() with partial data."""
    data = {"dn": 50, "pn": 16, "confidence_scores": {"dn": 1.0, "pn": 1.0}}
    params = ExtractedParameters.from_dict(data)
    assert params.dn == 50
    assert params.pn == 16
    assert params.valve_type is None
    assert params.confidence_scores == {"dn": 1.0, "pn": 1.0}


def test_from_dict_full():
    """Test from_dict() with all fields."""
    data = {
        "dn": 100,
        "pn": 25,
        "valve_type": "zwrotny",
        "material": "mosiądz",
        "drive_type": "elektryczny",
        "voltage": "230V",
        "manufacturer": "KSB",
        "confidence_scores": {"dn": 1.0, "pn": 1.0},
    }
    params = ExtractedParameters.from_dict(data)
    assert params.dn == 100
    assert params.pn == 25
    assert params.valve_type == "zwrotny"
    assert params.material == "mosiądz"
    assert params.drive_type == "elektryczny"
    assert params.voltage == "230V"
    assert params.manufacturer == "KSB"


def test_from_dict_ignores_computed_fields():
    """Test from_dict() ignores computed fields from to_dict() output."""
    # to_dict() includes computed fields, from_dict() should ignore them
    data = {
        "dn": 50,
        "confidence_scores": {"dn": 1.0},
        "has_parameters": True,  # Computed field - should be ignored
        "has_critical_parameters": True,  # Computed field - should be ignored
        "average_confidence": 1.0,  # Computed field - should be ignored
    }
    params = ExtractedParameters.from_dict(data)
    # Computed fields should be recalculated, not taken from input
    assert params.has_parameters() is True
    assert params.has_critical_parameters() is True


# ============================================================================
# SERIALIZATION ROUNDTRIP TESTS
# ============================================================================


def test_serialization_roundtrip_empty(empty_params):
    """Test serialization roundtrip for empty parameters."""
    dict_repr = empty_params.to_dict()
    restored = ExtractedParameters.from_dict(dict_repr)
    assert restored == empty_params


def test_serialization_roundtrip_partial(partial_params):
    """Test serialization roundtrip for partial parameters."""
    dict_repr = partial_params.to_dict()
    restored = ExtractedParameters.from_dict(dict_repr)
    assert restored == partial_params


def test_serialization_roundtrip_full(full_params):
    """Test serialization roundtrip for full parameters."""
    dict_repr = full_params.to_dict()
    restored = ExtractedParameters.from_dict(dict_repr)
    assert restored == full_params


# ============================================================================
# EDGE CASES & ERROR HANDLING - Validation
# ============================================================================


def test_validation_confidence_score_above_range():
    """Test that confidence score above 1.0 raises ValueError."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        ExtractedParameters(dn=50, confidence_scores={"dn": 1.5})


def test_validation_confidence_score_below_range():
    """Test that confidence score below 0.0 raises ValueError."""
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        ExtractedParameters(dn=50, confidence_scores={"dn": -0.1})


def test_validation_confidence_score_exactly_zero():
    """Test that confidence score of 0.0 is valid."""
    params = ExtractedParameters(dn=50, confidence_scores={"dn": 0.0})
    assert params.get_confidence("dn") == 0.0


def test_validation_confidence_score_exactly_one():
    """Test that confidence score of 1.0 is valid."""
    params = ExtractedParameters(dn=50, confidence_scores={"dn": 1.0})
    assert params.get_confidence("dn") == 1.0


# ============================================================================
# IMMUTABILITY TESTS
# ============================================================================


def test_immutability_cannot_modify_dn(partial_params):
    """Test that ExtractedParameters is immutable (frozen dataclass)."""
    with pytest.raises(AttributeError):
        partial_params.dn = 100  # type: ignore


def test_immutability_cannot_modify_confidence_scores_reference(partial_params):
    """Test that confidence_scores dict reference cannot be changed."""
    with pytest.raises(AttributeError):
        partial_params.confidence_scores = {"new": 0.5}  # type: ignore


# ============================================================================
# EQUALITY TESTS
# ============================================================================


def test_equality_same_values():
    """Test that two ExtractedParameters with same values are equal."""
    params1 = ExtractedParameters(dn=50, pn=16, confidence_scores={"dn": 1.0})
    params2 = ExtractedParameters(dn=50, pn=16, confidence_scores={"dn": 1.0})
    assert params1 == params2


def test_inequality_different_values():
    """Test that ExtractedParameters with different values are not equal."""
    params1 = ExtractedParameters(dn=50)
    params2 = ExtractedParameters(dn=100)
    assert params1 != params2


# ============================================================================
# EDGE CASES - Specific Scenarios
# ============================================================================


def test_only_voltage_without_drive_type():
    """Test that voltage without drive_type is valid (though unusual)."""
    params = ExtractedParameters(voltage="230V", confidence_scores={"voltage": 0.8})
    assert params.voltage == "230V"
    assert params.drive_type is None
    assert params.has_parameters() is True


def test_manufacturer_with_no_technical_params():
    """Test manufacturer without technical params."""
    params = ExtractedParameters(manufacturer="KSB")
    # has_parameters() should be False (manufacturer alone not sufficient)
    assert params.has_parameters() is False
    # But is_empty() should be False (manufacturer is set)
    assert params.is_empty() is False
