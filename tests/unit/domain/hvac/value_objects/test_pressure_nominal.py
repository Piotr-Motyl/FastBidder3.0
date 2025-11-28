"""
Tests for PressureNominal Value Object.
Covers: parsing, validation, standard classes, immutability.
"""

import pytest

from src.domain.hvac.value_objects.pressure_nominal import PressureNominal
from src.domain.shared.exceptions import InvalidPNValueError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_pn():
    """Fixture for a valid PN value."""
    return PressureNominal(16)


# ============================================================================
# HAPPY PATH TESTS - Creation
# ============================================================================


def test_create_valid_pn():
    """Test creating PressureNominal with valid value."""
    pn = PressureNominal(16)
    assert pn.value == 16


def test_create_boundary_min():
    """Test creating PN with minimum valid value (PN1)."""
    pn = PressureNominal(1)
    assert pn.value == 1


def test_create_boundary_max():
    """Test creating PN with maximum valid value (PN100)."""
    pn = PressureNominal(100)
    assert pn.value == 100


# ============================================================================
# HAPPY PATH TESTS - from_string() with various formats
# ============================================================================


@pytest.mark.parametrize(
    "input_str,expected_value",
    [
        # Standard notation
        ("PN16", 16),
        ("pn16", 16),
        ("Pn16", 16),
        ("PN16", 16),
        # With separators
        ("PN 16", 16),
        ("PN-16", 16),
        ("PN=16", 16),
        ("PN 25", 25),
        # With unit (bar)
        ("16 bar", 16),
        ("16bar", 16),
        ("25 Bar", 25),
        ("40 BAR", 40),
        # Context patterns
        ("ciśnienie 16", 16),
        ("pressure 16", 16),
        ("Pressure 25", 25),
        ("ciśnienie: 40", 40),
        # Numeric only
        ("16", 16),
        ("25", 25),
        ("100", 100),
    ],
)
def test_from_string_various_formats(input_str, expected_value):
    """Test parsing PN from various string formats."""
    pn = PressureNominal.from_string(input_str)
    assert pn.value == expected_value


def test_from_string_with_whitespace():
    """Test parsing PN with leading/trailing whitespace."""
    pn = PressureNominal.from_string("  PN16  ")
    assert pn.value == 16


def test_from_string_boundary_values():
    """Test parsing boundary PN values."""
    pn_min = PressureNominal.from_string("PN1")
    assert pn_min.value == 1

    pn_max = PressureNominal.from_string("PN100")
    assert pn_max.value == 100


# ============================================================================
# HAPPY PATH TESTS - to_string()
# ============================================================================


def test_to_string():
    """Test conversion to string format."""
    pn = PressureNominal(16)
    assert pn.to_string() == "PN16"
    assert str(pn) == "PN16"


def test_to_string_various_values():
    """Test to_string() with various PN values."""
    assert PressureNominal(6).to_string() == "PN6"
    assert PressureNominal(10).to_string() == "PN10"
    assert PressureNominal(100).to_string() == "PN100"


# ============================================================================
# HAPPY PATH TESTS - is_standard_class()
# ============================================================================


def test_is_standard_class_true():
    """Test that standard PN classes are recognized."""
    # Standard classes: 6, 10, 16, 25, 40, 63, 100
    assert PressureNominal(6).is_standard_class() is True
    assert PressureNominal(10).is_standard_class() is True
    assert PressureNominal(16).is_standard_class() is True
    assert PressureNominal(25).is_standard_class() is True
    assert PressureNominal(40).is_standard_class() is True
    assert PressureNominal(63).is_standard_class() is True
    assert PressureNominal(100).is_standard_class() is True


def test_is_standard_class_false():
    """Test that non-standard PN values are recognized."""
    assert PressureNominal(5).is_standard_class() is False
    assert PressureNominal(15).is_standard_class() is False
    assert PressureNominal(20).is_standard_class() is False
    assert PressureNominal(50).is_standard_class() is False


# ============================================================================
# EDGE CASES & ERROR HANDLING - Invalid creation
# ============================================================================


def test_create_invalid_pn_below_range():
    """Test that PN below 1 raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="must be between 1 and 100"):
        PressureNominal(0)


def test_create_invalid_pn_above_range():
    """Test that PN above 100 raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="must be between 1 and 100"):
        PressureNominal(101)


def test_create_invalid_pn_negative():
    """Test that negative PN raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="must be between 1 and 100"):
        PressureNominal(-10)


def test_create_invalid_pn_type():
    """Test that non-integer PN raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="must be integer"):
        PressureNominal(16.5)  # type: ignore


# ============================================================================
# EDGE CASES & ERROR HANDLING - Invalid from_string()
# ============================================================================


def test_from_string_empty():
    """Test that empty string raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="Cannot parse PN from empty string"):
        PressureNominal.from_string("")


def test_from_string_whitespace_only():
    """Test that whitespace-only string raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="Cannot parse PN from empty string"):
        PressureNominal.from_string("   ")


def test_from_string_invalid_format():
    """Test that invalid format raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="Cannot parse PN from text"):
        PressureNominal.from_string("XYZ")


def test_from_string_non_string_input():
    """Test that non-string input raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError, match="Cannot parse PN from empty or non-string input"):
        PressureNominal.from_string(None)  # type: ignore


def test_from_string_invalid_value_out_of_range():
    """Test that valid format but invalid value raises InvalidPNValueError."""
    with pytest.raises(InvalidPNValueError):
        PressureNominal.from_string("PN0")

    with pytest.raises(InvalidPNValueError):
        PressureNominal.from_string("PN150")


# ============================================================================
# IMMUTABILITY TESTS
# ============================================================================


def test_immutability(valid_pn):
    """Test that PressureNominal is immutable (frozen dataclass)."""
    with pytest.raises(AttributeError):
        valid_pn.value = 25  # type: ignore


# ============================================================================
# EQUALITY TESTS
# ============================================================================


def test_equality():
    """Test that two PressureNominal instances with same value are equal."""
    pn1 = PressureNominal(16)
    pn2 = PressureNominal(16)
    assert pn1 == pn2


def test_inequality():
    """Test that two PressureNominal instances with different values are not equal."""
    pn1 = PressureNominal(16)
    pn2 = PressureNominal(25)
    assert pn1 != pn2


def test_roundtrip_from_string_to_string():
    """Test that parsing and converting back yields consistent results."""
    original = "PN16"
    pn = PressureNominal.from_string(original)
    assert pn.to_string() == original


# ============================================================================
# REPR TESTS
# ============================================================================


def test_repr():
    """Test developer-friendly representation."""
    pn = PressureNominal(16)
    assert repr(pn) == "PressureNominal(value=16)"
