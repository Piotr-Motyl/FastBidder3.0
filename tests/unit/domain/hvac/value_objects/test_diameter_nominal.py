"""
Tests for DiameterNominal Value Object.

Covers:
- Creation and validation
- Parsing from various string formats
- DN/PN compatibility checks
- Immutability
- Edge cases and error handling
"""

import pytest

from src.domain.hvac.value_objects.diameter_nominal import DiameterNominal
from src.domain.hvac.value_objects.pressure_nominal import PressureNominal
from src.domain.shared.exceptions import InvalidDNValueError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_dn_50():
    """Fixture for a valid DN50 instance."""
    return DiameterNominal(50)


@pytest.fixture
def valid_dn_15():
    """Fixture for a valid DN15 instance (boundary - minimum)."""
    return DiameterNominal(15)


@pytest.fixture
def valid_dn_1000():
    """Fixture for a valid DN1000 instance (boundary - maximum)."""
    return DiameterNominal(1000)


# ============================================================================
# HAPPY PATH TESTS
# ============================================================================


def test_create_valid_dn():
    """Test creating DiameterNominal with valid value."""
    dn = DiameterNominal(50)
    assert dn.value == 50


def test_create_valid_dn_boundary_min():
    """Test creating DiameterNominal with minimum valid value (15)."""
    dn = DiameterNominal(15)
    assert dn.value == 15


def test_create_valid_dn_boundary_max():
    """Test creating DiameterNominal with maximum valid value (1000)."""
    dn = DiameterNominal(1000)
    assert dn.value == 1000


@pytest.mark.parametrize(
    "input_str,expected_value",
    [
        # Standard DN notation
        ("DN50", 50),
        ("dn50", 50),
        ("Dn50", 50),
        ("DN 50", 50),
        ("DN-50", 50),
        ("DN=50", 50),
        # Diameter symbol
        ("Ø50", 50),
        ("ø100", 100),
        # Fractional inch notation
        ('2"', 50),
        ('1/2"', 15),
        ('3/4"', 20),
        ('1"', 25),
        ('1 1/4"', 32),
        ('1 1/2"', 40),
        # Decimal inch notation
        ('1.25"', 32),
        ('1.5"', 40),
        ('2.0"', 50),
    ],
)
def test_from_string_various_formats(input_str, expected_value):
    """Test parsing DN from various string formats."""
    dn = DiameterNominal.from_string(input_str)
    assert dn.value == expected_value


def test_to_string(valid_dn_50):
    """Test conversion to standard DN string format."""
    assert valid_dn_50.to_string() == "DN50"


def test_str_representation(valid_dn_50):
    """Test __str__ method returns standard DN format."""
    assert str(valid_dn_50) == "DN50"


def test_repr_representation(valid_dn_50):
    """Test __repr__ method returns developer-friendly format."""
    assert repr(valid_dn_50) == "DiameterNominal(value=50)"


def test_is_standard_size_true():
    """Test that standard DN sizes are recognized."""
    assert DiameterNominal(50).is_standard_size() is True
    assert DiameterNominal(100).is_standard_size() is True
    assert DiameterNominal(15).is_standard_size() is True
    assert DiameterNominal(1000).is_standard_size() is True


def test_is_standard_size_false():
    """Test that non-standard DN sizes are recognized."""
    assert DiameterNominal(55).is_standard_size() is False
    assert DiameterNominal(99).is_standard_size() is False
    assert DiameterNominal(30).is_standard_size() is False


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================


def test_create_invalid_dn_below_range():
    """Test that DN below 15 raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="must be between 15 and 1000"):
        DiameterNominal(10)


def test_create_invalid_dn_above_range():
    """Test that DN above 1000 raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="must be between 15 and 1000"):
        DiameterNominal(1500)


def test_create_invalid_dn_boundary_below():
    """Test that DN=14 (just below minimum) raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="must be between 15 and 1000"):
        DiameterNominal(14)


def test_create_invalid_dn_boundary_above():
    """Test that DN=1001 (just above maximum) raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="must be between 15 and 1000"):
        DiameterNominal(1001)


def test_create_invalid_dn_non_integer():
    """Test that non-integer DN value raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="must be integer"):
        DiameterNominal(50.5)  # type: ignore


def test_from_string_empty():
    """Test that empty string raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="empty string"):
        DiameterNominal.from_string("")


def test_from_string_whitespace_only():
    """Test that whitespace-only string raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="empty string"):
        DiameterNominal.from_string("   ")


def test_from_string_invalid_format():
    """Test that invalid format raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="Cannot parse DN"):
        DiameterNominal.from_string("XYZ123")


def test_from_string_none():
    """Test that None input raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="empty or non-string input"):
        DiameterNominal.from_string(None)  # type: ignore


def test_from_string_invalid_inch():
    """Test that unsupported inch value raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="not found in conversion table"):
        DiameterNominal.from_string('5"')  # 5" is not in INCH_TO_DN


def test_from_string_dn_value_out_of_range():
    """Test that valid format but invalid DN value raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="Invalid DN value"):
        DiameterNominal.from_string("DN10")  # DN10 < 15 (minimum)


def test_from_string_symbol_invalid_value():
    """Test that diameter symbol with invalid value raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="Invalid DN value"):
        DiameterNominal.from_string("Ø10")  # Ø10 < 15 (minimum)


def test_from_string_decimal_inch_not_in_table():
    """Test that decimal inch not in conversion table raises InvalidDNValueError."""
    with pytest.raises(InvalidDNValueError, match="not found in conversion table"):
        DiameterNominal.from_string('3.5"')  # 3.5" not in INCH_TO_DN


# ============================================================================
# COMPATIBILITY TESTS
# ============================================================================


@pytest.mark.parametrize(
    "dn_value,pn_value,expected",
    [
        # DN15-DN25: max PN40
        (15, 16, True),
        (15, 40, True),
        (15, 100, False),  # exceeds max PN40
        (25, 25, True),
        (25, 63, False),  # exceeds max PN40
        # DN32-DN50: max PN63
        (32, 16, True),
        (50, 63, True),
        (50, 100, False),  # exceeds max PN63
        # DN65-DN300: max PN100
        (65, 16, True),
        (100, 100, True),
        (300, 100, True),
        # DN350-DN600: max PN63
        (350, 40, True),
        (350, 63, True),
        (500, 100, False),  # exceeds max PN63
        # DN700-DN1000: max PN40
        (700, 16, True),
        (800, 40, True),
        (1000, 63, False),  # exceeds max PN40
    ],
)
def test_is_compatible_with(dn_value, pn_value, expected):
    """Test DN/PN compatibility according to HVAC standards."""
    dn = DiameterNominal(dn_value)
    pn = PressureNominal(pn_value)
    assert dn.is_compatible_with(pn) is expected


def test_is_compatible_with_invalid_type():
    """Test that invalid type for PN raises TypeError."""
    dn = DiameterNominal(50)
    with pytest.raises(TypeError, match="Expected PressureNominal"):
        dn.is_compatible_with("PN16")  # type: ignore


# ============================================================================
# IMMUTABILITY TESTS
# ============================================================================


def test_immutability(valid_dn_50):
    """Test that DiameterNominal is immutable (frozen dataclass)."""
    with pytest.raises(AttributeError):
        valid_dn_50.value = 100  # type: ignore


# ============================================================================
# EQUALITY TESTS
# ============================================================================


def test_equality_same_value():
    """Test that two DN instances with same value are equal."""
    dn1 = DiameterNominal(50)
    dn2 = DiameterNominal(50)
    assert dn1 == dn2


def test_equality_different_value():
    """Test that two DN instances with different values are not equal."""
    dn1 = DiameterNominal(50)
    dn2 = DiameterNominal(100)
    assert dn1 != dn2


def test_equality_from_string():
    """Test that DN created directly and from string are equal."""
    dn1 = DiameterNominal(50)
    dn2 = DiameterNominal.from_string("DN50")
    assert dn1 == dn2
