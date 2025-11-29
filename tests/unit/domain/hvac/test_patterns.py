"""
Tests for HVAC Patterns module.
Covers: regex patterns, text normalization, canonical form matching, parameter extraction.
"""

import pytest
import re

from src.domain.hvac.patterns import (
    # Regex patterns
    DN_PATTERN,
    DN_WORD_PATTERN,
    INCH_PATTERN,
    PN_PATTERN,
    PN_WORD_PATTERN,
    VOLTAGE_PATTERN,
    # Helper functions
    normalize_text,
    find_canonical_form,
    extract_dn_from_text,
    extract_pn_from_text,
    extract_voltage_from_text,
    validate_patterns,
)


# ============================================================================
# TESTS - normalize_text()
# ============================================================================


def test_normalize_text_lowercase():
    """Test that normalize_text converts to lowercase."""
    assert normalize_text("ZAWÓR KULOWY DN50") == "zawór kulowy dn50"
    assert normalize_text("Valve DN100") == "valve dn100"


def test_normalize_text_strip_whitespace():
    """Test that normalize_text strips leading/trailing whitespace."""
    assert normalize_text("  zawór kulowy  ") == "zawór kulowy"
    assert normalize_text("\tzawór\t") == "zawór"


def test_normalize_text_remove_double_spaces():
    """Test that normalize_text removes multiple spaces."""
    assert normalize_text("zawór  kulowy   DN50") == "zawór kulowy dn50"
    assert normalize_text("valve    with     spaces") == "valve with spaces"


def test_normalize_text_empty_string():
    """Test that normalize_text handles empty string."""
    assert normalize_text("") == ""
    assert normalize_text("   ") == ""


def test_normalize_text_none():
    """Test that normalize_text handles None input."""
    assert normalize_text(None) == ""


# ============================================================================
# TESTS - find_canonical_form()
# ============================================================================


@pytest.fixture
def valve_dictionary():
    """Sample valve type dictionary for testing."""
    return {
        "kulowy": ["kurek kulowy", "zawór kulowy", "zawór odcinający kulowy"],
        "zwrotny": ["klapka zwrotna", "zawór zwrotny"],
        "grzybkowy": ["zawór grzybkowy"],
    }


def test_find_canonical_form_exact_match(valve_dictionary):
    """Test find_canonical_form with exact canonical match."""
    canonical, confidence = find_canonical_form("kulowy", valve_dictionary)
    assert canonical == "kulowy"
    assert confidence == 1.0


def test_find_canonical_form_exact_match_case_insensitive(valve_dictionary):
    """Test find_canonical_form is case-insensitive."""
    canonical, confidence = find_canonical_form("KULOWY", valve_dictionary)
    assert canonical == "kulowy"
    assert confidence == 1.0


def test_find_canonical_form_synonym_match(valve_dictionary):
    """Test find_canonical_form with synonym match."""
    canonical, confidence = find_canonical_form("zawór kulowy", valve_dictionary)
    assert canonical == "kulowy"
    assert confidence == 0.9


def test_find_canonical_form_synonym_with_spaces(valve_dictionary):
    """Test find_canonical_form handles extra spaces in synonym."""
    canonical, confidence = find_canonical_form(
        "  zawór   kulowy  ", valve_dictionary
    )
    assert canonical == "kulowy"
    assert confidence == 0.9


def test_find_canonical_form_not_found(valve_dictionary):
    """Test find_canonical_form when term not in dictionary."""
    canonical, confidence = find_canonical_form("nieznany typ", valve_dictionary)
    assert canonical is None
    assert confidence == 0.0


def test_find_canonical_form_empty_text(valve_dictionary):
    """Test find_canonical_form with empty text."""
    canonical, confidence = find_canonical_form("", valve_dictionary)
    assert canonical is None
    assert confidence == 0.0


def test_find_canonical_form_empty_dictionary():
    """Test find_canonical_form with empty dictionary."""
    canonical, confidence = find_canonical_form("test", {})
    assert canonical is None
    assert confidence == 0.0


# ============================================================================
# TESTS - DN Regex Patterns
# ============================================================================


@pytest.mark.parametrize(
    "text,expected_dn",
    [
        # Standard DN notation
        ("DN50", "50"),
        ("DN 50", "50"),
        ("dn50", "50"),
        ("Dn50", "50"),
        ("DN-50", "50"),
        ("DN=50", "50"),
        # With diameter symbol
        ("Ø50", "50"),
        ("ø100", "100"),
        # Mixed format
        ("Zawór DN50 PN16", "50"),
        ("DN100", "100"),
    ],
)
def test_dn_pattern_matches(text, expected_dn):
    """Test DN_PATTERN matches various formats."""
    match = DN_PATTERN.search(text)
    assert match is not None
    assert match.group(1) == expected_dn


def test_dn_pattern_no_match():
    """Test DN_PATTERN doesn't match invalid formats."""
    assert DN_PATTERN.search("valve without DN") is None
    assert DN_PATTERN.search("D50") is None  # Wrong prefix


@pytest.mark.parametrize(
    "text,expected_dn",
    [
        ("średnica 50", "50"),
        ("średnica 50mm", "50"),
        ("diameter 100", "100"),
        ("srednica 80 mm", "80"),
    ],
)
def test_dn_word_pattern_matches(text, expected_dn):
    """Test DN_WORD_PATTERN matches word formats."""
    match = DN_WORD_PATTERN.search(text)
    assert match is not None
    assert match.group(1) == expected_dn


# ============================================================================
# TESTS - PN Regex Patterns
# ============================================================================


@pytest.mark.parametrize(
    "text,expected_pn",
    [
        # Standard PN notation
        ("PN16", "16"),
        ("PN 16", "16"),
        ("pn16", "16"),
        ("Pn16", "16"),
        ("PN-16", "16"),
        ("PN=16", "16"),
        # Mixed format
        ("Zawór DN50 PN16", "16"),
        ("PN10", "10"),
        ("PN25", "25"),
    ],
)
def test_pn_pattern_matches(text, expected_pn):
    """Test PN_PATTERN matches various formats."""
    match = PN_PATTERN.search(text)
    assert match is not None
    assert match.group(1) == expected_pn


def test_pn_pattern_no_match():
    """Test PN_PATTERN doesn't match invalid formats."""
    assert PN_PATTERN.search("valve without PN") is None
    assert PN_PATTERN.search("P16") is None  # Wrong prefix


@pytest.mark.parametrize(
    "text,expected_pn",
    [
        ("ciśnienie 16", "16"),
        ("ciśnienie 16 bar", "16"),
        ("pressure 10", "10"),
        ("cisnienie 25 bar", "25"),
    ],
)
def test_pn_word_pattern_matches(text, expected_pn):
    """Test PN_WORD_PATTERN matches word formats."""
    match = PN_WORD_PATTERN.search(text)
    assert match is not None
    assert match.group(1) == expected_pn


# ============================================================================
# TESTS - VOLTAGE Regex Pattern
# ============================================================================


@pytest.mark.parametrize(
    "text,expected_voltage",
    [
        ("230V", "230"),
        ("24V", "24"),
        ("400V", "400"),
        ("230 V", "230"),
        ("napęd 230V", "230"),
    ],
)
def test_voltage_pattern_matches(text, expected_voltage):
    """Test VOLTAGE_PATTERN matches various formats."""
    match = VOLTAGE_PATTERN.search(text)
    assert match is not None
    assert match.group(1) == expected_voltage


def test_voltage_pattern_no_match():
    """Test VOLTAGE_PATTERN doesn't match invalid formats."""
    assert VOLTAGE_PATTERN.search("valve without voltage") is None
    assert VOLTAGE_PATTERN.search("5V") is None  # Too short (only 1 digit)


# ============================================================================
# TESTS - extract_dn_from_text()
# ============================================================================


def test_extract_dn_from_text_primary_pattern():
    """Test extract_dn_from_text with primary DN pattern."""
    dn, confidence = extract_dn_from_text("Zawór kulowy DN50 PN16")
    assert dn == 50
    assert confidence == 1.0


def test_extract_dn_from_text_word_pattern():
    """Test extract_dn_from_text with word pattern."""
    dn, confidence = extract_dn_from_text("Rura średnica 100mm")
    assert dn == 100
    assert confidence == 0.8


def test_extract_dn_from_text_not_found():
    """Test extract_dn_from_text when DN not found."""
    dn, confidence = extract_dn_from_text("Valve without DN specification")
    assert dn is None
    assert confidence == 0.0


def test_extract_dn_from_text_diameter_symbol():
    """Test extract_dn_from_text with Ø symbol."""
    dn, confidence = extract_dn_from_text("Zawór Ø80")
    assert dn == 80
    assert confidence == 1.0


# ============================================================================
# TESTS - extract_pn_from_text()
# ============================================================================


def test_extract_pn_from_text_primary_pattern():
    """Test extract_pn_from_text with primary PN pattern."""
    pn, confidence = extract_pn_from_text("Zawór kulowy DN50 PN16")
    assert pn == 16
    assert confidence == 1.0


def test_extract_pn_from_text_word_pattern():
    """Test extract_pn_from_text with word pattern."""
    pn, confidence = extract_pn_from_text("Ciśnienie 10 bar")
    assert pn == 10
    assert confidence == 0.8


def test_extract_pn_from_text_not_found():
    """Test extract_pn_from_text when PN not found."""
    pn, confidence = extract_pn_from_text("Valve without PN specification")
    assert pn is None
    assert confidence == 0.0


# ============================================================================
# TESTS - extract_voltage_from_text()
# ============================================================================


def test_extract_voltage_from_text_found():
    """Test extract_voltage_from_text when voltage is found."""
    voltage, confidence = extract_voltage_from_text("Napęd elektryczny 230V")
    assert voltage == "230V"
    assert confidence == 1.0


def test_extract_voltage_from_text_not_found():
    """Test extract_voltage_from_text when voltage not found."""
    voltage, confidence = extract_voltage_from_text("Manual valve")
    assert voltage is None
    assert confidence == 0.0


def test_extract_voltage_from_text_multiple_formats():
    """Test extract_voltage_from_text with various formats."""
    assert extract_voltage_from_text("24V")[0] == "24V"
    assert extract_voltage_from_text("400 V")[0] == "400V"


# ============================================================================
# TESTS - validate_patterns()
# ============================================================================


def test_validate_patterns_returns_true():
    """Test that all regex patterns are valid."""
    assert validate_patterns() is True


def test_all_patterns_are_compiled():
    """Test that all pattern constants are compiled regex objects."""
    patterns = [
        DN_PATTERN,
        DN_WORD_PATTERN,
        INCH_PATTERN,
        PN_PATTERN,
        PN_WORD_PATTERN,
        VOLTAGE_PATTERN,
    ]

    for pattern in patterns:
        assert isinstance(pattern, re.Pattern)
        # Test that pattern can be used
        pattern.search("test text")


# ============================================================================
# INTEGRATION TESTS - Real-world scenarios
# ============================================================================


def test_extraction_from_real_description():
    """Test parameter extraction from realistic HVAC description."""
    text = "Zawór kulowy DN50 PN16 mosiężny z napędem elektrycznym 230V"

    dn, dn_conf = extract_dn_from_text(text)
    assert dn == 50
    assert dn_conf == 1.0

    pn, pn_conf = extract_pn_from_text(text)
    assert pn == 16
    assert pn_conf == 1.0

    voltage, volt_conf = extract_voltage_from_text(text)
    assert voltage == "230V"
    assert volt_conf == 1.0


def test_extraction_polish_description():
    """Test extraction from Polish text with word patterns."""
    text = "Rura średnica 100mm ciśnienie 10 bar"

    dn, dn_conf = extract_dn_from_text(text)
    assert dn == 100
    assert dn_conf == 0.8  # Word pattern has lower confidence

    pn, pn_conf = extract_pn_from_text(text)
    assert pn == 10
    assert pn_conf == 0.8
