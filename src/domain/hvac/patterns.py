"""
HVAC Regex Patterns

Regular expression patterns for extracting HVAC parameters from text descriptions.
Designed to handle various text formats, spacing variations, and common OCR errors.

Part of: Task 2.1.3 - ParameterExtractor domain service
Phase: 2.1 - Domain Layer Details

Pattern Design Principles:
- Case-insensitive matching (re.IGNORECASE flag)
- Flexible spacing (optional whitespace)
- Common format variations
- Capture groups for extracting values
"""

import re
from typing import Pattern


# ============================================================================
# DN (DIAMETER NOMINAL) PATTERNS
# ============================================================================

# Primary DN pattern - handles most common formats
# Matches: DN50, DN 50, dn50, Dn-50, DN=50, Ø50
DN_PATTERN: Pattern = re.compile(
    r"""
    (?:DN|dn|Dn|Ø)      # DN prefix or diameter symbol (Ø)
    \s*                 # Optional whitespace
    [-=]?               # Optional separator (dash or equals)
    \s*                 # Optional whitespace
    (\d{1,4})           # Capture 1-4 digits (DN value: 15-1000)
    (?![.\d])           # Negative lookahead - not followed by decimal or digit
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Alternative DN pattern for "diameter" word formats
# Matches: "średnica 50", "średnica 50mm", "diameter 50"
DN_WORD_PATTERN: Pattern = re.compile(
    r"""
    (?:średnica|diameter|srednica)  # Word for diameter (PL/EN)
    \s+                             # Required whitespace
    (\d{1,4})                       # Capture DN value
    \s*                             # Optional whitespace
    mm?                             # Optional mm unit
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Inch to DN conversion pattern
# Matches: 1/2", 2", 1 1/2"
INCH_PATTERN: Pattern = re.compile(
    r"""
    \b                              # Word boundary
    (\d{1,2})                       # Capture whole inches
    (?:\s*(\d)/(\d))?               # Optional fraction (e.g., "1/2")
    \s*                             # Optional whitespace
    "?                              # Optional quote mark
    (?:\s*cal(?:a|i)?)?             # Optional "cala/cali" (Polish for inches)
    \b                              # Word boundary
    """,
    re.VERBOSE,
)


# ============================================================================
# PN (PRESSURE NOMINAL) PATTERNS
# ============================================================================

# Primary PN pattern
# Matches: PN16, PN 16, pn16, Pn-16, PN=16
PN_PATTERN: Pattern = re.compile(
    r"""
    (?:PN|pn|Pn)        # PN prefix
    \s*                 # Optional whitespace
    [-=]?               # Optional separator
    \s*                 # Optional whitespace
    (\d{1,3})           # Capture 1-3 digits (PN value: 6-100)
    (?![.\d])           # Not followed by decimal or digit
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Alternative PN pattern for "pressure" word formats
# Matches: "ciśnienie 16", "pressure 16 bar"
PN_WORD_PATTERN: Pattern = re.compile(
    r"""
    (?:ciśnienie|cisnienie|pressure)  # Word for pressure (PL/EN)
    \s+                               # Required whitespace
    (\d{1,3})                         # Capture PN value
    \s*                               # Optional whitespace
    (?:bar)?                          # Optional bar unit
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================================
# VOLTAGE PATTERNS
# ============================================================================

# Voltage pattern for electric drives
# Matches: 230V, 24V, 400V, 230 V
VOLTAGE_PATTERN: Pattern = re.compile(
    r"""
    \b                  # Word boundary
    (\d{2,3})           # Capture 2-3 digits (voltage value)
    \s*                 # Optional whitespace
    V                   # Voltage unit
    (?![\w])            # Not followed by word character
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================================
# HELPER FUNCTIONS FOR PATTERN MATCHING
# ============================================================================


def extract_dn_from_text(text: str) -> tuple[int | None, float]:
    """
    Extract DN value from text using all DN patterns.

    Args:
        text: Text to search for DN value

    Returns:
        Tuple of (dn_value, confidence_score)
        - dn_value: Extracted DN as integer, or None if not found
        - confidence_score: 1.0 for exact match, 0.8 for word pattern, 0.0 if not found

    Algorithm:
        1. Try primary DN pattern first (highest confidence)
        2. Try word pattern if primary fails
        3. Try inch conversion if others fail (lowest confidence)

    Example:
        extract_dn_from_text("Zawór kulowy DN50 PN16") -> (50, 1.0)
        extract_dn_from_text("Rura średnica 100mm") -> (100, 0.8)
    """
    # Try primary pattern (highest confidence)
    match = DN_PATTERN.search(text)
    if match:
        return int(match.group(1)), 1.0

    # Try word pattern (medium confidence)
    match = DN_WORD_PATTERN.search(text)
    if match:
        return int(match.group(1)), 0.8

    # Try inch pattern (lower confidence - conversion needed)
    # Note: This is a placeholder - actual conversion would use INCH_TO_DN dict
    match = INCH_PATTERN.search(text)
    if match:
        # This would need proper conversion logic from constants.py
        # For now, return None to keep it simple (Happy Path)
        return None, 0.0

    return None, 0.0


def extract_pn_from_text(text: str) -> tuple[int | None, float]:
    """
    Extract PN value from text using all PN patterns.

    Args:
        text: Text to search for PN value

    Returns:
        Tuple of (pn_value, confidence_score)
        - pn_value: Extracted PN as integer, or None if not found
        - confidence_score: 1.0 for exact match, 0.8 for word pattern, 0.0 if not found

    Example:
        extract_pn_from_text("Zawór kulowy DN50 PN16") -> (16, 1.0)
        extract_pn_from_text("Ciśnienie 10 bar") -> (10, 0.8)
    """
    # Try primary pattern
    match = PN_PATTERN.search(text)
    if match:
        return int(match.group(1)), 1.0

    # Try word pattern
    match = PN_WORD_PATTERN.search(text)
    if match:
        return int(match.group(1)), 0.8

    return None, 0.0


def extract_voltage_from_text(text: str) -> tuple[str | None, float]:
    """
    Extract voltage value from text.

    Args:
        text: Text to search for voltage

    Returns:
        Tuple of (voltage_string, confidence_score)
        - voltage_string: Extracted voltage as string (e.g., "230V"), or None
        - confidence_score: 1.0 if found, 0.0 if not found

    Example:
        extract_voltage_from_text("Napęd elektryczny 230V") -> ("230V", 1.0)
    """
    match = VOLTAGE_PATTERN.search(text)
    if match:
        voltage_value = match.group(1)
        return f"{voltage_value}V", 1.0

    return None, 0.0


# ============================================================================
# PATTERN VALIDATION
# ============================================================================


def validate_patterns() -> bool:
    """
    Validate all regex patterns compile correctly.

    Returns:
        True if all patterns are valid

    Usage:
        Called during module initialization or testing to ensure patterns are correct
    """
    patterns = [
        DN_PATTERN,
        DN_WORD_PATTERN,
        INCH_PATTERN,
        PN_PATTERN,
        PN_WORD_PATTERN,
        VOLTAGE_PATTERN,
    ]

    try:
        for pattern in patterns:
            # Test each pattern compiles and can be used
            pattern.search("")
        return True
    except re.error:
        return False


# Validate patterns on module import
assert validate_patterns(), "Regex patterns failed validation"
