"""
DiameterNominal (DN) — immutable value object for nominal pipe diameter (mm, range 15-1000).
Supports parsing from DN notation, diameter symbol Ø, and inch sizes.
"""

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from src.domain.shared.exceptions import InvalidDNValueError

# Import only for type checking to avoid circular dependency
if TYPE_CHECKING:
    from src.domain.hvac.value_objects.pressure_nominal import PressureNominal


# ============================================================================
# MODULE-LEVEL CONSTANTS (outside dataclass to avoid mutable default issues)
# ============================================================================

# Standard DN values according to ISO/EN standards
STANDARD_DN_SIZES: Final[tuple[int, ...]] = (
    15,
    20,
    25,
    32,
    40,
    50,
    65,
    80,
    100,
    125,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    900,
    1000,
)

# Mapping of inch sizes to DN values
# Format: "fractional_inches" -> DN_value
INCH_TO_DN: Final[dict[str, int]] = {
    "1/2": 15,
    "3/4": 20,
    "1": 25,
    "1.25": 32,
    "1 1/4": 32,
    "1.5": 40,
    "1 1/2": 40,
    "2": 50,
}

# Regex patterns for parsing (compiled once for performance)
# Standard DN notation: DN50, dn50, DN 50, DN-50, DN=50
DN_STANDARD_PATTERN: Final[re.Pattern] = re.compile(r"DN[=\s-]?(\d+)", re.IGNORECASE)
# Diameter symbol: Ø50, ø100
DN_SYMBOL_PATTERN: Final[re.Pattern] = re.compile(r"[Øø](\d+)")
# Inch notation - fractional: Matches "1/2"", "3/4"", "1 1/2""
# Pattern: (optional whole number + space) + numerator/denominator
INCH_FRACTIONAL_PATTERN: Final[re.Pattern] = re.compile(r'((?:\d+\s+)?\d+/\d+)\s*"')
# Inch notation - whole number: "2""
INCH_WHOLE_PATTERN: Final[re.Pattern] = re.compile(r'(\d+)\s*"')
# Inch notation - decimal: "1.5"", "2.0"", "1.25""
INCH_DECIMAL_PATTERN: Final[re.Pattern] = re.compile(r'(\d+\.\d+)\s*"')

# Valid range for DN values
MIN_DN_VALUE: Final[int] = 15
MAX_DN_VALUE: Final[int] = 1000


@dataclass(frozen=True)
class DiameterNominal:
    """DN value in mm (15-1000). Immutable; equality by value."""

    value: int

    def __post_init__(self) -> None:
        """Validate DN is an int in [15, 1000]. Raises InvalidDNValueError otherwise."""
        if not isinstance(self.value, int):
            raise InvalidDNValueError(
                f"DN value must be integer, got {type(self.value).__name__}"
            )

        if not (MIN_DN_VALUE <= self.value <= MAX_DN_VALUE):
            raise InvalidDNValueError(
                f"DN value must be between {MIN_DN_VALUE} and {MAX_DN_VALUE}, got {self.value}"
            )

    @classmethod
    def from_string(cls, text: str) -> "DiameterNominal":
        """
        Parse DN from string. Supported formats: DN50, DN 50, DN-50, DN=50, Ø50, 2", 1/2", 1.5".
        Raises InvalidDNValueError if text cannot be parsed.
        """
        # Validate input type
        if not isinstance(text, str):
            raise InvalidDNValueError(
                "Cannot parse DN from empty or non-string input",
                original_value=str(text) if text is not None else None,
            )

        # Normalize: strip whitespace
        text = text.strip()

        # Check if empty after strip
        if not text:
            raise InvalidDNValueError(
                "Cannot parse DN from empty string", original_value=text
            )

        # Try standard DN format (DN50, dn50, DN 50, DN-50, DN=50)
        match = DN_STANDARD_PATTERN.search(text)
        if match:
            try:
                dn_value = int(match.group(1))
                return cls(dn_value)
            except (ValueError, InvalidDNValueError) as e:
                raise InvalidDNValueError(
                    f"Invalid DN value in standard format: {match.group(1)}",
                    original_value=text,
                ) from e

        # Try diameter symbol (Ø50, ø100)
        match = DN_SYMBOL_PATTERN.search(text)
        if match:
            try:
                dn_value = int(match.group(1))
                return cls(dn_value)
            except (ValueError, InvalidDNValueError) as e:
                raise InvalidDNValueError(
                    f"Invalid DN value in symbol format: {match.group(1)}",
                    original_value=text,
                ) from e

        # Try fractional inch notation FIRST (1/2", 3/4", 1 1/2")
        # IMPORTANT: Try fractional BEFORE other inch patterns
        match = INCH_FRACTIONAL_PATTERN.search(text)
        if match:
            inch_str = match.group(1).strip()
            # Look up in INCH_TO_DN dictionary
            if inch_str in INCH_TO_DN:
                return cls(INCH_TO_DN[inch_str])
            # If not found, raise error
            raise InvalidDNValueError(
                f"Inch value '{inch_str}' not found in conversion table. "
                f"Supported inch values: {', '.join(INCH_TO_DN.keys())}",
                original_value=text,
            )

        # Try decimal inch notation (1.5", 2.0", 1.25")
        match = INCH_DECIMAL_PATTERN.search(text)
        if match:
            inch_str = match.group(1).strip()
            # Look up in INCH_TO_DN dictionary (try as-is first)
            if inch_str in INCH_TO_DN:
                return cls(INCH_TO_DN[inch_str])
            # Try removing trailing .0 (e.g., "2.0" -> "2")
            inch_str_simplified = inch_str.rstrip("0").rstrip(".")
            if inch_str_simplified in INCH_TO_DN:
                return cls(INCH_TO_DN[inch_str_simplified])
            # If not found, raise error
            raise InvalidDNValueError(
                f"Inch value '{inch_str}' not found in conversion table. "
                f"Supported inch values: {', '.join(INCH_TO_DN.keys())}",
                original_value=text,
            )

        # Try whole number inch notation (2")
        match = INCH_WHOLE_PATTERN.search(text)
        if match:
            inch_str = match.group(1).strip()
            # Look up in INCH_TO_DN dictionary
            if inch_str in INCH_TO_DN:
                return cls(INCH_TO_DN[inch_str])
            # If not found, raise error
            raise InvalidDNValueError(
                f"Inch value '{inch_str}' not found in conversion table. "
                f"Supported inch values: {', '.join(INCH_TO_DN.keys())}",
                original_value=text,
            )

        # If no pattern matched, raise error
        raise InvalidDNValueError(
            f"Cannot parse DN from text: '{text}'. "
            f"Supported formats: DN50, Ø50, 2\", etc.",
            original_value=text,
        )

    def to_string(self) -> str:
        """Return standard HVAC notation, e.g. "DN50"."""
        return f"DN{self.value}"

    def is_standard_size(self) -> bool:
        """True if value is in STANDARD_DN_SIZES (ISO/EN)."""
        return self.value in STANDARD_DN_SIZES

    def is_compatible_with(self, pn: "PressureNominal") -> bool:
        """
        Check DN/PN compatibility per HVAC pressure limits:
            DN15-25 → max PN40 | DN32-50 → max PN63 | DN65-300 → max PN100
            DN350-600 → max PN63 | DN700-1000 → max PN40
        """
        # Import here to avoid circular dependency at runtime
        from src.domain.hvac.value_objects.pressure_nominal import PressureNominal

        # Validate input type
        if not isinstance(pn, PressureNominal):
            raise TypeError(
                f"Expected PressureNominal instance, got {type(pn).__name__}"
            )

        # Apply HVAC compatibility rules based on DN range
        # DN15-DN25: max PN40
        if 15 <= self.value <= 25:
            return pn.value <= 40

        # DN32-DN50: max PN63
        if 32 <= self.value <= 50:
            return pn.value <= 63

        # DN65-DN300: max PN100
        if 65 <= self.value <= 300:
            return pn.value <= 100

        # DN350-DN600: max PN63
        if 350 <= self.value <= 600:
            return pn.value <= 63

        # DN700-DN1000: max PN40
        if 700 <= self.value <= 1000:
            return pn.value <= 40

        # If DN is outside all ranges (shouldn't happen due to validation in __post_init__)
        # but handle defensively
        return False

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"DiameterNominal(value={self.value})"
