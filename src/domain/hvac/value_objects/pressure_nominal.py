"""
PressureNominal (PN) — immutable value object for nominal pressure rating (bar, range 1-100).
Standard classes: PN6, PN10, PN16, PN25, PN40, PN63, PN100.
"""

import re
from dataclasses import dataclass
from typing import Final

from src.domain.shared.exceptions import InvalidPNValueError


# ============================================================================
# MODULE-LEVEL CONSTANTS (compiled once for performance)
# ============================================================================

# Standard PN notation: PN16, pn16, PN 16, PN-16, PN=16
PN_STANDARD_PATTERN: Final[re.Pattern] = re.compile(r"PN[=\s-]?(\d+)", re.IGNORECASE)

# PN with unit: "16 bar", "16bar", "16 Bar"
PN_BAR_PATTERN: Final[re.Pattern] = re.compile(r"(\d+)\s*bar", re.IGNORECASE)

# PN in context: "ciśnienie 16", "pressure 16", "ciśnienie: 16"
PN_CONTEXT_PATTERN: Final[re.Pattern] = re.compile(
    r"(?:ciśnienie|pressure)[:\s]+(\d+)", re.IGNORECASE
)

# Numeric only pattern (last resort): "16", " 25 "
PN_NUMERIC_PATTERN: Final[re.Pattern] = re.compile(r"^(\d+)$")


@dataclass(frozen=True)
class PressureNominal:
    """PN value in bar (1-100). Immutable; equality by value."""

    value: int

    # Standard PN classes according to ISO/EN standards
    STANDARD_CLASSES: Final[tuple[int, ...]] = (6, 10, 16, 25, 40, 63, 100)

    # Valid range for PN values
    MIN_VALUE: Final[int] = 1
    MAX_VALUE: Final[int] = 100

    def __post_init__(self) -> None:
        """Validate PN is an int in [1, 100]. Raises InvalidPNValueError otherwise."""
        if not isinstance(self.value, int):
            raise InvalidPNValueError(
                f"PN value must be integer, got {type(self.value).__name__}"
            )

        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise InvalidPNValueError(
                f"PN value must be between {self.MIN_VALUE} and {self.MAX_VALUE}, got {self.value}"
            )

    @classmethod
    def from_string(cls, text: str) -> "PressureNominal":
        """
        Parse PN from string. Supported formats: PN16, PN 16, PN-16, PN=16, 16 bar, ciśnienie 16, 16.
        Raises InvalidPNValueError if text cannot be parsed.
        """
        # Validate input type
        if not isinstance(text, str):
            raise InvalidPNValueError(
                "Cannot parse PN from empty or non-string input",
                original_value=str(text) if text is not None else None,
            )

        # Normalize: strip whitespace
        text = text.strip()

        # Check if empty after strip
        if not text:
            raise InvalidPNValueError(
                "Cannot parse PN from empty string", original_value=text
            )

        # Try standard PN format (PN16, pn16, PN 16, PN-16, PN=16)
        match = PN_STANDARD_PATTERN.search(text)
        if match:
            try:
                pn_value = int(match.group(1))
                return cls(pn_value)
            except (ValueError, InvalidPNValueError) as e:
                raise InvalidPNValueError(
                    f"Invalid PN value in standard format: {match.group(1)}",
                    original_value=text,
                ) from e

        # Try PN with unit (16 bar, 16bar, 16 Bar)
        match = PN_BAR_PATTERN.search(text)
        if match:
            try:
                pn_value = int(match.group(1))
                return cls(pn_value)
            except (ValueError, InvalidPNValueError) as e:
                raise InvalidPNValueError(
                    f"Invalid PN value in bar format: {match.group(1)}",
                    original_value=text,
                ) from e

        # Try context pattern (ciśnienie 16, pressure 16)
        match = PN_CONTEXT_PATTERN.search(text)
        if match:
            try:
                pn_value = int(match.group(1))
                return cls(pn_value)
            except (ValueError, InvalidPNValueError) as e:
                raise InvalidPNValueError(
                    f"Invalid PN value in context format: {match.group(1)}",
                    original_value=text,
                ) from e

        # Try numeric only (last resort, matches whole string only)
        match = PN_NUMERIC_PATTERN.match(text)
        if match:
            try:
                pn_value = int(match.group(1))
                return cls(pn_value)
            except (ValueError, InvalidPNValueError) as e:
                raise InvalidPNValueError(
                    f"Invalid PN value in numeric format: {match.group(1)}",
                    original_value=text,
                ) from e

        # If no pattern matched, raise error
        raise InvalidPNValueError(
            f"Cannot parse PN from text: '{text}'. "
            f"Supported formats: PN16, 16 bar, ciśnienie 16, 16, etc.",
            original_value=text,
        )

    def to_string(self) -> str:
        """Return standard HVAC notation, e.g. "PN16"."""
        return f"PN{self.value}"

    def is_standard_class(self) -> bool:
        """True if value is in STANDARD_CLASSES: 6, 10, 16, 25, 40, 63, 100."""
        return self.value in self.STANDARD_CLASSES

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"PressureNominal(value={self.value})"
