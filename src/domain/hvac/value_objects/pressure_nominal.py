"""
PressureNominal (PN) Value Object.

PN represents the nominal pressure rating of pipes, valves, and fittings in HVAC systems.
It is expressed in bar and follows ISO/EN standards.

This is an immutable Value Object following DDD principles.
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
    """
    Immutable Value Object representing PN (Pressure Nominal) in HVAC systems.

    PN is the nominal pressure rating of pipes, valves, and fittings expressed in bar.
    Valid range: 1-100 bar (standard HVAC pressure classes).

    This class is immutable (frozen) - once created, value cannot be changed.
    Equality is based on value, not object identity.

    Attributes:
        value: Pressure nominal value in bar (1-100)

    Examples:
        >>> pn = PressureNominal(16)
        >>> pn.value
        16
        >>> pn.to_string()
        'PN16'
        >>> pn2 = PressureNominal.from_string("PN16")
        >>> pn == pn2
        True
    """

    value: int

    # Standard PN classes according to ISO/EN standards
    STANDARD_CLASSES: Final[tuple[int, ...]] = (6, 10, 16, 25, 40, 63, 100)

    # Valid range for PN values
    MIN_VALUE: Final[int] = 1
    MAX_VALUE: Final[int] = 100

    def __post_init__(self) -> None:
        """
        Validate PN value after initialization.

        Called automatically by dataclass after __init__.
        Since the object is frozen, this is the only place where validation can happen.

        Raises:
            InvalidPNValueError: If value is outside valid range (1-100)
        """
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
        Parse PN value from string with various formats.

        Supported formats:
        - "PN16", "pn16", "Pn16" - standard notation
        - "PN 16", "PN-16", "PN=16" - with separators
        - "16 bar", "16bar" - with unit
        - "16" - numeric only

        Args:
            text: String containing PN value in any supported format

        Returns:
            PressureNominal instance with parsed value

        Raises:
            InvalidPNValueError: If text cannot be parsed or value is invalid

        Examples:
            >>> PressureNominal.from_string("PN16")
            PressureNominal(value=16)
            >>> PressureNominal.from_string("25 bar")
            PressureNominal(value=25)
            >>> PressureNominal.from_string("pn 40")
            PressureNominal(value=40)
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
        """
        Convert PN to standard string format.

        Returns PN in standard HVAC notation: "PN{value}" (uppercase, no spaces).

        Returns:
            String representation in format "PN16", "PN25", etc.

        Examples:
            >>> PressureNominal(16).to_string()
            'PN16'
            >>> PressureNominal(100).to_string()
            'PN100'
        """
        return f"PN{self.value}"

    def is_standard_class(self) -> bool:
        """
        Check if PN value is a standard pressure class.

        Standard pressure classes are defined in STANDARD_CLASSES constant
        according to ISO/EN specifications: PN6, PN10, PN16, PN25, PN40, PN63, PN100.

        Returns:
            True if value is in standard pressure classes, False otherwise

        Examples:
            >>> PressureNominal(16).is_standard_class()
            True
            >>> PressureNominal(15).is_standard_class()
            False
        """
        return self.value in self.STANDARD_CLASSES

    def __str__(self) -> str:
        """
        String representation for display purposes.

        Returns:
            Standard PN notation string
        """
        return self.to_string()

    def __repr__(self) -> str:
        """
        Developer-friendly representation for debugging.

        Returns:
            String showing class name and value
        """
        return f"PressureNominal(value={self.value})"
