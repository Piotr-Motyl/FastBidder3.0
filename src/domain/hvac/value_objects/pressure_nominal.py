"""
PressureNominal (PN) Value Object.

PN represents the nominal pressure rating of pipes, valves, and fittings in HVAC systems.
It is expressed in bar and follows ISO/EN standards.

This is an immutable Value Object following DDD principles.
"""

from dataclasses import dataclass
from typing import Final

from src.domain.shared.exceptions import InvalidPNValueError


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
        # Implementation will be in Phase 3
        # Contract: Parse various PN formats and return PressureNominal instance
        raise NotImplementedError("Implementation in Phase 3")

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
