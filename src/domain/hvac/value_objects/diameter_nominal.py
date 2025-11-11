"""
DiameterNominal (DN) Value Object.

DN represents the nominal diameter of pipes, valves, and fittings in HVAC systems.
It is expressed in millimeters and follows ISO/EN standards.

This is an immutable Value Object following DDD principles.
"""

from dataclasses import dataclass
from typing import Final

from domain.hvac.value_objects.pressure_nominal import PressureNominal
from src.domain.shared.exceptions import InvalidDNValueError


@dataclass(frozen=True)
class DiameterNominal:
    """
    Immutable Value Object representing DN (Diameter Nominal) in HVAC systems.

    DN is the nominal diameter of pipes, valves, and fittings expressed in millimeters.
    Valid range: 15-1000mm (standard HVAC sizes).

    This class is immutable (frozen) - once created, value cannot be changed.
    Equality is based on value, not object identity.

    Attributes:
        value: Diameter nominal value in millimeters (15-1000)

    Examples:
        >>> dn = DiameterNominal(50)
        >>> dn.value
        50
        >>> dn.to_string()
        'DN50'
        >>> dn2 = DiameterNominal.from_string("DN50")
        >>> dn == dn2
        True
    """

    value: int

    # Standard DN values according to ISO/EN standards
    STANDARD_VALUES: Final[tuple[int, ...]] = (
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

    # Valid range for DN values
    MIN_VALUE: Final[int] = 15
    MAX_VALUE: Final[int] = 1000

    def __post_init__(self) -> None:
        """
        Validate DN value after initialization.

        Called automatically by dataclass after __init__.
        Since the object is frozen, this is the only place where validation can happen.

        Raises:
            InvalidDNValueError: If value is outside valid range (15-1000)
        """
        if not isinstance(self.value, int):
            raise InvalidDNValueError(
                f"DN value must be integer, got {type(self.value).__name__}"
            )

        if not (self.MIN_VALUE <= self.value <= self.MAX_VALUE):
            raise InvalidDNValueError(
                f"DN value must be between {self.MIN_VALUE} and {self.MAX_VALUE}, got {self.value}"
            )

    @classmethod
    def from_string(cls, text: str) -> "DiameterNominal":
        """
        Parse DN value from string with various formats.

        Supported formats:
        - "DN50", "dn50", "Dn50" - standard notation
        - "DN 50", "DN-50", "DN=50" - with separators
        - "Ø50", "ø50" - diameter symbol
        - "1/2\"", "2\"" - inch notation

        Args:
            text: String containing DN value in any supported format

        Returns:
            DiameterNominal instance with parsed value

        Raises:
            InvalidDNValueError: If text cannot be parsed or value is invalid

        Examples:
            >>> DiameterNominal.from_string("DN50")
            DiameterNominal(value=50)
            >>> DiameterNominal.from_string("2\"")
            DiameterNominal(value=50)
            >>> DiameterNominal.from_string("ø100")
            DiameterNominal(value=100)
        """
        # Implementation will be in Phase 3
        # Contract: Parse various DN formats and return DiameterNominal instance
        raise NotImplementedError("Implementation in Phase 3")

    def to_string(self) -> str:
        """
        Convert DN to standard string format.

        Returns DN in standard HVAC notation: "DN{value}" (uppercase, no spaces).

        Returns:
            String representation in format "DN50", "DN100", etc.

        Examples:
            >>> DiameterNominal(50).to_string()
            'DN50'
            >>> DiameterNominal(150).to_string()
            'DN150'
        """
        return f"DN{self.value}"

    def is_standard_size(self) -> bool:
        """
        Check if DN value is a standard HVAC size.

        Standard sizes are defined in STANDARD_VALUES constant according to
        ISO/EN specifications.

        Returns:
            True if value is in standard sizes list, False otherwise

        Examples:
            >>> DiameterNominal(50).is_standard_size()
            True
            >>> DiameterNominal(55).is_standard_size()
            False
        """
        return self.value in self.STANDARD_VALUES

    def is_compatible_with(self, pn: "PressureNominal") -> bool:
        """
        Check if this DN is compatible with given PN (Pressure Nominal).

        According to HVAC standards, certain DN/PN combinations are not physically
        possible or safe. For example:
        - Small diameters (DN15-DN25) cannot handle very high pressures (PN100)
        - Very large diameters (DN800-DN1000) typically use lower pressure classes

        Business Rules:
        - DN15-DN25: max PN40
        - DN32-DN50: max PN63
        - DN65-DN300: max PN100
        - DN350-DN600: max PN63
        - DN700-DN1000: max PN40

        Args:
            pn: PressureNominal instance to check compatibility with

        Returns:
            True if DN/PN combination is valid, False otherwise

        Raises:
            IncompatibleDNPNError: If combination violates safety standards

        Examples:
            >>> dn = DiameterNominal(50)
            >>> pn = PressureNominal(16)
            >>> dn.is_compatible_with(pn)
            True
            >>> dn15 = DiameterNominal(15)
            >>> pn100 = PressureNominal(100)
            >>> dn15.is_compatible_with(pn100)
            False
        """
        # Import here to avoid circular dependency
        from src.domain.hvac.value_objects.pressure_nominal import PressureNominal

        # Implementation will be in Phase 3
        # Contract: Check DN/PN compatibility based on HVAC standards
        raise NotImplementedError("Implementation in Phase 3")

    def __str__(self) -> str:
        """
        String representation for display purposes.

        Returns:
            Standard DN notation string
        """
        return self.to_string()

    def __repr__(self) -> str:
        """
        Developer-friendly representation for debugging.

        Returns:
            String showing class name and value
        """
        return f"DiameterNominal(value={self.value})"
