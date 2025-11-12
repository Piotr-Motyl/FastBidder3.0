"""
ExtractedParameters Value Object

Represents the result of parameter extraction from HVAC description text.
This value object holds raw extracted values (not Value Objects) along with
confidence scores for each parameter.

Part of: Task 2.1.3 - ParameterExtractor domain service
Phase: 2.1 - Domain Layer Details
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass(frozen=True)
class ExtractedParameters:
    """
    Value Object representing extracted HVAC parameters from text description.

    Contains raw values (int/str) rather than Value Objects for separation of concerns.
    Each parameter has an associated confidence score (0.0-1.0) indicating extraction quality.

    Confidence Score Interpretation:
    - 1.0: Exact match with regex or dictionary
    - 0.5-0.9: Partial match or synonym match
    - 0.0: Parameter not found

    Business Rules:
    - All fields are Optional (None if not found)
    - Confidence scores stored separately for clarity
    - Raw values allow flexible validation in consuming code
    - Immutable after creation (frozen dataclass)

    Usage Example:
        params = ExtractedParameters(
            dn=50,
            pn=16,
            valve_type="kulowy",
            confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 0.9}
        )

        if params.has_parameters():
            print(f"Found DN{params.dn} PN{params.pn}")
    """

    # Core HVAC Parameters (Happy Path - DN/PN most important)
    dn: Optional[int] = None  # Diameter Nominal (15-1000mm)
    pn: Optional[int] = None  # Pressure Nominal (6-100 bar)

    # Equipment Type
    valve_type: Optional[str] = None  # e.g., "kulowy", "zwrotny", "grzybkowy"

    # Material Properties
    material: Optional[str] = None  # e.g., "mosiądz", "stal", "PP-R"

    # Drive/Actuation
    drive_type: Optional[str] = None  # e.g., "ręczny", "elektryczny", "pneumatyczny"
    voltage: Optional[str] = None  # e.g., "230V", "24V" (when drive is electric)

    # Manufacturer (optional)
    manufacturer: Optional[str] = None  # e.g., "KSB", "Danfoss", "Belimo"

    # Confidence Scores (0.0 - 1.0 for each parameter)
    confidence_scores: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate confidence scores after initialization.

        Business Rule: All confidence scores must be between 0.0 and 1.0
        """
        for param_name, score in self.confidence_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Confidence score for '{param_name}' must be between 0.0 and 1.0, "
                    f"got {score}"
                )

    def has_parameters(self) -> bool:
        """
        Check if any parameters were successfully extracted.

        Returns:
            True if at least one parameter (excluding manufacturer) was found

        Business Logic:
            Manufacturer alone is not sufficient - we need at least one technical parameter
        """
        return any(
            [
                self.dn is not None,
                self.pn is not None,
                self.valve_type is not None,
                self.material is not None,
                self.drive_type is not None,
                self.voltage is not None,
            ]
        )

    def has_critical_parameters(self) -> bool:
        """
        Check if critical parameters (DN or PN) were found.

        Returns:
            True if either DN or PN was extracted

        Business Logic:
            DN and PN are the most important parameters for HVAC matching.
            At least one of them is usually required for meaningful matching.
        """
        return self.dn is not None or self.pn is not None

    def get_confidence(self, parameter_name: str) -> float:
        """
        Get confidence score for a specific parameter.

        Args:
            parameter_name: Name of the parameter (e.g., "dn", "pn", "valve_type")

        Returns:
            Confidence score (0.0-1.0), or 0.0 if parameter not in scores

        Note:
            Returns 0.0 for missing parameters rather than raising KeyError
            for easier usage in scoring algorithms
        """
        return self.confidence_scores.get(parameter_name, 0.0)

    def get_average_confidence(self) -> float:
        """
        Calculate average confidence across all extracted parameters.

        Returns:
            Average confidence score, or 0.0 if no parameters extracted

        Business Logic:
            Only considers parameters that were actually found (have scores).
            Useful for overall extraction quality assessment.
        """
        if not self.confidence_scores:
            return 0.0

        return sum(self.confidence_scores.values()) / len(self.confidence_scores)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization/logging.

        Returns:
            Dictionary with all parameters and confidence scores

        Usage:
            Useful for JSON serialization, logging, or debugging
        """
        return {
            "dn": self.dn,
            "pn": self.pn,
            "valve_type": self.valve_type,
            "material": self.material,
            "drive_type": self.drive_type,
            "voltage": self.voltage,
            "manufacturer": self.manufacturer,
            "confidence_scores": self.confidence_scores,
            "has_parameters": self.has_parameters(),
            "has_critical_parameters": self.has_critical_parameters(),
            "average_confidence": self.get_average_confidence(),
        }
