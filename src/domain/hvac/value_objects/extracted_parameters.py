"""
ExtractedParameters — result of parameter extraction from HVAC description text.
Raw values (int/str) with confidence scores per parameter. Immutable frozen dataclass.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Dict


@dataclass(frozen=True)
class ExtractedParameters:
    """
    Extracted HVAC parameters with per-parameter confidence scores (0.0-1.0).

    All fields are Optional (None = not found). Confidence: 1.0 = exact match, 0.5-0.9 = synonym match.
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
        """Validate all confidence scores are in [0.0, 1.0]."""
        for param_name, score in self.confidence_scores.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Confidence score for '{param_name}' must be between 0.0 and 1.0, "
                    f"got {score}"
                )

    def has_parameters(self) -> bool:
        """True if at least one technical parameter (excluding manufacturer) was found."""
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
        """True if DN or PN was extracted (the two most important matching parameters)."""
        return self.dn is not None or self.pn is not None

    def get_confidence(self, parameter_name: str) -> float:
        """Return confidence score for a parameter, or 0.0 if not present."""
        return self.confidence_scores.get(parameter_name, 0.0)

    def get_average_confidence(self) -> float:
        """Average confidence across all extracted parameters, or 0.0 if none."""
        if not self.confidence_scores:
            return 0.0

        return sum(self.confidence_scores.values()) / len(self.confidence_scores)

    def is_empty(self) -> bool:
        """True if all parameters (including manufacturer) are None. Complement of has_parameters() + manufacturer."""
        return all(
            [
                self.dn is None,
                self.pn is None,
                self.valve_type is None,
                self.material is None,
                self.drive_type is None,
                self.voltage is None,
                self.manufacturer is None,
            ]
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization/logging. Includes derived fields (has_parameters, etc.)."""
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractedParameters":
        """Deserialize from dict. Ignores computed fields from to_dict() (has_parameters, etc.)."""
        # Extract only fields that are part of the dataclass
        # Ignore computed fields from to_dict() output
        return cls(
            dn=data.get("dn"),
            pn=data.get("pn"),
            valve_type=data.get("valve_type"),
            material=data.get("material"),
            drive_type=data.get("drive_type"),
            voltage=data.get("voltage"),
            manufacturer=data.get("manufacturer"),
            confidence_scores=data.get("confidence_scores", {}),
        )
