"""
HVAC Value Objects.

This module exports all Value Objects used in the HVAC domain.
Value Objects are immutable objects that represent domain concepts by their value,
not by their identity.

Available Value Objects:
    - DiameterNominal (DN): Nominal diameter in millimeters
    - PressureNominal (PN): Nominal pressure in bar
    - MatchScore: Hybrid matching score (parameter + semantic)
    - MatchResult: Complete matching result with metadata
"""

from src.domain.hvac.value_objects.diameter_nominal import DiameterNominal
from src.domain.hvac.value_objects.pressure_nominal import PressureNominal

# These will be imported when their contracts are ready in Phase 1
# For now, we import only the new Value Objects from Task 2.1.1
from src.domain.hvac.value_objects.match_score import MatchScore
from src.domain.hvac.value_objects.match_result import MatchResult

__all__ = [
    "DiameterNominal",
    "PressureNominal",
    "MatchScore",
    "MatchResult",
]
