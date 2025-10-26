"""
HVAC Value Objects Module

Immutable value objects representing HVAC domain concepts.
Value objects are defined by their attributes (no identity).

This module exports:
    - MatchScore: Hybrid scoring (parameter + semantic)
    - MatchResult: Result of matching operation

Future exports (Phase 2+):
    - DN: Diameter Nominal value object (15-1000mm)
    - PN: Pressure Nominal value object (1-100 bar)
    - Material: HVAC material types
    - ValveType: Valve type classifications
"""

from .match_result import MatchResult
from .match_score import MatchScore

__all__ = [
    "MatchScore",
    "MatchResult",
]
