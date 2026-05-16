"""
MatchingEngineProtocol — Protocol interface for hybrid HVAC description matching.
Hybrid scoring: 40% parameter matching + 60% semantic similarity.
Infrastructure layer provides the implementation.
"""

from typing import Optional, Protocol

from ..entities.hvac_description import HVACDescription
from ..value_objects.match_result import MatchResult


class MatchingEngineProtocol(Protocol):
    """
    Protocol for hybrid HVAC description matching.

    Scoring: final_score = 0.4 × parameter_score + 0.6 × semantic_score
    Returns MatchResult if final_score >= threshold, otherwise None.
    """

    async def match(
        self,
        working_description: HVACDescription,
        reference_descriptions: list[HVACDescription],
        threshold: float = 75.0,
    ) -> Optional[MatchResult]:
        """
        Find best matching reference for a working description.

        Args:
            working_description: Description from working file to be matched.
            reference_descriptions: Reference catalog items to match against.
            threshold: Minimum final_score to accept a match (0-100, default 75.0).

        Returns:
            MatchResult with score, confidence, and breakdown; or None if no match above threshold.
        """
        ...

    async def calculate_confidence(
        self, best_score: float, second_best_score: Optional[float]
    ) -> float:
        """
        Calculate confidence (0-1) from score gap between best and second-best match.

        Large gap → high confidence. None second_best_score → confidence from score alone.
        """
        ...
