"""
ConcreteMatchingEngine — legacy stub. All methods raise NotImplementedError.
Production matching is handled by HybridMatchingEngine in matching/hybrid_matching_engine.py.
"""

from typing import List, Optional
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.value_objects.match_result import MatchResult


class ConcreteMatchingEngine:
    """Unimplemented stub for MatchingEngineProtocol. Use HybridMatchingEngine instead."""

    def __init__(self):
        pass

    def match(
        self,
        working_item: HVACDescription,
        reference_catalog: List[HVACDescription],
        threshold: float = 75.0,
    ) -> Optional[MatchResult]:
        """Not implemented. Use HybridMatchingEngine.match() instead."""
        raise NotImplementedError("Use HybridMatchingEngine.")

    async def match_batch(
        self,
        working_items: List[HVACDescription],
        reference_catalog: List[HVACDescription],
        threshold: float = 75.0,
    ) -> List[Optional[MatchResult]]:
        """Not implemented."""
        raise NotImplementedError("Use HybridMatchingEngine.")

    def _extract_parameters(self, text: str) -> dict:
        raise NotImplementedError()

    def _calculate_parameter_score(self, params1: dict, params2: dict) -> float:
        raise NotImplementedError()

    def _calculate_semantic_score(self, text1: str, text2: str) -> float:
        raise NotImplementedError()
