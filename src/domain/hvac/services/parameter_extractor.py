"""
ParameterExtractor Domain Service

CONTRACT ONLY - Phase 1/2
Extracts technical parameters (DN, PN, material) from text descriptions.
Core business logic for HVAC-specific parsing.
"""

from typing import Any, Dict, Optional, Protocol


class ParameterExtractor(Protocol):
    """
    Service for extracting HVAC parameters from raw text.

    This is domain service because it contains business rules
    about what DN, PN, materials mean in HVAC context.

    Future implementation (Phase 2):
    - Regex patterns for DN (15-1000mm)
    - Regex patterns for PN (1-100 bar)
    - Material detection (PP-R, PVC, steel, etc.)
    - Valve type detection
    """

    def extract_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract all recognizable HVAC parameters.

        Args:
            text: Raw description like "Zawór kulowy DN50 PN16"

        Returns:
            Dict with extracted params: {"dn": 50, "pn": 16, "valve_type": "kulowy"}
            Empty dict if no parameters found.

        CONTRACT ONLY - Implementation in Phase 2.
        """
        ...
