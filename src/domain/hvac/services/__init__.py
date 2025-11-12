"""
HVAC Domain Services Module

Business operations that don't naturally fit into entities.
Domain services operate on multiple entities or coordinate complex logic.

This module exports:
    - MatchingEngineProtocol: Core matching service (Protocol interface)

Future exports (Phase 2+):
    - ParameterExtractor: Extract DN, PN, materials from text
    - ScoringEngine: Calculate hybrid match scores
    - ValidationService: Validate HVAC business rules
"""

from .matching_engine import MatchingEngineProtocol
from .parameter_extractor import ParameterExtractorProtocol
from .simple_matching_engine import SimpleMatchingEngine

__all__ = [
    "MatchingEngineProtocol",
    "ParameterExtractorProtocol",
    "SimpleMatchingEngine",
]
