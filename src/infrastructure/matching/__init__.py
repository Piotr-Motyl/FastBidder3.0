"""
Matching Engine Infrastructure Module

Concrete implementation of Domain MatchingEngine service.

Exports:
    - ConcreteMatchingEngine: Hybrid matching implementation (40% param + 60% semantic)
                              Explicitly implements MatchingEngineProtocol
"""

from .matching_engine import ConcreteMatchingEngine

__all__ = [
    "ConcreteMatchingEngine",
]
