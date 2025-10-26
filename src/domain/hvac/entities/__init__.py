"""
HVAC Entities Module

Domain entities with identity and lifecycle.
Entities are defined by their unique ID, not their attributes.

This module exports:
    - HVACDescription: Core entity representing HVAC product description

Future exports (Phase 2+):
    - MatchSession: Entity tracking entire matching session
    - PriceQuote: Entity representing final pricing result
"""

from .hvac_description import HVACDescription

__all__ = [
    "HVACDescription",
]
