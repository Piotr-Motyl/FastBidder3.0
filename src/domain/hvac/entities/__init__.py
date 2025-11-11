"""
HVAC Domain Entities.

This module exports all entities used in the HVAC domain.
Entities have identity and lifecycle - they are mutable objects tracked by ID.

Available Entities:
    - HVACDescription: Core entity representing HVAC product description
    - HVACDescriptionState: Enum representing entity lifecycle states
"""

from src.domain.hvac.entities.hvac_description import (
    HVACDescription,
    HVACDescriptionState,
)

__all__ = [
    "HVACDescription",
    "HVACDescriptionState",
]
