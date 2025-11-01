"""
Repository Implementations Module

Concrete implementations of Domain repository interfaces.

Exports:
    - HVACDescriptionRepository: Redis-based implementation
"""

from .hvac_description_repository import HVACDescriptionRepository

__all__ = [
    "HVACDescriptionRepository",
]
