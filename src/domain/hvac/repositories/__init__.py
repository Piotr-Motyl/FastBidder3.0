"""
HVAC Repository Interfaces Module

Repository pattern interfaces (contracts) for data persistence.
Defined in Domain Layer, implemented in Infrastructure Layer.

This module exports:
    - HVACDescriptionRepositoryProtocol: Repository interface for HVACDescription

Future exports (Phase 2+):
    - MatchResultRepositoryProtocol: Repository for storing match results
    - CacheRepositoryProtocol: Generic cache interface
"""

from .hvac_description_repository import HVACDescriptionRepositoryProtocol

__all__ = [
    "HVACDescriptionRepositoryProtocol",
]
