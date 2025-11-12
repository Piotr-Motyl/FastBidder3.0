"""
HVAC Subdomain Module

Core business logic for HVAC product matching.
Contains entities, value objects, services, and repository interfaces.

This module is the main entry point for the HVAC subdomain and re-exports
all public interfaces for use by Application Layer.

Exports:
    Entities:
        - HVACDescription: Core entity representing HVAC product description

    Value Objects:
        - MatchScore: Hybrid scoring (parameter + semantic)
        - MatchResult: Result of matching operation

    Services:
        - MatchingEngineProtocol: Core matching service (Protocol interface)

    Repository Interfaces:
        - HVACDescriptionRepositoryProtocol: Data persistence contract

Usage:
    >>> # Import from subdomain module
    >>> from src.domain.hvac import HVACDescription, MatchResult, MatchingEngineProtocol
    >>>
    >>> # Or import from specific submodules
    >>> from src.domain.hvac.entities import HVACDescription
    >>> from src.domain.hvac.value_objects import MatchScore, MatchResult
    >>> from src.domain.hvac.services import MatchingEngineProtocol
"""

# Entities
from .entities import HVACDescription

# Value Objects
from .value_objects import MatchResult, MatchScore

# Services
from .services import MatchingEngineProtocol

# Repository Interfaces
from .repositories import HVACDescriptionRepositoryProtocol

from . import constants
from . import patterns

from . import matching_config

__all__ = [
    # Entities
    "HVACDescription",
    # Value Objects
    "MatchScore",
    "MatchResult",
    # Services
    "MatchingEngineProtocol",
    # Repository Interfaces
    "HVACDescriptionRepositoryProtocol",
    "constants",
    "patterns",
    "matching_config",
]
