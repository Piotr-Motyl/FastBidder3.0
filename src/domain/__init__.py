"""
Domain Layer - Core Business Logic

Heart of the FastBidder application. Contains all business rules, entities,
value objects, and domain services. Framework-independent and highly testable.

Architecture:
    - Clean Architecture: Domain Layer is the center, no external dependencies
    - Domain-Driven Design: Entities, Value Objects, Services, Repositories
    - Dependency Inversion: Domain defines interfaces, Infrastructure implements

Subdomains:
    - hvac: HVAC product matching logic (Phase 1)
    - excel: Excel processing logic (future)
    - shared: Cross-subdomain concepts

This module exports the most commonly used domain objects for convenience.

Exports:
    From HVAC Subdomain:
        - HVACDescription: Core entity
        - MatchScore: Scoring value object
        - MatchResult: Match result value object
        - MatchingEngineProtocol: Matching service (Protocol interface)
        - HVACDescriptionRepositoryProtocol: Repository interface

    From Shared Domain:
        - DomainException: Base exception class

Usage:
    >>> # Preferred: Import from domain module
    >>> from src.domain import HVACDescription, MatchResult, DomainException
    >>>
    >>> # Alternative: Import from subdomain
    >>> from src.domain.hvac import HVACDescription, MatchingEngineProtocol
    >>>
    >>> # Alternative: Import from specific module
    >>> from src.domain.hvac.entities import HVACDescription

Phase 1 Scope:
    - HVAC subdomain: Entities, Value Objects, Services, Repository interfaces
    - Shared domain: Base exception class only

Future Phases:
    - Phase 2: ParameterExtractor service, DN/PN value objects
    - Phase 3: Excel subdomain entities and services
    - Phase 4: Domain events for audit trail
"""

# HVAC Subdomain
from .hvac import (
    HVACDescription,
    HVACDescriptionRepositoryProtocol,
    MatchingEngineProtocol,
    MatchResult,
    MatchScore,
)

# Shared Domain
from .shared import DomainException

__all__ = [
    # HVAC Subdomain
    "HVACDescription",
    "MatchScore",
    "MatchResult",
    "MatchingEngineProtocol",
    "HVACDescriptionRepositoryProtocol",
    # Shared Domain
    "DomainException",
]
