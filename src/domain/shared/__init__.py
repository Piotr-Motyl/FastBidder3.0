"""
Shared Domain Module

Shared domain concepts used across all subdomains (HVAC, Excel, etc.).
Contains base classes, common value objects, and domain exceptions.

This module exports:
    - DomainException: Base exception for all domain errors

Future exports (Phase 2+):
    - BaseEntity: Base class for all entities
    - BaseValueObject: Base class for all value objects
    - DomainEvent: Base class for domain events
"""

from .exceptions import DomainException

__all__ = [
    "DomainException",
]
