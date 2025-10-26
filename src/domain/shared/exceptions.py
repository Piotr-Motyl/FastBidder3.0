"""
Domain Layer Exceptions

This module defines the base exception class for the Domain Layer.
All domain-specific exceptions should inherit from DomainException.

Responsibility:
    - Base exception class for domain errors
    - Type-safe error handling across layers
    - Clear separation from framework exceptions

Architecture Notes:
    - Part of Shared Domain (used across all subdomains)
    - Phase 1: Only base class (minimal scope)
    - Phase 2/3: Will add specific exceptions (InvalidParameterError, MatchingFailedError, etc.)
"""


class DomainException(Exception):
    """
    Base exception for all domain layer errors.

    This exception serves as the root of the domain exception hierarchy.
    All domain-specific exceptions should inherit from this class to enable
    type-safe error handling in Application and API layers.

    Usage:
        - Catch this in Application Layer to handle all domain errors
        - API Layer converts to appropriate HTTP status codes
        - Infrastructure Layer should not raise DomainException (use own exceptions)

    Examples:
        >>> raise DomainException("Business rule violation")

        >>> try:
        ...     # domain operation
        ... except DomainException as e:
        ...     # handle domain error
        ...     logger.error(f"Domain error: {e}")

    Phase 1 Contract:
        - Only base class implemented
        - Future phases will add:
            * InvalidParameterError (for DN/PN validation)
            * MatchingFailedError (when no match found)
            * ValidationError (for entity validation)
            * EntityNotFoundError (for repository operations)
    """

    def __init__(self, message: str) -> None:
        """
        Initialize domain exception with error message.

        Args:
            message: Human-readable error description
        """
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        """String representation of the exception."""
        return f"{self.__class__.__name__}: {self.message}"

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"{self.__class__.__name__}(message={self.message!r})"
