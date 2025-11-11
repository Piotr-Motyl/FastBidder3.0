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


class InvalidDNValueError(DomainException):
    """
    Raised when DN (Diameter Nominal) value is invalid.

    This exception is raised when:
    - DN value is outside valid range (15-1000mm)
    - DN value is not a standard HVAC size
    - DN string cannot be parsed
    - DN format is unrecognized

    Examples:
        >>> raise InvalidDNValueError("DN must be 15-1000, got 10")
        >>> raise InvalidDNValueError("Cannot parse DN from 'XYZ123'")
    """

    def __init__(self, message: str, original_value: str | None = None) -> None:
        """
        Initialize DN validation error.

        Args:
            message: Error description
            original_value: Original input string that caused the error (optional)
        """
        self.original_value = original_value
        super().__init__(message)


class InvalidPNValueError(DomainException):
    """
    Raised when PN (Pressure Nominal) value is invalid.

    This exception is raised when:
    - PN value is outside valid range (1-100 bar)
    - PN value is not a standard pressure class
    - PN string cannot be parsed
    - PN format is unrecognized

    Examples:
        >>> raise InvalidPNValueError("PN must be 1-100, got 150")
        >>> raise InvalidPNValueError("Cannot parse PN from 'ABC'")
    """

    def __init__(self, message: str, original_value: str | None = None) -> None:
        """
        Initialize PN validation error.

        Args:
            message: Error description
            original_value: Original input string that caused the error (optional)
        """
        self.original_value = original_value
        super().__init__(message)


class IncompatibleDNPNError(DomainException):
    """
    Raised when DN and PN combination is not compatible.

    According to HVAC standards, certain DN/PN combinations are not physically
    possible or safe. For example, DN15 cannot have PN100 pressure rating.

    Examples:
        >>> raise IncompatibleDNPNError("DN15 cannot support PN100")
        >>> raise IncompatibleDNPNError("DN800 with PN6 is below safety limits")
    """

    def __init__(
        self, message: str, dn_value: int | None = None, pn_value: int | None = None
    ) -> None:
        """
        Initialize DN/PN compatibility error.

        Args:
            message: Error description
            dn_value: DN value that caused incompatibility (optional)
            pn_value: PN value that caused incompatibility (optional)
        """
        self.dn_value = dn_value
        self.pn_value = pn_value
        super().__init__(message)


class InvalidHVACDescriptionError(DomainException):
    """
    Raised when HVACDescription entity validation fails.

    This exception is raised when:
    - raw_text is empty or too short (< 3 characters)
    - raw_text is not a string
    - Invalid state transition attempted
    - Invalid price value (negative)
    - Invalid match_score type

    Examples:
        >>> raise InvalidHVACDescriptionError("raw_text must have at least 3 characters")
        >>> raise InvalidHVACDescriptionError("Price cannot be negative, got -100")
    """

    def __init__(self, message: str, field_name: str | None = None) -> None:
        """
        Initialize HVAC description validation error.

        Args:
            message: Error description
            field_name: Name of field that caused error (optional)
        """
        self.field_name = field_name
        super().__init__(message)
