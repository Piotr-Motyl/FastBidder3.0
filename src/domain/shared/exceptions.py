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


class InvalidProcessMatchingCommandError(DomainException):
    """
    Raised when ProcessMatchingCommand validation fails.

    This exception is raised when command validation detects business rule violations:
    - Same file_id used for working and reference files
    - Invalid column format (not A-ZZ range)
    - Invalid row ranges (start >= end, or exceeds max limit)
    - Negative or zero threshold value
    - Missing required configuration fields

    This exception can contain multiple validation errors to provide
    comprehensive feedback to the user.

    Attributes:
        errors: List of validation error messages (for multiple validation failures)

    Examples:
        >>> raise InvalidProcessMatchingCommandError("working_file.file_id cannot equal reference_file.file_id")

        >>> # Multiple errors
        >>> errors = [
        ...     "Invalid column 'AAA': must be in range A-ZZ",
        ...     "Invalid range: start_row (10) must be less than end_row (5)"
        ... ]
        >>> raise InvalidProcessMatchingCommandError("Multiple validation errors", errors=errors)

    Phase 2 Note:
        Used by ProcessMatchingCommand.validate_business_rules() to report
        all validation errors at once (better UX than fail-fast approach).
    """

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        """
        Initialize command validation error.

        Args:
            message: Primary error message
            errors: Optional list of specific validation errors
        """
        self.errors = errors or []
        if self.errors:
            # Build comprehensive message from all errors
            detailed_message = f"{message}:\n" + "\n".join(f"  - {err}" for err in self.errors)
            super().__init__(detailed_message)
        else:
            super().__init__(message)

    def add_error(self, error: str) -> None:
        """
        Add validation error to the list.

        Useful for collecting multiple validation errors before raising.

        Args:
            error: Validation error message to add
        """
        self.errors.append(error)

    def has_errors(self) -> bool:
        """
        Check if any validation errors exist.

        Returns:
            True if there are validation errors, False otherwise
        """
        return len(self.errors) > 0


class FileSizeExceededError(DomainException):
    """
    Raised when file size exceeds allowed limit.

    This exception is raised when:
    - Excel file size > MAX_FILE_SIZE_BYTES (10MB)
    - File is too large for processing
    - Memory constraints violated

    Used by:
    - ExcelReaderService._validate_file_size()
    - FileStorageService validation

    Attributes:
        file_size_bytes: Actual file size in bytes
        max_size_bytes: Maximum allowed size in bytes

    Examples:
        >>> raise FileSizeExceededError(
        ...     "File size 15MB exceeds maximum 10MB",
        ...     file_size_bytes=15 * 1024 * 1024,
        ...     max_size_bytes=10 * 1024 * 1024
        ... )
    """

    def __init__(
        self,
        message: str,
        file_size_bytes: int | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        """
        Initialize file size exceeded error.

        Args:
            message: Error description
            file_size_bytes: Actual file size in bytes (optional)
            max_size_bytes: Maximum allowed size in bytes (optional)
        """
        self.file_size_bytes = file_size_bytes
        self.max_size_bytes = max_size_bytes

        # Build detailed message with human-readable sizes
        if file_size_bytes and max_size_bytes:
            file_mb = file_size_bytes / (1024 * 1024)
            max_mb = max_size_bytes / (1024 * 1024)
            detailed_message = (
                f"{message} "
                f"(File: {file_mb:.2f}MB, Max: {max_mb:.2f}MB)"
            )
            super().__init__(detailed_message)
        else:
            super().__init__(message)


class ExcelParsingError(DomainException):
    """
    Raised when Excel file cannot be parsed.

    This exception is raised when:
    - Excel file has invalid format
    - Polars cannot read the file
    - Encoding issues (after UTF-8 and CP1250 fallback)
    - Corrupted Excel file
    - Unsupported Excel version

    Used by:
    - ExcelReaderService._load_excel_dataframe()

    Attributes:
        file_path: Path to Excel file that failed parsing (optional)
        original_error: Original exception from Polars (optional)

    Examples:
        >>> raise ExcelParsingError(
        ...     "Cannot parse Excel file: invalid format",
        ...     file_path="working_file.xlsx",
        ...     original_error=PolarsError("...")
        ... )
    """

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """
        Initialize Excel parsing error.

        Args:
            message: Error description
            file_path: Path to file that failed (optional)
            original_error: Original exception from Polars (optional)
        """
        self.file_path = file_path
        self.original_error = original_error

        # Build detailed message with file path and original error
        detailed_parts = [message]
        if file_path:
            detailed_parts.append(f"File: {file_path}")
        if original_error:
            detailed_parts.append(f"Original error: {type(original_error).__name__}: {original_error}")

        detailed_message = " | ".join(detailed_parts)
        super().__init__(detailed_message)


class ColumnNotFoundError(DomainException):
    """
    Raised when specified column doesn't exist in Excel DataFrame.

    This exception is raised when:
    - Specified column letter (e.g., "Z") doesn't exist in file
    - Column index out of bounds
    - Excel has fewer columns than expected

    Used by:
    - ExcelReaderService._validate_column_exists()

    Attributes:
        column: Column letter that was not found (e.g., "Z")
        available_columns: List of available column names in DataFrame (optional)

    Examples:
        >>> raise ColumnNotFoundError(
        ...     "Column 'Z' not found in Excel file",
        ...     column="Z",
        ...     available_columns=["A", "B", "C", "D", "E"]
        ... )
    """

    def __init__(
        self,
        message: str,
        column: str | None = None,
        available_columns: list[str] | None = None,
    ) -> None:
        """
        Initialize column not found error.

        Args:
            message: Error description
            column: Column letter that was not found (optional)
            available_columns: List of available columns (optional)
        """
        self.column = column
        self.available_columns = available_columns

        # Build detailed message with available columns
        detailed_parts = [message]
        if column:
            detailed_parts.append(f"Requested column: '{column}'")
        if available_columns:
            detailed_parts.append(f"Available columns: {', '.join(available_columns)}")

        detailed_message = " | ".join(detailed_parts)
        super().__init__(detailed_message)
