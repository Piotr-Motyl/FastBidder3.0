"""
ProcessMatchingCommand - CQRS Write Command (Phase 2 - Detailed Contract)

Encapsulates all data needed to trigger matching process with full validation.
Part of CQRS pattern - separates write operations from read operations.

Responsibility:
    - Data holder for matching process initiation
    - Full business rules validation (file IDs, columns, ranges, threshold)
    - Conversion from API request to domain command
    - Serialization for Celery task queue

Architecture Notes:
    - Part of Application Layer (orchestration)
    - Used by ProcessMatchingUseCase to trigger Celery task
    - Immutable data structure (Command pattern)
    - Serialized to JSON for Celery task queue
    - Does NOT validate file existence (Use Case responsibility)

Phase 2 Implementation:
    - Full validation in validate_business_rules()
    - Dedicated config classes (WorkingFileConfig, ReferenceFileConfig)
    - Support for matching strategy and report format
    - Excel column validation (A-ZZ format)
    - Row range validation (with 1000 row limit per file)
"""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from src.application.models import MatchingStrategy, ReportFormat
from src.domain.shared.exceptions import InvalidProcessMatchingCommandError


# ============================================================================
# FILE CONFIGURATION MODELS
# ============================================================================


class Range(BaseModel):
    """
    Range of rows in Excel file (1-based indexing like Excel).

    Represents start and end row numbers for description data.
    Uses Excel notation where first row is 1 (not 0).

    Attributes:
        start: Start row number (inclusive, 1-based)
        end: End row number (inclusive, 1-based)

    Validation:
        - Both start and end must be >= 1
        - start must be < end
        - Range size (end - start) must be <= 1000 rows

    Examples:
        >>> range = Range(start=2, end=100)
        >>> range.start
        2
        >>> range.end
        100
    """

    start: int = Field(ge=1, description="Start row (Excel notation, 1-based)")
    end: int = Field(ge=1, description="End row (Excel notation, 1-based)")

    def size(self) -> int:
        """
        Calculate number of rows in range.

        Returns:
            Number of rows (end - start + 1)

        Examples:
            >>> Range(start=2, end=10).size()
            9
            >>> Range(start=1, end=1).size()
            1
        """
        return self.end - self.start + 1


class WorkingFileConfig(BaseModel):
    """
    Configuration for working file (file to be priced).

    The working file contains descriptions without prices.
    This config specifies where to read descriptions and where to write results.

    Attributes:
        file_id: UUID of working file (string for JSON serialization)
        description_column: Column with descriptions (e.g., 'C', 'AB')
        description_range: Range of rows containing descriptions
        price_target_column: Column where matched prices will be written
        matching_report_column: Optional column for match report

    Validation (Phase 2):
        - file_id must be valid UUID format
        - Columns must be in A-ZZ range (not AAA or beyond)
        - description_range must be valid (start < end, max 1000 rows)
        - price_target_column and matching_report_column should differ

    Examples:
        >>> config = WorkingFileConfig(
        ...     file_id="a3bb189e-8bf9-3888-9912-ace4e6543002",
        ...     description_column="C",
        ...     description_range=Range(start=2, end=100),
        ...     price_target_column="F",
        ...     matching_report_column="G"
        ... )
    """

    file_id: str = Field(description="UUID of working file as string")
    description_column: str = Field(
        description="Column with descriptions (e.g., 'C', 'AB')"
    )
    description_range: Range = Field(description="Range of rows with descriptions")
    price_target_column: str = Field(
        description="Column where matched prices will be written"
    )
    matching_report_column: Optional[str] = Field(
        default=None, description="Optional column for match report"
    )

    @field_validator("file_id")
    @classmethod
    def validate_file_id_format(cls, value: str) -> str:
        """
        Validate that file_id is valid UUID format.

        Args:
            value: file_id string to validate

        Returns:
            Validated file_id string

        Raises:
            ValueError: If file_id is not valid UUID format
        """
        try:
            UUID(value)
        except ValueError as e:
            raise ValueError(f"file_id must be valid UUID format, got '{value}'") from e
        return value


class ReferenceFileConfig(BaseModel):
    """
    Configuration for reference file (price catalog).

    The reference file contains descriptions with prices.
    This config specifies where to read descriptions and prices.

    Attributes:
        file_id: UUID of reference file (string for JSON serialization)
        description_column: Column with descriptions (e.g., 'B', 'C')
        description_range: Range of rows containing descriptions
        price_source_column: Column with prices to copy

    Validation (Phase 2):
        - file_id must be valid UUID format
        - Columns must be in A-ZZ range (not AAA or beyond)
        - description_range must be valid (start < end, max 1000 rows)

    Examples:
        >>> config = ReferenceFileConfig(
        ...     file_id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ...     description_column="B",
        ...     description_range=Range(start=2, end=500),
        ...     price_source_column="D"
        ... )
    """

    file_id: str = Field(description="UUID of reference file as string")
    description_column: str = Field(
        description="Column with descriptions (e.g., 'B', 'C')"
    )
    description_range: Range = Field(description="Range of rows with descriptions")
    price_source_column: str = Field(description="Column with prices to copy")

    @field_validator("file_id")
    @classmethod
    def validate_file_id_format(cls, value: str) -> str:
        """
        Validate that file_id is valid UUID format.

        Args:
            value: file_id string to validate

        Returns:
            Validated file_id string

        Raises:
            ValueError: If file_id is not valid UUID format
        """
        try:
            UUID(value)
        except ValueError as e:
            raise ValueError(f"file_id must be valid UUID format, got '{value}'") from e
        return value


# ============================================================================
# MAIN COMMAND CLASS
# ============================================================================


class ProcessMatchingCommand(BaseModel):
    """
    Command containing all data needed for matching process (Phase 2 - Full Contract).

    Includes file configurations, matching parameters, and processing options.
    Designed for easy serialization to Celery (JSON format).

    Attributes:
        working_file: Configuration for file to be priced
        reference_file: Configuration for price catalog
        matching_threshold: Minimum similarity score to accept match (1.0-100.0)
        matching_strategy: Strategy for handling multiple matches (default: BEST_MATCH)
        report_format: Format of matching report in Excel (default: SIMPLE)

    Business Rules (Phase 2 - Validated in validate_business_rules()):
        - working_file.file_id != reference_file.file_id
        - All columns must be in A-ZZ range (validated format)
        - All ranges must be valid: start < end
        - Row ranges must not exceed 1000 rows per file
        - matching_threshold must be > 0 (Pydantic validates >= 1.0)

    Phase 2 Notes:
        - File existence validation is in Use Case (not Command)
        - Column existence in Excel validated in Infrastructure Layer
        - This is a data holder with business rule validation only
        - Minimal validation for happy path (edge cases in later phases)

    Examples:
        >>> command = ProcessMatchingCommand(
        ...     working_file=WorkingFileConfig(...),
        ...     reference_file=ReferenceFileConfig(...),
        ...     matching_threshold=80.0,
        ...     matching_strategy=MatchingStrategy.BEST_MATCH,
        ...     report_format=ReportFormat.SIMPLE
        ... )
        >>> command.validate_business_rules()  # Raises if invalid
    """

    working_file: WorkingFileConfig = Field(
        description="Configuration for file to be priced"
    )
    reference_file: ReferenceFileConfig = Field(
        description="Configuration for price catalog"
    )
    matching_threshold: float = Field(
        default=75.0,
        ge=1.0,
        le=100.0,
        description="Similarity threshold percentage (1.0-100.0)",
    )
    matching_strategy: MatchingStrategy = Field(
        default=MatchingStrategy.BEST_MATCH,
        description="Strategy for handling multiple matches",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.SIMPLE,
        description="Format of matching report in Excel output",
    )

    # Maximum rows per file (Phase 2 limit for happy path)
    MAX_ROWS_PER_FILE: int = 1000

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "working_file": {
                    "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                    "description_column": "C",
                    "description_range": {"start": 2, "end": 100},
                    "price_target_column": "F",
                    "matching_report_column": "G",
                },
                "reference_file": {
                    "file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "description_column": "B",
                    "description_range": {"start": 2, "end": 500},
                    "price_source_column": "D",
                },
                "matching_threshold": 80.0,
                "matching_strategy": "best_match",
                "report_format": "simple",
            }
        }

    @classmethod
    def from_api_request(cls, request: dict) -> "ProcessMatchingCommand":
        """
        Factory method to create command from API request.

        Converts API request structure to command structure.
        Handles conversion of nested dicts to config objects.
        Converts string enum values to proper Enum types.

        Args:
            request: Dict from ProcessMatchingRequest.dict()

        Returns:
            ProcessMatchingCommand ready for validation and use case execution

        Examples:
            >>> api_request = {
            ...     "working_file": {...},
            ...     "reference_file": {...},
            ...     "matching_threshold": 80.0,
            ...     "matching_strategy": "best_match",
            ...     "report_format": "simple"
            ... }
            >>> command = ProcessMatchingCommand.from_api_request(api_request)
        """
        # Convert string enum values to Enum instances
        strategy_str = request.get("matching_strategy", "best_match")
        strategy = MatchingStrategy(strategy_str) if strategy_str else MatchingStrategy.BEST_MATCH

        format_str = request.get("report_format", "simple")
        report_fmt = ReportFormat(format_str) if format_str else ReportFormat.SIMPLE

        return cls(
            working_file=WorkingFileConfig(**request["working_file"]),
            reference_file=ReferenceFileConfig(**request["reference_file"]),
            matching_threshold=request.get("matching_threshold", 75.0),
            matching_strategy=strategy,
            report_format=report_fmt,
        )

    def validate_business_rules(self) -> None:
        """
        Validate command against business rules (Phase 2 - Full Implementation).

        Collects all validation errors and raises InvalidProcessMatchingCommandError
        with complete list of errors. This provides better UX than fail-fast approach.

        Business Rules Validated:
            1. working_file.file_id != reference_file.file_id
            2. All columns in valid format (A-ZZ, not AAA)
            3. All row ranges valid (start < end)
            4. Row ranges within limit (max 1000 rows per file)

        Raises:
            InvalidProcessMatchingCommandError: If any business rule violated
                (contains list of all validation errors)

        Examples:
            >>> command = ProcessMatchingCommand(...)
            >>> try:
            ...     command.validate_business_rules()
            ... except InvalidProcessMatchingCommandError as e:
            ...     print(e.errors)  # List of all validation errors
        """
        errors: list[str] = []

        # Rule 1: Different file IDs
        if self.working_file.file_id == self.reference_file.file_id:
            errors.append(
                f"working_file.file_id and reference_file.file_id must be different, "
                f"both are '{self.working_file.file_id}'"
            )

        # Rule 2: Validate column formats for working file
        wf_columns = [
            ("description_column", self.working_file.description_column),
            ("price_target_column", self.working_file.price_target_column),
        ]
        if self.working_file.matching_report_column:
            wf_columns.append(
                ("matching_report_column", self.working_file.matching_report_column)
            )

        for field_name, column in wf_columns:
            if not self._is_valid_excel_column(column):
                errors.append(
                    f"working_file.{field_name}: Invalid column '{column}'. "
                    f"Must be in range A-ZZ (e.g., 'A', 'B', 'AA', 'AZ', 'ZZ')"
                )

        # Rule 2: Validate column formats for reference file
        rf_columns = [
            ("description_column", self.reference_file.description_column),
            ("price_source_column", self.reference_file.price_source_column),
        ]

        for field_name, column in rf_columns:
            if not self._is_valid_excel_column(column):
                errors.append(
                    f"reference_file.{field_name}: Invalid column '{column}'. "
                    f"Must be in range A-ZZ (e.g., 'A', 'B', 'AA', 'AZ', 'ZZ')"
                )

        # Rule 3 & 4: Validate working file range
        wf_range = self.working_file.description_range
        if wf_range.start > wf_range.end:
            errors.append(
                f"working_file.description_range: start ({wf_range.start}) "
                f"must be less than or equal to end ({wf_range.end})"
            )

        if wf_range.size() > self.MAX_ROWS_PER_FILE:
            errors.append(
                f"working_file.description_range: range size ({wf_range.size()} rows) "
                f"exceeds maximum allowed ({self.MAX_ROWS_PER_FILE} rows)"
            )

        # Rule 3 & 4: Validate reference file range
        rf_range = self.reference_file.description_range
        if rf_range.start > rf_range.end:
            errors.append(
                f"reference_file.description_range: start ({rf_range.start}) "
                f"must be less than or equal to end ({rf_range.end})"
            )

        if rf_range.size() > self.MAX_ROWS_PER_FILE:
            errors.append(
                f"reference_file.description_range: range size ({rf_range.size()} rows) "
                f"exceeds maximum allowed ({self.MAX_ROWS_PER_FILE} rows)"
            )

        # Raise if any validation errors found
        if errors:
            raise InvalidProcessMatchingCommandError(
                "Command validation failed", errors=errors
            )

    def to_celery_dict(self) -> dict[str, Any]:
        """
        Convert command to dictionary for Celery task serialization.

        Pydantic automatically handles JSON serialization, but this method
        makes serialization explicit and testable. Ensures UUID strings,
        enum values, and nested objects are properly serialized.

        Returns:
            Dictionary ready for JSON serialization and Celery queue

        Examples:
            >>> command = ProcessMatchingCommand(...)
            >>> celery_data = command.to_celery_dict()
            >>> # Can be serialized to JSON and sent to Celery
            >>> import json
            >>> json_data = json.dumps(celery_data)
        """
        return {
            "working_file": {
                "file_id": self.working_file.file_id,
                "description_column": self.working_file.description_column,
                "description_range": {
                    "start": self.working_file.description_range.start,
                    "end": self.working_file.description_range.end,
                },
                "price_target_column": self.working_file.price_target_column,
                "matching_report_column": self.working_file.matching_report_column,
            },
            "reference_file": {
                "file_id": self.reference_file.file_id,
                "description_column": self.reference_file.description_column,
                "description_range": {
                    "start": self.reference_file.description_range.start,
                    "end": self.reference_file.description_range.end,
                },
                "price_source_column": self.reference_file.price_source_column,
            },
            "matching_threshold": self.matching_threshold,
            "matching_strategy": self.matching_strategy.value,
            "report_format": self.report_format.value,
        }

    @staticmethod
    def _is_valid_excel_column(column: str) -> bool:
        """
        Validate Excel column format (A-ZZ range only).

        Valid formats:
        - Single letter: A-Z (26 columns)
        - Two letters: AA-ZZ (676 columns total, so A-ZZ = 702 columns)

        Invalid formats:
        - Empty string
        - More than 2 letters (AAA, AAAA, etc.)
        - Non-letter characters (A1, 1A, A$, etc.)
        - Lowercase (phase 2: accept only uppercase for simplicity)

        Args:
            column: Column string to validate (e.g., 'A', 'B', 'AA', 'ZZ')

        Returns:
            True if valid format, False otherwise

        Examples:
            >>> ProcessMatchingCommand._is_valid_excel_column('A')
            True
            >>> ProcessMatchingCommand._is_valid_excel_column('AA')
            True
            >>> ProcessMatchingCommand._is_valid_excel_column('ZZ')
            True
            >>> ProcessMatchingCommand._is_valid_excel_column('AAA')
            False
            >>> ProcessMatchingCommand._is_valid_excel_column('A1')
            False
            >>> ProcessMatchingCommand._is_valid_excel_column('')
            False

        Phase 2 Note:
            - Validates format only (not existence in actual Excel file)
            - Accepts uppercase only (lowercase normalization in later phases)
            - Limited to A-ZZ for happy path (extended range in later phases)
        """
        if not column:
            return False

        # Must be 1-2 characters
        if len(column) > 2:
            return False

        # All characters must be uppercase letters
        if not column.isalpha() or not column.isupper():
            return False

        return True

    @staticmethod
    def column_to_index(column: str) -> int:
        """
        Convert Excel column letter(s) to 0-based index.

        Conversion examples:
        - A → 0
        - B → 1
        - Z → 25
        - AA → 26
        - AB → 27
        - AZ → 51
        - ZZ → 701

        Args:
            column: Excel column in format 'A', 'B', 'AA', etc.

        Returns:
            0-based column index

        Raises:
            ValueError: If column format is invalid

        Examples:
            >>> ProcessMatchingCommand.column_to_index('A')
            0
            >>> ProcessMatchingCommand.column_to_index('B')
            1
            >>> ProcessMatchingCommand.column_to_index('AA')
            26
            >>> ProcessMatchingCommand.column_to_index('ZZ')
            701

        Phase 2 Note:
            - Helper method for Infrastructure Layer (ExcelReader)
            - Command keeps columns as strings (user-facing format)
            - Actual conversion happens in Infrastructure during Excel read
        """
        if not ProcessMatchingCommand._is_valid_excel_column(column):
            raise ValueError(
                f"Invalid Excel column format: '{column}'. Must be A-ZZ range."
            )

        # Convert column letters to 0-based index
        # A=0, B=1, ..., Z=25, AA=26, AB=27, ..., ZZ=701
        index = 0
        for i, char in enumerate(reversed(column)):
            # char value: A=1, B=2, ..., Z=26
            char_value = ord(char) - ord("A") + 1
            # Position multiplier: 1st position (from right) = 1, 2nd = 26
            index += char_value * (26**i)

        # Convert to 0-based (Excel A=1, we want A=0)
        return index - 1
