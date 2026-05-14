"""
ProcessMatchingCommand — CQRS write command for triggering matching.

Encapsulates request data + business-rule validation + Celery serialization.
File existence is validated by the use case (not here) — this is pure data + rules.
"""

from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.application.models import MatchingStrategy, ReportFormat
from src.domain.shared.exceptions import InvalidProcessMatchingCommandError
from src.shared.utils.excel import excel_column_to_index, is_valid_excel_column


class Range(BaseModel):
    """Inclusive Excel row range (1-based: first row is 1, not 0)."""

    start: int = Field(ge=1, description="Start row (Excel notation, 1-based)")
    end: int = Field(ge=1, description="End row (Excel notation, 1-based)")

    def size(self) -> int:
        return self.end - self.start + 1


class WorkingFileConfig(BaseModel):
    """File being priced. Specifies where to read descriptions and where to write results."""

    file_id: str = Field(description="UUID of working file as string")
    description_column: str = Field(description="Column with descriptions (e.g. 'C', 'AB')")
    description_range: Range
    price_target_column: str = Field(description="Column where matched prices are written")
    matching_report_column: Optional[str] = Field(
        default=None, description="Optional column for human-readable match report"
    )

    @field_validator("file_id")
    @classmethod
    def validate_file_id_format(cls, value: str) -> str:
        try:
            UUID(value)
        except ValueError as e:
            raise ValueError(f"file_id must be valid UUID format, got '{value}'") from e
        return value


class ReferenceFileConfig(BaseModel):
    """Price catalog. Specifies columns for descriptions and matching prices."""

    file_id: str = Field(description="UUID of reference file as string")
    description_column: str = Field(description="Column with descriptions (e.g. 'B', 'C')")
    description_range: Range
    price_source_column: str = Field(description="Column with prices to copy on match")

    @field_validator("file_id")
    @classmethod
    def validate_file_id_format(cls, value: str) -> str:
        try:
            UUID(value)
        except ValueError as e:
            raise ValueError(f"file_id must be valid UUID format, got '{value}'") from e
        return value


class ProcessMatchingCommand(BaseModel):
    """
    Encapsulates everything needed to trigger one matching job.

    Business rules validated by validate_business_rules():
      - working_file.file_id != reference_file.file_id
      - All columns valid A-ZZ format
      - All ranges valid (start <= end)
      - Range size <= MAX_ROWS_PER_FILE (1000)

    File existence is NOT validated here — that's the use case's job.
    """

    working_file: WorkingFileConfig
    reference_file: ReferenceFileConfig
    matching_threshold: float = Field(
        default=75.0,
        ge=1.0,
        le=100.0,
        description="Similarity threshold percentage (1.0–100.0)",
    )
    matching_strategy: MatchingStrategy = Field(default=MatchingStrategy.BEST_MATCH)
    report_format: ReportFormat = Field(default=ReportFormat.SIMPLE)

    MAX_ROWS_PER_FILE: int = 1000

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
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
        },
    )

    @classmethod
    def from_api_request(cls, request: dict) -> "ProcessMatchingCommand":
        """Build from dict (e.g. ProcessMatchingRequest.model_dump()), normalizing enums."""
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
        Validate all business rules and raise with the full error list.

        Collects all errors (rather than fail-fast) so the API can show them
        together in one 422 response — better UX than one-at-a-time corrections.
        """
        errors: list[str] = []

        # Rule 1: different file IDs
        if self.working_file.file_id == self.reference_file.file_id:
            errors.append(
                f"working_file.file_id and reference_file.file_id must be different, "
                f"both are '{self.working_file.file_id}'"
            )

        # Rule 2: column format (A-ZZ)
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
                    f"Must be in range A-ZZ"
                )

        rf_columns = [
            ("description_column", self.reference_file.description_column),
            ("price_source_column", self.reference_file.price_source_column),
        ]
        for field_name, column in rf_columns:
            if not self._is_valid_excel_column(column):
                errors.append(
                    f"reference_file.{field_name}: Invalid column '{column}'. "
                    f"Must be in range A-ZZ"
                )

        # Rule 3 + 4: ranges valid and within size limit
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

        if errors:
            raise InvalidProcessMatchingCommandError(
                "Command validation failed", errors=errors
            )

    def to_celery_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for Celery task kwargs."""
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
        """Valid: A-Z, AA-ZZ. Invalid: empty, 3+ letters, lowercase, non-letters."""
        return is_valid_excel_column(column)

    @staticmethod
    def column_to_index(column: str) -> int:
        """Excel letter → 0-based index. A=0, B=1, ..., Z=25, AA=26, ..., ZZ=701."""
        return excel_column_to_index(column)
