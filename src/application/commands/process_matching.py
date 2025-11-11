"""
ProcessMatchingCommand - CQRS Write Command

Encapsulates all data needed to trigger matching process.
Part of CQRS pattern - separates write operations from read operations.

Responsibility:
    - Data holder for matching process initiation
    - Validation of business rules (Phase 2+)
    - Conversion from API request to domain command

Architecture Notes:
    - Part of Application Layer (orchestration)
    - Used by ProcessMatchingUseCase to trigger Celery task
    - Immutable data structure (Command pattern)
    - Will be serialized to JSON for Celery task queue

Phase 1 Note:
    CONTRACT ONLY - minimal validation.
    Full business rules validation in Phase 2.
"""

from typing import Optional
from uuid import UUID
from pydantic import BaseModel, Field


class ProcessMatchingCommand(BaseModel):
    """
    Command containing all data needed for matching process.

    Includes file IDs and detailed column mappings from request.
    Designed for easy serialization to Celery (JSON format).

    Attributes:
        working_file: Configuration for file to be priced (as dict for Celery)
        reference_file: Configuration for price catalog (as dict for Celery)
        matching_threshold: Minimum similarity score to accept match

    Business Rules (Phase 2+):
        - working_file.file_id != reference_file.file_id
        - Both files must exist in storage
        - Both files must be valid Excel format
        - Specified columns must exist in files
        - Ranges must be valid (start < end)
        - Threshold should be > 0 for meaningful results
    """

    working_file: dict = Field(
        description="WorkingFileConfig as dict for Celery serialization"
    )
    reference_file: dict = Field(
        description="ReferenceFileConfig as dict for Celery serialization"
    )
    matching_threshold: float = Field(
        default=75.0, ge=1.0, le=100.0, description="Similarity threshold percentage"
    )

    class Config:
        # Allow dict for Celery JSON serialization
        arbitrary_types_allowed = True
        json_schema_extra = {
            "example": {
                "working_file": {
                    "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                    "description_column": "C",
                    "description_range": {"start": 2, "end": 10},
                    "price_target_column": "F",
                    "matching_report_column": "G",
                },
                "reference_file": {
                    "file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "description_column": "B",
                    "description_range": {"start": 2, "end": 20},
                    "price_source_column": "D",
                },
                "matching_threshold": 80.0,
            }
        }

    @classmethod
    def from_api_request(cls, request: dict) -> "ProcessMatchingCommand":
        """
        Factory method to create command from API request.

        Converts API request structure to command structure.
        Main transformation: ensures file configs are dicts for Celery.

        Args:
            request: Dict from ProcessMatchingRequest.dict()

        Returns:
            ProcessMatchingCommand ready for use case execution

        Note:
            Phase 3 will add validation of business rules here.
        """
        return cls(
            working_file=request["working_file"],
            reference_file=request["reference_file"],
            matching_threshold=request.get("matching_threshold", 75.0),
        )

    def validate_business_rules(self) -> None:
        """
        Validate command against business rules.

        CONTRACT ONLY - Implementation in Phase 2.

        Will check:
            - Different file IDs
            - Valid column references
            - Valid ranges
            - Threshold makes sense

        Raises:
            ValueError: If any business rule violated
        """
        # Phase 2 implementation
        pass
