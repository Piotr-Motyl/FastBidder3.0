"""
Process Matching Command - CQRS Write Operation

Responsibility:
    Encapsulates the intent to start a matching process.
    Immutable command object with validated parameters.

Architecture Notes:
    - Part of Application Layer (CQRS Commands)
    - Used by ProcessMatchingUseCase for orchestration
    - Converted from API Layer ProcessMatchingRequest
    - Write operation (triggers async Celery task)

Contains:
    - ProcessMatchingCommand: Command object with file IDs and threshold

Does NOT contain:
    - Business logic (belongs to Domain Layer)
    - HTTP handling (belongs to API Layer)
    - Celery task execution (belongs to tasks/)
    - File validation (will be in Use Case or Domain)

Phase 1 Note:
    This is a CONTRACT ONLY. Implementation in Phase 3.
"""

from typing import Any
from uuid import UUID
from pydantic import BaseModel, Field, model_validator


class ProcessMatchingCommand(BaseModel):
    """
    Command to trigger asynchronous matching process.

    Encapsulates all parameters required to start matching HVAC descriptions
    from working file against reference file with prices.

    CQRS Pattern:
        This is a COMMAND (write operation) that modifies system state
        by creating a new matching job. It does not return data directly,
        but triggers async processing that will eventually produce results.

    Attributes:
        wf_file_id: UUID of uploaded working file (Excel with descriptions to match)
        ref_file_id: UUID of uploaded reference file (Excel with products and prices)
        threshold: Similarity threshold percentage (0.0-100.0).
                   Only matches above this threshold will be included in results.
                   Default: 75.0 (from .env DEFAULT_THRESHOLD)

    Validation Rules:
        - wf_file_id and ref_file_id must be valid UUIDs
        - wf_file_id != ref_file_id (cannot match file against itself)
        - threshold must be between 0.0 and 100.0 (inclusive)

    Example:
        >>> command = ProcessMatchingCommand(
        ...     wf_file_id=UUID("a3bb189e-8bf9-3888-9912-ace4e6543002"),
        ...     ref_file_id=UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
        ...     threshold=80.0
        ... )
        >>> # Pass to use case
        >>> result = await use_case.execute(command)

    Factory Methods:
        Use from_request() to create from API Layer request model.

    Immutability:
        Command objects should be immutable after creation.
        Use Pydantic's frozen=True in Config if strict immutability needed.
    """

    wf_file_id: UUID = Field(description="UUID of working file (descriptions to match)")

    ref_file_id: UUID = Field(
        description="UUID of reference file (products with prices)"
    )

    threshold: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="Similarity threshold percentage (0-100)",
    )

    @model_validator(mode="after")
    def validate_different_files(self) -> "ProcessMatchingCommand":
        """
        Validate that working file and reference file are different.

        Business Rule:
            Cannot match a file against itself. This would be meaningless
            and waste computational resources.

        Returns:
            ProcessMatchingCommand: Self if validation passes

        Raises:
            ValueError: If wf_file_id == ref_file_id

        Note:
            This validation will be executed during command instantiation.
            Use Case will perform additional business validations (file existence, format).

            Uses Pydantic v2 @model_validator(mode='after') for whole-model validation.
        """
        if self.wf_file_id == self.ref_file_id:
            raise ValueError(
                "Working file and reference file must be different. "
                f"Both files have the same ID: {self.ref_file_id}"
            )
        return self

    @classmethod
    def from_request(cls, request: Any) -> "ProcessMatchingCommand":
        """
        Factory method to create command from API request.

        Converts API Layer ProcessMatchingRequest to Application Layer Command.
        This provides clean separation between layers.

        Args:
            request: ProcessMatchingRequest from API Layer
                    (or any object with wf_file_id, ref_file_id, threshold attributes)

        Returns:
            ProcessMatchingCommand: Validated command ready for use case

        Example:
            >>> from src.api.routers.matching import ProcessMatchingRequest
            >>> api_request = ProcessMatchingRequest(
            ...     wf_file_id=UUID("a3bb189e-8bf9-3888-9912-ace4e6543002"),
            ...     ref_file_id=UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479"),
            ...     threshold=80.0
            ... )
            >>> command = ProcessMatchingCommand.from_request(api_request)

        Note:
            Implementation in Phase 3. This is a contract showing the pattern.
        """
        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.2.1. "
            "This is a contract only (Phase 1 - Task 1.1.2)."
        )

    def validate_business_rules(self) -> None:
        """
        Placeholder for additional business rule validation.

        Business rules that will be validated:
        - Files exist in storage (via FileStorageService)
        - Files have valid Excel format (.xlsx, .xls)
        - Working file contains required columns (descriptions)
        - Reference file contains required columns (descriptions, prices)
        - No duplicate file pair currently processing (optional)

        This method will be called by Use Case before triggering Celery task.

        Raises:
            ValueError: If any business rule is violated

        Note:
            Implementation in Phase 3. Actual validation will use
            Infrastructure Layer services (FileStorageService, ExcelValidator).
        """
        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.2.1. "
            "This is a contract only (Phase 1 - Task 1.1.2)."
        )

    class Config:
        """Pydantic configuration for ProcessMatchingCommand."""

        json_schema_extra = {
            "example": {
                "wf_file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                "ref_file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "threshold": 80.0,
            }
        }
        # frozen = True  # Uncomment for strict immutability
