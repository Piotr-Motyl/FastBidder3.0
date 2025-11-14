"""
Process Matching Use Case - Application Orchestration

Responsibility:
    Orchestrates the matching process by coordinating multiple services.
    Implements Use Case pattern from Clean Architecture.

Architecture Notes:
    - Part of Application Layer (Services/Use Cases)
    - Depends on Domain Layer (via interfaces)
    - Depends on Infrastructure Layer (via dependency injection)
    - Coordinates Celery task execution
    - No direct HTTP handling (that's API Layer concern)

Contains:
    - ProcessMatchingUseCase: Main orchestration class
    - ProcessMatchingResult: Response DTO

Does NOT contain:
    - Domain business rules (delegated to Domain services)
    - HTTP handling (delegated to API Layer)
    - Direct Redis/Celery access (uses abstraction)
    - File parsing logic (delegated to Infrastructure)

Phase 1 Note:
    This is a CONTRACT ONLY. Implementation in Phase 3.
"""

from typing import Optional, Protocol
from uuid import UUID
from celery import Celery
from pydantic import BaseModel, Field

from src.application.commands.process_matching import ProcessMatchingCommand
from src.application.models import JobStatus


# ============================================================================
# PROTOCOL (Interface for dependency injection)
# ============================================================================


class FileStorageServiceProtocol(Protocol):
    """
    Protocol (interface) for file storage service.

    Defines the contract that Infrastructure Layer must implement.
    This follows Dependency Inversion Principle from SOLID.

    Methods:
        file_exists: Check if specific file exists in job storage
        get_file_metadata: Get file metadata (size, format, timestamps)

    Storage Structure (Phase 2):
        Files are organized by job_id and file_type:
        - {job_id}/input/working_file.xlsx
        - {job_id}/input/reference_file.xlsx
        - {job_id}/output/result.xlsx

    Note:
        Actual implementation in Infrastructure Layer:
        src/infrastructure/file_storage/file_storage_service.py

        Phase 2 Update:
        Changed from file_id to (job_id, file_type) for better architecture.
        Aligns with storage structure {job_id}/{input|output}/.
    """

    async def file_exists(self, job_id: UUID, file_type: str) -> bool:
        """
        Check if specific file exists in job storage.

        Args:
            job_id: UUID of the job
            file_type: Type of file ("working", "reference", "result")

        Returns:
            True if file exists, False otherwise
        """
        ...

    async def get_file_metadata(self, job_id: UUID, file_type: str) -> dict:
        """
        Get file metadata (size, format, timestamps).

        Args:
            job_id: UUID of the job
            file_type: Type of file ("working", "reference", "result")

        Returns:
            Dict with metadata (size, format, created_at, modified_at, etc.)
        """
        ...


# ============================================================================
# RESULT (Response DTO)
# ============================================================================


class ProcessMatchingResult(BaseModel):
    """
    Result DTO for process matching use case.

    Contains job metadata returned after triggering Celery task.
    This is Application Layer's representation, API Layer may convert
    to ProcessMatchingResponse for HTTP response.

    Attributes:
        job_id: UUID of created Celery task
        status: Initial job status (always "queued" immediately after creation)
        estimated_time: Estimated time to completion in seconds
        message: Human-readable message about job status

    Example:
        >>> result = ProcessMatchingResult(
        ...     job_id=UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"),
        ...     status=JobStatus.QUEUED,
        ...     estimated_time=45,
        ...     message="Matching job queued successfully"
        ... )
    """

    job_id: UUID = Field(description="Celery task ID for tracking progress")

    status: JobStatus = Field(
        default=JobStatus.QUEUED,
        description="Initial job status (always 'queued' after creation)",
    )

    estimated_time: int = Field(
        ge=0, description="Estimated time to completion in seconds (based on file size)"
    )

    message: str = Field(
        default="Matching job queued successfully. Use job_id to check status.",
        description="Human-readable status message",
    )

    class Config:
        """Pydantic configuration for ProcessMatchingResult."""

        json_schema_extra = {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "queued",
                "estimated_time": 45,
                "message": "Matching job queued successfully. Use job_id to check status.",
            }
        }


# ============================================================================
# USE CASE
# ============================================================================


class ProcessMatchingUseCase:
    """
    Orchestrates the matching process.

    Responsibility:
        Coordinates multiple services to trigger asynchronous matching:
        1. Validates business rules (file existence, format)
        2. Triggers Celery task with validated parameters
        3. Returns job metadata for status tracking

    Architecture Pattern:
        - Use Case from Clean Architecture
        - Depends on abstractions (Protocol), not implementations
        - Constructor injection for dependencies
        - Single Responsibility: orchestration only

    Dependencies:
        - celery_app: Celery application instance (for triggering tasks)
        - file_storage: FileStorageService for file validation (optional, Phase 3)

    Flow:
        API Layer → Use Case → Celery Task → Domain Services

        1. API receives request → creates Command
        2. Use Case validates Command business rules
        3. Use Case triggers Celery task
        4. Celery task executes async (orchestrates Domain services)
        5. Use Case returns job metadata immediately

    Example:
        >>> # In API Layer dependency injection:
        >>> def get_process_matching_use_case():
        ...     celery_app = get_celery_app()
        ...     file_storage = get_file_storage_service()
        ...     return ProcessMatchingUseCase(celery_app, file_storage)
        >>>
        >>> # Usage in API endpoint:
        >>> use_case = Depends(get_process_matching_use_case)
        >>> command = ProcessMatchingCommand.from_request(request)
        >>> result = await use_case.execute(command)
        >>> return ProcessMatchingResponse(**result.dict())
    """

    def __init__(
        self,
        celery_app: Celery,
        file_storage: Optional[FileStorageServiceProtocol] = None,
    ):
        """
        Initialize use case with dependencies.

        Args:
            celery_app: Celery application instance for task triggering
            file_storage: FileStorageService for file validation (optional, Phase 3)

        Architecture Note:
            Constructor injection follows Clean Architecture and SOLID principles.
            Dependencies are injected, not created internally.
            This enables easy testing (mock dependencies) and flexibility.

            file_storage is Optional because in Phase 1 we focus on contracts.
            In Phase 3, this will be required for business rule validation.
        """
        self.celery_app = celery_app
        self.file_storage = file_storage

    async def execute(self, command: ProcessMatchingCommand) -> ProcessMatchingResult:
        """
        Execute matching process orchestration.

        Main use case method that coordinates the entire matching process.

        Args:
            command: ProcessMatchingCommand with validated file IDs and threshold

        Returns:
            ProcessMatchingResult: Job metadata with job_id for status tracking

        Raises:
            FileNotFoundException: If wf_file_id or ref_file_id not found in storage
            InvalidFileFormatError: If files are not valid Excel format
            BusinessRuleViolationError: If any business rule is violated

        Process Flow:
            1. Validate command (already done by Pydantic)
            2. Validate business rules:
               - Files exist in storage (via file_storage service)
               - Files have valid Excel format
               - Files contain required columns
            3. Estimate processing time (based on file size)
            4. Trigger Celery task with command parameters
            5. Create and return ProcessMatchingResult with job metadata

        Business Rules Validated:
            - wf_file_id != ref_file_id (already in Command validator)
            - Both files exist in storage
            - Both files are valid Excel (.xlsx, .xls)
            - Working file contains 'Description' column
            - Reference file contains 'Description' and 'Price' columns
            - File sizes within limits (max 10MB per file)

        Celery Task Triggering:
            Task name: "process_matching"
            Parameters: wf_file_id (str), ref_file_id (str), threshold (float)
            Execution: Async (non-blocking)
            Return: Celery AsyncResult with task_id

        Example:
            >>> command = ProcessMatchingCommand(
            ...     wf_file_id=UUID("a3bb189e..."),
            ...     ref_file_id=UUID("f47ac10b..."),
            ...     threshold=80.0
            ... )
            >>> result = await use_case.execute(command)
            >>> print(result.job_id)  # UUID("3fa85f64...")
            >>> print(result.status)  # JobStatus.QUEUED

        Note:
            Implementation in Phase 3 - Task 3.2.1.
            This is a contract showing expected behavior and dependencies.
        """
        # CONTRACT ONLY - Implementation in Phase 3
        #
        # Implementation will:
        # 1. Validate command business rules:
        #    command.validate_business_rules()
        #
        # 2. Validate files exist (if file_storage available):
        #    if self.file_storage:
        #        await self._validate_files(command)
        #
        # 3. Estimate processing time (simple heuristic):
        #    estimated_time = await self._estimate_processing_time(
        #        UUID(command.working_file.file_id),
        #        UUID(command.reference_file.file_id)
        #    )
        #
        # 4. Trigger Celery task with full command data:
        #    from src.application.tasks.matching_tasks import process_matching_task
        #    celery_data = command.to_celery_dict()
        #    task_result = process_matching_task.delay(**celery_data)
        #    # OR explicitly:
        #    # task_result = process_matching_task.delay(
        #    #     working_file=celery_data['working_file'],
        #    #     reference_file=celery_data['reference_file'],
        #    #     matching_threshold=celery_data['matching_threshold'],
        #    #     matching_strategy=celery_data['matching_strategy'],
        #    #     report_format=celery_data['report_format']
        #    # )
        #
        # 5. Create and return result:
        #    return ProcessMatchingResult(
        #        job_id=UUID(task_result.id),
        #        status=JobStatus.QUEUED,
        #        estimated_time=estimated_time,
        #        message=f"Matching job queued successfully. Job ID: {task_result.id}"
        #    )
        #
        # Error Handling:
        #    - Wrap exceptions in domain-specific exceptions
        #    - Log errors for debugging
        #    - Return meaningful error messages

        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.2.1. "
            "This is a contract only (Phase 1 - Task 1.1.2)."
        )

    async def _validate_files(self, command: ProcessMatchingCommand) -> None:
        """
        Validate file existence and format.

        Private helper method for business rule validation.

        Args:
            command: ProcessMatchingCommand with file IDs to validate

        Raises:
            FileNotFoundException: If any file not found
            InvalidFileFormatError: If any file has invalid format

        Validation Steps:
            1. Check if wf_file_id exists in storage
            2. Check if ref_file_id exists in storage
            3. Validate wf file is Excel format
            4. Validate ref file is Excel format
            5. Check file sizes (max 10MB)

        Note:
            Implementation in Phase 3. Will use file_storage service.
        """
        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.2.1. "
            "This is a contract only (Phase 1 - Task 1.1.2)."
        )

    async def _estimate_processing_time(
        self, wf_file_id: UUID, ref_file_id: UUID
    ) -> int:
        """
        Estimate processing time based on file sizes.

        Private helper method for time estimation.

        Args:
            wf_file_id: Working file UUID
            ref_file_id: Reference file UUID

        Returns:
            int: Estimated processing time in seconds

        Estimation Logic:
            - Get file sizes from file_storage
            - ~1 second per 10 rows
            - File size → estimated row count
            - Add overhead for parameter extraction and matching
            - Minimum: 10 seconds
            - Maximum: 300 seconds (5 minutes)

        Example:
            >>> time = await use_case._estimate_processing_time(
            ...     wf_file_id=UUID("a3bb189e..."),
            ...     ref_file_id=UUID("f47ac10b...")
            ... )
            >>> print(time)  # ~45 seconds

        Note:
            Implementation in Phase 3. Simple heuristic for Phase 1.
        """
        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.2.1. "
            "This is a contract only (Phase 1 - Task 1.1.2)."
        )
