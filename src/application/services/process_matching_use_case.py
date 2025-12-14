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

import logging
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.application.commands.process_matching import ProcessMatchingCommand
from src.application.models import JobStatus
from src.application.ports.file_storage import FileStorageServiceProtocol
from src.application.tasks.matching_tasks import process_matching_task
from src.infrastructure.persistence.redis.progress_tracker import RedisProgressTracker

if TYPE_CHECKING:
    from celery import Celery

# Configure logger for this module
logger = logging.getLogger(__name__)


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
    Orchestrates the matching process (Phase 2 - Detailed Contract).

    Responsibility:
        Coordinates multiple services to trigger asynchronous matching:
        1. Validates business rules (file existence via file_storage)
        2. Estimates processing time (rows_count * 0.1s algorithm)
        3. Triggers Celery task with validated parameters
        4. Returns job metadata for status tracking

    Architecture Pattern:
        - Use Case from Clean Architecture
        - Depends on abstractions (Protocol), not implementations
        - Constructor injection for dependencies
        - Single Responsibility: orchestration only

    Dependencies (Phase 2):
        - celery_app: Celery application instance (for triggering tasks)
        - file_storage: FileStorageService for file validation and metadata extraction

    Flow:
        API Layer → Use Case → Celery Task → Domain Services

        1. API receives request → creates Command
        2. Use Case validates Command business rules (command.validate_business_rules())
        3. Use Case validates files exist (via file_storage.file_exists())
        4. Use Case extracts metadata for estimation (via file_storage.extract_file_metadata())
        5. Use Case estimates processing time (rows_count * 0.1s)
        6. Use Case triggers Celery task (via command.to_celery_dict())
        7. Celery task executes async (orchestrates Domain services)
        8. Use Case returns job metadata immediately (job_id, status, estimated_time)

    Phase 3+ Extensions (NOT in Phase 2):
        - Idempotency: Cache job_id by request hash to avoid duplicate jobs
        - Priority queue: Support LOW/NORMAL/HIGH priority levels
        - Quota check: Limit concurrent jobs per user (requires auth)
        - Dry run mode: Simulate processing without saving results
        - Callback URL: Webhook notification on job completion
        - Advanced metrics: position_in_queue, estimated_start, estimated_completion

    Example:
        >>> # In API Layer dependency injection:
        >>> def get_process_matching_use_case():
        ...     celery_app = get_celery_app()
        ...     file_storage = get_file_storage_service()
        ...     return ProcessMatchingUseCase(celery_app, file_storage)
        >>>
        >>> # Usage in API endpoint:
        >>> use_case = Depends(get_process_matching_use_case)
        >>> command = ProcessMatchingCommand(
        ...     working_file=request.working_file,
        ...     reference_file=request.reference_file,
        ...     matching_threshold=request.matching_threshold
        ... )
        >>> result = await use_case.execute(command)
        >>> return ProcessMatchingResponse(
        ...     job_id=str(result.job_id),
        ...     status=result.status,
        ...     estimated_time=result.estimated_time,
        ...     message=result.message
        ... )
    """

    def __init__(
        self,
        celery_app: "Celery",
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
        Execute matching process orchestration (Phase 2 - Detailed Contract).

        Main use case method that coordinates the entire matching process.
        Orchestrates validation, estimation, and Celery task triggering.

        Process Flow (10 steps):
            1. Validate command business rules using command.validate_business_rules()
               - Checks file_ids are different
               - Checks ranges are valid (start < end)
               - Checks columns are valid Excel format (A-ZZ)
               - Validates matching_threshold (1.0-100.0)
            2. Validate working file exists via file_storage.file_exists()
            3. Validate reference file exists via file_storage.file_exists()
            4. Extract working file metadata via file_storage.extract_file_metadata()
            5. Calculate estimated time using _estimate_processing_time()
            6. Serialize command to Celery format using command.to_celery_dict()
            7. Import process_matching_task from matching_tasks module
            8. Trigger Celery task with task_result = process_matching_task.delay(**celery_data)
            9. Create ProcessMatchingResult with job_id from task_result.id
            10. Return result to API Layer

        Args:
            command: ProcessMatchingCommand with validated file IDs and configuration

        Returns:
            ProcessMatchingResult: Job metadata with job_id for status tracking

        Raises:
            ValueError: If file IDs are identical or invalid format
            FileNotFoundError: If working or reference file not found in uploads storage
            InvalidProcessMatchingCommandError: If business rules validation fails

        Business Rules Validated (Phase 2 - Happy Path):
            - wf_file_id != ref_file_id (in command.validate_business_rules())
            - Both files exist in uploads storage (via file_storage.file_exists())
            - Phase 3+ will add: file format, size limits, required columns

        Celery Task Triggering (Phase 2):
            Task name: "process_matching" (from matching_tasks.py)
            Serialization: command.to_celery_dict() produces:
                {
                    'working_file': dict (WorkingFileConfig as dict),
                    'reference_file': dict (ReferenceFileConfig as dict),
                    'matching_threshold': float,
                    'matching_strategy': str,
                    'report_format': str
                }
            Execution: Async (non-blocking), returns immediately
            Return: Celery AsyncResult with task_id (UUID)

        Example:
            >>> command = ProcessMatchingCommand(
            ...     working_file=WorkingFileConfig(
            ...         file_id="a3bb189e-8bf9-3888-9912-ace4e6543002",
            ...         description_column="C",
            ...         description_range=Range(start=2, end=100),
            ...         price_target_column="F"
            ...     ),
            ...     reference_file=ReferenceFileConfig(
            ...         file_id="f47ac10b-58cc-4372-a567-0e02b2c3d479",
            ...         description_column="B",
            ...         description_range=Range(start=2, end=50),
            ...         price_source_column="D"
            ...     ),
            ...     matching_threshold=80.0
            ... )
            >>> result = await use_case.execute(command)
            >>> print(result.job_id)  # UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
            >>> print(result.status)  # JobStatus.QUEUED
            >>> print(result.estimated_time)  # 10 (seconds, based on 100 rows * 0.1)

        Implementation Note (Phase 3):
            import logging
            from uuid import UUID
            from src.application.tasks.matching_tasks import process_matching_task

            logger = logging.getLogger(__name__)

            # Step 1: Validate command business rules
            command.validate_business_rules()
            logger.debug(f"Command validation passed for files: {command.working_file.file_id}, {command.reference_file.file_id}")

            # Steps 2-3: Validate files exist (if file_storage available)
            if self.file_storage:
                await self._validate_files(command)
                logger.debug("File existence validation passed")

            # Steps 4-5: Estimate processing time
            estimated_time = await self._estimate_processing_time(
                UUID(command.working_file.file_id),
                UUID(command.reference_file.file_id)
            )
            logger.info(f"Estimated processing time: {estimated_time}s")

            # Steps 6-8: Trigger Celery task
            celery_data = command.to_celery_dict()
            task_result = process_matching_task.delay(**celery_data)
            logger.info(f"Celery task triggered: {task_result.id}")

            # Steps 9-10: Create and return result
            return ProcessMatchingResult(
                job_id=UUID(task_result.id),
                status=JobStatus.QUEUED,
                estimated_time=estimated_time,
                message=f"Matching job queued successfully. Check status at GET /jobs/{task_result.id}/status"
            )

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3 - Task 3.2.1.
        """
        # Import process_matching_task here to avoid circular imports
        from src.application.tasks.matching_tasks import process_matching_task

        # Step 1: Validate command business rules
        command.validate_business_rules()
        logger.debug(
            f"Command validation passed for files: "
            f"{command.working_file.file_id}, {command.reference_file.file_id}"
        )

        # Steps 2-3: Validate files exist (if file_storage available)
        if self.file_storage:
            await self._validate_files(command)
            logger.debug("File existence validation passed")

        # Steps 4-5: Estimate processing time
        if self.file_storage:
            estimated_time = await self._estimate_processing_time(
                UUID(command.working_file.file_id), UUID(command.reference_file.file_id)
            )
        else:
            # Fallback if no file_storage: use default estimation
            estimated_time = 30  # Default 30 seconds
        logger.info(f"Estimated processing time: {estimated_time}s")

        # Steps 6-7: Generate job_id and initialize in Redis BEFORE Celery
        # This ensures API can query status immediately without race condition
        job_id = str(uuid4())
        logger.info(f"Generated job_id: {job_id}")

        # Initialize job status in Redis BEFORE sending to Celery queue
        progress_tracker = RedisProgressTracker()
        progress_tracker.start_job(
            job_id=job_id,
            message="Job queued, waiting for worker to start processing",
            total_items=0,  # Unknown at this point, will be updated by worker
        )
        logger.info(f"Job {job_id} initialized in Redis with status QUEUED")

        # Step 8: Trigger Celery task with custom task_id (same as job_id)
        # Using apply_async() instead of delay() to specify custom task_id
        celery_data = command.to_celery_dict()
        task_result = process_matching_task.apply_async(
            kwargs=celery_data,
            task_id=job_id,  # Use our pre-generated job_id
        )
        logger.info(f"Celery task triggered with task_id: {task_result.id}")

        # Steps 9-10: Create and return result
        return ProcessMatchingResult(
            job_id=UUID(task_result.id),
            status=JobStatus.QUEUED,
            estimated_time=estimated_time,
            message=f"Matching job queued successfully. Check status at GET /jobs/{task_result.id}/status",
        )

    async def _validate_files(self, command: ProcessMatchingCommand) -> None:
        """
        Validate file existence in uploads storage (Phase 2 - Detailed Contract).

        Private helper method for business rule validation.
        Phase 2 focuses on happy path - checks only file existence.

        Process Flow (Phase 2 - Minimal):
            1. Get working file_id from command.working_file.file_id
            2. Check if working file exists via file_storage.file_exists()
               Note: file_storage operates on uploads/{file_id}/ directory structure
            3. If working file not found, raise FileNotFoundError
            4. Get reference file_id from command.reference_file.file_id
            5. Check if reference file exists via file_storage.file_exists()
            6. If reference file not found, raise FileNotFoundError
            7. Log successful validation (DEBUG level)

        Args:
            command: ProcessMatchingCommand with file IDs to validate

        Raises:
            FileNotFoundError: If working or reference file not found in uploads storage

        Phase 3+ Extensions (NOT in Phase 2):
            - Excel format validation (via extract_file_metadata())
            - File size validation (max 10MB)
            - Required columns validation (Description, Price)
            - File corruption check

        Example:
            >>> command = ProcessMatchingCommand(
            ...     working_file=WorkingFileConfig(file_id="abc-123", ...),
            ...     reference_file=ReferenceFileConfig(file_id="def-456", ...)
            ... )
            >>> await use_case._validate_files(command)
            # If both files exist, returns silently
            # If file missing, raises FileNotFoundError

        Implementation Note (Phase 3):
            import logging
            from uuid import UUID

            logger = logging.getLogger(__name__)

            # Validate working file exists
            wf_file_id = UUID(command.working_file.file_id)
            wf_exists = await self.file_storage.file_exists(wf_file_id, "working")
            if not wf_exists:
                raise FileNotFoundError(
                    f"Working file not found in uploads storage: {command.working_file.file_id}"
                )

            # Validate reference file exists
            ref_file_id = UUID(command.reference_file.file_id)
            ref_exists = await self.file_storage.file_exists(ref_file_id, "reference")
            if not ref_exists:
                raise FileNotFoundError(
                    f"Reference file not found in uploads storage: {command.reference_file.file_id}"
                )

            logger.debug(f"File existence validation passed for WF={wf_file_id}, REF={ref_file_id}")

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3 - Task 3.2.1.
        """
        # Validate working file exists in upload storage
        wf_file_id = UUID(command.working_file.file_id)
        wf_upload_dir = self.file_storage.get_uploaded_file_path(wf_file_id)

        # Check if upload directory exists and contains files
        if not wf_upload_dir.exists() or not any(wf_upload_dir.iterdir()):
            raise FileNotFoundError(
                f"Working file not found in uploads storage: {command.working_file.file_id}"
            )

        # Validate reference file exists in upload storage
        ref_file_id = UUID(command.reference_file.file_id)
        ref_upload_dir = self.file_storage.get_uploaded_file_path(ref_file_id)

        # Check if upload directory exists and contains files
        if not ref_upload_dir.exists() or not any(ref_upload_dir.iterdir()):
            raise FileNotFoundError(
                f"Reference file not found in uploads storage: {command.reference_file.file_id}"
            )

        logger.debug(
            f"File existence validation passed for WF={wf_file_id}, REF={ref_file_id}"
        )

    async def _estimate_processing_time(
        self, wf_file_id: UUID, ref_file_id: UUID
    ) -> int:
        """
        Estimate processing time based on working file rows count (Phase 2 - Detailed Contract).

        Private helper method for time estimation using simple algorithm:
        estimated_time = rows_count * 0.1s (from implementation_plan.md)

        Process Flow:
            1. Get working file path from uploads storage (uploads/{wf_file_id}/)
            2. Extract metadata using file_storage.extract_file_metadata(file_path)
            3. Get rows_count from metadata dict (first sheet only)
            4. Calculate estimated_time = rows_count * 0.1 seconds
            5. Apply bounds: min=10s, max=300s (5 minutes - Celery time_limit)
            6. Round to nearest integer
            7. Log estimation (DEBUG level)
            8. Return estimated_time

        Args:
            wf_file_id: Working file UUID (file to be matched)
            ref_file_id: Reference file UUID (not used in Phase 2 estimation)

        Returns:
            int: Estimated processing time in seconds (10-300 range)

        Estimation Algorithm (Phase 2):
            Formula: estimated_time = rows_count * 0.1
            Rationale:
                - Matching is O(n*m) where n=WF rows, m=REF rows
                - But primary iteration is over WF (n)
                - 0.1s per row is simple heuristic
                - Includes overhead for: file I/O, parameter extraction, matching, writing results

        Bounds:
            - Minimum: 10 seconds (even for tiny files, file I/O takes time)
            - Maximum: 300 seconds (Celery time_limit from matching_tasks.py)

        Examples:
            >>> # Small file: 50 rows
            >>> time = await use_case._estimate_processing_time(
            ...     wf_file_id=UUID("a3bb189e..."),
            ...     ref_file_id=UUID("f47ac10b...")
            ... )
            >>> print(time)  # 10 (50 * 0.1 = 5s, but min=10s)

            >>> # Medium file: 500 rows
            >>> time = await use_case._estimate_processing_time(...)
            >>> print(time)  # 50 (500 * 0.1 = 50s)

            >>> # Large file: 1000 rows (Phase 2 max)
            >>> time = await use_case._estimate_processing_time(...)
            >>> print(time)  # 100 (1000 * 0.1 = 100s)

            >>> # Huge file: 5000 rows (Phase 3+)
            >>> time = await use_case._estimate_processing_time(...)
            >>> print(time)  # 300 (5000 * 0.1 = 500s, but max=300s)

        Implementation Note (Phase 3):
            import logging

            logger = logging.getLogger(__name__)

            # Get working file path from uploads storage
            wf_path = self.file_storage.get_upload_file_path(wf_file_id)

            # Extract metadata (includes rows_count from first sheet)
            metadata = await self.file_storage.extract_file_metadata(wf_path)
            rows_count = metadata['rows_count']

            # Calculate estimated time using algorithm: rows_count * 0.1s
            estimated_time_raw = rows_count * 0.1

            # Apply bounds (min=10s, max=300s)
            estimated_time = max(10, min(300, int(estimated_time_raw)))

            logger.debug(
                f"Estimated processing time: {estimated_time}s "
                f"(based on {rows_count} rows * 0.1s/row)"
            )

            return estimated_time

        Phase 3+ Extensions:
            - Include reference file size in estimation (currently not used)
            - Machine learning model based on historical job durations
            - Adjust for matching_strategy (ALL_MATCHES slower than FIRST_MATCH)
            - Adjust for file complexity (number of columns, data types)

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3 - Task 3.2.1.
        """
        # Get working file directory from upload storage
        wf_upload_dir = self.file_storage.get_uploaded_file_path(wf_file_id)

        # Find the uploaded file (should be only one .xlsx file in directory)
        uploaded_files = list(wf_upload_dir.glob("*.xlsx"))
        if not uploaded_files:
            raise FileNotFoundError(
                f"No .xlsx file found in upload directory: {wf_upload_dir}"
            )
        wf_path = uploaded_files[0]

        # Extract metadata (includes rows_count from first sheet)
        metadata = await self.file_storage.extract_file_metadata(wf_path)
        rows_count = metadata["rows_count"]

        # Calculate estimated time using algorithm: rows_count * 0.1s
        estimated_time_raw = rows_count * 0.1

        # Apply bounds (min=10s, max=300s)
        estimated_time = max(10, min(300, int(estimated_time_raw)))

        logger.debug(
            f"Estimated processing time: {estimated_time}s "
            f"(based on {rows_count} rows * 0.1s/row)"
        )

        return estimated_time
