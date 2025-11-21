"""
GetJobStatusQuery - CQRS Read Query

Query object and handler for retrieving job status from Redis.
Part of CQRS pattern - separates read operations from write operations.

Responsibility:
    - Query: Data holder with job_id to query
    - Handler: Executes query against Redis to get status

Architecture Notes:
    - Part of Application Layer (orchestration)
    - Query is simple DTO (Data Transfer Object)
    - Handler contains logic to fetch from Infrastructure
    - Follows CQRS pattern for read operations
"""

from typing import Optional, Dict, Any
from uuid import UUID
from pydantic import BaseModel, Field


class GetJobStatusQuery(BaseModel):
    """
    Query object containing job ID to retrieve status for.

    Simple data holder following CQRS Query pattern.
    Immutable once created.

    Attributes:
        job_id: Celery task ID to query status for
    """

    job_id: UUID = Field(description="Celery task ID returned from matching process")


class JobStatusResult(BaseModel):
    """
    Result DTO returned by GetJobStatusQueryHandler.

    Contains all job status information from Redis.
    This is Application Layer DTO, converted to API response model later.

    Attributes:
        job_id: Original task ID
        status: Current status (queued, processing, completed, failed)
        progress: Percentage complete (0-100)
        message: Human-readable status message
        result_ready: Whether results can be downloaded
        current_step: Current processing step (optional)
        error_details: Error information if failed (optional)
        created_at: When job was created (optional)
        updated_at: Last status update time (optional)
    """

    job_id: UUID
    status: str  # JobStatus enum value as string
    progress: int = Field(ge=0, le=100)
    message: str
    result_ready: bool = False
    current_step: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None  # ISO datetime string
    updated_at: Optional[str] = None  # ISO datetime string


class GetJobStatusQueryHandler:
    """
    Handler for retrieving job status from Redis.

    CONTRACT ONLY - Implementation in Phase 3.
    This handler will be injected with RedisProgressTracker dependency.

    Architecture:
        API Layer → QueryHandler → Infrastructure (Redis)

    Dependencies (Phase 3):
        - RedisProgressTracker: For fetching job progress from Redis
        - Optional: JobRepository for detailed job info from database

    Responsibility:
        - Fetch job status from Redis by key
        - Convert Redis data to JobStatusResult
        - Handle missing jobs (expired or not found)

    Usage:
        handler = GetJobStatusQueryHandler(redis_tracker)
        result = await handler.handle(query)
    """

    def __init__(self, progress_tracker=None):
        """
        Initialize with dependencies.

        Args:
            progress_tracker: RedisProgressTracker instance (injected)

        Note:
            Phase 3 will inject actual RedisProgressTracker.
            For now, accepts None for contract phase.
        """
        self.progress_tracker = progress_tracker

    async def handle(self, query: GetJobStatusQuery) -> JobStatusResult:
        """
        Retrieve job status from Redis (Phase 2 - Detailed Contract).

        Main handler method that orchestrates job status retrieval from Redis.
        Delegates to RedisProgressTracker and converts infrastructure data to
        Application Layer DTO.

        Process Flow (8 steps):
            1. Extract job_id from query object
            2. Call progress_tracker.get_status(job_id) to fetch from Redis
            3. Check if status data exists (raise JobNotFoundException if None)
            4. Extract status string from progress_data and convert to JobStatus enum
            5. Map Redis progress_data fields to JobStatusResult fields
            6. Calculate result_ready flag (status == "completed")
            7. Format error_details from errors list (join with newlines)
            8. Return JobStatusResult with all mapped fields

        Args:
            query: GetJobStatusQuery with job_id

        Returns:
            JobStatusResult with current status and progress

        Raises:
            JobNotFoundException: If job_id not found or expired (TTL exceeded)

        Redis Data Structure (from RedisProgressTracker.get_status()):
            Key: "progress:{job_id}"
            Value: JSON dict with structure:
            {
                "status": "processing",              # str: queued/processing/completed/failed/cancelled
                "progress": 45,                      # int: 0-100
                "message": "Matching descriptions",  # str: human-readable current step
                "current_item": 450,                 # int: current record being processed
                "total_items": 1000,                 # int: total records to process
                "stage": "MATCHING",                 # str: stage name (START, MATCHING, etc.)
                "eta_seconds": 120,                  # int: estimated time to completion
                "memory_mb": 512.5,                  # float: memory usage
                "errors": [],                        # list[str]: error messages
                "last_heartbeat": "2025-01-11T10:30:45.123"  # str: ISO timestamp
            }

        Field Mapping (Redis progress_data → JobStatusResult):
            - job_id: query.job_id (from input, not from Redis)
            - status: JobStatus(progress_data["status"])  # Convert string to enum
            - progress: progress_data["progress"]  # Direct copy (0-100)
            - message: progress_data["message"]  # Direct copy
            - result_ready: (progress_data["status"] == "completed")  # Boolean flag
            - current_step: progress_data["stage"]  # Stage name (e.g., "MATCHING")
            - error_details: "\n".join(progress_data["errors"]) if errors else None
            - created_at: None  # Phase 2: Not tracked yet, will be added in Phase 3
            - updated_at: progress_data["last_heartbeat"]  # ISO timestamp string

        TTL Configuration:
            - Progress data: 3600 seconds (1 hour) - auto-expire after 1h
            - Completed jobs: 86400 seconds (24 hours) - kept longer for history

        Examples:
            >>> # Processing job
            >>> query = GetJobStatusQuery(job_id=UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"))
            >>> result = await handler.handle(query)
            >>> print(result.status)  # JobStatus.PROCESSING
            >>> print(result.progress)  # 45
            >>> print(result.current_step)  # "MATCHING"
            >>> print(result.result_ready)  # False

            >>> # Completed job
            >>> query = GetJobStatusQuery(job_id=UUID("abc-123..."))
            >>> result = await handler.handle(query)
            >>> print(result.status)  # JobStatus.COMPLETED
            >>> print(result.progress)  # 100
            >>> print(result.result_ready)  # True

            >>> # Failed job with errors
            >>> query = GetJobStatusQuery(job_id=UUID("def-456..."))
            >>> result = await handler.handle(query)
            >>> print(result.status)  # JobStatus.FAILED
            >>> print(result.error_details)  # "File not found: working.xlsx\nInvalid format"

            >>> # Job not found (expired or never existed)
            >>> query = GetJobStatusQuery(job_id=UUID("xyz-789..."))
            >>> result = await handler.handle(query)  # Raises JobNotFoundException

        Implementation Note (Phase 3):
            import logging
            from src.application.models import JobStatus

            logger = logging.getLogger(__name__)

            # Step 1: Extract job_id
            job_id_str = str(query.job_id)
            logger.debug(f"Retrieving status for job: {job_id_str}")

            # Step 2: Fetch from Redis via progress_tracker
            progress_data = self.progress_tracker.get_status(job_id_str)

            # Step 3: Check if exists
            if not progress_data:
                logger.warning(f"Job not found in Redis: {job_id_str}")
                raise JobNotFoundException(query.job_id)

            logger.debug(f"Retrieved progress data: status={progress_data['status']}, progress={progress_data['progress']}%")

            # Step 4: Convert status string to enum
            try:
                status_enum = JobStatus(progress_data["status"])
            except ValueError as e:
                logger.error(f"Invalid status value from Redis: {progress_data['status']}")
                raise ValueError(f"Invalid status in Redis: {progress_data['status']}")

            # Step 5-7: Map fields
            result_ready = (progress_data["status"] == "completed")
            error_details = None
            if progress_data.get("errors"):
                error_details = "\n".join(progress_data["errors"])

            # Step 8: Create and return result
            result = JobStatusResult(
                job_id=query.job_id,
                status=status_enum.value,  # Convert enum to string for Pydantic
                progress=progress_data["progress"],
                message=progress_data["message"],
                result_ready=result_ready,
                current_step=progress_data.get("stage"),  # May be None in early Phase
                error_details=error_details,
                created_at=None,  # Phase 2: Not tracked yet
                updated_at=progress_data.get("last_heartbeat")
            )

            logger.info(f"Job {job_id_str} status retrieved: {status_enum.value} ({progress_data['progress']}%)")
            return result

        Error Handling (Phase 2 - Minimal):
            - JobNotFoundException: When progress_data is None (job not found/expired)
            - ValueError: When status value from Redis is invalid (shouldn't happen in happy path)
            - Exception: Catch-all for unexpected errors (Redis connection, etc.)

        Phase 3+ Extensions (NOT in Phase 2):
            - created_at tracking: Store job creation timestamp in separate Redis key
            - Metadata enrichment: Include worker info, queue position, estimated start
            - History access: Return last 10 progress updates for debugging
            - Cache layer: Add in-memory cache for frequently polled jobs
            - Fallback: If Redis fails, try fallback file storage

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3 - Task 3.4.1.
        """
        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.4.1. "
            "This is a detailed contract (Phase 2 - Task 2.4.3)."
        )


class JobNotFoundException(Exception):
    """
    Raised when job_id not found in Redis.

    Can happen when:
        - Job never existed
        - Job expired (TTL exceeded)
        - Redis was restarted without persistence
    """

    def __init__(self, job_id: UUID):
        self.job_id = job_id
        super().__init__(f"Job {job_id} not found or expired")
