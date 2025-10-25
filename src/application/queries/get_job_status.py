"""
Get Job Status Query - CQRS Read Operation

Responsibility:
    Query for retrieving asynchronous job status and progress.
    Implements CQRS pattern for read operations.

Architecture Notes:
    - Part of Application Layer (CQRS Queries)
    - Query is immutable data holder
    - Handler contains execution logic
    - Read-only operation (no state modification)
    - Retrieves data from Redis (via Infrastructure Layer)

Contains:
    - GetJobStatusQuery: Query object with job_id
    - GetJobStatusQueryHandler: Executes query and returns result
    - JobStatusResult: Response DTO with job status data

Does NOT contain:
    - Direct Redis access (uses RedisProgressTracker from Infrastructure)
    - HTTP handling (belongs to API Layer)
    - Business logic (pure data retrieval)

Phase 1 Note:
    This is a CONTRACT ONLY. Implementation in Phase 3.
"""

from typing import Optional, Protocol
from uuid import UUID
from pydantic import BaseModel, Field

from src.application.models import JobStatus


# ============================================================================
# PROTOCOL (Interface for dependency injection)
# ============================================================================


class RedisProgressTrackerProtocol(Protocol):
    """
    Protocol (interface) for Redis progress tracker.

    Defines the contract that Infrastructure Layer must implement.
    This follows Dependency Inversion Principle from SOLID.

    Methods:
        get_status: Retrieve job status from Redis

    Note:
        Actual implementation will be in Infrastructure Layer:
        src/infrastructure/persistence/redis/progress_tracker.py
    """

    async def get_status(self, job_id: str) -> Optional[dict]:
        """
        Retrieve job status from Redis.

        Args:
            job_id: Celery task ID as string

        Returns:
            dict with status data or None if job not found
        """
        ...


# ============================================================================
# QUERY
# ============================================================================


class GetJobStatusQuery(BaseModel):
    """
    Query to retrieve status of asynchronous job.

    CQRS Pattern:
        This is a QUERY (read operation) that retrieves data without
        modifying system state. Pure data retrieval from Redis.

    Attributes:
        job_id: UUID of Celery task to query

    Example:
        >>> query = GetJobStatusQuery(
        ...     job_id=UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6")
        ... )
        >>> handler = GetJobStatusQueryHandler(redis_tracker)
        >>> result = await handler.handle(query)

    Immutability:
        Query objects are immutable data holders.
    """

    job_id: UUID = Field(description="Celery task ID to query status for")

    class Config:
        """Pydantic configuration for GetJobStatusQuery."""

        json_schema_extra = {
            "example": {"job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"}
        }


# ============================================================================
# RESULT (Response DTO)
# ============================================================================


class JobStatusResult(BaseModel):
    """
    Result DTO for job status query.

    Contains all information about job progress and current state.
    This is Application Layer's representation, API Layer may convert
    to JobStatusResponse for HTTP response.

    Attributes:
        job_id: Celery task ID
        status: Current job status (queued/processing/completed/failed/cancelled)
        progress: Completion percentage (0-100), None if not processing
        message: Human-readable status message
        result_ready: Flag indicating if results are available for download
        current_step: Current processing step description (optional)
        error_details: Error information if status=failed (optional)
        created_at: ISO timestamp when job was created
        updated_at: ISO timestamp of last status update

    Redis Storage:
        - Key pattern: job:{job_id}:status
        - TTL: 1 hour for active jobs, 24 hours for completed

    Example:
        >>> result = JobStatusResult(
        ...     job_id=UUID("3fa85f64..."),
        ...     status=JobStatus.PROCESSING,
        ...     progress=45,
        ...     message="Matching descriptions (45/100)",
        ...     result_ready=False,
        ...     current_step="Parameter extraction",
        ...     created_at="2025-10-11T10:30:00Z",
        ...     updated_at="2025-10-11T10:30:45Z"
        ... )
    """

    job_id: UUID = Field(description="Celery task ID")

    status: JobStatus = Field(
        description="Current job status (queued/processing/completed/failed/cancelled)"
    )

    progress: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Completion percentage (0-100), None if not processing",
    )

    message: str = Field(description="Human-readable status message with details")

    result_ready: bool = Field(
        default=False,
        description="Flag indicating if results are available for download",
    )

    current_step: Optional[str] = Field(
        default=None, description="Current processing step for detailed progress"
    )

    error_details: Optional[str] = Field(
        default=None, description="Detailed error information if status=failed"
    )

    created_at: str = Field(description="ISO 8601 timestamp when job was created")

    updated_at: str = Field(description="ISO 8601 timestamp of last status update")

    class Config:
        """Pydantic configuration for JobStatusResult."""

        json_schema_extra = {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "processing",
                "progress": 45,
                "message": "Processing: Matching descriptions (45/100)",
                "result_ready": False,
                "current_step": "Parameter extraction",
                "error_details": None,
                "created_at": "2025-10-11T10:30:00Z",
                "updated_at": "2025-10-11T10:30:45Z",
            }
        }


# ============================================================================
# QUERY HANDLER
# ============================================================================


class GetJobStatusQueryHandler:
    """
    Handles execution of GetJobStatusQuery.

    Responsibility:
        Retrieves job status from Redis and formats response.
        Implements CQRS pattern - separates query (data) from handler (logic).

    Architecture Pattern:
        - Handler receives query object
        - Uses Infrastructure Layer services (RedisProgressTracker)
        - Returns Application Layer DTO (JobStatusResult)
        - No business logic - pure data retrieval and formatting

    Dependencies:
        - redis_progress_tracker: RedisProgressTracker from Infrastructure Layer

    Flow:
        1. Receive GetJobStatusQuery with job_id
        2. Query Redis for job metadata (via RedisProgressTracker)
        3. Format data into JobStatusResult
        4. Return result to caller (API Layer)

    Example:
        >>> # In API Layer dependency injection:
        >>> def get_job_status_query_handler():
        ...     redis_tracker = get_redis_progress_tracker()
        ...     return GetJobStatusQueryHandler(redis_tracker)
        >>>
        >>> # Usage in API endpoint:
        >>> handler = Depends(get_job_status_query_handler)
        >>> query = GetJobStatusQuery(job_id=job_id)
        >>> result = await handler.handle(query)
    """

    def __init__(self, redis_progress_tracker: RedisProgressTrackerProtocol):
        """
        Initialize query handler with dependencies.

        Args:
            redis_progress_tracker: RedisProgressTracker from Infrastructure Layer
                                   (to be implemented in Task 1.1.4)

        Architecture Note:
            Constructor injection follows Clean Architecture.
            Dependencies are abstractions (Protocol), not concrete implementations.
            This enables easy testing and follows Dependency Inversion Principle.
        """
        self.redis_progress_tracker = redis_progress_tracker

    async def handle(self, query: GetJobStatusQuery) -> JobStatusResult:
        """
        Execute query and return job status.

        Retrieves job status from Redis and formats into JobStatusResult.

        Args:
            query: GetJobStatusQuery with job_id to query

        Returns:
            JobStatusResult: Formatted job status data

        Raises:
            JobNotFoundException: If job_id not found in Redis (expired or never existed)
            RedisConnectionError: If Redis connection fails

        Process Flow:
            1. Validate query (Pydantic already did this)
            2. Query Redis for job:{job_id}:status key
            3. Parse JSON data from Redis
            4. Format into JobStatusResult DTO
            5. Return result

        Redis Data Structure:
            Key: job:{job_id}:status
            Value: JSON with fields:
                - status: str
                - progress: int (optional)
                - message: str
                - current_step: str (optional)
                - error_details: str (optional)
                - created_at: str (ISO timestamp)
                - updated_at: str (ISO timestamp)
            TTL: 1h for active, 24h for completed

        Example:
            >>> query = GetJobStatusQuery(job_id=UUID("3fa85f64..."))
            >>> result = await handler.handle(query)
            >>> print(result.status)  # JobStatus.PROCESSING
            >>> print(result.progress)  # 45

        Note:
            Implementation in Phase 3 - Task 3.4.1.
            This is a contract showing expected behavior.
        """
        # CONTRACT ONLY - Implementation in Phase 3
        #
        # Implementation will:
        # 1. Call redis_progress_tracker.get_status(str(query.job_id))
        # 2. Check if data exists (raise JobNotFoundException if not)
        # 3. Parse JSON data from Redis
        # 4. Convert status string to JobStatus enum
        # 5. Create JobStatusResult from parsed data
        # 6. Return result
        #
        # Example implementation:
        # try:
        #     status_data = await self.redis_progress_tracker.get_status(
        #         job_id=str(query.job_id)
        #     )
        #     if not status_data:
        #         raise JobNotFoundException(
        #             f"Job {query.job_id} not found or expired"
        #         )
        #
        #     return JobStatusResult(
        #         job_id=query.job_id,
        #         status=JobStatus(status_data["status"]),  # Convert str to enum
        #         progress=status_data.get("progress"),
        #         message=status_data["message"],
        #         result_ready=status_data.get("result_ready", False),
        #         current_step=status_data.get("current_step"),
        #         error_details=status_data.get("error_details"),
        #         created_at=status_data["created_at"],
        #         updated_at=status_data["updated_at"]
        #     )
        # except RedisError as e:
        #     raise RedisConnectionError(f"Failed to query Redis: {e}")

        raise NotImplementedError(
            "Implementation in Phase 3 - Task 3.4.1. "
            "This is a contract only (Phase 1 - Task 1.1.2)."
        )
