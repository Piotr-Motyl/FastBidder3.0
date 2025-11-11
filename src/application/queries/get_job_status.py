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
        Retrieve job status from Redis.

        Process:
            1. Use job_id as Redis key
            2. Fetch status data from Redis
            3. Parse and validate data
            4. Convert to JobStatusResult
            5. Handle missing/expired jobs

        Args:
            query: GetJobStatusQuery with job_id

        Returns:
            JobStatusResult with current status and progress

        Raises:
            JobNotFoundException: If job_id not found or expired (TTL exceeded)

        Redis Key Format:
            Key: f"job:{job_id}"
            Value: JSON with status, progress, message, timestamps
            TTL: 3600 seconds (1 hour) for progress
                 86400 seconds (24 hours) for completed jobs

        CONTRACT ONLY - Implementation in Phase 3.
        """
        # Implementation in Phase 3 will:
        # 1. Check Redis for key f"job:{query.job_id}"
        # 2. Parse JSON value
        # 3. Return JobStatusResult
        # 4. Handle missing keys with JobNotFoundException

        raise NotImplementedError(
            "Implementation in Phase 3. Will fetch from Redis using job_id as key."
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
