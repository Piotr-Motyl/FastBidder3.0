"""
API Router for Job Status Tracking

Responsibility:
    HTTP interface for querying status of asynchronous Celery jobs.
    Provides real-time progress updates for long-running operations.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Depends on Application Layer (GetJobStatusQueryHandler - Task 1.1.2)
    - Read-only operations (CQRS Query pattern)
    - No business logic - pure HTTP concerns
    - Cross-cutting concern (used by matching, upload, and future operations)

Contains:
    - GET /jobs/{job_id}/status - Query job progress and status

Does NOT contain:
    - Business logic (delegated to Domain Layer)
    - Direct Redis access (delegated to Infrastructure Layer via Application Layer)
    - Job creation (belongs to specific domain routers: matching.py, upload.py)
    - Result retrieval (separate router: results.py)

Phase 1 Note:
    This is a CONTRACT ONLY. Implementation will be added in Phase 3.
    All endpoints raise NotImplementedError.

CRITICAL FIX:
    Fixed dependency injection - now returns GetJobStatusQueryHandler (not Query).
    This was a critical architectural error in the original version.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Path, Depends
from pydantic import BaseModel, Field

# Import shared models - now from Application Layer (correct dependency direction)
from src.application.models import JobStatus


# ============================================================================
# RESPONSE MODEL (API Layer specific)
# ============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error response model for all API errors.

    Provides consistent error structure across all endpoints.

    Attributes:
        code: Machine-readable error code (e.g., "JOB_NOT_FOUND")
        message: Human-readable error message
        details: Optional additional error details
    """

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict] = Field(
        default=None, description="Additional error context"
    )


class JobStatusResponse(BaseModel):
    """
    Response model for job status query.

    This is API Layer's HTTP representation.
    Converted from Application Layer's JobStatusResult.

    Attributes:
        job_id: Celery task ID
        status: Current job status (enum)
        progress: Completion percentage (0-100)
        message: Human-readable status message
        result_ready: Flag indicating if results available
        current_step: Current processing step (optional)
        error_details: Error information if failed (optional)
        created_at: ISO timestamp when job created
        updated_at: ISO timestamp of last update
    """

    job_id: UUID = Field(description="Celery task ID")

    status: JobStatus = Field(description="Current job status")

    progress: Optional[int] = Field(
        default=None, ge=0, le=100, description="Completion percentage (0-100)"
    )

    message: str = Field(description="Human-readable status message with details")

    result_ready: bool = Field(
        default=False, description="Flag indicating if results available for download"
    )

    current_step: Optional[str] = Field(
        default=None, description="Current processing step"
    )

    error_details: Optional[str] = Field(
        default=None, description="Detailed error information if status=failed"
    )

    created_at: str = Field(description="ISO 8601 timestamp when job was created")

    updated_at: str = Field(description="ISO 8601 timestamp of last status update")

    class Config:
        """Pydantic configuration."""

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
# ROUTER CONFIGURATION
# ============================================================================


router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Not Found - Job ID not found or expired",
        },
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity - Invalid job ID format",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# ============================================================================
# DEPENDENCY INJECTION - CRITICAL FIX
# ============================================================================


async def get_job_status_query_handler():
    """
    Dependency injection for GetJobStatusQueryHandler.

    CRITICAL FIX:
        Original version returned GetJobStatusQuery (data holder).
        Corrected version returns GetJobStatusQueryHandler (executor).

    Returns:
        GetJobStatusQueryHandler: Application Layer query handler with dependencies

    Note:
        Implementation in Phase 3 (Task 3.4.1).
        Will inject actual handler with Redis dependencies.

    Architecture Pattern:
        API Layer → Query Handler → Infrastructure Service

        The handler is injected here with all its dependencies:
        - RedisProgressTracker from Infrastructure Layer

    Example implementation (Phase 3):
        from src.application.queries import GetJobStatusQueryHandler
        from src.infrastructure.persistence.redis import get_redis_progress_tracker

        redis_tracker = get_redis_progress_tracker()
        return GetJobStatusQueryHandler(redis_tracker)
    """
    # Implementation in Phase 3
    raise NotImplementedError(
        "GetJobStatusQueryHandler not implemented yet - will be added in Task 1.1.2/3.4.1"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get(
    "/{job_id}/status",
    status_code=status.HTTP_200_OK,
    response_model=JobStatusResponse,
    summary="Get status of asynchronous job",
    description=(
        "Retrieves current status and progress of an asynchronous job. "
        "Client should poll this endpoint every 2-5 seconds during processing. "
        "Works for any async job (matching, upload, embedding generation, etc.)."
    ),
    responses={
        200: {
            "description": "Success - Job status retrieved",
            "model": JobStatusResponse,
        },
        404: {
            "description": "Not Found - Job ID not found or expired (TTL exceeded)",
            "model": ErrorResponse,
        },
        422: {
            "description": "Unprocessable Entity - Invalid job ID format",
            "model": ErrorResponse,
        },
    },
)
async def get_job_status(
    job_id: UUID = Path(
        description="Celery task ID returned from POST /matching/process or other async endpoints"
    ),
    handler=Depends(get_job_status_query_handler),
) -> JobStatusResponse:
    """
    Get current status and progress of asynchronous job.

    CRITICAL FIX:
        Parameter renamed from 'query' to 'handler' to reflect correct architecture.
        The injected dependency is GetJobStatusQueryHandler, not GetJobStatusQuery.

    This endpoint queries Redis for job status and progress information.
    Used by frontend to display real-time progress during matching process.

    **Process Flow:**
    1. Validate job_id format (UUID)
    2. Create GetJobStatusQuery with job_id
    3. Delegate to Application Layer handler (GetJobStatusQueryHandler)
    4. Handler retrieves status from Redis (via RedisProgressTracker)
    5. Convert JobStatusResult to JobStatusResponse
    6. Return formatted status response

    **Status Lifecycle:**
    - queued: Job accepted, waiting in Celery queue
    - processing: Job being executed, progress 0-100%
    - completed: Job finished, results available for download
    - failed: Job failed, error details available
    - cancelled: Job cancelled by user or system timeout

    **Progress Tracking:**
    - Progress updates written to Redis by Celery worker every 10%
    - TTL: 1 hour for progress data (active jobs)
    - TTL: 24 hours for completed job metadata (history)
    - Key pattern: `job:{job_id}:status`

    Args:
        job_id: UUID of the Celery task from async endpoint
        handler: Injected GetJobStatusQueryHandler from Application Layer

    Returns:
        JobStatusResponse: HTTP response with job status and progress

    Raises:
        HTTPException 404: If job_id not found in Redis
        HTTPException 422: If job_id is not a valid UUID
        HTTPException 500: If Redis connection fails

    Example:
        >>> response = await client.get(
        ...     "/api/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6/status"
        ... )
        >>> print(response.json())
        {
            "job_id": "3fa85f64...",
            "status": "processing",
            "progress": 45,
            ...
        }

    Architecture Note:
        This endpoint is read-only (CQRS Query pattern).
        Delegates to Application Layer GetJobStatusQueryHandler.
        No direct Redis access - follows dependency inversion principle.
    """
    # CONTRACT ONLY - Implementation in Phase 3
    #
    # Implementation will:
    # 1. Create query object from path parameter
    # 2. Call handler.handle(query)
    # 3. Convert JobStatusResult to JobStatusResponse
    # 4. Return response
    #
    # Example implementation:
    # from src.application.queries import GetJobStatusQuery
    #
    # try:
    #     query = GetJobStatusQuery(job_id=job_id)
    #     result = await handler.handle(query)
    #
    #     # Convert Application Layer DTO to API Layer Response
    #     return JobStatusResponse(
    #         job_id=result.job_id,
    #         status=result.status,
    #         progress=result.progress,
    #         message=result.message,
    #         result_ready=result.result_ready,
    #         current_step=result.current_step,
    #         error_details=result.error_details,
    #         created_at=result.created_at,
    #         updated_at=result.updated_at
    #     )
    # except JobNotFoundException:
    #     raise HTTPException(
    #         status_code=404,
    #         detail=ErrorResponse(
    #             code="JOB_NOT_FOUND",
    #             message=f"Job with ID {job_id} not found or expired"
    #         ).dict()
    #     )

    raise NotImplementedError(
        "Implementation in Phase 3 - Task 3.4.1. "
        "This is a contract only (Phase 1 - Task 1.1.1)."
    )
