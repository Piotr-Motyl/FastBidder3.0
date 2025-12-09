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

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Path, Depends
from pydantic import BaseModel, Field

# Import shared models - now from Application Layer (correct dependency direction)
from src.application.queries.get_job_status import (
    GetJobStatusQueryHandler,
    GetJobStatusQuery,
    JobNotFoundException,
)
from src.application.models import JobStatus
from src.infrastructure.persistence.redis.progress_tracker import RedisProgressTracker

# Import shared API schemas
from src.api.schemas.common import ErrorResponse

# Configure logger
logger = logging.getLogger(__name__)


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

    progress: int = Field(
        default=0, ge=0, le=100, description="Completion percentage (0-100)"
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

    created_at: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp when job was created"
    )

    updated_at: Optional[str] = Field(
        default=None, description="ISO 8601 timestamp of last status update"
    )

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
    # Create RedisProgressTracker instance
    redis_tracker = RedisProgressTracker()

    # Create and return GetJobStatusQueryHandler with injected dependencies
    return GetJobStatusQueryHandler(progress_tracker=redis_tracker)


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
        "Works for any async job (matching, upload, embedding generation, etc.). "
        "Response includes Cache-Control: no-cache for real-time updates."
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
        500: {
            "description": "Internal Server Error - Redis connection failure",
            "model": ErrorResponse,
        },
    },
)
async def get_job_status(
    job_id: UUID = Path(..., description="Celery task ID returned from async endpoint"),
    handler: GetJobStatusQueryHandler = Depends(get_job_status_query_handler),
) -> JobStatusResponse:
    """
    Get current status and progress of asynchronous job (Phase 2 - Detailed Contract).

    This endpoint is a thin wrapper around Application Layer query handler.
    All business logic is delegated to GetJobStatusQueryHandler following Clean Architecture.

    CRITICAL FIX:
        Parameter renamed from 'query' to 'handler' to reflect correct architecture.
        The injected dependency is GetJobStatusQueryHandler, not GetJobStatusQuery.

    This endpoint queries Redis for job status and progress information.
    Used by frontend to display real-time progress during matching process.

    Process Flow (10 steps):
        1. Receive job_id from URL path parameter (FastAPI validates UUID format)
        2. Create GetJobStatusQuery object with job_id
        3. Delegate to handler.handle(query) in Application Layer
        4. Handler calls RedisProgressTracker.get_status(job_id) in Infrastructure Layer
        5. Handler converts Redis progress_data to JobStatusResult DTO
        6. Convert JobStatusResult to JobStatusResponse (API Layer model)
        7. Set Cache-Control: no-cache header for real-time updates
        8. Return HTTP 200 OK with JobStatusResponse
        9. Handle JobNotFoundException → HTTP 404 Not Found
        10. Handle unexpected errors → HTTP 500 Internal Server Error

    Args:
        job_id: UUID of the Celery task from async endpoint (e.g., from POST /matching/process)
        handler: Injected GetJobStatusQueryHandler from Application Layer

    Returns:
        JobStatusResponse: HTTP response with job status and progress

    Raises:
        HTTPException 404: If job_id not found in Redis (expired or never existed)
        HTTPException 422: If job_id is not a valid UUID format (handled by FastAPI automatically)
        HTTPException 500: If Redis connection fails or unexpected error

    Status Lifecycle:
        - QUEUED: Job accepted and waiting in Celery queue (progress=0%)
        - PROCESSING: Job being executed by Celery worker (progress=1-99%)
        - COMPLETED: Job finished successfully, results available for download (progress=100%)
        - FAILED: Job failed with error details available in error_details field
        - CANCELLED: Job cancelled by user or system timeout

    Progress Tracking:
        - Progress updates written to Redis by Celery worker every 10% or 100 records (whichever first)
        - TTL: 1 hour (3600s) for active progress data
        - TTL: 24 hours (86400s) for completed job metadata
        - Key pattern: "progress:{job_id}"
        - Heartbeat: Updated every 30s to show task is alive

    Polling Recommendations:
        - Poll every 2-5 seconds during PROCESSING status
        - Stop polling when status is COMPLETED, FAILED, or CANCELLED
        - Use exponential backoff for long-running jobs (2s → 5s → 10s)
        - Set max polling timeout (e.g., 5 minutes for UI responsiveness)

    Error Handling (Phase 2 - Minimal):
        JobNotFoundException → 404 Not Found (job expired or never existed)
        ValueError → 500 Internal Server Error (invalid status value from Redis)
        Exception → 500 Internal Server Error (catch-all for unexpected errors)

    Architecture Note:
        - API Layer responsibility: HTTP concerns only (status codes, error mapping, response headers)
        - Application Layer responsibility: Business logic (query handling, data conversion)
        - Infrastructure Layer responsibility: Redis operations (data retrieval)
        - No direct Redis access - follows dependency inversion principle
        - Read-only operation (CQRS Query pattern)

    Phase 3+ Extensions (NOT in Phase 2):
        - WebSocket alternative: GET /api/jobs/{job_id}/stream for real-time push updates
        - Long polling: Support ?wait=30 query parameter to wait up to 30s for status change
        - Rate limiting: Max 60 requests per minute per job_id (prevent polling abuse)
        - Cache-Control header: Set programmatically with Response object
        - Progress history: Include last 10 progress updates in response for debugging
        - Partial results: If FAILED, include how much was processed before error
        - Stack trace: Include full error stack trace only in DEBUG mode
        - ETA calculation: Show estimated time remaining based on current progress rate

    Examples:
        >>> # Example 1: Job in QUEUED status (just started)
        >>> curl -X GET "http://localhost:8000/api/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6/status"
        {
          "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
          "status": "queued",
          "progress": 0,
          "message": "Job queued, waiting for worker",
          "result_ready": false,
          "current_step": "START",
          "error_details": null,
          "created_at": null,
          "updated_at": "2025-01-11T10:30:00.000Z"
        }

        >>> # Example 2: Job in PROCESSING status (45% complete)
        >>> curl -X GET "http://localhost:8000/api/jobs/abc-123-def-456/status"
        {
          "job_id": "abc-123-def-456",
          "status": "processing",
          "progress": 45,
          "message": "Matching descriptions (450/1000)",
          "result_ready": false,
          "current_step": "MATCHING",
          "error_details": null,
          "created_at": null,
          "updated_at": "2025-01-11T10:32:15.500Z"
        }

        >>> # Example 3: Job in COMPLETED status (100% done)
        >>> curl -X GET "http://localhost:8000/api/jobs/xyz-789-uvw-012/status"
        {
          "job_id": "xyz-789-uvw-012",
          "status": "completed",
          "progress": 100,
          "message": "Matching completed successfully. 950 matches found.",
          "result_ready": true,
          "current_step": "COMPLETE",
          "error_details": null,
          "created_at": null,
          "updated_at": "2025-01-11T10:35:00.123Z"
        }

        >>> # Example 4: Job in FAILED status (with errors)
        >>> curl -X GET "http://localhost:8000/api/jobs/err-404-not-found/status"
        {
          "job_id": "err-404-not-found",
          "status": "failed",
          "progress": 30,
          "message": "Job failed: File not found",
          "result_ready": false,
          "current_step": "FILES_LOADED",
          "error_details": "File not found: working_file.xlsx\\nInvalid Excel format",
          "created_at": null,
          "updated_at": "2025-01-11T10:31:45.999Z"
        }

        >>> # Example 5: Using Python requests
        >>> import requests
        >>> import time
        >>>
        >>> job_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        >>> url = f"http://localhost:8000/api/jobs/{job_id}/status"
        >>>
        >>> # Poll until completed
        >>> while True:
        ...     response = requests.get(url)
        ...     data = response.json()
        ...
        ...     print(f"Status: {data['status']}, Progress: {data['progress']}%, Step: {data['current_step']}")
        ...
        ...     if data['status'] in ['completed', 'failed', 'cancelled']:
        ...         break
        ...
        ...     time.sleep(2)  # Poll every 2 seconds
        Status: processing, Progress: 10%, Step: FILES_LOADED
        Status: processing, Progress: 45%, Step: MATCHING
        Status: processing, Progress: 90%, Step: SAVING_RESULTS
        Status: completed, Progress: 100%, Step: COMPLETE

        >>> # Example 6: Job not found (404 error)
        >>> curl -X GET "http://localhost:8000/api/jobs/00000000-0000-0000-0000-000000000000/status"
        {
          "code": "JOB_NOT_FOUND",
          "message": "Job with ID 00000000-0000-0000-0000-000000000000 not found or expired",
          "details": {"job_id": "00000000-0000-0000-0000-000000000000"}
        }

    Implementation Note (Phase 3):
        import logging
        from src.application.queries.get_job_status import GetJobStatusQuery, JobNotFoundException

        logger = logging.getLogger(__name__)

        try:
            # Step 2: Create query object from path parameter
            query = GetJobStatusQuery(job_id=job_id)
            logger.debug(f"Querying status for job: {job_id}")

            # Step 3-5: Execute query handler
            result = await handler.handle(query)
            logger.info(f"Job {job_id} status retrieved: {result.status} ({result.progress}%)")

            # Step 6: Convert Application Layer DTO to API Layer Response
            response = JobStatusResponse(
                job_id=result.job_id,
                status=result.status,  # Already JobStatus enum value
                progress=result.progress,
                message=result.message,
                result_ready=result.result_ready,
                current_step=result.current_step,
                error_details=result.error_details,
                created_at=result.created_at,
                updated_at=result.updated_at
            )

            # Step 7: Cache-Control header (Phase 3 - set via Response object)
            # response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            # response.headers["Pragma"] = "no-cache"
            # response.headers["Expires"] = "0"

            # Step 8: Return response
            return response

        except JobNotFoundException as e:
            # Step 9: Job not found → 404
            logger.warning(f"Job not found: {job_id}")
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    code="JOB_NOT_FOUND",
                    message=f"Job with ID {job_id} not found or expired",
                    details={"job_id": str(job_id)}
                ).dict()
            )

        except ValueError as e:
            # Invalid status value from Redis (shouldn't happen in happy path)
            logger.error(f"Invalid status value for job {job_id}: {e}")
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    code="INVALID_STATUS",
                    message="Invalid job status value in storage",
                    details={"job_id": str(job_id), "error": str(e)}
                ).dict()
            )

        except Exception as e:
            # Step 10: Unexpected error → 500
            logger.error(f"Unexpected error retrieving job status for {job_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred while retrieving job status",
                    details={"job_id": str(job_id), "error": str(e)}
                ).dict()
            )

    Phase 2 Contract:
        This method defines the interface contract with detailed documentation.
        Actual implementation will be added in Phase 3 - Task 3.4.1.
    """
    # Implementation based on Phase 2 contract
    try:
        # Step 2: Create query object from path parameter
        query = GetJobStatusQuery(job_id=job_id)
        logger.debug(f"Querying status for job: {job_id}")

        # Step 3-5: Execute query handler
        result = await handler.handle(query)
        logger.info(f"Job {job_id} status retrieved: {result.status} ({result.progress}%)")

        # Step 6: Convert Application Layer DTO to API Layer Response
        response = JobStatusResponse(
            job_id=result.job_id,
            status=result.status,  # Already JobStatus enum value
            progress=result.progress,
            message=result.message,
            result_ready=result.result_ready,
            current_step=result.current_step,
            error_details=result.error_details,
            created_at=result.created_at,
            updated_at=result.updated_at,
        )

        # Step 7: Cache-Control header (Phase 3 - set via Response object)
        # Note: In FastAPI, headers are typically set using Response parameter
        # For now, we return the response directly. Cache headers can be added later.

        # Step 8: Return response
        return response

    except JobNotFoundException as e:
        # Step 9: Job not found → 404
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(
            status_code=404,
            detail={
                "code": "JOB_NOT_FOUND",
                "message": f"Job with ID {job_id} not found or expired",
                "details": {"job_id": str(job_id)},
            },
        )

    except ValueError as e:
        # Invalid status value from Redis (shouldn't happen in happy path)
        logger.error(f"Invalid status value for job {job_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INVALID_STATUS",
                "message": "Invalid job status value in storage",
                "details": {"job_id": str(job_id), "error": str(e)},
            },
        )

    except Exception as e:
        # Step 10: Unexpected error → 500
        logger.error(
            f"Unexpected error retrieving job status for {job_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred while retrieving job status",
                "details": {"job_id": str(job_id), "error": str(e)},
            },
        )
