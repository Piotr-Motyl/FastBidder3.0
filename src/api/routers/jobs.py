"""
API Router for Job Status Tracking

Responsibility:
    HTTP interface for querying status of asynchronous Celery jobs.
    Provides real-time progress updates for long-running operations.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Depends on Application Layer (GetJobStatusQuery - to be implemented in Task 1.1.2)
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
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Path, Depends
from pydantic import BaseModel, Field

# Import shared models from matching router
# In real implementation, these would be in a separate models.py file
from .matching import JobStatus, ErrorResponse


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class JobStatusResponse(BaseModel):
    """
    Response model for job status query.

    Provides detailed information about job progress and current state.
    Client should poll this endpoint every 2-5 seconds during processing.

    Attributes:
        job_id: Celery task ID
        status: Current job status (queued/processing/completed/failed/cancelled)
        progress: Completion percentage (0-100). Only present when status=processing.
        message: Human-readable status message with current step information
        result_ready: Flag indicating if results are available for download
        current_step: Description of current processing step (optional)
        error_details: Error information if status=failed (optional)
        created_at: ISO timestamp when job was created
        updated_at: ISO timestamp of last status update

    Examples:
        Processing:
        {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "processing",
            "progress": 45,
            "message": "Processing: Matching descriptions (45/100)",
            "result_ready": false,
            "current_step": "Parameter extraction",
            "created_at": "2025-10-07T10:30:00Z",
            "updated_at": "2025-10-07T10:30:45Z"
        }

        Completed:
        {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "completed",
            "progress": 100,
            "message": "Matching completed successfully. 87 matches found.",
            "result_ready": true,
            "created_at": "2025-10-07T10:30:00Z",
            "updated_at": "2025-10-07T10:31:30Z"
        }

        Failed:
        {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "failed",
            "progress": 35,
            "message": "Job failed during parameter extraction",
            "result_ready": false,
            "error_details": "Excel file missing required column 'Description'",
            "created_at": "2025-10-07T10:30:00Z",
            "updated_at": "2025-10-07T10:30:15Z"
        }

    Redis Storage:
        - Progress data TTL: 1 hour (frequent updates)
        - Completed job metadata TTL: 24 hours (for history)
        - Key pattern: job:{job_id}:status

    Polling Strategy:
        - Poll every 2-5 seconds while status is "queued" or "processing"
        - Stop polling when status is "completed", "failed", or "cancelled"
        - Show progress bar based on progress field (0-100)
        - Display current_step for detailed user feedback
    """

    job_id: UUID = Field(description="Celery task ID")
    status: JobStatus = Field(description="Current job status")
    progress: Optional[int] = Field(
        default=None,
        ge=0,
        le=100,
        description="Completion percentage (0-100). Only present during processing.",
    )
    message: str = Field(description="Human-readable status message with details")
    result_ready: bool = Field(
        default=False,
        description="Flag indicating if results are available for download at GET /results/{job_id}/download",
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Description of current processing step for detailed progress",
    )
    error_details: Optional[str] = Field(
        default=None,
        description="Detailed error information if status=failed (debug information)",
    )
    created_at: str = Field(description="ISO 8601 timestamp when job was created")
    updated_at: str = Field(
        description="ISO 8601 timestamp of last status update (for detecting stale jobs)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "processing",
                "progress": 45,
                "message": "Processing: Matching descriptions (45/100)",
                "result_ready": False,
                "current_step": "Parameter extraction",
                "error_details": None,
                "created_at": "2025-10-07T10:30:00Z",
                "updated_at": "2025-10-07T10:30:45Z",
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
# DEPENDENCY INJECTION PLACEHOLDERS
# ============================================================================


async def get_job_status_query():
    """
    Dependency injection for GetJobStatusQuery.

    Returns:
        GetJobStatusQuery: Application Layer query for retrieving job status

    Note:
        Implementation in Phase 2 (Task 1.1.2 - Application Layer contracts).
        Will inject actual query from Application Layer.

    Architecture Pattern:
        This follows CQRS pattern - read operations are separate from writes.
        The query will handle:
        - Retrieving job metadata from Redis (via RedisProgressTracker)
        - Formatting status data for API response
        - Handling expired jobs (TTL cleanup)
        - Validating job_id existence

    Infrastructure Dependencies:
        - RedisProgressTracker (Infrastructure Layer)
        - Job metadata stored with TTL (1h for progress, 24h for results)
    """
    # Implementation in Phase 3
    raise NotImplementedError(
        "GetJobStatusQuery not implemented yet - will be added in Task 1.1.2"
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
    query=Depends(get_job_status_query),
) -> JobStatusResponse:
    """
    Get current status and progress of asynchronous job.

    This endpoint queries Redis for job status and progress information.
    Used by frontend to display real-time progress during matching process.

    **Process Flow:**
    1. Validate job_id format (UUID)
    2. Create GetJobStatusQuery with job_id
    3. Delegate to Application Layer query (GetJobStatusQuery)
    4. Query retrieves status from Redis (via RedisProgressTracker)
    5. Return formatted status response

    **Status Lifecycle:**
    - queued: Job accepted, waiting in Celery queue (no progress data yet)
    - processing: Job being executed, progress 0-100%
    - completed: Job finished, results available for download
    - failed: Job failed, error details available
    - cancelled: Job cancelled by user or system timeout

    **Progress Tracking:**
    - Progress updates written to Redis by Celery worker every 10%
    - TTL: 1 hour for progress data (active jobs)
    - TTL: 24 hours for completed job metadata (history)
    - Key pattern: `job:{job_id}:status`

    **Polling Strategy:**
    Client should implement exponential backoff:
    - Initial poll: Immediately after job creation
    - While status="queued": Poll every 2 seconds
    - While status="processing": Poll every 3-5 seconds based on progress rate
    - Stop polling: When status is "completed", "failed", or "cancelled"

    **Job Expiration:**
    - Active jobs (queued/processing): TTL 1 hour
    - Completed jobs: TTL 24 hours
    - After TTL: Returns 404 Not Found

    Args:
        job_id: UUID of the Celery task from async endpoint (e.g., POST /matching/process)
        query: Injected GetJobStatusQuery from Application Layer

    Returns:
        JobStatusResponse with detailed status, progress, and metadata

    Raises:
        HTTPException 404: If job_id not found in Redis (expired or never existed)
        HTTPException 422: If job_id is not a valid UUID
        HTTPException 500: If Redis connection fails or unexpected error

    Example:
        >>> # Initial poll after job creation
        >>> response = await client.get(
        ...     "/api/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6/status"
        ... )
        >>> print(response.status_code)
        200
        >>> print(response.json())
        {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "processing",
            "progress": 45,
            "message": "Processing: Matching descriptions (45/100)",
            "result_ready": false,
            "current_step": "Parameter extraction",
            ...
        }

        >>> # Poll again after 3 seconds
        >>> response = await client.get(
        ...     "/api/jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6/status"
        ... )
        >>> data = response.json()
        >>> if data["status"] == "completed":
        ...     # Download results
        ...     result_response = await client.get(
        ...         f"/api/results/{job_id}/download"
        ...     )

    Architecture Note:
        This endpoint is read-only (CQRS Query pattern).
        Delegates to Application Layer GetJobStatusQuery.
        No direct Redis access - follows dependency inversion principle.
        Cross-cutting concern - works for any async job type.

    Frontend Integration:
        ```javascript
        // React polling example
        const pollJobStatus = async (jobId) => {
            const response = await fetch(`/api/jobs/${jobId}/status`);
            const data = await response.json();

            if (data.status === 'processing') {
                updateProgressBar(data.progress);
                displayMessage(data.current_step);
                setTimeout(() => pollJobStatus(jobId), 3000); // Poll every 3s
            } else if (data.status === 'completed') {
                downloadResults(jobId);
            } else if (data.status === 'failed') {
                showError(data.error_details);
            }
        };
        ```
    """
    # CONTRACT ONLY - Implementation in Phase 3
    #
    # Implementation will:
    # 1. Validate job_id format (already done by Pydantic)
    # 2. Create GetJobStatusQuery with job_id
    # 3. Call query.execute()
    # 4. Return JobStatusResponse with status data from Redis
    #
    # Example implementation:
    # try:
    #     query_obj = GetJobStatusQuery(job_id=job_id)
    #     result = await query.execute(query_obj)
    #     return result
    # except JobNotFoundException:
    #     raise HTTPException(
    #         status_code=404,
    #         detail=ErrorResponse(
    #             code="JOB_NOT_FOUND",
    #             message=f"Job with ID {job_id} not found or expired",
    #             details={"job_id": str(job_id), "ttl_hours": 24}
    #         )
    #     )
    # except RedisConnectionError:
    #     raise HTTPException(
    #         status_code=500,
    #         detail=ErrorResponse(
    #             code="REDIS_CONNECTION_ERROR",
    #             message="Failed to retrieve job status from Redis"
    #         )
    #     )

    raise NotImplementedError(
        "Implementation in Phase 3 - Task 3.4.1. "
        "This is a contract only (Phase 1 - Task 1.1.1)."
    )
