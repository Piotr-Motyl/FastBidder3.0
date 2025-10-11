"""
API Router for HVAC Matching Process

Responsibility:
    HTTP interface for triggering asynchronous matching process.
    Thin layer that delegates to Application Layer use cases via dependency injection.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Depends on Application Layer (ProcessMatchingUseCase - to be implemented in Task 1.1.2)
    - Returns 202 Accepted for async operations (REST best practice)
    - No business logic - pure HTTP concerns
    - Uses dependency injection pattern (NOT direct Celery calls!)

Contains:
    - POST /matching/process - Trigger async matching job

Does NOT contain:
    - Business logic (delegated to Domain Layer)
    - File processing (delegated to Infrastructure Layer)
    - Celery task execution (delegated to Application Layer)
    - Direct database access (delegated to Infrastructure Layer)
    - Job status tracking (separate router: jobs.py)

Phase 1 Note:
    This is a CONTRACT ONLY. Implementation will be added in Phase 3.
    All endpoints raise NotImplementedError.
"""

from typing import Optional, Dict, Any
from uuid import UUID
from enum import Enum

from fastapi import APIRouter, status, HTTPException, Depends
from pydantic import BaseModel, Field


# ============================================================================
# ENUMS
# ============================================================================


class JobStatus(str, Enum):
    """
    Status of asynchronous matching job.

    Represents the lifecycle of a Celery task from queue to completion.
    Used across multiple routers (matching.py, jobs.py).

    Attributes:
        QUEUED: Job accepted and waiting in Celery queue
        PROCESSING: Job currently being executed by Celery worker
        COMPLETED: Job finished successfully with results available
        FAILED: Job failed with error details available
        CANCELLED: Job cancelled by user or system
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class ProcessMatchingRequest(BaseModel):
    """
    Request model for triggering matching process.

    Attributes:
        wf_file_id: UUID of uploaded working file (Excel with descriptions to match)
        ref_file_id: UUID of uploaded reference file (Excel with products and prices)
        threshold: Similarity threshold percentage (0.0-100.0).
                   Only matches above this threshold will be included in results.
                   Default: 75.0 (from .env DEFAULT_THRESHOLD)

    Example:
        {
            "wf_file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
            "ref_file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "threshold": 80.0
        }

    Validation:
        - wf_file_id and ref_file_id must be valid UUIDs
        - wf_file_id != ref_file_id (cannot match file against itself)
        - threshold must be between 0.0 and 100.0 (inclusive)

    Business Rules (validated by Application Layer):
        - Both files must exist in storage
        - Files must have valid Excel format (.xlsx, .xls)
        - Working file must contain descriptions column
        - Reference file must contain descriptions and prices columns
    """

    wf_file_id: UUID = Field(description="UUID of working file uploaded to system")
    ref_file_id: UUID = Field(description="UUID of reference file uploaded to system")
    threshold: float = Field(
        default=75.0,
        ge=0.0,
        le=100.0,
        description="Similarity threshold percentage. Matches below this value will be filtered out.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "wf_file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                "ref_file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "threshold": 80.0,
            }
        }


class ProcessMatchingResponse(BaseModel):
    """
    Response model for successfully triggered matching process.

    Returned with HTTP 202 Accepted to indicate async job has been queued.
    Client should poll GET /jobs/{job_id}/status to track progress.

    Attributes:
        job_id: Unique identifier for the Celery task (can be used to query status)
        status: Current job status (always "queued" in immediate response)
        estimated_time: Estimated time to completion in seconds (rough estimate based on file size)
        message: Human-readable message about job status

    Example:
        {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "queued",
            "estimated_time": 45,
            "message": "Matching job queued successfully. Use job_id to check status at GET /jobs/{job_id}/status"
        }
    """

    job_id: UUID = Field(description="Celery task ID for tracking job progress")
    status: JobStatus = Field(
        default=JobStatus.QUEUED, description="Current status of the job"
    )
    estimated_time: int = Field(
        ge=0,
        description="Estimated time to completion in seconds (based on file size and historical data)",
    )
    message: str = Field(
        default="Matching job queued successfully. Use job_id to check status.",
        description="Human-readable status message",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "queued",
                "estimated_time": 45,
                "message": "Matching job queued successfully. Check status at GET /jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6/status",
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response model for all API errors.

    Provides consistent error structure across all endpoints.

    Attributes:
        code: Machine-readable error code (e.g., "INVALID_FILE_ID", "FILE_NOT_FOUND")
        message: Human-readable error message
        details: Optional additional error details (validation errors, stack trace in dev mode)

    Example:
        {
            "code": "FILE_NOT_FOUND",
            "message": "Working file with ID a3bb189e-8bf9-3888-9912-ace4e6543002 not found",
            "details": {
                "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                "checked_locations": ["/tmp/fastbidder/uploads"]
            }
        }
    """

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error context (validation errors, debug info)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "code": "FILE_NOT_FOUND",
                "message": "Working file not found in storage",
                "details": {"file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002"},
            }
        }


# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================


router = APIRouter(
    prefix="/matching",
    tags=["matching"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Bad Request - Invalid input parameters",
        },
        404: {"model": ErrorResponse, "description": "Not Found - Resource not found"},
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity - Validation error",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# ============================================================================
# DEPENDENCY INJECTION PLACEHOLDERS
# ============================================================================


async def get_process_matching_use_case():
    """
    Dependency injection for ProcessMatchingUseCase.

    Returns:
        ProcessMatchingUseCase: Application Layer use case for triggering matching

    Note:
        Implementation in Phase 2 (Task 1.1.2 - Application Layer contracts).
        Will inject actual use case from Application Layer.

    Architecture Pattern:
        This follows Dependency Inversion Principle from Clean Architecture.
        API Layer depends on abstractions (use case interface), not concrete implementations.
        The use case will handle:
        - Creating ProcessMatchingCommand from request
        - Validating file existence
        - Triggering Celery task via process_matching_task
        - Returning job metadata
    """
    # Implementation in Phase 3
    raise NotImplementedError(
        "ProcessMatchingUseCase not implemented yet - will be added in Task 1.1.2"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessMatchingResponse,
    summary="Trigger asynchronous matching process",
    description=(
        "Triggers an asynchronous Celery task to match HVAC descriptions "
        "from working file against reference file with prices. "
        "Returns immediately with job_id for status tracking."
    ),
    responses={
        202: {
            "description": "Accepted - Job queued successfully",
            "model": ProcessMatchingResponse,
        },
        400: {
            "description": "Bad Request - Invalid file IDs or same files provided",
            "model": ErrorResponse,
        },
        404: {
            "description": "Not Found - One or both file IDs not found",
            "model": ErrorResponse,
        },
        422: {
            "description": "Unprocessable Entity - Validation error (e.g., invalid threshold)",
            "model": ErrorResponse,
        },
    },
)
async def process_matching(
    request: ProcessMatchingRequest, use_case=Depends(get_process_matching_use_case)
) -> ProcessMatchingResponse:
    """
    Trigger asynchronous HVAC matching process.

    This endpoint accepts two file IDs (working file and reference file) and
    a similarity threshold, then queues an asynchronous Celery task to perform
    the matching process.

    **Process Flow:**
    1. Validate input parameters (file IDs, threshold)
    2. Create ProcessMatchingCommand with validated data
    3. Delegate to Application Layer use case (ProcessMatchingUseCase)
    4. Use case validates file existence and format
    5. Use case triggers Celery task via process_matching_task
    6. Return 202 Accepted with job_id for status tracking

    **Business Rules:**
    - Working file and reference file must be different
    - Files must exist in storage (validated by Application Layer)
    - Files must be valid Excel format (.xlsx, .xls)
    - Threshold must be between 0.0 and 100.0
    - Only one matching job per file pair at a time (validated by Application Layer)

    **Async Processing:**
    - Job is queued immediately (returns in < 100ms)
    - Client must poll GET /jobs/{job_id}/status for progress
    - Recommended polling interval: 2-5 seconds
    - Job progress updated every 10% in Redis

    Args:
        request: ProcessMatchingRequest with file IDs and threshold
        use_case: Injected ProcessMatchingUseCase from Application Layer

    Returns:
        ProcessMatchingResponse with job_id, status, and estimated completion time

    Raises:
        HTTPException 400: If file IDs are identical or invalid format
        HTTPException 404: If one or both files not found in storage
        HTTPException 422: If threshold validation fails
        HTTPException 500: If unexpected error during job creation

    Example:
        >>> response = await client.post(
        ...     "/api/matching/process",
        ...     json={
        ...         "wf_file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
        ...         "ref_file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ...         "threshold": 80.0
        ...     }
        ... )
        >>> print(response.status_code)
        202
        >>> print(response.json())
        {
            "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            "status": "queued",
            "estimated_time": 45,
            "message": "Matching job queued successfully..."
        }

    Architecture Note:
        This endpoint is a thin wrapper around Application Layer.
        All business logic is delegated to ProcessMatchingUseCase.
        No direct Celery task invocation - follows dependency inversion principle.

    CQRS Pattern:
        This endpoint implements a COMMAND (write operation).
        It modifies system state by creating a new matching job.
        Read operations (job status) are in separate router (jobs.py).
    """
    # CONTRACT ONLY - Implementation in Phase 3
    #
    # Implementation will:
    # 1. Validate wf_file_id != ref_file_id
    # 2. Create ProcessMatchingCommand from request
    # 3. Call use_case.execute(command)
    # 4. Return ProcessMatchingResponse with job metadata
    # NotImplementedError`   `
    # Example implementation:
    # if request.wf_file_id == request.ref_file_id:
    #     raise HTTPException(
    #         status_code=400,
    #         detail=ErrorResponse(
    #             code="IDENTICAL_FILES",
    #             message="Working file and reference file must be different"
    #         )
    #     )
    #
    # command = ProcessMatchingCommand(
    #     wf_file_id=request.wf_file_id,
    #     ref_file_id=request.ref_file_id,
    #     threshold=request.threshold
    # )
    # result = await use_case.execute(command)
    # return result

    raise NotImplementedError(
        "Implementation in Phase 3 - Task 3.4.1. "
        "This is a contract only (Phase 1 - Task 1.1.1)."
    )
