"""
API Router for HVAC Matching Process

Responsibility:
    HTTP interface for triggering asynchronous matching process.
    Thin layer that delegates to Application Layer use cases via dependency injection.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Depends on Application Layer (ProcessMatchingUseCase - Task 1.1.2)
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

FIX:
    JobStatus now imported from Application Layer (correct dependency direction).
"""

from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Depends
from pydantic import BaseModel, Field

# Import JobStatus from Application Layer (correct dependency direction)
from src.application.models import JobStatus


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class ProcessMatchingRequest(BaseModel):
    """
    Request model for triggering matching process.

    Attributes:
        wf_file_id: UUID of uploaded working file
        ref_file_id: UUID of uploaded reference file
        threshold: Similarity threshold percentage (0.0-100.0)

    Validation:
        - wf_file_id and ref_file_id must be valid UUIDs
        - wf_file_id != ref_file_id (validated in Command layer)
        - threshold must be between 0.0 and 100.0 (inclusive)
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

    Attributes:
        job_id: Unique identifier for the Celery task
        status: Current job status (always "queued" in immediate response)
        estimated_time: Estimated time to completion in seconds
        message: Human-readable message about job status
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

    Attributes:
        code: Machine-readable error code
        message: Human-readable error message
        details: Optional additional error details
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
# DEPENDENCY INJECTION
# ============================================================================


async def get_process_matching_use_case():
    """
    Dependency injection for ProcessMatchingUseCase.

    Returns:
        ProcessMatchingUseCase: Application Layer use case for triggering matching

    Note:
        Implementation in Phase 3 (Task 3.4.1).
        Will inject actual use case with all dependencies:
        - Celery app
        - FileStorageService

    Example implementation (Phase 3):
        from src.application.services import ProcessMatchingUseCase
        from src.application.tasks import celery_app
        from src.infrastructure.file_storage import get_file_storage_service

        file_storage = get_file_storage_service()
        return ProcessMatchingUseCase(celery_app, file_storage)
    """
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
            "description": "Unprocessable Entity - Validation error",
            "model": ErrorResponse,
        },
    },
)
async def process_matching(
    request: ProcessMatchingRequest, use_case=Depends(get_process_matching_use_case)
) -> ProcessMatchingResponse:
    """
    Trigger asynchronous HVAC matching process.

    Process Flow:
    1. Validate input parameters (file IDs, threshold)
    2. Create ProcessMatchingCommand from request
    3. Delegate to Application Layer use case
    4. Use case validates file existence and format
    5. Use case triggers Celery task
    6. Return 202 Accepted with job_id for status tracking

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

    Architecture Note:
        This endpoint is a thin wrapper around Application Layer.
        All business logic is delegated to ProcessMatchingUseCase.
        No direct Celery task invocation - follows dependency inversion principle.
    """
    # CONTRACT ONLY - Implementation in Phase 3
    #
    # Implementation will:
    # 1. Convert request to command
    # 2. Call use_case.execute(command)
    # 3. Convert result to response
    # 4. Return response
    #
    # Example implementation:
    # from src.application.commands import ProcessMatchingCommand
    #
    # try:
    #     command = ProcessMatchingCommand(
    #         wf_file_id=request.wf_file_id,
    #         ref_file_id=request.ref_file_id,
    #         threshold=request.threshold
    #     )
    #     result = await use_case.execute(command)
    #
    #     return ProcessMatchingResponse(
    #         job_id=result.job_id,
    #         status=result.status,
    #         estimated_time=result.estimated_time,
    #         message=result.message
    #     )
    # except ValueError as e:
    #     raise HTTPException(status_code=400, detail=str(e))

    raise NotImplementedError(
        "Implementation in Phase 3 - Task 3.4.1. "
        "This is a contract only (Phase 1 - Task 1.1.1)."
    )
