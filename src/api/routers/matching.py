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
"""

from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Depends
from pydantic import BaseModel, Field

from src.application.models import JobStatus


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================


class Range(BaseModel):
    """Range of rows in Excel (1-based indexing like Excel)."""

    start: int = Field(ge=1, description="Start row (Excel notation, 1-based)")
    end: int = Field(ge=1, description="End row (Excel notation, 1-based)")


class WorkingFileConfig(BaseModel):
    """Configuration for working file (file to be priced)."""

    file_id: str = Field(description="UUID of working file as string")
    description_column: str = Field(description="Column with descriptions (e.g., 'C')")
    description_range: Range = Field(description="Range of rows with descriptions")
    price_target_column: str = Field(description="Column where prices will be written")
    matching_report_column: Optional[str] = Field(
        default=None, description="Column for match report (score + matched item name)"
    )


class ReferenceFileConfig(BaseModel):
    """Configuration for reference file (price catalog)."""

    file_id: str = Field(description="UUID of reference file as string")
    description_column: str = Field(description="Column with descriptions (e.g., 'B')")
    description_range: Range = Field(description="Range of rows with descriptions")
    price_source_column: str = Field(description="Column with prices to copy")


class ProcessMatchingRequest(BaseModel):
    """
    Request to trigger matching process with column mappings.

    User specifies exactly which columns and ranges to use in each file.
    This gives full control over Excel structure handling.

    Validation Notes (Phase 2+):
        - file_ids must exist in storage
        - file_ids must be different
        - ranges must be valid (start < end)
        - columns must exist in files
        - threshold > 0 for meaningful results

    Phase 2 Extensions:
        - matching_strategy: Strategy for handling multiple matches
        - report_format: Format of matching report in Excel
    """

    working_file: WorkingFileConfig
    reference_file: ReferenceFileConfig
    matching_threshold: float = Field(
        default=75.0,
        ge=1.0,
        le=100.0,
        description="Similarity threshold percentage. Matches below this value will be ignored.",
    )
    matching_strategy: Optional[str] = Field(
        default="best_match",
        description="Strategy for multiple matches: 'first_match', 'best_match', 'all_matches'",
    )
    report_format: Optional[str] = Field(
        default="simple",
        description="Report format: 'simple', 'detailed', 'debug'",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "working_file": {
                    "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                    "description_column": "C",
                    "description_range": {"start": 2, "end": 10},
                    "price_target_column": "F",
                    "matching_report_column": "G",
                },
                "reference_file": {
                    "file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "description_column": "B",
                    "description_range": {"start": 2, "end": 20},
                    "price_source_column": "D",
                },
                "matching_threshold": 80.0,
                "matching_strategy": "best_match",
                "report_format": "simple",
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

    job_id: str = Field(description="Celery task ID for tracking job progress")

    status: JobStatus = Field(
        default=JobStatus.QUEUED, description="Current status of the job"
    )

    estimated_time: int = Field(
        description="Estimated time to completion in seconds (based on file size and historical data)"
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
        Will inject actual use case with all its dependencies:
        - FileStorageService for validating files exist
        - Celery app for triggering async tasks
        - ProgressTracker for initializing job tracking

    The use case will validate business rules and trigger Celery task.
    """
    # Implementation in Phase 3
    raise NotImplementedError(
        "ProcessMatchingUseCase not implemented yet - will be added in Task 1.1.2/3.4.1"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessMatchingResponse,
    summary="Trigger async matching process",
    description=(
        "Initiates asynchronous matching process between working file "
        "(to be priced) and reference file (price catalog). "
        "Returns immediately with job_id for status tracking. "
        "Process runs in background via Celery."
    ),
    responses={
        202: {
            "description": "Success - Job queued for processing",
            "model": ProcessMatchingResponse,
        },
        400: {
            "description": "Bad Request - Invalid parameters (e.g., identical file IDs)",
        },
        404: {"description": "Not Found - One or both files not found in storage"},
        422: {"description": "Unprocessable Entity - Request validation failed"},
    },
)
async def process_matching(
    request: ProcessMatchingRequest,
    use_case=Depends(get_process_matching_use_case),
) -> ProcessMatchingResponse:
    """
    Trigger asynchronous matching process with column mappings.

    Process Flow:
        1. Validate input parameters (file IDs, column mappings, threshold)
        2. Create ProcessMatchingCommand from request
        3. Delegate to Application Layer use case
        4. Use case validates file existence and format
        5. Use case triggers Celery task
        6. Return 202 Accepted with job_id for status tracking

    Args:
        request: ProcessMatchingRequest with file configs and threshold
        use_case: Injected ProcessMatchingUseCase from Application Layer

    Returns:
        ProcessMatchingResponse with job_id, status, and estimated completion time

    Raises:
        HTTPException 400: If file IDs are identical or invalid format
        HTTPException 404: If one or both files not found in storage
        HTTPException 422: If threshold or column validation fails
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
    #         working_file=request.working_file.dict(),
    #         reference_file=request.reference_file.dict(),
    #         matching_threshold=request.matching_threshold
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
