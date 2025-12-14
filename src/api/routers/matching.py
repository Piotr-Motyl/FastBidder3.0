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

import logging
from typing import Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Depends
from pydantic import BaseModel, Field

from src.application.models import JobStatus, MatchingStrategy, ReportFormat
from src.application.commands.process_matching import (
    ProcessMatchingCommand,
    WorkingFileConfig,
    ReferenceFileConfig,
    Range,
)
from src.application.services.process_matching_use_case import ProcessMatchingUseCase
from src.infrastructure.file_storage.file_storage_service import FileStorageService

# Import shared API schemas
from src.api.schemas.common import ErrorResponse

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================
# Note: WorkingFileConfig, ReferenceFileConfig, Range imported from Application layer


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
    matching_strategy: MatchingStrategy = Field(
        default=MatchingStrategy.BEST_MATCH,
        description="Strategy for multiple matches: FIRST_MATCH, BEST_MATCH, ALL_MATCHES",
    )
    report_format: ReportFormat = Field(
        default=ReportFormat.SIMPLE,
        description="Report format: SIMPLE, DETAILED, DEBUG",
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
    # Create FileStorageService instance
    file_storage = FileStorageService()

    # Create and return ProcessMatchingUseCase with injected dependencies
    # Note: celery_app is not needed as dependency - use case imports it directly
    return ProcessMatchingUseCase(file_storage=file_storage, celery_app=None)


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
        "Process runs in background via Celery. "
        "Estimated time based on file size (rows_count * 0.1s algorithm)."
    ),
    responses={
        202: {
            "description": "Success - Job queued for processing",
            "model": ProcessMatchingResponse,
        },
        400: {
            "description": "Bad Request - Invalid parameters (e.g., identical file IDs)",
        },
        404: {"description": "Not Found - One or both files not found in uploads storage"},
        422: {"description": "Unprocessable Entity - Request validation failed"},
        500: {"description": "Internal Server Error - Unexpected error during job creation"},
    },
)
async def process_matching(
    request: ProcessMatchingRequest,
    use_case=Depends(get_process_matching_use_case),
) -> ProcessMatchingResponse:
    """
    Trigger asynchronous matching process with column mappings (Phase 2 - Detailed Contract).

    This endpoint is a thin wrapper around Application Layer use case.
    All business logic is delegated to ProcessMatchingUseCase following Clean Architecture.

    Process Flow (10 steps):
        1. Receive ProcessMatchingRequest from API client (FastAPI validates Pydantic model)
        2. Convert request to ProcessMatchingCommand (domain command)
        3. Delegate to use_case.execute(command)
        4. Use case validates command business rules (file IDs different, ranges valid, etc.)
        5. Use case validates files exist in uploads storage
        6. Use case extracts working file metadata for estimation
        7. Use case calculates estimated_time (rows_count * 0.1s)
        8. Use case triggers Celery task (process_matching_task.delay())
        9. Convert ProcessMatchingResult to ProcessMatchingResponse
        10. Return HTTP 202 Accepted with job_id, status, estimated_time

    Args:
        request: ProcessMatchingRequest with file configs and threshold
        use_case: Injected ProcessMatchingUseCase from Application Layer

    Returns:
        ProcessMatchingResponse with job_id, status="queued", estimated_time

    Raises:
        HTTPException 400: If file IDs are identical or invalid UUID format
        HTTPException 404: If working or reference file not found in uploads storage
        HTTPException 422: If business rules validation fails (invalid ranges, columns, threshold)
        HTTPException 500: If unexpected error during job creation (Celery connection, etc.)

    Error Handling (Phase 2 - Minimal):
        ValueError → 400 Bad Request (file IDs identical, invalid UUID)
        FileNotFoundError → 404 Not Found (file not in uploads storage)
        Exception → 500 Internal Server Error (catch-all for unexpected errors)

    Architecture Note:
        - API Layer responsibility: HTTP concerns only (status codes, error mapping)
        - Application Layer responsibility: Business logic, orchestration, validation
        - No direct Celery task invocation - follows dependency inversion principle
        - No direct file storage access - delegated to use case

    Phase 3+ Extensions (NOT in Phase 2):
        - Idempotency: Return existing job_id if request hash already processed
        - Priority queue: Support ?priority=HIGH query parameter
        - Quota check: Reject if user has too many concurrent jobs (requires auth)
        - Dry run mode: Support ?dry_run=true for simulation without saving
        - Callback URL: Support callback_url in request for webhook notification
        - Advanced response: Include position_in_queue, estimated_start, estimated_completion

    Examples:
        >>> # Using cURL
        >>> curl -X POST "http://localhost:8000/api/matching/process" \\
        ...      -H "Content-Type: application/json" \\
        ...      -d '{
        ...        "working_file": {
        ...          "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
        ...          "description_column": "C",
        ...          "description_range": {"start": 2, "end": 100},
        ...          "price_target_column": "F",
        ...          "matching_report_column": "G"
        ...        },
        ...        "reference_file": {
        ...          "file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ...          "description_column": "B",
        ...          "description_range": {"start": 2, "end": 50},
        ...          "price_source_column": "D"
        ...        },
        ...        "matching_threshold": 80.0,
        ...        "matching_strategy": "best_match",
        ...        "report_format": "simple"
        ...      }'
        {
          "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
          "status": "queued",
          "estimated_time": 10,
          "message": "Matching job queued successfully. Check status at GET /jobs/3fa85f64-.../status"
        }

        >>> # Using Python requests
        >>> import requests
        >>> response = requests.post(
        ...     "http://localhost:8000/api/matching/process",
        ...     json={
        ...         "working_file": {
        ...             "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
        ...             "description_column": "C",
        ...             "description_range": {"start": 2, "end": 100},
        ...             "price_target_column": "F"
        ...         },
        ...         "reference_file": {
        ...             "file_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
        ...             "description_column": "B",
        ...             "description_range": {"start": 2, "end": 50},
        ...             "price_source_column": "D"
        ...         },
        ...         "matching_threshold": 75.0
        ...     }
        ... )
        >>> data = response.json()
        >>> print(data["job_id"])  # Use this for status tracking
        3fa85f64-5717-4562-b3fc-2c963f66afa6

        >>> # Then poll for status
        >>> status_response = requests.get(
        ...     f"http://localhost:8000/api/jobs/{data['job_id']}/status"
        ... )

    Implementation Note (Phase 3):
        import logging
        from src.application.commands.process_matching import ProcessMatchingCommand

        logger = logging.getLogger(__name__)

        try:
            # Step 2: Convert request to Command
            command = ProcessMatchingCommand(
                working_file=request.working_file,
                reference_file=request.reference_file,
                matching_threshold=request.matching_threshold,
                matching_strategy=request.matching_strategy,
                report_format=request.report_format
            )
            logger.debug(f"Created command for WF={request.working_file.file_id}, REF={request.reference_file.file_id}")

            # Step 3: Execute use case (validates, estimates, triggers Celery)
            result = await use_case.execute(command)
            logger.info(f"Job queued: {result.job_id}, estimated_time={result.estimated_time}s")

            # Step 9: Convert result to response
            return ProcessMatchingResponse(
                job_id=str(result.job_id),  # Convert UUID to string for JSON
                status=result.status,
                estimated_time=result.estimated_time,
                message=result.message
            )

        except ValueError as e:
            # File IDs identical or invalid UUID format
            logger.warning(f"Bad request: {e}")
            raise HTTPException(
                status_code=400,
                detail=ErrorResponse(
                    code="INVALID_PARAMETERS",
                    message=str(e),
                    details={"request": request.dict()}
                ).dict()
            )

        except FileNotFoundError as e:
            # File not found in uploads storage
            logger.warning(f"File not found: {e}")
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    code="FILE_NOT_FOUND",
                    message=str(e),
                    details={"working_file_id": request.working_file.file_id, "reference_file_id": request.reference_file.file_id}
                ).dict()
            )

        except Exception as e:
            # Unexpected error (Celery connection, etc.)
            logger.error(f"Unexpected error during job creation: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred during job creation",
                    details={"error": str(e)}
                ).dict()
            )

    Phase 2 Contract:
        This method defines the interface contract with detailed documentation.
        Actual implementation will be added in Phase 3 - Task 3.4.1.
    """
    # Implementation based on Phase 2 contract
    try:
        # Step 2: Convert request to Command
        # Convert API layer Pydantic models to dict for Application layer
        command = ProcessMatchingCommand(
            working_file=request.working_file.model_dump(),
            reference_file=request.reference_file.model_dump(),
            matching_threshold=request.matching_threshold,
            matching_strategy=request.matching_strategy,
            report_format=request.report_format,
        )
        logger.debug(
            f"Created command for WF={request.working_file.file_id}, "
            f"REF={request.reference_file.file_id}"
        )

        # Step 3: Execute use case (validates, estimates, triggers Celery)
        result = await use_case.execute(command)
        logger.info(
            f"Job queued: {result.job_id}, estimated_time={result.estimated_time}s"
        )

        # Step 9: Convert result to response
        return ProcessMatchingResponse(
            job_id=str(result.job_id),  # Convert UUID to string for JSON
            status=result.status,
            estimated_time=result.estimated_time,
            message=result.message,
        )

    except ValueError as e:
        # File IDs identical or invalid UUID format
        logger.warning(f"Bad request: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_PARAMETERS",
                "message": str(e),
                "details": {
                    "working_file_id": request.working_file.file_id,
                    "reference_file_id": request.reference_file.file_id,
                },
            },
        )

    except FileNotFoundError as e:
        # File not found in uploads storage
        logger.warning(f"File not found: {e}")
        raise HTTPException(
            status_code=404,
            detail={
                "code": "FILE_NOT_FOUND",
                "message": str(e),
                "details": {
                    "working_file_id": request.working_file.file_id,
                    "reference_file_id": request.reference_file.file_id,
                },
            },
        )

    except Exception as e:
        # Unexpected error (Celery connection, etc.)
        logger.error(f"Unexpected error during job creation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during job creation",
                "details": {"error": str(e)},
            },
        )
