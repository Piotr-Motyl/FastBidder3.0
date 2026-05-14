"""HTTP interface for triggering async HVAC matching (POST /matching/process).

Returns 202 Accepted with a job_id; client polls /jobs/{job_id}/status until
status=completed, then downloads via /results/{job_id}/download.
"""

import logging

from fastapi import APIRouter, status, HTTPException, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.application.models import JobStatus, MatchingStrategy, ReportFormat
from src.application.commands.process_matching import (
    ProcessMatchingCommand,
    WorkingFileConfig,
    ReferenceFileConfig,
)
from src.api.dependencies import get_process_matching_use_case
from src.api.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class ProcessMatchingRequest(BaseModel):
    """Matching request with explicit column mappings (user picks which columns
    and row ranges to read from each file, plus where to write results)."""

    working_file: WorkingFileConfig
    reference_file: ReferenceFileConfig
    matching_threshold: float = Field(
        default=75.0,
        ge=1.0,
        le=100.0,
        description="Similarity threshold percentage. Matches below this value are ignored.",
    )
    matching_strategy: MatchingStrategy = Field(default=MatchingStrategy.BEST_MATCH)
    report_format: ReportFormat = Field(default=ReportFormat.SIMPLE)

    model_config = ConfigDict(
        json_schema_extra={
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
    )


class ProcessMatchingResponse(BaseModel):
    """Returned with 202 Accepted; client uses job_id to track progress."""

    job_id: str = Field(description="Celery task ID for tracking job progress")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    estimated_time: int = Field(description="Estimated time to completion in seconds")
    message: str = Field(
        default="Matching job queued successfully. Use job_id to check status."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "queued",
                "estimated_time": 45,
                "message": "Matching job queued successfully. Check status at GET /jobs/3fa85f64-5717-4562-b3fc-2c963f66afa6/status",
            }
        }
    )


router = APIRouter(
    prefix="/matching",
    tags=["matching"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input parameters"},
        404: {"model": ErrorResponse, "description": "File not found"},
        422: {"model": ErrorResponse, "description": "Request validation error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


@router.post(
    "/process",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ProcessMatchingResponse,
    summary="Trigger async matching process",
    description=(
        "Initiates async matching between working file (to be priced) and reference "
        "file (price catalog). Returns immediately with job_id; processing runs in "
        "background via Celery. Estimated time = rows_count * 0.1s, clamped to [10, 300]."
    ),
    responses={
        202: {"description": "Job queued for processing", "model": ProcessMatchingResponse},
        400: {"description": "Invalid parameters (e.g. identical file IDs)"},
        404: {"description": "One or both files not found in uploads storage"},
        422: {"description": "Request validation failed"},
        500: {"description": "Unexpected error during job creation"},
    },
)
async def process_matching(
    request: ProcessMatchingRequest,
    use_case=Depends(get_process_matching_use_case),
) -> ProcessMatchingResponse:
    """Thin HTTP wrapper around ProcessMatchingUseCase. Maps exceptions → HTTP codes."""
    try:
        command = ProcessMatchingCommand(
            working_file=request.working_file,
            reference_file=request.reference_file,
            matching_threshold=request.matching_threshold,
            matching_strategy=request.matching_strategy,
            report_format=request.report_format,
        )
        logger.debug(
            f"Created command for WF={request.working_file.file_id}, "
            f"REF={request.reference_file.file_id}"
        )

        result = await use_case.execute(command)
        logger.info(
            f"Job queued: {result.job_id}, estimated_time={result.estimated_time}s"
        )

        return ProcessMatchingResponse(
            job_id=str(result.job_id),
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
        logger.error(f"Unexpected error during job creation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during job creation",
                "details": {"error": str(e)},
            },
        )
