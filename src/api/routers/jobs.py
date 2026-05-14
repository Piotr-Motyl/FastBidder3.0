"""HTTP interface for querying async job status (GET /jobs/{job_id}/status)."""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Path, Depends
from pydantic import BaseModel, ConfigDict, Field

from src.application.queries.get_job_status import (
    GetJobStatusQueryHandler,
    GetJobStatusQuery,
    JobNotFoundException,
)
from src.application.models import JobStatus
from src.api.dependencies import get_job_status_query_handler
from src.api.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class JobStatusResponse(BaseModel):
    """HTTP representation of job status. Mapped from Application's JobStatusResult."""

    job_id: UUID
    status: JobStatus
    progress: int = Field(default=0, ge=0, le=100)
    message: str
    result_ready: bool = False
    current_step: Optional[str] = None
    error_details: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    using_ai: bool = False
    ai_model: Optional[str] = None

    model_config = ConfigDict(
        json_schema_extra={
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
                "using_ai": True,
                "ai_model": "paraphrase-multilingual-MiniLM-L12-v2",
            }
        }
    )


router = APIRouter(
    prefix="/jobs",
    tags=["jobs"],
    responses={
        404: {"model": ErrorResponse, "description": "Job ID not found or expired"},
        422: {"model": ErrorResponse, "description": "Invalid job ID format"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


@router.get(
    "/{job_id}/status",
    status_code=status.HTTP_200_OK,
    response_model=JobStatusResponse,
    summary="Get status of asynchronous job",
    description=(
        "Retrieves current status and progress of an async job. Poll every "
        "2-5 seconds during processing; stop when status is completed, failed, "
        "or cancelled. TTL: 1h for active progress, 24h for completed jobs."
    ),
    responses={
        200: {"description": "Job status retrieved", "model": JobStatusResponse},
        404: {"description": "Job not found or expired", "model": ErrorResponse},
        422: {"description": "Invalid job ID format", "model": ErrorResponse},
        500: {"description": "Redis connection failure", "model": ErrorResponse},
    },
)
async def get_job_status(
    job_id: UUID = Path(..., description="Celery task ID returned from async endpoint"),
    handler: GetJobStatusQueryHandler = Depends(get_job_status_query_handler),
) -> JobStatusResponse:
    """Read-only query; delegates to Application's GetJobStatusQueryHandler."""
    try:
        query = GetJobStatusQuery(job_id=job_id)
        logger.debug(f"Querying status for job: {job_id}")

        result = await handler.handle(query)
        logger.info(f"Job {job_id} status retrieved: {result.status} ({result.progress}%)")

        return JobStatusResponse(
            job_id=result.job_id,
            status=JobStatus(result.status),
            progress=result.progress,
            message=result.message,
            result_ready=result.result_ready,
            current_step=result.current_step,
            error_details=result.error_details,
            created_at=result.created_at,
            updated_at=result.updated_at,
            using_ai=result.using_ai,
            ai_model=result.ai_model,
        )

    except JobNotFoundException:
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
        # Invalid status string from Redis — shouldn't happen on happy path.
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
