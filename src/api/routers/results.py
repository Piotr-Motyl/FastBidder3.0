"""HTTP interface for downloading result files (GET /results/{job_id}/download).

KISS: direct dependency injection on infrastructure services (no Query/Handler
pattern for simple file serving).
"""

import logging
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Path, Depends
from fastapi.responses import FileResponse

from src.infrastructure.file_storage.file_storage_service import FileStorageService
from src.infrastructure.persistence.redis.progress_tracker import RedisProgressTracker
from src.api.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/results",
    tags=["results"],
    responses={
        404: {"model": ErrorResponse, "description": "Job or result file not found"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


def get_file_storage_service() -> FileStorageService:
    return FileStorageService()


def get_progress_tracker() -> RedisProgressTracker:
    return RedisProgressTracker()


@router.get(
    "/{job_id}/download",
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    summary="Download result file for completed job",
    description=(
        "Downloads the result Excel file for a completed matching job. "
        "Client must poll GET /jobs/{job_id}/status until status=completed, "
        "then download via this endpoint."
    ),
    responses={
        200: {
            "description": "Result file download",
            "content": {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        404: {
            "description": "Job not found, not completed, or result file missing",
            "model": ErrorResponse,
        },
        422: {"description": "Invalid job ID format", "model": ErrorResponse},
        500: {"description": "File system or Redis error", "model": ErrorResponse},
    },
)
async def download_result(
    job_id: UUID = Path(..., description="Job ID from POST /matching/process"),
    file_storage: FileStorageService = Depends(get_file_storage_service),
    progress_tracker: RedisProgressTracker = Depends(get_progress_tracker),
) -> FileResponse:
    """
    Serve the result.xlsx for a completed job.

    404 paths (each surface a distinct `code` for the client to dispatch on):
      - JOB_NOT_FOUND: Redis has no entry (expired or never existed)
      - JOB_NOT_COMPLETED: job exists but status != completed
      - RESULT_FILE_NOT_FOUND: status=completed but file missing (job aborted)
    """
    try:
        job_id_str = str(job_id)
        progress_data = progress_tracker.get_status(job_id_str)

        if not progress_data:
            logger.warning(f"Job not found for download: {job_id_str}")
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    code="JOB_NOT_FOUND",
                    message=f"Job with ID {job_id} not found or expired",
                    details={"job_id": job_id_str},
                ).model_dump(),
            )

        job_status = progress_data.get("status")
        if job_status != "completed":
            logger.warning(
                f"Attempted download of non-completed job {job_id_str}: status={job_status}"
            )
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    code="JOB_NOT_COMPLETED",
                    message=(
                        f"Job is not completed yet. Current status: {job_status}. "
                        f"Poll GET /jobs/{job_id}/status until completed."
                    ),
                    details={"job_id": job_id_str, "current_status": job_status},
                ).model_dump(),
            )

        result_path = file_storage.get_result_file_path(job_id)

        if not file_storage.result_file_exists(job_id):
            # Edge case: status=completed but file missing (e.g. job aborted mid-write)
            logger.error(
                f"Result file not found for completed job {job_id_str}: {result_path}"
            )
            raise HTTPException(
                status_code=404,
                detail=ErrorResponse(
                    code="RESULT_FILE_NOT_FOUND",
                    message=(
                        f"Result file not found for job {job_id}. "
                        "The job may have failed or been interrupted."
                    ),
                    details={"job_id": job_id_str},
                ).model_dump(),
            )

        logger.info(f"Serving result file for job {job_id_str}: {result_path}")

        return FileResponse(
            path=result_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="result.xlsx",
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(
            f"Unexpected error downloading result for job {job_id}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred while downloading result file",
                details={"job_id": str(job_id), "error": str(e)},
            ).model_dump(),
        )
