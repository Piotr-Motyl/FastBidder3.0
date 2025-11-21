"""
API Router for Result File Downloads

Responsibility:
    HTTP interface for downloading completed job result files.
    Handles file retrieval and serves Excel files to clients.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Depends on Infrastructure Layer (FileStorageService, RedisProgressTracker)
    - KISS approach: Direct dependency injection, no Query/Handler pattern
    - Read-only operations (file downloads)
    - No business logic - pure HTTP file serving

Contains:
    - GET /results/{job_id}/download - Download result Excel file

Does NOT contain:
    - Business logic (delegated to Domain Layer)
    - File processing (delegated to Infrastructure Layer)
    - Job creation (belongs to matching.py router)
    - Status tracking (belongs to jobs.py router)

Phase 1 Note:
    This is a CONTRACT ONLY. Implementation will be added in Phase 3.
    All endpoints raise NotImplementedError.
"""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, status, HTTPException, Path, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Import from infrastructure (KISS approach - no Application Layer for simple file serving)
from src.infrastructure.file_storage.file_storage_service import FileStorageService
from src.infrastructure.persistence.redis.progress_tracker import RedisProgressTracker
from src.application.models import JobStatus


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class ErrorResponse(BaseModel):
    """
    Standard error response model for all API errors.

    Provides consistent error structure across all endpoints.

    Attributes:
        code: Machine-readable error code
        message: Human-readable error message
        details: Optional additional error details
    """

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict] = Field(
        default=None, description="Additional error context"
    )


# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================


router = APIRouter(
    prefix="/results",
    tags=["results"],
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Not Found - Job or result file not found",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================


def get_file_storage_service() -> FileStorageService:
    """
    Dependency injection for FileStorageService.

    Returns:
        FileStorageService: Infrastructure service for file operations

    Note:
        Implementation in Phase 3 (Task 3.4.1).
        Will inject actual service with proper configuration.

    Example implementation (Phase 3):
        return FileStorageService()
    """
    # Implementation in Phase 3
    raise NotImplementedError(
        "FileStorageService dependency not implemented yet - will be added in Task 3.4.1"
    )


def get_progress_tracker() -> RedisProgressTracker:
    """
    Dependency injection for RedisProgressTracker.

    Returns:
        RedisProgressTracker: Infrastructure service for job status tracking

    Note:
        Implementation in Phase 3 (Task 3.4.1).
        Will inject actual tracker with Redis connection.

    Example implementation (Phase 3):
        return RedisProgressTracker()
    """
    # Implementation in Phase 3
    raise NotImplementedError(
        "RedisProgressTracker dependency not implemented yet - will be added in Task 3.4.1"
    )


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.get(
    "/{job_id}/download",
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
    summary="Download result file for completed job",
    description=(
        "Downloads the result Excel file for a completed matching job. "
        "Result file contains matched descriptions with prices. "
        "Client must poll GET /jobs/{job_id}/status until status=completed, "
        "then download results using this endpoint."
    ),
    responses={
        200: {
            "description": "Success - Result file download",
            "content": {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        404: {
            "description": "Not Found - Job not found, job not completed, or result file missing",
            "model": ErrorResponse,
        },
        422: {
            "description": "Unprocessable Entity - Invalid job ID format",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal Server Error - File system error or Redis connection failure",
            "model": ErrorResponse,
        },
    },
)
async def download_result(
    job_id: UUID = Path(..., description="Job ID from POST /matching/process"),
    file_storage: FileStorageService = Depends(get_file_storage_service),
    progress_tracker: RedisProgressTracker = Depends(get_progress_tracker),
) -> FileResponse:
    """
    Download result Excel file for completed job (Phase 2 - Detailed Contract).

    This endpoint serves the result file generated by matching process.
    Uses KISS approach with direct dependency injection (no Query/Handler pattern).

    Process Flow (10 steps):
        1. Receive job_id from URL path parameter (FastAPI validates UUID format)
        2. Verify job exists in Redis via progress_tracker.get_status(job_id)
        3. Verify job status is COMPLETED (not QUEUED, PROCESSING, FAILED, CANCELLED)
        4. Get result file path via file_storage.get_result_file_path(job_id)
        5. Verify result file exists via file_storage.result_file_exists(job_id)
        6. Prepare filename for Content-Disposition header (result.xlsx)
        7. Prepare Content-Type header (application/vnd.openxmlformats-officedocument.spreadsheetml.sheet)
        8. Create FileResponse with file path and headers
        9. Return FileResponse (FastAPI streams file to client)
        10. Handle errors (404 if not found/not completed, 500 if file system error)

    Args:
        job_id: UUID of the job from POST /matching/process
        file_storage: Injected FileStorageService from Infrastructure Layer
        progress_tracker: Injected RedisProgressTracker from Infrastructure Layer

    Returns:
        FileResponse: Excel file with matched descriptions and prices

    Raises:
        HTTPException 404: If job not found, job not completed, or result file missing
        HTTPException 422: If job_id is not a valid UUID format (handled by FastAPI automatically)
        HTTPException 500: If file system error or Redis connection failure

    File Details:
        - Filename: result.xlsx (Phase 2 - static name)
        - Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        - Content-Disposition: attachment; filename="result.xlsx"
        - Max size: ~10MB (based on MAX_FILE_SIZE_MB config)
        - Format: Excel .xlsx with matched prices and reports

    Workflow:
        1. Client POSTs to /matching/process → receives job_id
        2. Client polls GET /jobs/{job_id}/status until status="completed"
        3. Client GETs /results/{job_id}/download → receives Excel file
        4. Client saves file locally for analysis

    Error Handling (Phase 2 - Minimal):
        - Job not found in Redis → 404 Not Found (JOB_NOT_FOUND)
        - Job not completed → 404 Not Found (JOB_NOT_COMPLETED)
        - Result file not found → 404 Not Found (RESULT_FILE_NOT_FOUND)
        - File system error → 500 Internal Server Error (FILE_SYSTEM_ERROR)
        - Redis connection error → 500 Internal Server Error (REDIS_ERROR)

    Architecture Note:
        - API Layer responsibility: HTTP file serving, headers, error mapping
        - Infrastructure Layer responsibility: File system operations, Redis lookups
        - KISS approach: No Application Layer Query/Handler (simple CRUD read)
        - Direct dependency injection for simplicity
        - Uses FastAPI FileResponse for efficient file streaming

    Phase 3+ Extensions (NOT in Phase 2):
        - Authorization: Check if user has permission to access job_id
        - Format choice: Support ?format=xlsx or ?format=csv query parameter
        - Compression: Support Accept-Encoding: gzip for large files (auto-compression)
        - Resume support: Support Range headers for interrupted downloads
        - Dynamic filename: {original_name}_matched_{timestamp}.xlsx (requires metadata tracking)
        - Expiry tracking: Return 410 Gone if result expired (>24h old)
        - Stats tracking: Log download count and last download timestamp
        - Preview mode: Support ?preview=true to return first 100 rows as JSON
        - Watermark: Add "Generated by FastBidder" footer to Excel file
        - Cache headers: Set Cache-Control for browser caching strategy

    Examples:
        >>> # Example 1: Successful download (cURL)
        >>> curl -X GET "http://localhost:8000/api/results/3fa85f64-5717-4562-b3fc-2c963f66afa6/download" \\
        ...      -o result.xlsx
        # Downloads result.xlsx file (binary Excel data)
        # Response headers:
        # Content-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
        # Content-Disposition: attachment; filename="result.xlsx"
        # Content-Length: 1048576

        >>> # Example 2: Python requests (download to file)
        >>> import requests
        >>>
        >>> job_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        >>> url = f"http://localhost:8000/api/results/{job_id}/download"
        >>>
        >>> # Download file
        >>> response = requests.get(url)
        >>> if response.status_code == 200:
        ...     with open("matched_results.xlsx", "wb") as f:
        ...         f.write(response.content)
        ...     print(f"Downloaded {len(response.content)} bytes")
        ... else:
        ...     print(f"Error: {response.status_code}")
        Downloaded 1048576 bytes

        >>> # Example 3: Full workflow (poll status → download)
        >>> import requests
        >>> import time
        >>>
        >>> job_id = "3fa85f64-5717-4562-b3fc-2c963f66afa6"
        >>> status_url = f"http://localhost:8000/api/jobs/{job_id}/status"
        >>> download_url = f"http://localhost:8000/api/results/{job_id}/download"
        >>>
        >>> # Poll until completed
        >>> while True:
        ...     status_resp = requests.get(status_url)
        ...     data = status_resp.json()
        ...     print(f"Status: {data['status']}, Progress: {data['progress']}%")
        ...     if data['status'] == 'completed':
        ...         break
        ...     time.sleep(2)
        Status: processing, Progress: 45%
        Status: processing, Progress: 90%
        Status: completed, Progress: 100%
        >>>
        >>> # Download result
        >>> download_resp = requests.get(download_url)
        >>> with open("result.xlsx", "wb") as f:
        ...     f.write(download_resp.content)
        >>> print("Download complete!")

        >>> # Example 4: Job not completed (404 error)
        >>> curl -X GET "http://localhost:8000/api/results/processing-job-id/download"
        {
          "code": "JOB_NOT_COMPLETED",
          "message": "Job is not completed yet. Current status: processing. Poll GET /jobs/{job_id}/status until completed.",
          "details": {"job_id": "processing-job-id", "current_status": "processing"}
        }

        >>> # Example 5: Job not found (404 error)
        >>> curl -X GET "http://localhost:8000/api/results/00000000-0000-0000-0000-000000000000/download"
        {
          "code": "JOB_NOT_FOUND",
          "message": "Job with ID 00000000-0000-0000-0000-000000000000 not found or expired",
          "details": {"job_id": "00000000-0000-0000-0000-000000000000"}
        }

        >>> # Example 6: Result file missing (404 error - defensive check)
        >>> curl -X GET "http://localhost:8000/api/results/completed-but-no-file/download"
        {
          "code": "RESULT_FILE_NOT_FOUND",
          "message": "Result file not found for job completed-but-no-file. The job may have failed or been interrupted.",
          "details": {"job_id": "completed-but-no-file"}
        }

    Implementation Note (Phase 3):
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Step 2: Verify job exists in Redis
            job_id_str = str(job_id)
            progress_data = progress_tracker.get_status(job_id_str)

            if not progress_data:
                logger.warning(f"Job not found for download: {job_id_str}")
                raise HTTPException(
                    status_code=404,
                    detail=ErrorResponse(
                        code="JOB_NOT_FOUND",
                        message=f"Job with ID {job_id} not found or expired",
                        details={"job_id": job_id_str}
                    ).dict()
                )

            # Step 3: Verify job status is COMPLETED
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
                        details={"job_id": job_id_str, "current_status": job_status}
                    ).dict()
                )

            # Step 4-5: Get result file path and verify existence
            result_path = file_storage.get_result_file_path(job_id)

            if not file_storage.result_file_exists(job_id):
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
                        details={"job_id": job_id_str}
                    ).dict()
                )

            # Step 6-8: Prepare response with headers
            filename = "result.xlsx"  # Phase 2: static filename
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

            logger.info(f"Serving result file for job {job_id_str}: {result_path}")

            # Step 9: Return FileResponse (FastAPI handles file streaming)
            return FileResponse(
                path=result_path,
                media_type=media_type,
                filename=filename,  # Sets Content-Disposition: attachment; filename="..."
            )

        except HTTPException:
            # Re-raise HTTPExceptions (already formatted)
            raise

        except Exception as e:
            # Step 10: Handle unexpected errors
            logger.error(
                f"Unexpected error downloading result for job {job_id}: {e}",
                exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail=ErrorResponse(
                    code="INTERNAL_SERVER_ERROR",
                    message="An unexpected error occurred while downloading result file",
                    details={"job_id": str(job_id), "error": str(e)}
                ).dict()
            )

    Phase 2 Contract:
        This method defines the interface contract with detailed documentation.
        Actual implementation will be added in Phase 3 - Task 3.4.1.
    """
    raise NotImplementedError(
        "Implementation in Phase 3 - Task 3.4.1. "
        "This is a detailed contract (Phase 2 - Task 2.4.4)."
    )
