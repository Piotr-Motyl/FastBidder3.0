"""
GetJobStatusQuery — CQRS read query for job status from Redis.

Query is an immutable DTO; QueryHandler delegates to ProgressTrackerProtocol
and converts the infrastructure dict into a JobStatusResult DTO.
"""

import logging
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.application.models import JobStatus
from src.application.ports.progress_tracker import ProgressTrackerProtocol

logger = logging.getLogger(__name__)


class GetJobStatusQuery(BaseModel):
    """Immutable query holding the job_id to look up."""

    model_config = ConfigDict(frozen=True)

    job_id: UUID = Field(description="Celery task ID returned from matching process")


class JobStatusResult(BaseModel):
    """Application-layer DTO with current job status. Converted to API model upstream."""

    job_id: UUID
    status: str  # JobStatus enum value as string
    progress: int = Field(ge=0, le=100)
    message: str
    result_ready: bool = False
    current_step: Optional[str] = None
    error_details: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    using_ai: bool = False
    ai_model: Optional[str] = None


class GetJobStatusQueryHandler:
    """Fetch job status from Redis via ProgressTrackerProtocol."""

    def __init__(self, progress_tracker: ProgressTrackerProtocol):
        self.progress_tracker = progress_tracker

    async def handle(self, query: GetJobStatusQuery) -> JobStatusResult:
        """
        Look up job status by id.

        Raises:
            JobNotFoundException: if job_id is not in Redis (expired or never existed).
            ValueError: if status string from Redis isn't a valid JobStatus enum value.
        """
        job_id_str = str(query.job_id)
        progress_data = self.progress_tracker.get_status(job_id_str)

        if not progress_data:
            logger.warning(f"Job not found in Redis: {job_id_str}")
            raise JobNotFoundException(query.job_id)

        try:
            status_enum = JobStatus(progress_data["status"])
        except ValueError:
            logger.error(f"Invalid status value from Redis: {progress_data['status']}")
            raise ValueError(f"Invalid status in Redis: {progress_data['status']}")

        result_ready = progress_data["status"] == "completed"
        error_details = None
        if progress_data.get("errors"):
            error_details = "\n".join(progress_data["errors"])

        result = JobStatusResult(
            job_id=query.job_id,
            status=status_enum.value,
            progress=progress_data["progress"],
            message=progress_data["message"],
            result_ready=result_ready,
            current_step=progress_data.get("stage"),
            error_details=error_details,
            created_at=None,
            updated_at=progress_data.get("last_heartbeat"),
            using_ai=progress_data.get("using_ai", False),
            ai_model=progress_data.get("ai_model"),
        )

        logger.info(
            f"Job {job_id_str} status retrieved: {status_enum.value} "
            f"({progress_data['progress']}%)"
        )
        return result


class JobNotFoundException(Exception):
    """Raised when job_id is missing from Redis (never existed, expired, or restart without persistence)."""

    def __init__(self, job_id: UUID):
        self.job_id = job_id
        super().__init__(f"Job {job_id} not found or expired")
