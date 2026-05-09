"""
Progress Tracker Protocol — interface implemented by Infrastructure
(RedisProgressTracker). Used by Application Layer use cases and Celery tasks.
"""

from typing import Optional, Protocol


class ProgressTrackerProtocol(Protocol):
    """Interface for job progress tracking. Concrete impl: RedisProgressTracker."""

    def start_job(
        self,
        job_id: str,
        message: str = "Job started",
        total_items: int = 0,
    ) -> None:
        """Initialize job entry with QUEUED/PROCESSING status."""
        ...

    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str,
        current_item: int = 0,
        total_items: int = 0,
        stage: str = "",
        eta_seconds: int = 0,
        memory_mb: float = 0.0,
        errors: Optional[list[str]] = None,
    ) -> None:
        """Update job progress percentage with extended metadata."""
        ...

    def complete_job(
        self,
        job_id: str,
        result: Optional[dict] = None,
    ) -> None:
        """Mark job as completed with optional result metadata."""
        ...

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed with error details."""
        ...

    def get_status(self, job_id: str) -> Optional[dict]:
        """Retrieve current job status dict, or None if not found."""
        ...
