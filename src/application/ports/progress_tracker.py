"""
Progress Tracker Protocol

Defines the interface that Infrastructure Layer must implement
for job progress tracking. Used by Application Layer use cases and API layer.

Architecture Notes:
    - Part of Application Layer (Ports/Interfaces)
    - Implements Dependency Inversion Principle (SOLID)
    - Infrastructure Layer provides concrete implementation (RedisProgressTracker)
    - Enables easy testing with mock implementations
"""

from typing import Optional, Protocol


class ProgressTrackerProtocol(Protocol):
    """
    Protocol (interface) for job progress tracking.

    Defines the contract that Infrastructure Layer's RedisProgressTracker
    must implement. Follows Dependency Inversion Principle.

    Used by:
        - ProcessMatchingUseCase: to initialize job status before Celery task
        - GetJobStatusQueryHandler: to retrieve current job status
        - Celery task: to update progress during processing
    """

    def start_job(
        self,
        job_id: str,
        message: str = "Job started",
        total_items: int = 0,
    ) -> None:
        """Initialize job entry with QUEUED status."""
        ...

    def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str = "",
        current_step: Optional[str] = None,
        items_processed: Optional[int] = None,
        items_matched: Optional[int] = None,
    ) -> None:
        """Update job progress percentage and status message."""
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
