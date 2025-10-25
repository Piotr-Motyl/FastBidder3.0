"""
Shared Application Models

Responsibility:
    Contains shared models used across Application Layer.
    Prevents circular dependencies and code duplication.

Architecture Notes:
    - Part of Application Layer (Shared)
    - Used by Commands, Queries, and Services
    - Enums and common DTOs that don't belong to specific modules

Contains:
    - JobStatus: Enum for job lifecycle states

Does NOT contain:
    - Business logic (belongs to Domain Layer)
    - HTTP models (belongs to API Layer)
    - Infrastructure details (belongs to Infrastructure Layer)

Phase 1 Note:
    Shared models for cross-module usage in Application Layer.
"""

from enum import Enum


class JobStatus(str, Enum):
    """
    Status of asynchronous job lifecycle.

    Represents the complete lifecycle of a Celery task from creation
    to completion or failure. Used across Application Layer and API Layer.

    Architecture Note:
        This enum is in Application Layer (not API Layer) because:
        - Multiple modules need it (Commands, Queries, Tasks)
        - API Layer can import from Application Layer (allowed by Clean Architecture)
        - Prevents circular dependencies

    Attributes:
        QUEUED: Job accepted and waiting in Celery queue
        PROCESSING: Job currently being executed by Celery worker
        COMPLETED: Job finished successfully with results available
        FAILED: Job failed with error details available
        CANCELLED: Job cancelled by user or system

    Usage:
        >>> from src.application.models import JobStatus
        >>> status = JobStatus.PROCESSING
        >>> print(status.value)  # "processing"
        >>> if status == JobStatus.COMPLETED:
        ...     print("Job done!")
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
