"""
Application Layer Package

Responsibility:
    Coordinates use cases, manages asynchronous task processing with Celery,
    and implements CQRS pattern.

Architecture Notes:
    - Orchestration layer between API and Domain
    - Contains Commands (write), Queries (read), and Use Cases
    - Celery tasks for long-running operations
    - Shared models (enums, DTOs)

Contains:
    - commands/: CQRS write operations
    - queries/: CQRS read operations
    - services/: Use Cases (orchestration)
    - tasks/: Celery async tasks
    - models: Shared Application Layer models

Does NOT contain:
    - Domain business rules (in Domain Layer)
    - HTTP handling (in API Layer)
    - Infrastructure details (in Infrastructure Layer)

Phase 1 Note:
    Currently contains contracts only. Implementation in Phase 3.
"""

# Re-export commonly used models for convenience
from src.application.models import JobStatus

__all__ = [
    "JobStatus",
]
