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


class MatchingStrategy(str, Enum):
    """
    Strategy for handling multiple matches during matching process.

    When a working file description matches multiple reference descriptions,
    this enum defines which match(es) to return.

    Architecture Note:
        Used by ProcessMatchingCommand to configure matching behavior.
        Application Layer enum shared across Commands and Tasks.

    Attributes:
        FIRST_MATCH: Return first match found (fastest, minimal processing)
        BEST_MATCH: Return match with highest score (default, balanced)
        ALL_MATCHES: Return all matches above threshold (comprehensive)

    Usage:
        >>> from src.application.models import MatchingStrategy
        >>> strategy = MatchingStrategy.BEST_MATCH
        >>> if strategy == MatchingStrategy.ALL_MATCHES:
        ...     print("Return all matches")

    Business Rules:
        - FIRST_MATCH: Stops after finding first match >= threshold
        - BEST_MATCH: Evaluates all candidates, returns highest score
        - ALL_MATCHES: Returns list of all matches >= threshold

    Phase 2 Note:
        Default strategy is BEST_MATCH (highest quality results).
        FIRST_MATCH useful for performance optimization in later phases.
        ALL_MATCHES useful for manual verification workflows.
    """

    FIRST_MATCH = "first_match"
    BEST_MATCH = "best_match"
    ALL_MATCHES = "all_matches"


class ReportFormat(str, Enum):
    """
    Format of matching report generated in Excel output.

    Defines level of detail in matching report column.
    Used to balance readability vs debugging information.

    Architecture Note:
        Used by ProcessMatchingCommand to configure output verbosity.
        Application Layer enum shared across Commands and Tasks.

    Attributes:
        SIMPLE: Basic match info (score + matched description)
        DETAILED: Include parameter breakdown (DN, PN, material)
        DEBUG: Full debug info (all scores, parameters, confidence)

    Usage:
        >>> from src.application.models import ReportFormat
        >>> format = ReportFormat.SIMPLE
        >>> if format == ReportFormat.DEBUG:
        ...     print("Include debug information")

    Report Examples:
        SIMPLE:
            "Matched: Zawór DN50 PN16 | Score: 95.2%"

        DETAILED:
            "Matched: Zawór DN50 PN16 | Score: 95.2% | DN: 50 (match) | PN: 16 (match) | Material: brass"

        DEBUG:
            "Matched: Zawór DN50 PN16 | Final: 95.2% | Param: 100% (DN:1.0, PN:1.0) | Semantic: 92% | Confidence: High"

    Phase 2 Note:
        Default format is SIMPLE (clean output for end users).
        DETAILED for manual verification workflows.
        DEBUG for troubleshooting and model tuning.
    """

    SIMPLE = "simple"
    DETAILED = "detailed"
    DEBUG = "debug"
