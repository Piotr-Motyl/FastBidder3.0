"""
Shared Application-layer enums (used by Commands, Queries, Tasks).

Lives here (not in API or Domain) to avoid circular imports.
"""

from enum import Enum


class JobStatus(str, Enum):
    """Lifecycle states of an async job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MatchingStrategy(str, Enum):
    """How to handle multiple matching candidates."""

    FIRST_MATCH = "first_match"   # stop at first candidate >= threshold
    BEST_MATCH = "best_match"     # default — return highest-scoring candidate
    ALL_MATCHES = "all_matches"   # return every candidate >= threshold


class ReportFormat(str, Enum):
    """Verbosity of the matching report column in the Excel output."""

    SIMPLE = "simple"       # score + matched description
    DETAILED = "detailed"   # + parameter breakdown
    DEBUG = "debug"         # + per-component scores, confidence
