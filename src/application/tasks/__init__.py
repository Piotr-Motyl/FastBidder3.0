"""
Celery Tasks

Responsibility:
    Asynchronous task definitions for long-running operations.
    Progress tracking and result storage in Redis.

Contains:
    - celery_app.py - Celery configuration
    - Async task definitions (matching, embedding, file processing)
    - Progress tracking utilities

Does NOT contain:
    - Business logic (delegates to Domain services)
    - Synchronous operations (use Application services instead)
"""

from .celery_app import celery_app, health_check
from .matching_tasks import process_matching_task

__all__ = ["celery_app", "health_check", "process_matching_task"]
