"""
Celery tasks package.

This package contains all Celery task definitions and the main
celery_app instance used for async processing.

For Task 0.1.2, only the basic celery_app and health_check task
are included to test Redis-Celery connectivity.

Usage:
    Start worker:
        celery -A src.application.tasks worker --loglevel=info

    Run task:
        from src.application.tasks import health_check
        result = health_check.delay()
"""

from .celery_app import celery_app, health_check

__all__ = ["celery_app", "health_check"]
