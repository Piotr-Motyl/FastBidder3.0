"""
Celery application initialization - MINIMAL setup for Task 0.1.2.

This module creates the simplest possible Celery app to test:
1. Connection to Redis broker
2. Result backend storage
3. Basic task execution (health_check)

Architecture Note:
- Part of Application Layer (orchestration)
- Uses environment variables for configuration
- No business logic - pure infrastructure setup
"""

import os
from datetime import datetime
from celery import Celery
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create Celery instance with minimal config from environment
celery_app = Celery(
    "fastbidder",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

# Basic configuration
celery_app.conf.update(
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    result_expires=3600,  # Results expire after 1 hour
)

# Autodiscover tasks from src.application.tasks module
# This will automatically find and register all @celery_app.task decorated functions
# in matching_tasks.py and other task modules
celery_app.autodiscover_tasks(["src.application.tasks"])


@celery_app.task(name="health_check")
def health_check() -> dict:
    """
    Simple health check task to verify Celery-Redis connection.

    This task:
    - Tests broker connection (Redis)
    - Tests result backend storage
    - Validates task execution pipeline

    Returns:
        dict: Status information with timestamp
            - status (str): "ok" if healthy
            - message (str): Human-readable status message
            - timestamp (str): ISO format timestamp
            - worker (str): Worker hostname that executed the task

    Example:
        >>> from src.application.tasks.celery_app import health_check
        >>> result = health_check.delay()
        >>> print(result.get(timeout=5))
        {
            'status': 'ok',
            'message': 'Celery worker is healthy',
            'timestamp': '2025-10-03T10:30:45.123456',
            'worker': 'celery@hostname'
        }
    """
    return {
        "status": "ok",
        "message": "Celery worker is healthy",
        "timestamp": datetime.now().isoformat(),
        "worker": (
            celery_app.current_task.request.hostname
            if celery_app.current_task
            else "unknown"
        ),
    }
