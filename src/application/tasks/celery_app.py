"""
Celery app initialization. Connects to Redis broker (DB0) and result backend (DB1).
Auto-discovers tasks from src.application.tasks.
"""

import os
from datetime import datetime
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

celery_app = Celery(
    "fastbidder",
    broker=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)

celery_app.conf.update(
    task_track_started=True,
    task_time_limit=300,  # 5 min hard limit
    result_expires=3600,  # results kept 1h
)

celery_app.autodiscover_tasks(["src.application.tasks"])


@celery_app.task(name="health_check")
def health_check() -> dict:
    """Verify broker + result backend + worker are up."""
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
