"""
Centralized Dependency Injection for FastAPI

Single place for all DI wiring. Routers import their dependencies from here.
If constructor signatures of infrastructure services change, this is the only
file that needs updating.
"""

from fastapi import Depends

from src.application.ports.progress_tracker import ProgressTrackerProtocol
from src.application.queries.get_job_status import GetJobStatusQueryHandler
from src.application.services.process_matching_use_case import ProcessMatchingUseCase
from src.infrastructure.file_storage.file_storage_service import FileStorageService
from src.infrastructure.persistence.redis.progress_tracker import RedisProgressTracker


def get_progress_tracker() -> ProgressTrackerProtocol:
    return RedisProgressTracker()


def get_file_storage() -> FileStorageService:
    return FileStorageService()


async def get_process_matching_use_case(
    file_storage: FileStorageService = Depends(get_file_storage),
    progress_tracker: ProgressTrackerProtocol = Depends(get_progress_tracker),
) -> ProcessMatchingUseCase:
    return ProcessMatchingUseCase(
        file_storage=file_storage,
        celery_app=None,
        progress_tracker=progress_tracker,
    )


async def get_job_status_query_handler(
    progress_tracker: ProgressTrackerProtocol = Depends(get_progress_tracker),
) -> GetJobStatusQueryHandler:
    return GetJobStatusQueryHandler(progress_tracker=progress_tracker)
