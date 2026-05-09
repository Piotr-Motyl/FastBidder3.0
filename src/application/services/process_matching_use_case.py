"""
ProcessMatchingUseCase — orchestrates the async matching pipeline.

Validates command + files, estimates processing time, triggers Celery task,
and returns job metadata. No HTTP/business-logic concerns.
"""

import logging
from typing import TYPE_CHECKING, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.application.commands.process_matching import ProcessMatchingCommand
from src.application.models import JobStatus
from src.application.ports.file_storage import FileStorageServiceProtocol
from src.application.ports.progress_tracker import ProgressTrackerProtocol
from src.application.tasks.matching_tasks import process_matching_task

if TYPE_CHECKING:
    from celery import Celery

logger = logging.getLogger(__name__)


class ProcessMatchingResult(BaseModel):
    """DTO returned by ProcessMatchingUseCase.execute()."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                "status": "queued",
                "estimated_time": 45,
                "message": "Matching job queued successfully. Use job_id to check status.",
            }
        }
    )

    job_id: UUID = Field(description="Celery task ID for tracking progress")
    status: JobStatus = Field(default=JobStatus.QUEUED)
    estimated_time: int = Field(ge=0, description="Estimated time to completion (seconds)")
    message: str = Field(default="Matching job queued successfully. Use job_id to check status.")


class ProcessMatchingUseCase:
    """
    Orchestrates the matching process:
      1. Validate command business rules
      2. Validate files exist (via file_storage)
      3. Estimate processing time (rows_count * 0.1s, clamped 10–300s)
      4. Pre-initialize job in progress tracker (no race condition with status endpoint)
      5. Trigger Celery task (custom task_id == job_id)
    """

    def __init__(
        self,
        celery_app: Optional["Celery"] = None,
        file_storage: Optional[FileStorageServiceProtocol] = None,
        progress_tracker: Optional[ProgressTrackerProtocol] = None,
    ):
        # celery_app kept for backwards-compat; task is invoked via module-level import.
        self.celery_app = celery_app
        self.file_storage = file_storage
        self.progress_tracker = progress_tracker

    async def execute(self, command: ProcessMatchingCommand) -> ProcessMatchingResult:
        """Run the full orchestration. Returns job metadata immediately."""
        # Step 1: command-level business rules
        command.validate_business_rules()
        logger.debug(
            f"Command validation passed for files: "
            f"{command.working_file.file_id}, {command.reference_file.file_id}"
        )

        # Steps 2–3: file existence
        if self.file_storage:
            await self._validate_files(command)

        # Steps 4–5: time estimate
        if self.file_storage:
            estimated_time = await self._estimate_processing_time(
                UUID(command.working_file.file_id), UUID(command.reference_file.file_id)
            )
        else:
            estimated_time = 30  # fallback when no file_storage injected (tests)
        logger.info(f"Estimated processing time: {estimated_time}s")

        # Step 6: generate job_id and pre-init in tracker BEFORE Celery
        # so GET /jobs/{id}/status responds immediately (no race).
        job_id = str(uuid4())
        logger.info(f"Generated job_id: {job_id}")

        if self.progress_tracker:
            self.progress_tracker.start_job(
                job_id=job_id,
                message="Job queued, waiting for worker to start processing",
                total_items=0,
            )
        else:
            logger.warning(
                f"No progress_tracker injected — job {job_id} not pre-initialized in Redis"
            )

        # Step 7: trigger Celery with custom task_id (== job_id)
        celery_data = command.to_celery_dict()
        process_matching_task.apply_async(  # type: ignore[attr-defined]
            kwargs=celery_data,
            task_id=job_id,
        )
        logger.info(f"Celery task triggered with task_id: {job_id}")

        return ProcessMatchingResult(
            job_id=UUID(job_id),
            status=JobStatus.QUEUED,
            estimated_time=estimated_time,
            message=f"Matching job queued successfully. Check status at GET /jobs/{job_id}/status",
        )

    async def _validate_files(self, command: ProcessMatchingCommand) -> None:
        """Check that working/reference upload directories exist and are non-empty."""
        assert self.file_storage is not None  # guaranteed by execute()

        wf_file_id = UUID(command.working_file.file_id)
        wf_upload_dir = self.file_storage.get_uploaded_file_path(wf_file_id)
        if not wf_upload_dir.exists() or not any(wf_upload_dir.iterdir()):
            raise FileNotFoundError(
                f"Working file not found in uploads storage: {command.working_file.file_id}"
            )

        ref_file_id = UUID(command.reference_file.file_id)
        ref_upload_dir = self.file_storage.get_uploaded_file_path(ref_file_id)
        if not ref_upload_dir.exists() or not any(ref_upload_dir.iterdir()):
            raise FileNotFoundError(
                f"Reference file not found in uploads storage: {command.reference_file.file_id}"
            )

        logger.debug(
            f"File existence validation passed for WF={wf_file_id}, REF={ref_file_id}"
        )

    async def _estimate_processing_time(
        self, wf_file_id: UUID, ref_file_id: UUID
    ) -> int:
        """Estimate processing time = rows_count * 0.1s, clamped to [10, 300]."""
        assert self.file_storage is not None  # guaranteed by execute()

        wf_upload_dir = self.file_storage.get_uploaded_file_path(wf_file_id)
        uploaded_files = list(wf_upload_dir.glob("*.xlsx"))
        if not uploaded_files:
            raise FileNotFoundError(
                f"No .xlsx file found in upload directory: {wf_upload_dir}"
            )
        wf_path = uploaded_files[0]

        metadata = await self.file_storage.extract_file_metadata(wf_path)
        rows_count = metadata["rows_count"]

        estimated_time = max(10, min(300, int(rows_count * 0.1)))
        logger.debug(
            f"Estimated processing time: {estimated_time}s "
            f"(based on {rows_count} rows * 0.1s/row)"
        )
        return estimated_time
