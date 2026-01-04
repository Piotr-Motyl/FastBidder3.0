"""
Tests for ProcessMatchingUseCase.

Covers:
- Use case execution flow
- Command validation
- File existence validation
- Time estimation
- Celery task triggering
- Redis job initialization
- Error handling
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.application.commands.process_matching import (
    ProcessMatchingCommand,
    Range,
    ReferenceFileConfig,
    WorkingFileConfig,
)
from src.application.models import JobStatus, MatchingStrategy, ReportFormat
from src.application.services.process_matching_use_case import (
    ProcessMatchingResult,
    ProcessMatchingUseCase,
)
from src.domain.shared.exceptions import InvalidProcessMatchingCommandError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_celery_app():
    """Create mock Celery application."""
    return MagicMock()


@pytest.fixture
def mock_file_storage():
    """Create mock FileStorageService."""
    storage = MagicMock()
    storage.get_uploaded_file_path = MagicMock()
    storage.extract_file_metadata = AsyncMock()
    return storage


@pytest.fixture
def use_case(mock_celery_app, mock_file_storage):
    """Create ProcessMatchingUseCase with mocked dependencies."""
    return ProcessMatchingUseCase(
        celery_app=mock_celery_app, file_storage=mock_file_storage
    )


@pytest.fixture
def valid_command():
    """Create valid ProcessMatchingCommand for testing."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    return ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
            matching_report_column="G",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
        matching_threshold=80.0,
        matching_strategy=MatchingStrategy.BEST_MATCH,
        report_format=ReportFormat.SIMPLE,
    )


@pytest.fixture
def temp_upload_dir(tmp_path):
    """Create temporary upload directory with Excel file."""
    upload_dir = tmp_path / "uploads" / str(uuid4())
    upload_dir.mkdir(parents=True)
    # Create dummy xlsx file
    xlsx_file = upload_dir / "test.xlsx"
    xlsx_file.write_text("dummy excel content")
    return upload_dir


# ============================================================================
# HAPPY PATH TESTS - ProcessMatchingResult
# ============================================================================


def test_result_creates_with_valid_data():
    """Test ProcessMatchingResult creates with valid data."""
    job_id = uuid4()

    result = ProcessMatchingResult(
        job_id=job_id, status=JobStatus.QUEUED, estimated_time=45
    )

    assert result.job_id == job_id
    assert result.status == JobStatus.QUEUED
    assert result.estimated_time == 45
    assert "queued successfully" in result.message.lower()


def test_result_uses_default_values():
    """Test ProcessMatchingResult uses default values."""
    job_id = uuid4()

    result = ProcessMatchingResult(job_id=job_id, estimated_time=30)

    assert result.status == JobStatus.QUEUED  # Default
    assert "queued successfully" in result.message.lower()  # Default


# ============================================================================
# HAPPY PATH TESTS - execute()
# ============================================================================


@pytest.mark.asyncio
async def test_execute_validates_command_business_rules(use_case):
    """Test execute() validates command business rules."""
    # Invalid command: same file IDs
    same_id = str(uuid4())
    invalid_command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=same_id,
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=same_id,  # Same ID - invalid
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
    )

    with pytest.raises(
        InvalidProcessMatchingCommandError,
        match="working_file.file_id and reference_file.file_id must be different",
    ):
        await use_case.execute(invalid_command)


@pytest.mark.asyncio
async def test_execute_validates_file_existence(
    use_case, valid_command, mock_file_storage
):
    """Test execute() validates files exist in uploads storage."""
    # Mock file_storage to return non-existent path
    non_existent_path = Path("/non/existent/path")
    mock_file_storage.get_uploaded_file_path.return_value = non_existent_path

    with pytest.raises(FileNotFoundError, match="Working file not found"):
        await use_case.execute(valid_command)


@pytest.mark.asyncio
async def test_execute_estimates_processing_time(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() estimates processing time based on file metadata."""
    # Mock file storage to return temp upload dir
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir

    # Mock metadata extraction
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 100}

    # Mock Celery task
    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        mock_task_result.id = str(uuid4())
        mock_task.apply_async.return_value = mock_task_result

        # Mock RedisProgressTracker
        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Estimated time should be 100 * 0.1 = 10 seconds (min is 10s)
            assert result.estimated_time == 10


@pytest.mark.asyncio
async def test_execute_applies_min_estimation_bound(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() applies minimum estimation of 10 seconds."""
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir

    # Small file: 50 rows * 0.1 = 5s, but min is 10s
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 50}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        mock_task_result.id = str(uuid4())
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Should be minimum 10 seconds
            assert result.estimated_time == 10


@pytest.mark.asyncio
async def test_execute_applies_max_estimation_bound(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() applies maximum estimation of 300 seconds."""
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir

    # Large file: 5000 rows * 0.1 = 500s, but max is 300s
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 5000}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        mock_task_result.id = str(uuid4())
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Should be maximum 300 seconds
            assert result.estimated_time == 300


@pytest.mark.asyncio
async def test_execute_initializes_job_in_redis(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() initializes job in Redis before triggering Celery."""
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 100}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        mock_task_result.id = str(uuid4())
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Verify Redis tracker was initialized
            mock_tracker_class.assert_called_once()
            # Verify start_job was called
            mock_tracker.start_job.assert_called_once()


@pytest.mark.asyncio
async def test_execute_triggers_celery_task(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() triggers Celery task with command data."""
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 100}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        job_id = str(uuid4())
        mock_task_result.id = job_id
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Verify Celery task was triggered
            mock_task.apply_async.assert_called_once()

            # Verify task was called with command data
            call_kwargs = mock_task.apply_async.call_args[1]
            assert "kwargs" in call_kwargs
            celery_data = call_kwargs["kwargs"]
            assert "working_file" in celery_data
            assert "reference_file" in celery_data
            assert "matching_threshold" in celery_data


@pytest.mark.asyncio
async def test_execute_uses_custom_task_id(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() uses custom task_id matching job_id."""
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 100}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        mock_task_result.id = str(uuid4())
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Verify task_id was set in apply_async
            call_kwargs = mock_task.apply_async.call_args[1]
            assert "task_id" in call_kwargs
            # task_id should match result.job_id
            assert UUID(call_kwargs["task_id"]) == result.job_id


@pytest.mark.asyncio
async def test_execute_returns_result_with_job_metadata(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() returns ProcessMatchingResult with job metadata."""
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 200}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        job_id = str(uuid4())
        mock_task_result.id = job_id
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case.execute(valid_command)

            # Verify result structure
            assert isinstance(result, ProcessMatchingResult)
            # Result.job_id comes from the generated UUID, not from mock
            # Verify it's a valid UUID by checking it's in the message
            assert str(result.job_id) in result.message
            assert result.status == JobStatus.QUEUED
            assert result.estimated_time == 20  # 200 rows * 0.1 = 20s


@pytest.mark.asyncio
async def test_execute_works_without_file_storage(valid_command):
    """Test execute() works with file_storage=None (fallback to default estimation)."""
    mock_celery_app = MagicMock()
    use_case_no_storage = ProcessMatchingUseCase(
        celery_app=mock_celery_app, file_storage=None
    )

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        mock_task_result.id = str(uuid4())
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            result = await use_case_no_storage.execute(valid_command)

            # Should use default estimation of 30 seconds
            assert result.estimated_time == 30


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_execute_raises_file_not_found_for_working_file(
    use_case, valid_command, mock_file_storage
):
    """Test execute() raises FileNotFoundError when working file missing."""
    # Mock working file as non-existent
    wf_path = Path("/non/existent/working")
    mock_file_storage.get_uploaded_file_path.side_effect = [
        wf_path,  # First call for working file
    ]

    with pytest.raises(FileNotFoundError, match="Working file not found"):
        await use_case.execute(valid_command)


@pytest.mark.asyncio
async def test_execute_raises_file_not_found_for_reference_file(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Test execute() raises FileNotFoundError when reference file missing."""
    # Mock working file as existent, reference file as non-existent
    rf_path = Path("/non/existent/reference")
    mock_file_storage.get_uploaded_file_path.side_effect = [
        temp_upload_dir,  # First call for working file (exists)
        rf_path,  # Second call for reference file (doesn't exist)
    ]

    with pytest.raises(FileNotFoundError, match="Reference file not found"):
        await use_case.execute(valid_command)


@pytest.mark.asyncio
async def test_execute_raises_when_no_xlsx_file_in_directory(
    use_case, valid_command, mock_file_storage, tmp_path
):
    """Test execute() raises FileNotFoundError when no .xlsx file in upload directory."""
    # Create upload directory with a non-xlsx file
    # (empty directory would fail at _validate_files stage)
    upload_dir = tmp_path / "upload_dir"
    upload_dir.mkdir()
    # Add a non-xlsx file so directory is not empty
    (upload_dir / "readme.txt").write_text("test")

    mock_file_storage.get_uploaded_file_path.return_value = upload_dir

    with pytest.raises(FileNotFoundError, match="No .xlsx file found in upload directory"):
        await use_case.execute(valid_command)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_full_execution_flow_happy_path(
    use_case, valid_command, mock_file_storage, temp_upload_dir
):
    """Integration test: Full execution flow from command to result."""
    # Setup mocks
    mock_file_storage.get_uploaded_file_path.return_value = temp_upload_dir
    mock_file_storage.extract_file_metadata.return_value = {"rows_count": 150}

    with patch(
        "src.application.services.process_matching_use_case.process_matching_task"
    ) as mock_task:
        mock_task_result = MagicMock()
        job_id = str(uuid4())
        mock_task_result.id = job_id
        mock_task.apply_async.return_value = mock_task_result

        with patch(
            "src.application.services.process_matching_use_case.RedisProgressTracker"
        ) as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            # Execute
            result = await use_case.execute(valid_command)

            # Verify complete flow
            # 1. Command validated (no exception)
            # 2. Files validated (working + reference in _validate_files)
            # 3. Working file path retrieved again in _estimate_processing_time
            assert mock_file_storage.get_uploaded_file_path.call_count == 3

            # 4. Metadata extracted
            mock_file_storage.extract_file_metadata.assert_called_once()

            # 4. Redis initialized
            mock_tracker.start_job.assert_called_once()

            # 5. Celery task triggered
            mock_task.apply_async.assert_called_once()

            # 6. Result returned with valid job_id
            # job_id is generated internally, not from mock
            assert isinstance(result.job_id, UUID)
            assert result.status == JobStatus.QUEUED
            assert result.estimated_time == 15  # 150 * 0.1 = 15s
            assert str(result.job_id) in result.message
