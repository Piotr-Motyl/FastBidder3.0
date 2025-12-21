"""
Tests for GetJobStatusQuery and GetJobStatusQueryHandler.

Covers:
- Query creation and validation
- Handler initialization
- Job status retrieval from Redis
- Status mapping to JobStatusResult
- Error handling for missing jobs
- Field mapping and transformation
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.application.models import JobStatus
from src.application.queries.get_job_status import (
    GetJobStatusQuery,
    GetJobStatusQueryHandler,
    JobNotFoundException,
    JobStatusResult,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_progress_tracker():
    """Create mock RedisProgressTracker."""
    return MagicMock()


@pytest.fixture
def handler(mock_progress_tracker):
    """Create GetJobStatusQueryHandler with mocked tracker."""
    return GetJobStatusQueryHandler(progress_tracker=mock_progress_tracker)


@pytest.fixture
def sample_job_id():
    """Create sample job ID for testing."""
    return uuid4()


@pytest.fixture
def processing_progress_data():
    """Create sample progress data for processing job."""
    return {
        "status": "processing",
        "progress": 45,
        "message": "Matching descriptions",
        "current_item": 450,
        "total_items": 1000,
        "stage": "MATCHING",
        "eta_seconds": 120,
        "memory_mb": 512.5,
        "errors": [],
        "last_heartbeat": "2025-01-11T10:30:45.123",
    }


@pytest.fixture
def completed_progress_data():
    """Create sample progress data for completed job."""
    return {
        "status": "completed",
        "progress": 100,
        "message": "Matching completed successfully",
        "stage": "COMPLETE",
        "errors": [],
        "last_heartbeat": "2025-01-11T11:00:00.000",
    }


@pytest.fixture
def failed_progress_data():
    """Create sample progress data for failed job."""
    return {
        "status": "failed",
        "progress": 60,
        "message": "Processing failed",
        "stage": "MATCHING",
        "errors": ["File not found: working.xlsx", "Invalid format in column C"],
        "last_heartbeat": "2025-01-11T10:45:00.000",
    }


# ============================================================================
# HAPPY PATH TESTS - GetJobStatusQuery
# ============================================================================


def test_query_creates_with_valid_job_id(sample_job_id):
    """Test GetJobStatusQuery creates with valid job ID."""
    query = GetJobStatusQuery(job_id=sample_job_id)

    assert query.job_id == sample_job_id


def test_query_validates_uuid_format():
    """Test GetJobStatusQuery validates job_id is UUID format."""
    # Valid UUID
    query = GetJobStatusQuery(job_id=uuid4())
    assert query.job_id is not None

    # Invalid UUID should raise validation error
    with pytest.raises(Exception):  # Pydantic ValidationError
        GetJobStatusQuery(job_id="not-a-uuid")


# ============================================================================
# HAPPY PATH TESTS - JobStatusResult
# ============================================================================


def test_result_creates_with_valid_data(sample_job_id):
    """Test JobStatusResult creates with valid data."""
    result = JobStatusResult(
        job_id=sample_job_id,
        status="processing",
        progress=45,
        message="Matching descriptions",
        result_ready=False,
        current_step="MATCHING",
    )

    assert result.job_id == sample_job_id
    assert result.status == "processing"
    assert result.progress == 45
    assert result.message == "Matching descriptions"
    assert result.result_ready is False
    assert result.current_step == "MATCHING"


def test_result_validates_progress_range(sample_job_id):
    """Test JobStatusResult validates progress is 0-100."""
    # Below minimum
    with pytest.raises(Exception):  # Pydantic ValidationError
        JobStatusResult(
            job_id=sample_job_id,
            status="processing",
            progress=-10,  # Invalid
            message="Test",
        )

    # Above maximum
    with pytest.raises(Exception):  # Pydantic ValidationError
        JobStatusResult(
            job_id=sample_job_id,
            status="processing",
            progress=150,  # Invalid
            message="Test",
        )


def test_result_accepts_boundary_progress_values(sample_job_id):
    """Test JobStatusResult accepts boundary values 0 and 100."""
    # Minimum value
    result_min = JobStatusResult(
        job_id=sample_job_id, status="queued", progress=0, message="Queued"
    )
    assert result_min.progress == 0

    # Maximum value
    result_max = JobStatusResult(
        job_id=sample_job_id, status="completed", progress=100, message="Done"
    )
    assert result_max.progress == 100


def test_result_uses_default_values(sample_job_id):
    """Test JobStatusResult uses default values for optional fields."""
    result = JobStatusResult(
        job_id=sample_job_id, status="queued", progress=0, message="Queued"
    )

    assert result.result_ready is False  # Default
    assert result.current_step is None  # Default
    assert result.error_details is None  # Default
    assert result.created_at is None  # Default
    assert result.updated_at is None  # Default


# ============================================================================
# HAPPY PATH TESTS - GetJobStatusQueryHandler initialization
# ============================================================================


def test_handler_initializes_with_progress_tracker(mock_progress_tracker):
    """Test GetJobStatusQueryHandler initializes with progress tracker."""
    handler = GetJobStatusQueryHandler(progress_tracker=mock_progress_tracker)

    assert handler.progress_tracker == mock_progress_tracker


def test_handler_initializes_with_none_tracker():
    """Test GetJobStatusQueryHandler allows None tracker (for testing)."""
    handler = GetJobStatusQueryHandler(progress_tracker=None)

    assert handler.progress_tracker is None


# ============================================================================
# HAPPY PATH TESTS - handle() for processing job
# ============================================================================


@pytest.mark.asyncio
async def test_handle_retrieves_processing_job_status(
    handler, sample_job_id, processing_progress_data, mock_progress_tracker
):
    """Test handle() retrieves status for processing job."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = processing_progress_data

    result = await handler.handle(query)

    # Verify get_status was called with correct job_id
    mock_progress_tracker.get_status.assert_called_once_with(str(sample_job_id))

    # Verify result mapping
    assert result.job_id == sample_job_id
    assert result.status == "processing"
    assert result.progress == 45
    assert result.message == "Matching descriptions"
    assert result.result_ready is False  # Not completed
    assert result.current_step == "MATCHING"
    assert result.error_details is None  # No errors
    assert result.updated_at == "2025-01-11T10:30:45.123"


@pytest.mark.asyncio
async def test_handle_maps_all_fields_correctly(
    handler, sample_job_id, processing_progress_data, mock_progress_tracker
):
    """Test handle() maps all Redis fields to JobStatusResult correctly."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = processing_progress_data

    result = await handler.handle(query)

    # Field-by-field verification
    assert result.job_id == sample_job_id  # From query
    assert result.status == processing_progress_data["status"]  # Direct copy
    assert result.progress == processing_progress_data["progress"]  # Direct copy
    assert result.message == processing_progress_data["message"]  # Direct copy
    assert result.current_step == processing_progress_data["stage"]  # Mapped
    assert (
        result.updated_at == processing_progress_data["last_heartbeat"]
    )  # Mapped


# ============================================================================
# HAPPY PATH TESTS - handle() for completed job
# ============================================================================


@pytest.mark.asyncio
async def test_handle_retrieves_completed_job_status(
    handler, sample_job_id, completed_progress_data, mock_progress_tracker
):
    """Test handle() retrieves status for completed job."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = completed_progress_data

    result = await handler.handle(query)

    assert result.status == "completed"
    assert result.progress == 100
    assert result.result_ready is True  # Completed → result ready


@pytest.mark.asyncio
async def test_handle_sets_result_ready_true_for_completed(
    handler, sample_job_id, completed_progress_data, mock_progress_tracker
):
    """Test handle() sets result_ready=True only for completed jobs."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = completed_progress_data

    result = await handler.handle(query)

    # result_ready should be True for completed status
    assert result.result_ready is True


# ============================================================================
# HAPPY PATH TESTS - handle() for failed job
# ============================================================================


@pytest.mark.asyncio
async def test_handle_retrieves_failed_job_status(
    handler, sample_job_id, failed_progress_data, mock_progress_tracker
):
    """Test handle() retrieves status for failed job."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = failed_progress_data

    result = await handler.handle(query)

    assert result.status == "failed"
    assert result.progress == 60
    assert result.result_ready is False  # Failed → no result
    assert result.error_details is not None


@pytest.mark.asyncio
async def test_handle_formats_error_details_from_errors_list(
    handler, sample_job_id, failed_progress_data, mock_progress_tracker
):
    """Test handle() formats error_details from errors list."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = failed_progress_data

    result = await handler.handle(query)

    # error_details should be newline-joined string
    assert "File not found: working.xlsx" in result.error_details
    assert "Invalid format in column C" in result.error_details
    assert "\n" in result.error_details  # Newline separator


@pytest.mark.asyncio
async def test_handle_sets_error_details_none_when_no_errors(
    handler, sample_job_id, processing_progress_data, mock_progress_tracker
):
    """Test handle() sets error_details=None when errors list is empty."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    # processing_progress_data has empty errors list
    mock_progress_tracker.get_status.return_value = processing_progress_data

    result = await handler.handle(query)

    assert result.error_details is None


# ============================================================================
# ERROR HANDLING TESTS - handle()
# ============================================================================


@pytest.mark.asyncio
async def test_handle_raises_job_not_found_when_redis_returns_none(
    handler, sample_job_id, mock_progress_tracker
):
    """Test handle() raises JobNotFoundException when Redis returns None."""
    query = GetJobStatusQuery(job_id=sample_job_id)
    mock_progress_tracker.get_status.return_value = None  # Job not found

    with pytest.raises(JobNotFoundException, match=str(sample_job_id)):
        await handler.handle(query)


@pytest.mark.asyncio
async def test_handle_raises_when_tracker_is_none(sample_job_id):
    """Test handle() raises RuntimeError when progress_tracker is None."""
    handler_no_tracker = GetJobStatusQueryHandler(progress_tracker=None)
    query = GetJobStatusQuery(job_id=sample_job_id)

    with pytest.raises(RuntimeError, match="RedisProgressTracker not initialized"):
        await handler_no_tracker.handle(query)


@pytest.mark.asyncio
async def test_handle_raises_value_error_for_invalid_status(
    handler, sample_job_id, mock_progress_tracker
):
    """Test handle() raises ValueError for invalid status from Redis."""
    query = GetJobStatusQuery(job_id=sample_job_id)

    # Invalid status value
    invalid_data = {
        "status": "invalid_status",  # Not in JobStatus enum
        "progress": 50,
        "message": "Test",
    }
    mock_progress_tracker.get_status.return_value = invalid_data

    with pytest.raises(ValueError, match="Invalid status in Redis"):
        await handler.handle(query)


# ============================================================================
# ERROR HANDLING TESTS - JobNotFoundException
# ============================================================================


def test_job_not_found_exception_creates_with_job_id(sample_job_id):
    """Test JobNotFoundException creates with job_id."""
    exc = JobNotFoundException(job_id=sample_job_id)

    assert exc.job_id == sample_job_id
    assert str(sample_job_id) in str(exc)
    assert "not found or expired" in str(exc)


# ============================================================================
# EDGE CASES TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_handle_handles_missing_optional_fields(
    handler, sample_job_id, mock_progress_tracker
):
    """Test handle() handles missing optional fields in progress_data."""
    query = GetJobStatusQuery(job_id=sample_job_id)

    # Minimal progress data (missing optional fields)
    minimal_data = {
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        # stage, errors, last_heartbeat missing
    }
    mock_progress_tracker.get_status.return_value = minimal_data

    result = await handler.handle(query)

    assert result.status == "queued"
    assert result.progress == 0
    assert result.current_step is None  # Missing stage
    assert result.error_details is None  # Missing errors
    assert result.updated_at is None  # Missing last_heartbeat


@pytest.mark.asyncio
async def test_handle_supports_all_job_status_values(
    handler, sample_job_id, mock_progress_tracker
):
    """Test handle() supports all JobStatus enum values."""
    query = GetJobStatusQuery(job_id=sample_job_id)

    # Test each status value
    statuses = ["queued", "processing", "completed", "failed", "cancelled"]

    for status_value in statuses:
        progress_data = {
            "status": status_value,
            "progress": 50,
            "message": f"Job {status_value}",
        }
        mock_progress_tracker.get_status.return_value = progress_data

        result = await handler.handle(query)

        assert result.status == status_value


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_full_query_flow_from_creation_to_result(
    mock_progress_tracker, processing_progress_data
):
    """Integration test: Full flow from query creation to result."""
    job_id = uuid4()

    # Step 1: Create query
    query = GetJobStatusQuery(job_id=job_id)
    assert query.job_id == job_id

    # Step 2: Create handler
    handler = GetJobStatusQueryHandler(progress_tracker=mock_progress_tracker)

    # Step 3: Mock Redis response
    mock_progress_tracker.get_status.return_value = processing_progress_data

    # Step 4: Execute query
    result = await handler.handle(query)

    # Step 5: Verify complete result
    assert isinstance(result, JobStatusResult)
    assert result.job_id == job_id
    assert result.status == "processing"
    assert result.progress == 45
    assert result.message == "Matching descriptions"
    assert result.result_ready is False
    assert result.current_step == "MATCHING"
    assert result.updated_at == "2025-01-11T10:30:45.123"
