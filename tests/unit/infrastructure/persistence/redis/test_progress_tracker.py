"""
Tests for RedisProgressTracker.

Covers:
- Job lifecycle (start, update, complete, fail)
- Progress tracking with extended metadata
- History tracking (last 10 updates)
- Heartbeat mechanism
- TTL configuration
- Atomic operations (MULTI/EXEC)
- Error recovery with fallback
- Cleanup old jobs
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import RedisError

from src.infrastructure.persistence.redis.progress_tracker import (
    RedisProgressTracker,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_redis():
    """Create mock Redis client for testing."""
    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    redis_mock.pipeline.return_value = redis_mock  # Pipeline returns itself
    redis_mock.execute.return_value = [True, True, True]  # Mock pipeline execution
    return redis_mock


@pytest.fixture
def tracker(mock_redis):
    """Create RedisProgressTracker with mocked Redis."""
    with patch("src.infrastructure.persistence.redis.progress_tracker.Redis") as redis_class:
        redis_class.return_value = mock_redis
        tracker = RedisProgressTracker(
            redis_host="localhost", redis_port=6379, redis_db=0
        )
        return tracker


@pytest.fixture
def job_id():
    """Sample job ID for testing."""
    return "test-job-123"


@pytest.fixture
def temp_fallback_dir(tmp_path):
    """Create temporary fallback directory."""
    fallback_dir = tmp_path / "fallback"
    fallback_dir.mkdir()
    return fallback_dir


# ============================================================================
# HAPPY PATH TESTS - Key Generation
# ============================================================================


def test_get_progress_key(tracker, job_id):
    """Test progress key generation."""
    key = tracker._get_progress_key(job_id)
    assert key == f"progress:{job_id}"
    assert isinstance(key, str)


def test_get_result_key(tracker, job_id):
    """Test result key generation."""
    key = tracker._get_result_key(job_id)
    assert key == f"result:{job_id}"
    assert isinstance(key, str)


def test_get_history_key(tracker, job_id):
    """Test history key generation."""
    key = tracker._get_history_key(job_id)
    assert key == f"progress:{job_id}:history"
    assert isinstance(key, str)


# ============================================================================
# HAPPY PATH TESTS - start_job()
# ============================================================================


def test_start_job_creates_initial_progress(tracker, mock_redis, job_id):
    """Test start_job creates initial progress with status 'processing'."""
    tracker.start_job(job_id, message="Starting job", total_items=100)

    # Verify pipeline was used (MULTI/EXEC)
    mock_redis.pipeline.assert_called_once()

    # Verify setex was called for progress key
    calls = mock_redis.setex.call_args_list
    assert len(calls) >= 1

    # Check first call (progress data)
    progress_key, ttl, data_json = calls[0][0]
    assert progress_key == f"progress:{job_id}"
    assert ttl == tracker.progress_ttl

    # Verify progress data structure
    progress_data = json.loads(data_json)
    assert progress_data["status"] == "processing"
    assert progress_data["progress"] == 0
    assert progress_data["message"] == "Starting job"
    assert progress_data["total_items"] == 100
    assert progress_data["stage"] == "START"
    assert "last_heartbeat" in progress_data


def test_start_job_creates_history_entry(tracker, mock_redis, job_id):
    """Test start_job creates initial history entry."""
    tracker.start_job(job_id, message="Starting job")

    # Verify lpush was called for history
    mock_redis.lpush.assert_called()

    # Get the history entry that was pushed
    lpush_calls = mock_redis.lpush.call_args_list
    assert len(lpush_calls) >= 1

    history_key, history_json = lpush_calls[0][0]
    assert history_key == f"progress:{job_id}:history"

    history_entry = json.loads(history_json)
    assert history_entry["progress"] == 0
    assert history_entry["message"] == "Starting job"
    assert history_entry["stage"] == "START"
    assert "timestamp" in history_entry


def test_start_job_sets_ttl_on_history(tracker, mock_redis, job_id):
    """Test start_job sets TTL on history key."""
    tracker.start_job(job_id, message="Starting job")

    # Verify expire was called for history key
    mock_redis.expire.assert_called()
    expire_calls = mock_redis.expire.call_args_list

    # Find the expire call for history key
    history_key = f"progress:{job_id}:history"
    history_expire = [c for c in expire_calls if c[0][0] == history_key]
    assert len(history_expire) == 1
    assert history_expire[0][0][1] == tracker.progress_ttl


def test_start_job_uses_atomic_operation(tracker, mock_redis, job_id):
    """Test start_job uses pipeline for atomic operation."""
    tracker.start_job(job_id)

    # Verify pipeline was created and executed
    mock_redis.pipeline.assert_called_once()
    mock_redis.execute.assert_called()


# ============================================================================
# HAPPY PATH TESTS - update_progress()
# ============================================================================


def test_update_progress_updates_data(tracker, mock_redis, job_id):
    """Test update_progress updates progress data."""
    # Mock get_status to return existing data
    mock_redis.get.return_value = json.dumps({"status": "processing", "progress": 0})

    tracker.update_progress(
        job_id,
        progress=50,
        message="Processing items",
        current_item=50,
        total_items=100,
        stage="MATCHING",
        eta_seconds=120,
        memory_mb=256.5,
    )

    # Verify setex was called
    calls = mock_redis.setex.call_args_list
    assert len(calls) >= 1

    progress_key, ttl, data_json = calls[0][0]
    progress_data = json.loads(data_json)

    assert progress_data["progress"] == 50
    assert progress_data["message"] == "Processing items"
    assert progress_data["current_item"] == 50
    assert progress_data["total_items"] == 100
    assert progress_data["stage"] == "MATCHING"
    assert progress_data["eta_seconds"] == 120
    assert progress_data["memory_mb"] == 256.5


def test_update_progress_adds_history_entry(tracker, mock_redis, job_id):
    """Test update_progress adds entry to history."""
    mock_redis.get.return_value = json.dumps({"status": "processing"})

    tracker.update_progress(
        job_id, progress=50, message="Half done", stage="MATCHING"
    )

    # Verify lpush was called
    lpush_calls = mock_redis.lpush.call_args_list
    assert len(lpush_calls) >= 1

    history_key, history_json = lpush_calls[0][0]
    history_entry = json.loads(history_json)

    assert history_entry["progress"] == 50
    assert history_entry["message"] == "Half done"
    assert history_entry["stage"] == "MATCHING"


def test_update_progress_trims_history_to_max_entries(tracker, mock_redis, job_id):
    """Test update_progress trims history to max 10 entries."""
    mock_redis.get.return_value = json.dumps({"status": "processing"})

    tracker.update_progress(job_id, progress=50, message="Test")

    # Verify ltrim was called with correct range (0, 9 for max 10 entries)
    mock_redis.ltrim.assert_called()
    ltrim_calls = mock_redis.ltrim.call_args_list

    history_key = f"progress:{job_id}:history"
    ltrim_for_history = [c for c in ltrim_calls if c[0][0] == history_key]
    assert len(ltrim_for_history) >= 1

    # Check trim range (0 to max_history_entries - 1)
    assert ltrim_for_history[0][0][1] == 0
    assert ltrim_for_history[0][0][2] == tracker.max_history_entries - 1


def test_update_progress_validates_range(tracker, job_id):
    """Test update_progress raises ValueError for invalid progress."""
    with pytest.raises(ValueError, match="Progress must be 0-100"):
        tracker.update_progress(job_id, progress=150, message="Invalid")

    with pytest.raises(ValueError, match="Progress must be 0-100"):
        tracker.update_progress(job_id, progress=-10, message="Invalid")


def test_update_progress_accepts_boundary_values(tracker, mock_redis, job_id):
    """Test update_progress accepts boundary values 0 and 100."""
    mock_redis.get.return_value = json.dumps({"status": "processing"})

    # Should not raise
    tracker.update_progress(job_id, progress=0, message="Start")
    tracker.update_progress(job_id, progress=100, message="Complete")


# ============================================================================
# HAPPY PATH TESTS - heartbeat()
# ============================================================================


def test_heartbeat_updates_timestamp(tracker, mock_redis, job_id):
    """Test heartbeat updates last_heartbeat timestamp."""
    # Mock existing progress
    existing_data = {
        "status": "processing",
        "progress": 50,
        "message": "Working",
        "last_heartbeat": "2025-01-01T00:00:00",
    }
    mock_redis.get.return_value = json.dumps(existing_data)

    tracker.heartbeat(job_id)

    # Verify setex was called
    calls = mock_redis.setex.call_args_list
    assert len(calls) >= 1

    progress_key, ttl, data_json = calls[0][0]
    updated_data = json.loads(data_json)

    # Heartbeat timestamp should be updated
    assert updated_data["last_heartbeat"] != "2025-01-01T00:00:00"
    # Other fields should be preserved
    assert updated_data["progress"] == 50
    assert updated_data["message"] == "Working"


def test_heartbeat_handles_unknown_job(tracker, mock_redis, job_id):
    """Test heartbeat handles unknown job gracefully."""
    mock_redis.get.return_value = None

    # Should not raise - just log warning
    tracker.heartbeat(job_id)

    # Verify setex was NOT called (no update for unknown job)
    mock_redis.setex.assert_not_called()


# ============================================================================
# HAPPY PATH TESTS - complete_job()
# ============================================================================


def test_complete_job_sets_status_completed(tracker, mock_redis, job_id):
    """Test complete_job sets status to 'completed' and progress to 100."""
    result_data = {"matches_count": 950, "result_file_id": "result-uuid"}

    tracker.complete_job(job_id, result=result_data)

    # Verify setex calls (progress + result)
    calls = mock_redis.setex.call_args_list
    assert len(calls) >= 2  # One for progress, one for result

    # Check progress data
    progress_key, ttl, progress_json = calls[0][0]
    progress_data = json.loads(progress_json)

    assert progress_data["status"] == "completed"
    assert progress_data["progress"] == 100
    assert progress_data["stage"] == "COMPLETE"


def test_complete_job_stores_result_with_longer_ttl(tracker, mock_redis, job_id):
    """Test complete_job stores result with longer TTL than progress."""
    result_data = {"test": "data"}

    tracker.complete_job(job_id, result=result_data)

    calls = mock_redis.setex.call_args_list

    # Find result key call
    result_calls = [c for c in calls if "result:" in c[0][0]]
    assert len(result_calls) == 1

    result_key, ttl, result_json = result_calls[0][0]
    assert result_key == f"result:{job_id}"
    assert ttl == tracker.result_ttl  # Longer TTL
    assert ttl > tracker.progress_ttl


def test_complete_job_compresses_large_result(tracker, mock_redis, job_id):
    """Test complete_job compresses result larger than 1MB."""
    # Create large result (>1MB)
    large_result = {"data": "x" * (1024 * 1024 + 1000)}  # >1MB

    tracker.complete_job(job_id, result=large_result)

    # Find result storage call
    calls = mock_redis.setex.call_args_list
    result_calls = [c for c in calls if "result:" in c[0][0]]
    assert len(result_calls) == 1

    result_key, ttl, result_json = result_calls[0][0]
    stored_data = json.loads(result_json)

    # Should have compression flag
    assert "compressed" in stored_data
    assert stored_data["compressed"] is True
    assert "data" in stored_data


def test_complete_job_does_not_compress_small_result(tracker, mock_redis, job_id):
    """Test complete_job does not compress result smaller than 1MB."""
    small_result = {"matches_count": 100}

    tracker.complete_job(job_id, result=small_result)

    # Find result storage call
    calls = mock_redis.setex.call_args_list
    result_calls = [c for c in calls if "result:" in c[0][0]]
    assert len(result_calls) == 1

    result_key, ttl, result_json = result_calls[0][0]
    stored_data = json.loads(result_json)

    # Should be stored directly without compression
    assert stored_data == small_result


# ============================================================================
# HAPPY PATH TESTS - fail_job()
# ============================================================================


def test_fail_job_sets_status_failed(tracker, mock_redis, job_id):
    """Test fail_job sets status to 'failed' and adds error."""
    # Mock current progress
    mock_redis.get.return_value = json.dumps(
        {"status": "processing", "progress": 50, "stage": "MATCHING", "errors": []}
    )

    tracker.fail_job(job_id, error_message="File not found")

    # Verify setex was called
    calls = mock_redis.setex.call_args_list
    assert len(calls) >= 1

    progress_key, ttl, data_json = calls[0][0]
    progress_data = json.loads(data_json)

    assert progress_data["status"] == "failed"
    assert progress_data["progress"] == 50  # Preserved
    assert "File not found" in progress_data["errors"]


def test_fail_job_appends_to_existing_errors(tracker, mock_redis, job_id):
    """Test fail_job appends error to existing errors list."""
    # Mock progress with existing error
    mock_redis.get.return_value = json.dumps(
        {
            "status": "processing",
            "progress": 30,
            "stage": "MATCHING",
            "errors": ["Previous error"],
        }
    )

    tracker.fail_job(job_id, error_message="New error")

    calls = mock_redis.setex.call_args_list
    progress_data = json.loads(calls[0][0][2])

    # Should have both errors
    assert len(progress_data["errors"]) == 2
    assert "Previous error" in progress_data["errors"]
    assert "New error" in progress_data["errors"]


# ============================================================================
# HAPPY PATH TESTS - get_status()
# ============================================================================


def test_get_status_returns_progress_data(tracker, mock_redis, job_id):
    """Test get_status returns progress data from Redis."""
    progress_data = {
        "status": "processing",
        "progress": 75,
        "message": "Almost done",
        "stage": "SAVING",
    }
    mock_redis.get.return_value = json.dumps(progress_data)

    result = tracker.get_status(job_id)

    assert result == progress_data
    mock_redis.get.assert_called_once_with(f"progress:{job_id}")


def test_get_status_returns_none_for_unknown_job(tracker, mock_redis, job_id):
    """Test get_status returns None for unknown job."""
    mock_redis.get.return_value = None

    result = tracker.get_status(job_id)

    assert result is None


# ============================================================================
# HAPPY PATH TESTS - get_history()
# ============================================================================


def test_get_history_returns_history_entries(tracker, mock_redis, job_id):
    """Test get_history returns list of history entries."""
    history_entries = [
        json.dumps(
            {"timestamp": "2025-01-01T10:00:00", "progress": 100, "stage": "COMPLETE"}
        ),
        json.dumps(
            {"timestamp": "2025-01-01T09:55:00", "progress": 75, "stage": "MATCHING"}
        ),
        json.dumps(
            {"timestamp": "2025-01-01T09:50:00", "progress": 50, "stage": "MATCHING"}
        ),
    ]
    mock_redis.lrange.return_value = history_entries

    result = tracker.get_history(job_id)

    assert len(result) == 3
    assert result[0]["progress"] == 100
    assert result[1]["progress"] == 75
    assert result[2]["progress"] == 50


def test_get_history_limits_to_max_entries(tracker, mock_redis, job_id):
    """Test get_history queries with correct range (0 to max-1)."""
    tracker.get_history(job_id)

    mock_redis.lrange.assert_called_once_with(
        f"progress:{job_id}:history", 0, tracker.max_history_entries - 1
    )


def test_get_history_returns_empty_list_for_unknown_job(tracker, mock_redis, job_id):
    """Test get_history returns empty list for unknown job."""
    mock_redis.lrange.return_value = []

    result = tracker.get_history(job_id)

    assert result == []
    assert isinstance(result, list)


# ============================================================================
# HAPPY PATH TESTS - delete_status()
# ============================================================================


def test_delete_status_deletes_all_keys(tracker, mock_redis, job_id):
    """Test delete_status deletes progress, result, and history keys."""
    tracker.delete_status(job_id)

    # Verify delete was called for all 3 keys
    mock_redis.delete.assert_called()
    delete_calls = mock_redis.delete.call_args_list

    deleted_keys = [call[0][0] for call in delete_calls]
    assert f"progress:{job_id}" in deleted_keys
    assert f"result:{job_id}" in deleted_keys
    assert f"progress:{job_id}:history" in deleted_keys


def test_delete_status_uses_pipeline(tracker, mock_redis, job_id):
    """Test delete_status uses pipeline for atomic deletion."""
    tracker.delete_status(job_id)

    mock_redis.pipeline.assert_called_once()
    mock_redis.execute.assert_called()


# ============================================================================
# HAPPY PATH TESTS - cleanup_old_jobs()
# ============================================================================


def test_cleanup_old_jobs_scans_progress_keys(tracker, mock_redis):
    """Test cleanup_old_jobs scans for progress keys."""
    mock_redis.scan_iter.return_value = []

    tracker.cleanup_old_jobs()

    # Verify scan_iter was called for progress keys
    scan_calls = [
        c for c in mock_redis.scan_iter.call_args_list if "progress:*" in str(c)
    ]
    assert len(scan_calls) >= 1


def test_cleanup_old_jobs_sets_ttl_on_keys_without_ttl(tracker, mock_redis):
    """Test cleanup_old_jobs sets TTL on keys that don't have one."""
    # Mock scan to return keys without TTL
    mock_redis.scan_iter.return_value = [
        "progress:job-1",
        "progress:job-2",
        "result:job-3",
    ]
    mock_redis.ttl.return_value = -1  # No TTL set

    tracker.cleanup_old_jobs()

    # Verify expire was called for keys without TTL
    assert mock_redis.expire.call_count >= 2  # At least for 2 progress keys


# ============================================================================
# ERROR HANDLING TESTS - Fallback mechanism
# ============================================================================


def test_start_job_writes_fallback_on_redis_error(tracker, mock_redis, job_id, temp_fallback_dir):
    """Test start_job writes to fallback file when Redis fails."""
    tracker.fallback_dir = temp_fallback_dir
    mock_redis.pipeline.side_effect = RedisError("Connection lost")

    # Should not raise - graceful degradation
    tracker.start_job(job_id, message="Test")

    # Verify fallback file was created
    fallback_file = temp_fallback_dir / f"progress_{job_id}.json"
    assert fallback_file.exists()

    # Check fallback content
    with fallback_file.open("r") as f:
        data = json.load(f)
        assert data["status"] == "processing"
        assert data["message"] == "Test"


def test_update_progress_writes_fallback_on_redis_error(
    tracker, mock_redis, job_id, temp_fallback_dir
):
    """Test update_progress writes to fallback file when Redis fails."""
    tracker.fallback_dir = temp_fallback_dir
    # First get_status call succeeds, then pipeline fails
    mock_redis.get.return_value = json.dumps({"status": "processing", "progress": 0})
    mock_redis.pipeline.side_effect = RedisError("Connection lost")

    # Should not raise
    tracker.update_progress(job_id, progress=50, message="Test")

    # Verify fallback file was created
    fallback_file = temp_fallback_dir / f"progress_{job_id}.json"
    assert fallback_file.exists()


def test_get_status_reads_fallback_on_redis_error(
    tracker, mock_redis, job_id, temp_fallback_dir
):
    """Test get_status reads from fallback file when Redis fails."""
    tracker.fallback_dir = temp_fallback_dir

    # Create fallback file
    fallback_data = {"status": "processing", "progress": 50}
    fallback_file = temp_fallback_dir / f"progress_{job_id}.json"
    with fallback_file.open("w") as f:
        json.dump(fallback_data, f)

    # Make Redis fail
    mock_redis.get.side_effect = RedisError("Connection lost")

    # Should read from fallback
    result = tracker.get_status(job_id)

    assert result == fallback_data


def test_get_history_returns_empty_on_redis_error(tracker, mock_redis, job_id):
    """Test get_history returns empty list when Redis fails."""
    mock_redis.lrange.side_effect = RedisError("Connection lost")

    result = tracker.get_history(job_id)

    assert result == []
    assert isinstance(result, list)


# ============================================================================
# ERROR HANDLING TESTS - Edge cases
# ============================================================================


def test_heartbeat_handles_redis_error_gracefully(tracker, mock_redis, job_id):
    """Test heartbeat doesn't raise on Redis error (not critical)."""
    mock_redis.get.side_effect = RedisError("Connection lost")

    # Should not raise
    tracker.heartbeat(job_id)


def test_delete_status_handles_redis_error_gracefully(tracker, mock_redis, job_id):
    """Test delete_status doesn't raise on Redis error."""
    mock_redis.pipeline.side_effect = RedisError("Connection lost")

    # Should not raise
    tracker.delete_status(job_id)


def test_cleanup_old_jobs_returns_zero_on_redis_error(tracker, mock_redis):
    """Test cleanup_old_jobs returns 0 when Redis fails."""
    mock_redis.scan_iter.side_effect = RedisError("Connection lost")

    count = tracker.cleanup_old_jobs()

    assert count == 0


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_tracker_uses_env_variables_for_redis_config():
    """Test tracker reads Redis config from environment variables."""
    with patch.dict(
        "os.environ",
        {"REDIS_HOST": "redis.example.com", "REDIS_PORT": "6380"},
    ):
        with patch("src.infrastructure.persistence.redis.progress_tracker.Redis"):
            tracker = RedisProgressTracker()

            assert tracker.redis_host == "redis.example.com"
            assert tracker.redis_port == 6380


def test_tracker_uses_env_variables_for_ttl():
    """Test tracker reads TTL config from environment variables."""
    with patch.dict(
        "os.environ",
        {"REDIS_PROGRESS_TTL": "7200", "REDIS_RESULT_TTL": "172800"},
    ):
        with patch("src.infrastructure.persistence.redis.progress_tracker.Redis"):
            tracker = RedisProgressTracker()

            assert tracker.progress_ttl == 7200
            assert tracker.result_ttl == 172800


def test_tracker_uses_default_ttl_when_env_not_set():
    """Test tracker uses default TTL when env variables not set."""
    with patch.dict("os.environ", {}, clear=True):
        with patch("src.infrastructure.persistence.redis.progress_tracker.Redis"):
            tracker = RedisProgressTracker()

            assert tracker.progress_ttl == 3600  # Default 1h
            assert tracker.result_ttl == 86400  # Default 24h
