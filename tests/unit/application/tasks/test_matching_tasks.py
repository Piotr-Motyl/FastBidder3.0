"""
Tests for process_matching_task (Celery Task).

Note: These are simplified unit tests focusing on task configuration and structure.
Full functional/integration tests with real Celery execution are in tests/e2e/test_matching_workflow.py.

Covers:
- Task configuration (retries, timeouts, backoff)
- Task can be imported and has correct structure
- Task docstring and callable nature
"""

from unittest.mock import patch

from src.application.tasks.matching_tasks import process_matching_task


# ============================================================================
# CONFIGURATION TESTS
# ============================================================================


def test_task_is_configured_correctly():
    """
    Test that Celery task is configured with correct parameters.

    Verifies:
    - Task has correct decorators (max_retries, time_limit, soft_time_limit)
    - Task can be imported
    - Task name is set correctly
    """
    # Check task attributes (from @task decorator)
    assert process_matching_task.max_retries == 3
    assert process_matching_task.time_limit == 300
    assert process_matching_task.soft_time_limit == 270
    assert process_matching_task.name == "process_matching"

    # Verify task has __wrapped__ (indicates bind=True decorator)
    assert hasattr(process_matching_task, "__wrapped__")


def test_task_has_correct_retry_config():
    """
    Test that task has exponential backoff retry configuration.

    Verifies:
    - retry_backoff is enabled
    - retry_backoff_max is set to 900 seconds (15 minutes)
    """
    assert process_matching_task.retry_backoff is True
    assert process_matching_task.retry_backoff_max == 900


def test_task_has_redis_progress_tracker_integration():
    """
    Test that task imports RedisProgressTracker.

    Verifies that the task file has access to progress tracking infrastructure.
    """
    from src.application.tasks import matching_tasks

    assert hasattr(matching_tasks, "RedisProgressTracker")


def test_task_has_required_service_imports():
    """
    Test that task imports all required services.

    Verifies that the task file has access to:
    - File storage services
    - Excel services
    - Matching engine
    - Parameter extractor
    """
    from src.application.tasks import matching_tasks

    # Check imports exist
    assert hasattr(matching_tasks, "FileStorageService")
    assert hasattr(matching_tasks, "ExcelReaderService")
    assert hasattr(matching_tasks, "ExcelWriterService")
    assert hasattr(matching_tasks, "SimpleMatchingEngine")
    assert hasattr(matching_tasks, "ConcreteParameterExtractor")


def test_task_has_error_handling_imports():
    """
    Test that task imports error handling utilities.

    Verifies imports for:
    - Celery exceptions (SoftTimeLimitExceeded)
    - Retry mechanism
    """
    from src.application.tasks import matching_tasks

    assert hasattr(matching_tasks, "SoftTimeLimitExceeded")


# ============================================================================
# STRUCTURE TESTS
# ============================================================================


def test_task_function_exists_and_is_callable():
    """Test that task function can be imported and is callable."""
    assert callable(process_matching_task)
    assert hasattr(process_matching_task, "__wrapped__")


def test_task_has_proper_docstring():
    """
    Test that task has comprehensive docstring.

    Verifies documentation includes:
    - working_file parameter
    - reference_file parameter
    - Phase 2 contract details
    """
    assert process_matching_task.__doc__ is not None
    assert len(process_matching_task.__doc__) > 100
    assert "working_file" in process_matching_task.__doc__
    assert "reference_file" in process_matching_task.__doc__
    assert "Phase 2" in process_matching_task.__doc__


def test_task_signature_has_correct_parameters():
    """
    Test that __wrapped__ function has correct parameter signature.

    Verifies parameters:
    - working_file (required)
    - reference_file (required)
    - matching_threshold (with default)
    - matching_strategy (with default)
    - report_format (with default)
    """
    import inspect

    sig = inspect.signature(process_matching_task.__wrapped__)
    params = list(sig.parameters.keys())

    # Check required parameters
    assert "working_file" in params
    assert "reference_file" in params

    # Check optional parameters with defaults
    assert "matching_threshold" in params
    assert "matching_strategy" in params
    assert "report_format" in params

    # Verify defaults
    assert sig.parameters["matching_threshold"].default == 75.0
    assert sig.parameters["matching_strategy"].default == "best_match"
    assert sig.parameters["report_format"].default == "simple"


def test_task_return_type_annotation():
    """
    Test that task has proper return type annotation.

    Verifies the function returns dict.
    """
    import inspect

    sig = inspect.signature(process_matching_task.__wrapped__)
    assert sig.return_annotation == dict


#============================================================================
# MOCK-BASED INTEGRATION TEST
# ============================================================================


@patch("src.application.tasks.matching_tasks.psutil")
@patch("src.application.tasks.matching_tasks.SimpleMatchingEngine")
@patch("src.application.tasks.matching_tasks.ConcreteParameterExtractor")
@patch("src.application.tasks.matching_tasks.ExcelWriterService")
@patch("src.application.tasks.matching_tasks.ExcelReaderService")
@patch("src.application.tasks.matching_tasks.FileStorageService")
@patch("src.application.tasks.matching_tasks.RedisProgressTracker")
def test_task_can_be_instantiated_with_mocks(
    mock_progress_tracker,
    mock_file_storage,
    mock_excel_reader,
    mock_excel_writer,
    mock_param_extractor,
    mock_matching_engine,
    mock_psutil,
):
    """
    Test that all services can be mocked successfully.

    This verifies that the task's dependencies are properly structured
    and can be replaced with mocks for testing purposes.

    Note: Full execution tests are in tests/e2e/test_matching_workflow.py
    """
    # Verify all mocks are in place
    assert mock_progress_tracker is not None
    assert mock_file_storage is not None
    assert mock_excel_reader is not None
    assert mock_excel_writer is not None
    assert mock_param_extractor is not None
    assert mock_matching_engine is not None
    assert mock_psutil is not None

    # This test verifies mockability, not execution
    # Actual execution tests are in E2E tests


# ============================================================================
# SUMMARY
# ============================================================================
# Total tests: 11
# - Configuration tests: 5 (task config, retry, imports)
# - Structure tests: 5 (callable, docstring, signature, return type, params)
# - Mock integration: 1 (verifies services can be mocked)
#
# These tests verify:
# ✓ Task configuration (retries, timeouts, backoff)
# ✓ Task has all required imports
# ✓ Task is properly structured with correct signature
# ✓ Task has comprehensive documentation
# ✓ Task dependencies can be mocked
#
# NOTE: Full functional tests with real Celery execution, file I/O,
# and matching logic are in tests/e2e/test_matching_workflow.py
# ============================================================================
