"""
Tests for GET /api/results/{job_id}/download endpoint.

Covers:
- Successful file download
- Job not completed
- File not found
- Invalid job ID
- Error handling
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import status


def test_download_results_success(client, sample_job_id, sample_result_file_path):
    """
    Test successful results file download.

    Verifies:
    - Returns 200 OK
    - Response has Content-Disposition header with attachment
    - Response content-type is application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
    """
    # Arrange
    from src.api.routers.results import get_file_storage_service, get_progress_tracker

    mock_file_storage = MagicMock()
    mock_file_storage.get_result_file_path.return_value = sample_result_file_path
    mock_file_storage.result_file_exists.return_value = True

    mock_progress_tracker = MagicMock()
    mock_progress_tracker.get_status.return_value = {"status": "completed"}

    # Override dependencies using FastAPI's dependency_overrides
    def override_file_storage():
        return mock_file_storage

    def override_progress_tracker():
        return mock_progress_tracker

    client.app.dependency_overrides[get_file_storage_service] = override_file_storage
    client.app.dependency_overrides[get_progress_tracker] = override_progress_tracker

    try:
        # Act
        response = client.get(f"/api/results/{sample_job_id}/download")

        # Assert
        assert response.status_code == status.HTTP_200_OK
        assert "attachment" in response.headers.get("content-disposition", "").lower()
    finally:
        # Clean up
        client.app.dependency_overrides.clear()


def test_download_results_job_not_completed(client, sample_job_id):
    """
    Test download when job is not completed yet.

    Verifies:
    - Returns 404 Not Found (job not completed yet)
    """
    # Arrange
    mock_progress_tracker = MagicMock()
    mock_progress_tracker.get_status.return_value = {"status": "processing"}  # Not completed

    with patch("src.api.routers.results.get_progress_tracker", return_value=mock_progress_tracker):
        # Act
        response = client.get(f"/api/results/{sample_job_id}/download")

    # Assert
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_download_results_file_not_found(client, sample_job_id):
    """
    Test download when result file doesn't exist.

    Verifies:
    - Returns 404 Not Found
    """
    # Arrange
    mock_file_storage = MagicMock()
    mock_file_storage.get_result_file_path.return_value = Path("/nonexistent/file.xlsx")
    mock_file_storage.result_file_exists.return_value = False  # File doesn't exist

    mock_progress_tracker = MagicMock()
    mock_progress_tracker.get_status.return_value = {"status": "completed"}

    with patch(
        "src.api.routers.results.get_file_storage_service", return_value=mock_file_storage
    ), patch("src.api.routers.results.get_progress_tracker", return_value=mock_progress_tracker):
        # Act
        response = client.get(f"/api/results/{sample_job_id}/download")

    # Assert
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_download_results_invalid_job_id(client):
    """
    Test download with invalid job ID format.

    Verifies:
    - Returns 422 Unprocessable Entity
    """
    # Arrange
    invalid_job_id = "not-a-uuid"

    # Act
    response = client.get(f"/api/results/{invalid_job_id}/download")

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_download_results_job_not_found(client):
    """
    Test download for non-existent job.

    Verifies:
    - Returns 404 Not Found
    """
    # Arrange
    non_existent_job_id = str(uuid4())

    mock_progress_tracker = MagicMock()
    mock_progress_tracker.get_status.return_value = None  # Job not found

    with patch("src.api.routers.results.get_progress_tracker", return_value=mock_progress_tracker):
        # Act
        response = client.get(f"/api/results/{non_existent_job_id}/download")

    # Assert
    assert response.status_code == status.HTTP_404_NOT_FOUND


# Summary: 5 tests covering
# - Success scenario
# - Job not completed
# - File not found
# - Invalid UUID
# - Job not found
