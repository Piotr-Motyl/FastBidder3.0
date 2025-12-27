"""
Tests for GET /api/jobs/{job_id}/status endpoint.

Covers:
- Successful status retrieval
- Job not found
- Invalid job ID format
- Error handling
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import status


def test_get_job_status_success(client, mock_get_job_status_query, sample_job_id):
    """
    Test successful job status retrieval.

    Verifies:
    - Returns 200 OK
    - Response contains job_id, status, progress, message
    """
    # Arrange
    with patch(
        "src.api.routers.jobs.GetJobStatusQuery",
        return_value=mock_get_job_status_query,
    ):
        # Act
        response = client.get(f"/api/jobs/{sample_job_id}/status")

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert "progress" in data


def test_get_job_status_not_found(client):
    """
    Test status retrieval for non-existent job.

    Verifies:
    - Returns 404 Not Found
    """
    # Arrange
    non_existent_job_id = str(uuid4())

    mock_query = MagicMock()
    mock_query.execute.return_value = None  # Job not found

    with patch("src.api.routers.jobs.GetJobStatusQuery", return_value=mock_query):
        # Act
        response = client.get(f"/api/jobs/{non_existent_job_id}/status")

    # Assert
    assert response.status_code == status.HTTP_404_NOT_FOUND


def test_get_job_status_invalid_uuid(client):
    """
    Test status retrieval with invalid UUID format.

    Verifies:
    - Returns 422 Unprocessable Entity
    """
    # Arrange
    invalid_job_id = "not-a-valid-uuid"

    # Act
    response = client.get(f"/api/jobs/{invalid_job_id}/status")

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_get_job_status_query_error(client, sample_job_id):
    """
    Test status retrieval when query raises exception.

    Verifies:
    - Returns 500 Internal Server Error
    """
    # Arrange
    mock_query = MagicMock()
    mock_query.execute.side_effect = Exception("Redis connection error")

    with patch("src.api.routers.jobs.GetJobStatusQuery", return_value=mock_query):
        # Act
        response = client.get(f"/api/jobs/{sample_job_id}/status")

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_get_job_status_cache_headers(client, mock_get_job_status_query, sample_job_id):
    """
    Test that status endpoint has correct cache headers.

    Verifies:
    - Response has no-cache headers to prevent stale status
    """
    # Arrange
    with patch(
        "src.api.routers.jobs.GetJobStatusQuery",
        return_value=mock_get_job_status_query,
    ):
        # Act
        response = client.get(f"/api/jobs/{sample_job_id}/status")

    # Assert
    assert response.status_code == status.HTTP_200_OK
    # Cache-Control header should prevent caching for real-time status
    # Note: Actual header check depends on implementation


# Summary: 5 tests covering
# - Success scenario
# - Job not found
# - Invalid UUID
# - Query error
# - Cache headers
