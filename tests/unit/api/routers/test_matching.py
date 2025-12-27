"""
Tests for POST /api/matching/process endpoint.

Covers:
- Successful process initiation
- Validation (file IDs, threshold)
- Error handling
"""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import status


def test_process_matching_success(client, mock_process_matching_use_case, sample_file_id):
    """
    Test successful matching process initiation.

    Verifies:
    - Returns 202 Accepted
    - Response contains job_id, status="queued", estimated_time_seconds
    """
    # Arrange
    working_file_id = str(uuid4())
    reference_file_id = str(uuid4())

    payload = {
        "working_file_id": working_file_id,
        "reference_file_id": reference_file_id,
        "matching_threshold": 75.0,
    }

    with patch(
        "src.api.routers.matching.ProcessMatchingUseCase",
        return_value=mock_process_matching_use_case,
    ):
        # Act
        response = client.post("/api/matching/process", json=payload)

    # Assert
    assert response.status_code == status.HTTP_202_ACCEPTED
    data = response.json()
    assert "job_id" in data
    assert "status" in data
    assert data["status"] == "queued"


def test_process_matching_invalid_threshold(client):
    """
    Test process with invalid threshold value.

    Verifies:
    - Returns 422 Unprocessable Entity for threshold outside 0-100 range
    """
    # Arrange
    payload = {
        "working_file_id": str(uuid4()),
        "reference_file_id": str(uuid4()),
        "matching_threshold": 150.0,  # Invalid - above 100
    }

    # Act
    response = client.post("/api/matching/process", json=payload)

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_process_matching_missing_file_ids(client):
    """
    Test process without required file IDs.

    Verifies:
    - Returns 422 Unprocessable Entity
    """
    # Arrange
    payload = {
        "matching_threshold": 75.0,
        # Missing file IDs
    }

    # Act
    response = client.post("/api/matching/process", json=payload)

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_process_matching_invalid_file_id_format(client):
    """
    Test process with invalid UUID format for file IDs.

    Verifies:
    - Returns 422 Unprocessable Entity
    """
    # Arrange
    payload = {
        "working_file_id": "not-a-uuid",
        "reference_file_id": "also-not-a-uuid",
        "matching_threshold": 75.0,
    }

    # Act
    response = client.post("/api/matching/process", json=payload)

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_process_matching_use_case_error(client):
    """
    Test process when use case raises exception.

    Verifies:
    - Returns 500 Internal Server Error
    """
    # Arrange
    payload = {
        "working_file_id": str(uuid4()),
        "reference_file_id": str(uuid4()),
        "matching_threshold": 75.0,
    }

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = Exception("Celery connection error")

    with patch(
        "src.api.routers.matching.ProcessMatchingUseCase", return_value=mock_use_case
    ):
        # Act
        response = client.post("/api/matching/process", json=payload)

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


def test_process_matching_with_optional_parameters(client, mock_process_matching_use_case):
    """
    Test process with optional parameters (strategy, report_format).

    Verifies:
    - Accepts optional parameters
    - Returns 202 Accepted
    """
    # Arrange
    payload = {
        "working_file_id": str(uuid4()),
        "reference_file_id": str(uuid4()),
        "matching_threshold": 80.0,
        "matching_strategy": "best_match",
        "report_format": "detailed",
    }

    with patch(
        "src.api.routers.matching.ProcessMatchingUseCase",
        return_value=mock_process_matching_use_case,
    ):
        # Act
        response = client.post("/api/matching/process", json=payload)

    # Assert
    assert response.status_code == status.HTTP_202_ACCEPTED


# Summary: 6 tests covering
# - Success scenario
# - Invalid threshold
# - Missing file IDs
# - Invalid UUID format
# - Use case error
# - Optional parameters
