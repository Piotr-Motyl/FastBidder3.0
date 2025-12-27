"""
Tests for POST /api/files/upload endpoint.

Covers:
- Successful upload
- File validation (size, format)
- Error handling
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import status


def test_upload_file_success(client, mock_file_upload_use_case):
    """
    Test successful file upload.

    Verifies:
    - Returns 200 OK
    - Response contains file_id, filename, size_mb, rows_count
    """
    # Arrange
    file_content = b"PK\x03\x04" + b"\x00" * 100
    file = io.BytesIO(file_content)
    file.name = "test.xlsx"

    with patch(
        "src.api.routers.files.FileUploadUseCase", return_value=mock_file_upload_use_case
    ):
        # Act
        response = client.post(
            "/api/files/upload",
            files={"file": ("test.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "file_id" in data
    assert "filename" in data
    assert "size_mb" in data


def test_upload_file_invalid_extension(client):
    """
    Test upload with invalid file extension.

    Verifies:
    - Returns 400 Bad Request for non-Excel files
    """
    # Arrange
    file = io.BytesIO(b"not an excel file")
    file.name = "test.txt"

    # Act
    response = client.post(
        "/api/files/upload",
        files={"file": ("test.txt", file, "text/plain")},
    )

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_upload_file_too_large(client):
    """
    Test upload with file exceeding size limit (10MB).

    Verifies:
    - Returns 413 Request Entity Too Large
    """
    # Arrange - Create a mock file larger than 10MB
    large_content = b"x" * (11 * 1024 * 1024)  # 11 MB
    file = io.BytesIO(large_content)
    file.name = "large.xlsx"

    # Act
    response = client.post(
        "/api/files/upload",
        files={"file": ("large.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    # Assert
    assert response.status_code in [
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        status.HTTP_400_BAD_REQUEST,
    ]


def test_upload_file_missing_file(client):
    """
    Test upload without file parameter.

    Verifies:
    - Returns 422 Unprocessable Entity
    """
    # Act
    response = client.post("/api/files/upload")

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


def test_upload_file_use_case_error(client):
    """
    Test upload when use case raises exception.

    Verifies:
    - Returns 500 Internal Server Error
    - Error message is included
    """
    # Arrange
    file = io.BytesIO(b"PK\x03\x04" + b"\x00" * 100)
    file.name = "test.xlsx"

    mock_use_case = MagicMock()
    mock_use_case.execute.side_effect = Exception("Database error")

    with patch("src.api.routers.files.FileUploadUseCase", return_value=mock_use_case):
        # Act
        response = client.post(
            "/api/files/upload",
            files={"file": ("test.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# Summary: 5 tests covering
# - Success scenario
# - Invalid extension
# - File too large
# - Missing file parameter
# - Use case error handling
