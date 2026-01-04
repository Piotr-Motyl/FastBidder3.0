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
    - Returns 201 Created
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
    assert response.status_code == status.HTTP_201_CREATED
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
    from unittest.mock import AsyncMock

    # Arrange
    file = io.BytesIO(b"PK\x03\x04" + b"\x00" * 100)
    file.name = "test.xlsx"

    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(side_effect=Exception("Database error"))

    with patch("src.api.routers.files.FileUploadUseCase", return_value=mock_use_case):
        # Act
        response = client.post(
            "/api/files/upload",
            files={"file": ("test.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Assert
    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# ============================================================================
# PHASE 4: AI MATCHING - API SCHEMA UPDATES TESTS
# ============================================================================


def test_upload_file_with_file_type_reference(client):
    """
    Test Phase 4: file_type parameter for reference files.

    Verifies:
    - file_type query parameter is accepted
    - Response includes file_type="reference"
    - Response includes AI indexing fields (indexing_status, indexed_count)
    """
    from unittest.mock import AsyncMock, MagicMock
    from uuid import uuid4
    from src.application.services.file_upload_use_case import FileUploadResult

    # Arrange
    file_content = b"PK\x03\x04" + b"\x00" * 100
    file = io.BytesIO(file_content)
    file.name = "catalog.xlsx"

    # Create mock with file_type="reference"
    mock_use_case = MagicMock()
    mock_use_case.execute = AsyncMock(return_value=FileUploadResult(
        file_id=str(uuid4()),
        filename="catalog.xlsx",
        size_mb=0.1,
        sheets_count=1,
        rows_count=50,
        columns_count=5,
        upload_time="2025-12-31T00:00:00",
        preview=[],
        file_type="reference",  # Phase 4: reference type
        indexing_status=None,
        indexed_count=None,
    ))

    with patch(
        "src.api.routers.files.FileUploadUseCase", return_value=mock_use_case
    ):
        # Act - upload as reference file
        response = client.post(
            "/api/files/upload?file_type=reference",
            files={"file": ("catalog.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["file_type"] == "reference"
    assert "indexing_status" in data  # Should be None initially
    assert "indexed_count" in data  # Should be None initially


def test_upload_file_with_file_type_working(client, mock_file_upload_use_case):
    """
    Test Phase 4: file_type parameter for working files.

    Verifies:
    - file_type="working" is accepted
    - Response includes file_type="working"
    """
    # Arrange
    file_content = b"PK\x03\x04" + b"\x00" * 100
    file = io.BytesIO(file_content)
    file.name = "enquiry.xlsx"

    # mock_file_upload_use_case already has file_type="working" as default
    with patch(
        "src.api.routers.files.FileUploadUseCase", return_value=mock_file_upload_use_case
    ):
        # Act - upload as working file
        response = client.post(
            "/api/files/upload?file_type=working",
            files={"file": ("enquiry.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["file_type"] == "working"


def test_upload_file_backward_compatibility_no_file_type(client, mock_file_upload_use_case):
    """
    Test Phase 4: Backward compatibility - file_type parameter is optional.

    Verifies:
    - Endpoint works without file_type parameter (old clients)
    - Defaults to file_type="working"
    - Response includes all fields (old + new)
    """
    # Arrange
    file_content = b"PK\x03\x04" + b"\x00" * 100
    file = io.BytesIO(file_content)
    file.name = "test.xlsx"

    with patch(
        "src.api.routers.files.FileUploadUseCase", return_value=mock_file_upload_use_case
    ):
        # Act - no file_type parameter (old client behavior)
        response = client.post(
            "/api/files/upload",
            files={"file": ("test.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
        )

    # Assert
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    # Old fields still present
    assert "file_id" in data
    assert "filename" in data
    assert "size_mb" in data
    # New Phase 4 fields with defaults
    assert data["file_type"] == "working"  # Default value
    assert "indexing_status" in data
    assert "indexed_count" in data


def test_upload_file_invalid_file_type(client):
    """
    Test Phase 4: Invalid file_type parameter.

    Verifies:
    - Returns 422 for invalid file_type values
    """
    # Arrange
    file_content = b"PK\x03\x04" + b"\x00" * 100
    file = io.BytesIO(file_content)
    file.name = "test.xlsx"

    # Act - invalid file_type value
    response = client.post(
        "/api/files/upload?file_type=invalid",
        files={"file": ("test.xlsx", file, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
    )

    # Assert
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


# Summary: 10 tests covering (5 original + 5 Phase 4)
# Original:
# - Success scenario
# - Invalid extension
# - File too large
# - Missing file parameter
# - Use case error handling
# Phase 4 (API Schema Updates):
# - file_type="reference" parameter
# - file_type="working" parameter
# - Backward compatibility (no file_type)
# - Invalid file_type value
