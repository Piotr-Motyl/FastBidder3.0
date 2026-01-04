"""
Common fixtures for API unit tests.

Provides shared test utilities:
- FastAPI TestClient
- Mock dependencies
- Sample data
"""

import io
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app


@pytest.fixture
def client():
    """
    FastAPI TestClient for testing endpoints.

    Returns TestClient configured with the FastAPI app.
    """
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_job_id():
    """Generate a sample job ID (UUID)."""
    return str(uuid4())


@pytest.fixture
def sample_file_id():
    """Generate a sample file ID (UUID)."""
    return str(uuid4())


@pytest.fixture
def mock_excel_file():
    """
    Create a mock Excel file for upload testing.

    Returns a BytesIO object that simulates an .xlsx file.
    """
    # Simple mock Excel file content
    file_content = b"PK\x03\x04" + b"\x00" * 100  # Minimal xlsx header
    file = io.BytesIO(file_content)
    file.name = "test_file.xlsx"
    return file


@pytest.fixture
def mock_file_upload_use_case():
    """Mock for FileUploadUseCase (async)."""
    from unittest.mock import AsyncMock
    from src.application.services.file_upload_use_case import FileUploadResult

    mock = MagicMock()
    mock.execute = AsyncMock(return_value=FileUploadResult(
        file_id=str(uuid4()),
        filename="test.xlsx",
        size_mb=0.5,
        sheets_count=1,
        rows_count=100,
        columns_count=10,
        upload_time="2025-12-21T12:00:00",
        preview=[],
        file_type="working",  # Phase 4: default file type
        indexing_status=None,  # Phase 4: AI indexing fields
        indexed_count=None,
    ))
    return mock


@pytest.fixture
def mock_process_matching_use_case():
    """Mock for ProcessMatchingUseCase (async)."""
    from unittest.mock import AsyncMock
    from src.application.services.process_matching_use_case import ProcessMatchingResult

    mock = MagicMock()
    mock.execute = AsyncMock(return_value=ProcessMatchingResult(
        job_id=str(uuid4()),
        status="queued",
        estimated_time=30,
        message="Job queued successfully",
    ))
    return mock


@pytest.fixture
def mock_get_job_status_query():
    """Mock for GetJobStatusQuery."""
    mock = MagicMock()
    mock.execute.return_value = {
        "job_id": str(uuid4()),
        "status": "completed",
        "progress": 100,
        "message": "Processing completed",
        "created_at": "2025-12-21T12:00:00",
        "updated_at": "2025-12-21T12:01:00",
    }
    return mock


@pytest.fixture
def sample_result_file_path(tmp_path):
    """
    Create a temporary Excel file for result download testing.

    Returns path to a temporary .xlsx file.
    """
    result_file = tmp_path / "result.xlsx"
    result_file.write_bytes(b"PK\x03\x04" + b"\x00" * 200)
    return result_file
