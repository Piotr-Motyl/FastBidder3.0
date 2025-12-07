"""
Tests for FileStorageService.

Covers file upload, storage, metadata extraction, cleanup, and all CRUD operations.
"""
import pytest
import time
from pathlib import Path
from uuid import uuid4
import polars as pl

from src.infrastructure.file_storage.file_storage_service import FileStorageService
from src.domain.shared.exceptions import (
    FileSizeExceededError,
    ExcelParsingError,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def service(tmp_path):
    """Fixture for FileStorageService instance with temp directory."""
    return FileStorageService(
        base_dir=str(tmp_path / "fastbidder"),
        max_size_mb=10,
        allowed_extensions=[".xlsx", ".xls"],
    )


@pytest.fixture
def job_id():
    """Fixture for test job ID."""
    return uuid4()


@pytest.fixture
def file_id():
    """Fixture for test file ID."""
    return uuid4()


@pytest.fixture
def small_excel_file(tmp_path):
    """Create a small test Excel file."""
    df = pl.DataFrame({
        "Description": ["Item 1", "Item 2", "Item 3"],
        "Price": [10.50, 20.75, 15.00],
        "Quantity": [5, 3, 10],
    })

    file_path = tmp_path / "test_file.xlsx"
    df.write_excel(file_path)

    return file_path


# ============================================================================
# TESTS - Helper Methods
# ============================================================================


def test_get_subdirectory_working(service):
    """Test _get_subdirectory() maps 'working' to 'input'."""
    assert service._get_subdirectory("working") == "input"


def test_get_subdirectory_reference(service):
    """Test _get_subdirectory() maps 'reference' to 'input'."""
    assert service._get_subdirectory("reference") == "input"


def test_get_subdirectory_result(service):
    """Test _get_subdirectory() maps 'result' to 'output'."""
    assert service._get_subdirectory("result") == "output"


def test_get_subdirectory_invalid(service):
    """Test _get_subdirectory() raises ValueError for invalid file_type."""
    with pytest.raises(ValueError, match="Unknown file_type"):
        service._get_subdirectory("invalid")


def test_get_filename_for_type_working(service):
    """Test _get_filename_for_type() returns correct filename for 'working'."""
    assert service._get_filename_for_type("working") == "working_file.xlsx"


def test_get_filename_for_type_reference(service):
    """Test _get_filename_for_type() returns correct filename for 'reference'."""
    assert service._get_filename_for_type("reference") == "reference_file.xlsx"


def test_get_filename_for_type_result(service):
    """Test _get_filename_for_type() returns correct filename for 'result'."""
    assert service._get_filename_for_type("result") == "result.xlsx"


def test_get_filename_for_type_invalid(service):
    """Test _get_filename_for_type() raises ValueError for invalid file_type."""
    with pytest.raises(ValueError, match="Unknown file_type"):
        service._get_filename_for_type("invalid")


def test_ensure_directory_exists_creates_directory(service, tmp_path):
    """Test _ensure_directory_exists() creates directory."""
    test_dir = tmp_path / "test" / "nested" / "dir"

    service._ensure_directory_exists(test_dir)

    assert test_dir.exists()
    assert test_dir.is_dir()


def test_ensure_directory_exists_already_exists(service, tmp_path):
    """Test _ensure_directory_exists() handles existing directory."""
    test_dir = tmp_path / "existing"
    test_dir.mkdir()

    # Should not raise error
    service._ensure_directory_exists(test_dir)

    assert test_dir.exists()


def test_set_permissions(service, tmp_path):
    """Test _set_permissions() sets permissions on file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    # Should not raise error (may not work on Windows)
    service._set_permissions(test_file, mode=0o644)


def test_atomic_write_file(service, tmp_path):
    """Test _atomic_write_file() writes file atomically."""
    test_file = tmp_path / "atomic.txt"
    test_data = b"test content"

    service._atomic_write_file(test_file, test_data)

    assert test_file.exists()
    assert test_file.read_bytes() == test_data


def test_atomic_write_file_overwrites(service, tmp_path):
    """Test _atomic_write_file() overwrites existing file."""
    test_file = tmp_path / "overwrite.txt"
    test_file.write_text("old content")

    new_data = b"new content"
    service._atomic_write_file(test_file, new_data)

    assert test_file.read_bytes() == new_data


def test_is_directory_old_recent_directory(service, tmp_path):
    """Test _is_directory_old() returns False for recent directory."""
    test_dir = tmp_path / "recent"
    test_dir.mkdir()

    # Directory just created, should not be old
    is_old = service._is_directory_old(test_dir, hours=24)

    assert is_old is False


def test_is_directory_old_old_directory(service, tmp_path):
    """Test _is_directory_old() returns True for old directory."""
    test_dir = tmp_path / "old"
    test_dir.mkdir()

    # Modify directory timestamp to be older than threshold
    # Set mtime to 48 hours ago
    old_time = time.time() - (48 * 3600)
    import os
    os.utime(test_dir, (old_time, old_time))

    # Should be old (threshold: 24 hours)
    is_old = service._is_directory_old(test_dir, hours=24)

    assert is_old is True


def test_is_directory_old_nonexistent(service, tmp_path):
    """Test _is_directory_old() returns False for nonexistent directory."""
    nonexistent = tmp_path / "nonexistent"

    is_old = service._is_directory_old(nonexistent, hours=24)

    assert is_old is False


# ============================================================================
# TESTS - get_file_path()
# ============================================================================


def test_get_file_path_working(service, job_id):
    """Test get_file_path() returns correct path for working file."""
    path = service.get_file_path(job_id, "working")

    expected = service.base_dir / str(job_id) / "input" / "working_file.xlsx"
    assert path == expected


def test_get_file_path_result(service, job_id):
    """Test get_file_path() returns correct path for result file."""
    path = service.get_file_path(job_id, "result")

    expected = service.base_dir / str(job_id) / "output" / "result.xlsx"
    assert path == expected


# ============================================================================
# TESTS - file_exists()
# ============================================================================


@pytest.mark.asyncio
async def test_file_exists_true(service, job_id):
    """Test file_exists() returns True when file exists."""
    # Create file
    file_path = service.get_file_path(job_id, "working")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"test")

    exists = await service.file_exists(job_id, "working")

    assert exists is True


@pytest.mark.asyncio
async def test_file_exists_false(service, job_id):
    """Test file_exists() returns False when file doesn't exist."""
    exists = await service.file_exists(job_id, "working")

    assert exists is False


# ============================================================================
# TESTS - get_file_metadata()
# ============================================================================


@pytest.mark.asyncio
async def test_get_file_metadata_success(service, job_id, small_excel_file):
    """Test get_file_metadata() returns metadata for existing file."""
    # Upload file
    file_data = small_excel_file.read_bytes()
    await service.upload_file(job_id, file_data, "test.xlsx", "working")

    # Get metadata
    metadata = await service.get_file_metadata(job_id, "working")

    assert metadata["size"] > 0
    assert metadata["size_mb"] > 0
    assert metadata["format"] == "xlsx"
    assert metadata["exists"] is True
    assert metadata["file_type"] == "working"
    assert "created_at" in metadata
    assert "modified_at" in metadata


@pytest.mark.asyncio
async def test_get_file_metadata_not_found(service, job_id):
    """Test get_file_metadata() raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        await service.get_file_metadata(job_id, "working")


# ============================================================================
# TESTS - upload_file()
# ============================================================================


@pytest.mark.asyncio
async def test_upload_file_success(service, job_id, small_excel_file):
    """Test upload_file() saves file successfully."""
    file_data = small_excel_file.read_bytes()

    path = await service.upload_file(job_id, file_data, "test.xlsx", "working")

    assert path.exists()
    assert path.read_bytes() == file_data
    assert path.name == "working_file.xlsx"


@pytest.mark.asyncio
async def test_upload_file_invalid_extension(service, job_id):
    """Test upload_file() raises ValueError for invalid extension."""
    file_data = b"test data"

    with pytest.raises(ValueError, match="Invalid extension"):
        await service.upload_file(job_id, file_data, "test.pdf", "working")


@pytest.mark.asyncio
async def test_upload_file_too_large(service, job_id):
    """Test upload_file() raises FileSizeExceededError for large file."""
    # Create 11MB file (exceeds 10MB limit)
    large_data = b"x" * (11 * 1024 * 1024)

    with pytest.raises(FileSizeExceededError):
        await service.upload_file(job_id, large_data, "large.xlsx", "working")


@pytest.mark.asyncio
async def test_upload_file_reference(service, job_id, small_excel_file):
    """Test upload_file() handles reference file type."""
    file_data = small_excel_file.read_bytes()

    path = await service.upload_file(job_id, file_data, "ref.xlsx", "reference")

    assert path.exists()
    assert path.name == "reference_file.xlsx"


# ============================================================================
# TESTS - cleanup_job()
# ============================================================================


@pytest.mark.asyncio
async def test_cleanup_job_success(service, job_id, small_excel_file):
    """Test cleanup_job() deletes job directory."""
    # Upload file to create job directory
    file_data = small_excel_file.read_bytes()
    await service.upload_file(job_id, file_data, "test.xlsx", "working")

    job_dir = service._get_job_dir(job_id)
    assert job_dir.exists()

    # Cleanup
    await service.cleanup_job(job_id)

    assert not job_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_job_nonexistent(service, job_id):
    """Test cleanup_job() handles nonexistent job gracefully."""
    # Should not raise error
    await service.cleanup_job(job_id)


@pytest.mark.asyncio
async def test_cleanup_job_deletes_all_files(service, job_id, small_excel_file):
    """Test cleanup_job() deletes all files in job directory."""
    file_data = small_excel_file.read_bytes()

    # Upload working and reference files
    await service.upload_file(job_id, file_data, "working.xlsx", "working")
    await service.upload_file(job_id, file_data, "ref.xlsx", "reference")

    job_dir = service._get_job_dir(job_id)
    input_dir = job_dir / "input"
    assert (input_dir / "working_file.xlsx").exists()
    assert (input_dir / "reference_file.xlsx").exists()

    # Cleanup
    await service.cleanup_job(job_id)

    assert not job_dir.exists()


# ============================================================================
# TESTS - cleanup_old_jobs()
# ============================================================================


@pytest.mark.asyncio
async def test_cleanup_old_jobs_no_old_jobs(service, job_id, small_excel_file):
    """Test cleanup_old_jobs() skips recent jobs."""
    # Upload file (creates recent job directory)
    file_data = small_excel_file.read_bytes()
    await service.upload_file(job_id, file_data, "test.xlsx", "working")

    # Cleanup old jobs (threshold: 24 hours)
    count = await service.cleanup_old_jobs(hours=24)

    assert count == 0
    assert service._get_job_dir(job_id).exists()


@pytest.mark.asyncio
async def test_cleanup_old_jobs_removes_old(service, job_id, small_excel_file, tmp_path):
    """Test cleanup_old_jobs() removes old job directories."""
    # Create old job directory
    old_job_id = uuid4()
    file_data = small_excel_file.read_bytes()
    await service.upload_file(old_job_id, file_data, "test.xlsx", "working")

    old_job_dir = service._get_job_dir(old_job_id)

    # Make directory old (48 hours ago)
    old_time = time.time() - (48 * 3600)
    import os
    os.utime(old_job_dir, (old_time, old_time))

    # Cleanup jobs older than 24 hours
    count = await service.cleanup_old_jobs(hours=24)

    assert count == 1
    assert not old_job_dir.exists()


@pytest.mark.asyncio
async def test_cleanup_old_jobs_skips_uploads_directory(service, tmp_path):
    """Test cleanup_old_jobs() skips 'uploads' directory."""
    # Create uploads directory
    uploads_dir = service.base_dir / "uploads"
    uploads_dir.mkdir(parents=True)

    # Make it old
    old_time = time.time() - (48 * 3600)
    import os
    os.utime(uploads_dir, (old_time, old_time))

    # Cleanup
    count = await service.cleanup_old_jobs(hours=1)

    # uploads directory should still exist
    assert uploads_dir.exists()


# ============================================================================
# TESTS - save_uploaded_file()
# ============================================================================


@pytest.mark.asyncio
async def test_save_uploaded_file_success(service, file_id, small_excel_file):
    """Test save_uploaded_file() saves file to uploads directory."""
    file_data = small_excel_file.read_bytes()

    path = await service.save_uploaded_file(file_id, file_data, "catalog.xlsx")

    assert path.exists()
    assert path.read_bytes() == file_data
    assert path.parent.name == str(file_id)
    assert path.name == "catalog.xlsx"  # Original filename preserved


@pytest.mark.asyncio
async def test_save_uploaded_file_invalid_extension(service, file_id):
    """Test save_uploaded_file() rejects invalid extension."""
    file_data = b"test"

    with pytest.raises(ValueError, match="Invalid extension"):
        await service.save_uploaded_file(file_id, file_data, "file.pdf")


@pytest.mark.asyncio
async def test_save_uploaded_file_too_large(service, file_id):
    """Test save_uploaded_file() rejects large files."""
    large_data = b"x" * (11 * 1024 * 1024)

    with pytest.raises(FileSizeExceededError):
        await service.save_uploaded_file(file_id, large_data, "large.xlsx")


# ============================================================================
# TESTS - extract_file_metadata()
# ============================================================================


@pytest.mark.asyncio
async def test_extract_file_metadata_success(service, small_excel_file):
    """Test extract_file_metadata() extracts metadata from Excel file."""
    metadata = await service.extract_file_metadata(small_excel_file)

    assert metadata["filename"] == small_excel_file.name
    assert metadata["size"] > 0
    assert metadata["size_mb"] > 0
    assert metadata["sheets_count"] == 1
    assert metadata["rows_count"] == 4  # Includes header row
    assert metadata["columns_count"] == 3
    assert "created_at" in metadata


@pytest.mark.asyncio
async def test_extract_file_metadata_not_found(service, tmp_path):
    """Test extract_file_metadata() raises FileNotFoundError."""
    nonexistent = tmp_path / "nonexistent.xlsx"

    with pytest.raises(FileNotFoundError):
        await service.extract_file_metadata(nonexistent)


@pytest.mark.asyncio
async def test_extract_file_metadata_invalid_excel(service, tmp_path):
    """Test extract_file_metadata() raises ExcelParsingError for invalid file."""
    # Create non-Excel file
    invalid_file = tmp_path / "invalid.xlsx"
    invalid_file.write_text("Not an Excel file")

    with pytest.raises(ExcelParsingError):
        await service.extract_file_metadata(invalid_file)


# ============================================================================
# TESTS - extract_file_preview()
# ============================================================================


@pytest.mark.asyncio
async def test_extract_file_preview_success(service, small_excel_file):
    """Test extract_file_preview() extracts preview rows."""
    preview = await service.extract_file_preview(small_excel_file, rows=3)

    # Should get some rows
    assert len(preview) > 0
    # Check that data is present (check any row has Description column)
    assert "Description" in preview[0]
    assert "Price" in preview[0]


@pytest.mark.asyncio
async def test_extract_file_preview_default_rows(service, small_excel_file):
    """Test extract_file_preview() uses default 5 rows."""
    preview = await service.extract_file_preview(small_excel_file)

    # Should get some rows from the file
    assert len(preview) > 0
    assert isinstance(preview, list)
    assert isinstance(preview[0], dict)


@pytest.mark.asyncio
async def test_extract_file_preview_not_found(service, tmp_path):
    """Test extract_file_preview() raises FileNotFoundError."""
    nonexistent = tmp_path / "nonexistent.xlsx"

    with pytest.raises(FileNotFoundError):
        await service.extract_file_preview(nonexistent)


# ============================================================================
# TESTS - get_result_file_path() & result_file_exists()
# ============================================================================


def test_get_result_file_path(service, job_id):
    """Test get_result_file_path() returns correct path."""
    path = service.get_result_file_path(job_id)

    expected = service.base_dir / str(job_id) / "output" / "result.xlsx"
    assert path == expected


def test_result_file_exists_true(service, job_id):
    """Test result_file_exists() returns True when result exists."""
    # Create result file
    result_path = service.get_result_file_path(job_id)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_bytes(b"result")

    exists = service.result_file_exists(job_id)

    assert exists is True


def test_result_file_exists_false(service, job_id):
    """Test result_file_exists() returns False when result doesn't exist."""
    exists = service.result_file_exists(job_id)

    assert exists is False
