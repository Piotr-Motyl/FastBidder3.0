"""
Tests for FileUploadUseCase with ReferenceIndexer integration.

Tests cover:
- Working file uploads (no indexing)
- Reference file uploads (with indexing)
- Indexing error handling
- Helper method for description extraction
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4

from src.application.services.file_upload_use_case import (
    FileUploadUseCase,
    FileUploadResult,
)
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.infrastructure.ai.vector_store.reference_indexer import IndexingResult


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_file_storage():
    """Mocked FileStorageService."""
    mock = Mock()

    # Mock save_uploaded_file to return a path
    async def mock_save(file_id, file_data, filename):
        return Path(f"/fake/uploads/{file_id}/{filename}")

    mock.save_uploaded_file = AsyncMock(side_effect=mock_save)

    # Mock extract_file_metadata to return sample metadata
    async def mock_metadata(file_path):
        return {
            "filename": "test_catalog.xlsx",
            "size_mb": 1.23,
            "sheets_count": 1,
            "rows_count": 100,
            "columns_count": 8,
        }

    mock.extract_file_metadata = AsyncMock(side_effect=mock_metadata)

    # Mock extract_file_preview to return sample preview
    async def mock_preview(file_path, rows):
        return [
            {"Description": "Zawór DN50", "Price": 123.45},
            {"Description": "Rura DN100", "Price": 234.56},
        ]

    mock.extract_file_preview = AsyncMock(side_effect=mock_preview)

    return mock


@pytest.fixture
def mock_reference_indexer():
    """Mocked ReferenceIndexer."""
    mock = Mock()

    # Default: successful indexing
    def mock_index_file(file_id, descriptions, skip_if_indexed=True):
        return IndexingResult(
            file_id=file_id,
            total_descriptions=len(descriptions),
            indexed_count=len(descriptions),
            failed_count=0,
            errors=[],
            indexing_time_seconds=0.5,
        )

    mock.index_file = Mock(side_effect=mock_index_file)

    return mock


@pytest.fixture
def sample_file_data():
    """Sample file bytes."""
    return b"fake excel file content"


# ============================================================================
# WORKING FILE TESTS (no indexing)
# ============================================================================


@pytest.mark.asyncio
async def test_working_file_upload_no_indexing(mock_file_storage, sample_file_data):
    """Test working file upload - should not be indexed."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)

    # Act
    result = await use_case.execute(
        file_data=sample_file_data, filename="working.xlsx", file_type="working"
    )

    # Assert
    assert isinstance(result, FileUploadResult)
    assert result.file_type == "working"
    assert result.indexing_status == "skipped"
    assert result.indexed_count is None


@pytest.mark.asyncio
async def test_working_file_default_file_type(mock_file_storage, sample_file_data):
    """Test that file_type defaults to 'working'."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)

    # Act
    result = await use_case.execute(
        file_data=sample_file_data, filename="default.xlsx"
    )

    # Assert
    assert result.file_type == "working"
    assert result.indexing_status == "skipped"


# ============================================================================
# REFERENCE FILE TESTS (with indexing)
# ============================================================================


@pytest.mark.asyncio
async def test_reference_file_successful_indexing(
    mock_file_storage, mock_reference_indexer, sample_file_data
):
    """Test reference file upload with successful indexing."""
    # Arrange
    use_case = FileUploadUseCase(
        file_storage=mock_file_storage, reference_indexer=mock_reference_indexer
    )

    # Mock _extract_descriptions_from_file to return sample descriptions
    file_id = uuid4()
    sample_descriptions = [
        HVACDescription(
            raw_text="Zawór DN50 PN16", source_row_number=1, file_id=file_id
        ),
        HVACDescription(raw_text="Rura DN100", source_row_number=2, file_id=file_id),
    ]

    with patch.object(
        use_case, "_extract_descriptions_from_file", return_value=sample_descriptions
    ):
        # Act
        result = await use_case.execute(
            file_data=sample_file_data, filename="reference.xlsx", file_type="reference"
        )

    # Assert
    assert result.file_type == "reference"
    assert result.indexing_status == "success"
    assert result.indexed_count == 2
    assert mock_reference_indexer.index_file.called


@pytest.mark.asyncio
async def test_reference_file_partial_indexing_failure(
    mock_file_storage, mock_reference_indexer, sample_file_data
):
    """Test reference file with partial indexing failure."""
    # Arrange
    use_case = FileUploadUseCase(
        file_storage=mock_file_storage, reference_indexer=mock_reference_indexer
    )

    # Mock indexer to return partial failure
    file_id = uuid4()

    def partial_failure(file_id, descriptions, skip_if_indexed=True):
        return IndexingResult(
            file_id=file_id,
            total_descriptions=10,
            indexed_count=7,
            failed_count=3,
            errors=["Error 1", "Error 2", "Error 3"],
        )

    mock_reference_indexer.index_file.side_effect = partial_failure

    # Mock description extraction
    sample_descriptions = [
        HVACDescription(raw_text=f"Desc {i}", source_row_number=i, file_id=file_id)
        for i in range(1, 11)
    ]

    with patch.object(
        use_case, "_extract_descriptions_from_file", return_value=sample_descriptions
    ):
        # Act
        result = await use_case.execute(
            file_data=sample_file_data, filename="reference.xlsx", file_type="reference"
        )

    # Assert
    assert result.file_type == "reference"
    assert result.indexing_status == "partial"
    assert result.indexed_count == 7


@pytest.mark.asyncio
async def test_reference_file_complete_indexing_failure(
    mock_file_storage, mock_reference_indexer, sample_file_data
):
    """Test reference file with complete indexing failure (0 indexed)."""
    # Arrange
    use_case = FileUploadUseCase(
        file_storage=mock_file_storage, reference_indexer=mock_reference_indexer
    )

    # Mock indexer to return complete failure
    file_id = uuid4()

    def complete_failure(file_id, descriptions, skip_if_indexed=True):
        return IndexingResult(
            file_id=file_id,
            total_descriptions=10,
            indexed_count=0,
            failed_count=10,
            errors=["All failed"],
        )

    mock_reference_indexer.index_file.side_effect = complete_failure

    # Mock description extraction
    sample_descriptions = [
        HVACDescription(raw_text=f"Desc {i}", source_row_number=i, file_id=file_id)
        for i in range(1, 11)
    ]

    with patch.object(
        use_case, "_extract_descriptions_from_file", return_value=sample_descriptions
    ):
        # Act
        result = await use_case.execute(
            file_data=sample_file_data, filename="reference.xlsx", file_type="reference"
        )

    # Assert
    assert result.file_type == "reference"
    assert result.indexing_status == "failed"
    assert result.indexed_count == 0


@pytest.mark.asyncio
async def test_reference_file_without_indexer(mock_file_storage, sample_file_data):
    """Test reference file upload without indexer configured."""
    # Arrange - No indexer provided
    use_case = FileUploadUseCase(file_storage=mock_file_storage, reference_indexer=None)

    # Act
    result = await use_case.execute(
        file_data=sample_file_data, filename="reference.xlsx", file_type="reference"
    )

    # Assert
    assert result.file_type == "reference"
    assert result.indexing_status == "skipped"
    assert result.indexed_count == 0


@pytest.mark.asyncio
async def test_reference_file_indexing_exception_does_not_block_upload(
    mock_file_storage, mock_reference_indexer, sample_file_data
):
    """Test that indexing exception doesn't block file upload."""
    # Arrange
    use_case = FileUploadUseCase(
        file_storage=mock_file_storage, reference_indexer=mock_reference_indexer
    )

    # Mock indexer to raise exception
    mock_reference_indexer.index_file.side_effect = RuntimeError("Indexing crashed")

    # Mock description extraction
    file_id = uuid4()
    sample_descriptions = [
        HVACDescription(raw_text="Test", source_row_number=1, file_id=file_id)
    ]

    with patch.object(
        use_case, "_extract_descriptions_from_file", return_value=sample_descriptions
    ):
        # Act
        result = await use_case.execute(
            file_data=sample_file_data, filename="reference.xlsx", file_type="reference"
        )

    # Assert - Upload succeeded despite indexing failure
    assert result.file_type == "reference"
    assert result.indexing_status == "failed"
    assert result.indexed_count == 0
    assert result.file_id is not None  # Upload succeeded


# ============================================================================
# DESCRIPTION EXTRACTION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_extract_descriptions_from_file(mock_file_storage):
    """Test _extract_descriptions_from_file helper method."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)
    file_id = uuid4()
    fake_path = Path("/fake/file.xlsx")

    # Mock pandas DataFrame
    import pandas as pd

    mock_df = pd.DataFrame(
        {
            "Description": [
                "Zawór kulowy DN50 PN16",
                "Rura stalowa DN100",
                "",  # Empty - should be skipped
                "Kolano 90° DN50",
            ]
        }
    )

    with patch("pandas.read_excel", return_value=mock_df):
        # Act
        descriptions = await use_case._extract_descriptions_from_file(
            fake_path, file_id
        )

    # Assert
    assert len(descriptions) == 3  # Empty row skipped
    assert all(isinstance(d, HVACDescription) for d in descriptions)
    assert descriptions[0].raw_text == "Zawór kulowy DN50 PN16"
    assert descriptions[0].source_row_number == 2  # Row 1 is header
    assert descriptions[0].file_id == file_id
    assert descriptions[1].raw_text == "Rura stalowa DN100"
    assert descriptions[1].source_row_number == 3
    assert descriptions[2].raw_text == "Kolano 90° DN50"
    assert descriptions[2].source_row_number == 5  # Row 4 was empty


@pytest.mark.asyncio
async def test_extract_descriptions_skips_nan_values(mock_file_storage):
    """Test that NaN values are skipped during extraction."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)
    file_id = uuid4()
    fake_path = Path("/fake/file.xlsx")

    import pandas as pd
    import numpy as np

    mock_df = pd.DataFrame(
        {
            "Description": [
                "Valid description",
                np.nan,  # NaN - should be skipped
                "Another valid description",
            ]
        }
    )

    with patch("pandas.read_excel", return_value=mock_df):
        # Act
        descriptions = await use_case._extract_descriptions_from_file(
            fake_path, file_id
        )

    # Assert
    assert len(descriptions) == 2
    assert descriptions[0].raw_text == "Valid description"
    assert descriptions[1].raw_text == "Another valid description"


@pytest.mark.asyncio
async def test_extract_descriptions_skips_too_short_text(mock_file_storage):
    """Test that too short descriptions are skipped."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)
    file_id = uuid4()
    fake_path = Path("/fake/file.xlsx")

    import pandas as pd

    mock_df = pd.DataFrame(
        {"Description": ["Valid description text", "AB", "Another valid text"]}  # Too short
    )

    with patch("pandas.read_excel", return_value=mock_df):
        # Act
        descriptions = await use_case._extract_descriptions_from_file(
            fake_path, file_id
        )

    # Assert
    assert len(descriptions) == 2  # "AB" was skipped
    assert descriptions[0].raw_text == "Valid description text"
    assert descriptions[1].raw_text == "Another valid text"


# ============================================================================
# METADATA TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_upload_result_contains_all_metadata(
    mock_file_storage, sample_file_data
):
    """Test that FileUploadResult contains all required fields."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)

    # Act
    result = await use_case.execute(
        file_data=sample_file_data, filename="test.xlsx", file_type="working"
    )

    # Assert - Check all fields are present
    assert result.file_id is not None
    assert result.filename == "test_catalog.xlsx"
    assert result.size_mb == 1.23
    assert result.sheets_count == 1
    assert result.rows_count == 100
    assert result.columns_count == 8
    assert result.upload_time is not None
    assert result.preview is not None
    assert len(result.preview) == 2
    assert result.file_type == "working"
    assert result.indexing_status == "skipped"
    assert result.indexed_count is None


@pytest.mark.asyncio
async def test_file_storage_methods_called_correctly(
    mock_file_storage, sample_file_data
):
    """Test that FileStorageService methods are called with correct arguments."""
    # Arrange
    use_case = FileUploadUseCase(file_storage=mock_file_storage)

    # Act
    await use_case.execute(
        file_data=sample_file_data, filename="test.xlsx", file_type="working"
    )

    # Assert
    assert mock_file_storage.save_uploaded_file.called
    assert mock_file_storage.extract_file_metadata.called
    assert mock_file_storage.extract_file_preview.called

    # Check preview called with rows=5
    preview_call_args = mock_file_storage.extract_file_preview.call_args
    assert preview_call_args[1]["rows"] == 5
