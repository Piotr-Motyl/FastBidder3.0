"""
Tests for ExcelReaderService.

Covers Excel reading, column extraction, caching, and HVACDescription entity creation.
"""
import pytest
from pathlib import Path
from uuid import uuid4
import polars as pl

from src.infrastructure.file_storage.excel_reader import ExcelReaderService
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.shared.exceptions import (
    FileSizeExceededError,
    ExcelParsingError,
    ColumnNotFoundError,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def reader():
    """Fixture for ExcelReaderService instance."""
    return ExcelReaderService()


@pytest.fixture
def temp_excel_file(tmp_path):
    """
    Create a temporary Excel file for testing.

    Structure:
    - Column A: Index (1, 2, 3, ...)
    - Column B: Descriptions (with some empty rows)
    - Column C: Extra data
    - 10 rows total
    """
    # Create test DataFrame
    df = pl.DataFrame({
        "Index": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Description": [
            "Zawór kulowy DN50 PN16",  # Row 1
            "",  # Row 2 - empty (will be skipped)
            "Zawór zwrotny DN25 PN10",  # Row 3
            "   ",  # Row 4 - whitespace only (will be skipped)
            "Pompa obiegowa DN32",  # Row 5
            None,  # Row 6 - None (will be skipped)
            "Filtr siatkowy DN40 PN16",  # Row 7
            "Kolano 90° DN50",  # Row 8
            "",  # Row 9 - empty
            "Zawór regulacyjny DN20 PN25",  # Row 10
        ],
        "Extra": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    })

    # Save to temp Excel file
    file_path = tmp_path / "test_file.xlsx"
    df.write_excel(file_path)

    return file_path


@pytest.fixture
def small_excel_file(tmp_path):
    """Create a small Excel file (< 10MB) for size validation tests."""
    df = pl.DataFrame({
        "A": [1, 2, 3],
        "B": ["Test 1", "Test 2", "Test 3"],
    })

    file_path = tmp_path / "small_file.xlsx"
    df.write_excel(file_path)

    return file_path


# ============================================================================
# TESTS - _column_letter_to_index()
# ============================================================================


def test_column_letter_to_index_single_letter():
    """Test _column_letter_to_index() with single letters (A-Z)."""
    assert ExcelReaderService._column_letter_to_index("A") == 0
    assert ExcelReaderService._column_letter_to_index("B") == 1
    assert ExcelReaderService._column_letter_to_index("C") == 2
    assert ExcelReaderService._column_letter_to_index("Z") == 25


def test_column_letter_to_index_double_letter():
    """Test _column_letter_to_index() with double letters (AA, AB, ...)."""
    assert ExcelReaderService._column_letter_to_index("AA") == 26
    assert ExcelReaderService._column_letter_to_index("AB") == 27
    assert ExcelReaderService._column_letter_to_index("AZ") == 51
    assert ExcelReaderService._column_letter_to_index("BA") == 52


def test_column_letter_to_index_case_insensitive():
    """Test that _column_letter_to_index() handles lowercase letters."""
    assert ExcelReaderService._column_letter_to_index("a") == 0
    assert ExcelReaderService._column_letter_to_index("b") == 1
    assert ExcelReaderService._column_letter_to_index("aa") == 26


def test_column_letter_to_index_invalid_input():
    """Test _column_letter_to_index() with invalid input."""
    with pytest.raises(ValueError, match="must contain only letters"):
        ExcelReaderService._column_letter_to_index("A1")

    with pytest.raises(ValueError, match="must contain only letters"):
        ExcelReaderService._column_letter_to_index("")

    with pytest.raises(ValueError, match="must contain only letters"):
        ExcelReaderService._column_letter_to_index("123")


# ============================================================================
# TESTS - _validate_file_size()
# ============================================================================


def test_validate_file_size_small_file_ok(reader, small_excel_file):
    """Test _validate_file_size() with file < 10MB (should pass)."""
    # Should not raise any exception
    reader._validate_file_size(small_excel_file)


def test_validate_file_size_file_not_found(reader):
    """Test _validate_file_size() with non-existent file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        reader._validate_file_size(Path("/nonexistent/file.xlsx"))


def test_validate_file_size_file_too_large(reader, tmp_path):
    """Test _validate_file_size() with file > 10MB."""
    # Create a large file (> 10MB)
    large_file = tmp_path / "large_file.txt"

    # Write 11MB of data
    with open(large_file, "wb") as f:
        f.write(b"x" * (11 * 1024 * 1024))  # 11MB

    with pytest.raises(FileSizeExceededError, match="exceeds maximum allowed size"):
        reader._validate_file_size(large_file)


# ============================================================================
# TESTS - _load_excel_dataframe()
# ============================================================================


def test_load_excel_dataframe_success(reader, temp_excel_file):
    """Test _load_excel_dataframe() loads Excel file successfully."""
    df = reader._load_excel_dataframe(temp_excel_file, None)

    assert isinstance(df, pl.DataFrame)
    assert len(df) == 10  # 10 rows
    assert "Description" in df.columns


def test_load_excel_dataframe_caching(reader, temp_excel_file):
    """Test that _load_excel_dataframe() caches DataFrame."""
    # First load
    df1 = reader._load_excel_dataframe(temp_excel_file, None)

    # Second load (should use cache)
    df2 = reader._load_excel_dataframe(temp_excel_file, None)

    # Should be the same object from cache
    assert df1 is df2


def test_load_excel_dataframe_invalid_file(reader, tmp_path):
    """Test _load_excel_dataframe() with invalid Excel file."""
    # Create a non-Excel file
    invalid_file = tmp_path / "invalid.xlsx"
    invalid_file.write_text("This is not an Excel file")

    with pytest.raises(ExcelParsingError, match="Cannot parse Excel file"):
        reader._load_excel_dataframe(invalid_file, None)


# ============================================================================
# TESTS - _validate_column_exists()
# ============================================================================


def test_validate_column_exists_column_found(reader):
    """Test _validate_column_exists() with existing column."""
    df = pl.DataFrame({"A": [1], "B": [2], "C": [3]})

    # Should not raise exception
    reader._validate_column_exists(df, "A")
    reader._validate_column_exists(df, "B")
    reader._validate_column_exists(df, "C")


def test_validate_column_exists_column_not_found(reader):
    """Test _validate_column_exists() with non-existent column."""
    df = pl.DataFrame({"A": [1], "B": [2], "C": [3]})  # Only 3 columns

    with pytest.raises(ColumnNotFoundError, match="Column 'Z'"):
        reader._validate_column_exists(df, "Z")


# ============================================================================
# TESTS - _extract_column_range()
# ============================================================================


def test_extract_column_range_all_rows(reader):
    """Test _extract_column_range() extracts all rows."""
    df = pl.DataFrame({
        "A": [1, 2, 3],
        "B": ["Text 1", "Text 2", "Text 3"],
    })

    result = reader._extract_column_range(df, "B", start_row=1, end_row=3)

    assert len(result) == 3
    assert result[0] == ("Text 1", 1)
    assert result[1] == ("Text 2", 2)
    assert result[2] == ("Text 3", 3)


def test_extract_column_range_skip_empty(reader):
    """Test _extract_column_range() skips empty rows."""
    df = pl.DataFrame({
        "A": [1, 2, 3, 4],
        "B": ["Text 1", "", "Text 3", None],
    })

    result = reader._extract_column_range(df, "B", start_row=1, end_row=4)

    # Should skip rows 2 (empty string) and 4 (None)
    assert len(result) == 2
    assert result[0] == ("Text 1", 1)
    assert result[1] == ("Text 3", 3)


def test_extract_column_range_row_range(reader):
    """Test _extract_column_range() with specific row range."""
    df = pl.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["Row 1", "Row 2", "Row 3", "Row 4", "Row 5"],
    })

    # Extract rows 2-4 (Excel 1-based)
    result = reader._extract_column_range(df, "B", start_row=2, end_row=4)

    assert len(result) == 3
    assert result[0] == ("Row 2", 2)
    assert result[1] == ("Row 3", 3)
    assert result[2] == ("Row 4", 4)


def test_extract_column_range_invalid_range(reader):
    """Test _extract_column_range() with invalid row range (start > end)."""
    df = pl.DataFrame({"A": [1, 2, 3]})

    with pytest.raises(ValueError, match="start_row.*must be <= end_row"):
        reader._extract_column_range(df, "A", start_row=5, end_row=2)


# ============================================================================
# TESTS - _create_hvac_descriptions()
# ============================================================================


def test_create_hvac_descriptions_basic(reader):
    """Test _create_hvac_descriptions() creates entities correctly."""
    data = [
        ("Zawór DN50 PN16", 2),
        ("Zawór DN25 PN10", 4),
    ]
    file_id = uuid4()

    descriptions = reader._create_hvac_descriptions(data, file_id)

    assert len(descriptions) == 2
    assert isinstance(descriptions[0], HVACDescription)
    assert descriptions[0].raw_text == "Zawór DN50 PN16"
    assert descriptions[0].source_row_number == 2
    assert descriptions[0].file_id == file_id


def test_create_hvac_descriptions_empty_list(reader):
    """Test _create_hvac_descriptions() with empty list."""
    descriptions = reader._create_hvac_descriptions([], None)

    assert descriptions == []


# ============================================================================
# TESTS - read_descriptions() - INTEGRATION
# ============================================================================


def test_read_descriptions_success(reader, temp_excel_file):
    """Integration test: read_descriptions() reads Excel file successfully."""
    file_id = uuid4()

    descriptions = reader.read_descriptions(
        file_path=temp_excel_file,
        description_column="B",  # Column B has descriptions
        start_row=1,
        end_row=10,
        sheet_name=None,
        file_id=file_id,
    )

    # Should extract 6 non-empty descriptions (rows 1, 3, 5, 7, 8, 10)
    # Skips: 2 (empty), 4 (whitespace), 6 (None), 9 (empty)
    assert len(descriptions) == 6

    # Check first description
    assert descriptions[0].raw_text == "Zawór kulowy DN50 PN16"
    assert descriptions[0].source_row_number == 1
    assert descriptions[0].file_id == file_id

    # Check third description (row 5 in Excel, index 2 in result)
    assert descriptions[2].raw_text == "Pompa obiegowa DN32"
    assert descriptions[2].source_row_number == 5


def test_read_descriptions_column_range(reader, temp_excel_file):
    """Test read_descriptions() with specific row range."""
    descriptions = reader.read_descriptions(
        file_path=temp_excel_file,
        description_column="B",
        start_row=3,  # Start from row 3
        end_row=7,     # End at row 7
        sheet_name=None,
        file_id=None,
    )

    # Should extract rows 3, 5, 7 (rows 4 and 6 are empty/None)
    assert len(descriptions) == 3
    assert descriptions[0].source_row_number == 3
    assert descriptions[1].source_row_number == 5
    assert descriptions[2].source_row_number == 7


def test_read_descriptions_file_too_large(reader, tmp_path):
    """Test read_descriptions() with file > 10MB."""
    # Create a large file
    large_file = tmp_path / "large.xlsx"
    with open(large_file, "wb") as f:
        f.write(b"x" * (11 * 1024 * 1024))  # 11MB

    with pytest.raises(FileSizeExceededError):
        reader.read_descriptions(
            file_path=large_file,
            description_column="B",
            start_row=1,
            end_row=10,
        )


def test_read_descriptions_column_not_found(reader, temp_excel_file):
    """Test read_descriptions() with non-existent column."""
    with pytest.raises(ColumnNotFoundError, match="Column 'Z'"):
        reader.read_descriptions(
            file_path=temp_excel_file,
            description_column="Z",  # Column Z doesn't exist
            start_row=1,
            end_row=10,
        )


def test_read_descriptions_cache_reuse(reader, temp_excel_file):
    """Test that read_descriptions() reuses cached DataFrame."""
    # First call - loads DataFrame
    descriptions1 = reader.read_descriptions(
        file_path=temp_excel_file,
        description_column="B",
        start_row=1,
        end_row=10,
    )

    # Cache should have 1 entry
    assert len(reader._dataframe_cache) == 1

    # Second call - uses cache
    descriptions2 = reader.read_descriptions(
        file_path=temp_excel_file,
        description_column="B",
        start_row=1,
        end_row=10,
    )

    # Results should be the same
    assert len(descriptions1) == len(descriptions2)

    # Cache should still have 1 entry (not duplicated)
    assert len(reader._dataframe_cache) == 1


# ============================================================================
# TESTS - _clear_cache()
# ============================================================================


def test_clear_cache(reader, temp_excel_file):
    """Test _clear_cache() removes all cached DataFrames."""
    # Load some DataFrames to populate cache
    reader._load_excel_dataframe(temp_excel_file, None)

    assert len(reader._dataframe_cache) == 1

    # Clear cache
    reader._clear_cache()

    assert len(reader._dataframe_cache) == 0
