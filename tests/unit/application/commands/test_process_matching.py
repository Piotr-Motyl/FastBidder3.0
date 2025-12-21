"""
Tests for ProcessMatchingCommand.

Covers:
- Command creation and validation
- Business rules validation
- Excel column validation
- Range validation
- from_api_request factory method
- to_celery_dict serialization
- column_to_index conversion
"""

from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.application.commands.process_matching import (
    ProcessMatchingCommand,
    Range,
    ReferenceFileConfig,
    WorkingFileConfig,
)
from src.application.models import MatchingStrategy, ReportFormat
from src.domain.shared.exceptions import InvalidProcessMatchingCommandError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def valid_working_file_config():
    """Create valid working file configuration."""
    return WorkingFileConfig(
        file_id=str(uuid4()),
        description_column="C",
        description_range=Range(start=2, end=100),
        price_target_column="F",
        matching_report_column="G",
    )


@pytest.fixture
def valid_reference_file_config():
    """Create valid reference file configuration."""
    return ReferenceFileConfig(
        file_id=str(uuid4()),
        description_column="B",
        description_range=Range(start=2, end=500),
        price_source_column="D",
    )


@pytest.fixture
def valid_command(valid_working_file_config, valid_reference_file_config):
    """Create valid ProcessMatchingCommand."""
    return ProcessMatchingCommand(
        working_file=valid_working_file_config,
        reference_file=valid_reference_file_config,
        matching_threshold=80.0,
        matching_strategy=MatchingStrategy.BEST_MATCH,
        report_format=ReportFormat.SIMPLE,
    )


# ============================================================================
# HAPPY PATH TESTS - Range
# ============================================================================


def test_range_creates_valid_range():
    """Test Range creates valid range with start and end."""
    range_obj = Range(start=2, end=100)

    assert range_obj.start == 2
    assert range_obj.end == 100


def test_range_calculates_size_correctly():
    """Test Range.size() calculates number of rows correctly."""
    range_obj = Range(start=2, end=10)

    assert range_obj.size() == 9  # 10 - 2 + 1 = 9


def test_range_size_single_row():
    """Test Range.size() for single row range."""
    range_obj = Range(start=1, end=1)

    assert range_obj.size() == 1


def test_range_validates_minimum_value():
    """Test Range validates start and end must be >= 1."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Range(start=0, end=10)

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Range(start=2, end=0)


# ============================================================================
# HAPPY PATH TESTS - WorkingFileConfig
# ============================================================================


def test_working_file_config_creates_valid_config():
    """Test WorkingFileConfig creates valid configuration."""
    file_id = str(uuid4())

    config = WorkingFileConfig(
        file_id=file_id,
        description_column="C",
        description_range=Range(start=2, end=100),
        price_target_column="F",
        matching_report_column="G",
    )

    assert config.file_id == file_id
    assert config.description_column == "C"
    assert config.price_target_column == "F"
    assert config.matching_report_column == "G"


def test_working_file_config_validates_uuid_format():
    """Test WorkingFileConfig validates file_id is valid UUID format."""
    with pytest.raises(ValidationError, match="file_id must be valid UUID format"):
        WorkingFileConfig(
            file_id="not-a-uuid",
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
        )


def test_working_file_config_optional_report_column():
    """Test WorkingFileConfig allows optional matching_report_column."""
    config = WorkingFileConfig(
        file_id=str(uuid4()),
        description_column="C",
        description_range=Range(start=2, end=100),
        price_target_column="F",
        # matching_report_column not provided
    )

    assert config.matching_report_column is None


# ============================================================================
# HAPPY PATH TESTS - ReferenceFileConfig
# ============================================================================


def test_reference_file_config_creates_valid_config():
    """Test ReferenceFileConfig creates valid configuration."""
    file_id = str(uuid4())

    config = ReferenceFileConfig(
        file_id=file_id,
        description_column="B",
        description_range=Range(start=2, end=500),
        price_source_column="D",
    )

    assert config.file_id == file_id
    assert config.description_column == "B"
    assert config.price_source_column == "D"


def test_reference_file_config_validates_uuid_format():
    """Test ReferenceFileConfig validates file_id is valid UUID format."""
    with pytest.raises(ValidationError, match="file_id must be valid UUID format"):
        ReferenceFileConfig(
            file_id="invalid-uuid",
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        )


# ============================================================================
# HAPPY PATH TESTS - ProcessMatchingCommand
# ============================================================================


def test_command_creates_with_valid_data(valid_command):
    """Test ProcessMatchingCommand creates successfully with valid data."""
    assert valid_command.working_file is not None
    assert valid_command.reference_file is not None
    assert valid_command.matching_threshold == 80.0
    assert valid_command.matching_strategy == MatchingStrategy.BEST_MATCH
    assert valid_command.report_format == ReportFormat.SIMPLE


def test_command_uses_default_values():
    """Test ProcessMatchingCommand uses default values for optional fields."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
        # threshold, strategy, format not provided - should use defaults
    )

    assert command.matching_threshold == 75.0  # Default
    assert command.matching_strategy == MatchingStrategy.BEST_MATCH  # Default
    assert command.report_format == ReportFormat.SIMPLE  # Default


def test_command_validates_threshold_range():
    """Test ProcessMatchingCommand validates threshold is 1.0-100.0."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    # Below minimum
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ProcessMatchingCommand(
            working_file=WorkingFileConfig(
                file_id=wf_id,
                description_column="C",
                description_range=Range(start=2, end=100),
                price_target_column="F",
            ),
            reference_file=ReferenceFileConfig(
                file_id=rf_id,
                description_column="B",
                description_range=Range(start=2, end=500),
                price_source_column="D",
            ),
            matching_threshold=0.5,  # Too low
        )

    # Above maximum
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        ProcessMatchingCommand(
            working_file=WorkingFileConfig(
                file_id=wf_id,
                description_column="C",
                description_range=Range(start=2, end=100),
                price_target_column="F",
            ),
            reference_file=ReferenceFileConfig(
                file_id=rf_id,
                description_column="B",
                description_range=Range(start=2, end=500),
                price_source_column="D",
            ),
            matching_threshold=150.0,  # Too high
        )


def test_command_accepts_boundary_threshold_values():
    """Test ProcessMatchingCommand accepts boundary values 1.0 and 100.0."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    # Minimum value
    cmd_min = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
        matching_threshold=1.0,
    )
    assert cmd_min.matching_threshold == 1.0

    # Maximum value
    cmd_max = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
        matching_threshold=100.0,
    )
    assert cmd_max.matching_threshold == 100.0


# ============================================================================
# HAPPY PATH TESTS - validate_business_rules()
# ============================================================================


def test_validate_business_rules_passes_for_valid_command(valid_command):
    """Test validate_business_rules() passes for valid command."""
    # Should not raise
    valid_command.validate_business_rules()


def test_validate_business_rules_fails_for_same_file_ids():
    """Test validate_business_rules() fails when file IDs are identical."""
    same_id = str(uuid4())

    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=same_id,  # Same ID
            description_column="C",
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=same_id,  # Same ID
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
    )

    with pytest.raises(
        InvalidProcessMatchingCommandError,
        match="working_file.file_id and reference_file.file_id must be different",
    ):
        command.validate_business_rules()


def test_validate_business_rules_fails_for_invalid_columns():
    """Test validate_business_rules() fails for invalid column formats."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    # Invalid working file column (AAA - too long)
    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="AAA",  # Invalid: > 2 chars
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
    )

    with pytest.raises(
        InvalidProcessMatchingCommandError,
        match="Invalid column 'AAA'",
    ):
        command.validate_business_rules()


def test_validate_business_rules_fails_for_lowercase_columns():
    """Test validate_business_rules() fails for lowercase column letters."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="c",  # Invalid: lowercase
            description_range=Range(start=2, end=100),
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
    )

    with pytest.raises(
        InvalidProcessMatchingCommandError,
        match="Invalid column 'c'",
    ):
        command.validate_business_rules()


def test_validate_business_rules_fails_for_invalid_range():
    """Test validate_business_rules() fails when start > end."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="C",
            description_range=Range(start=100, end=2),  # Invalid: start > end
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
    )

    with pytest.raises(
        InvalidProcessMatchingCommandError,
        match="start.*must be less than or equal to end",
    ):
        command.validate_business_rules()


def test_validate_business_rules_fails_for_exceeding_max_rows():
    """Test validate_business_rules() fails when range exceeds 1000 rows."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=wf_id,
            description_column="C",
            description_range=Range(start=1, end=1500),  # 1500 rows > 1000 max
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=rf_id,
            description_column="B",
            description_range=Range(start=2, end=500),
            price_source_column="D",
        ),
    )

    with pytest.raises(
        InvalidProcessMatchingCommandError,
        match="exceeds maximum allowed \\(1000 rows\\)",
    ):
        command.validate_business_rules()


def test_validate_business_rules_collects_multiple_errors():
    """Test validate_business_rules() collects and returns all errors."""
    same_id = str(uuid4())

    command = ProcessMatchingCommand(
        working_file=WorkingFileConfig(
            file_id=same_id,  # Error 1: Same ID
            description_column="AAA",  # Error 2: Invalid column
            description_range=Range(start=100, end=2),  # Error 3: Invalid range
            price_target_column="F",
        ),
        reference_file=ReferenceFileConfig(
            file_id=same_id,
            description_column="B",
            description_range=Range(start=1, end=1500),  # Error 4: Too many rows
            price_source_column="D",
        ),
    )

    with pytest.raises(InvalidProcessMatchingCommandError) as exc_info:
        command.validate_business_rules()

    # Should have multiple error messages
    error = exc_info.value
    assert len(error.errors) >= 3  # At least 3 errors


# ============================================================================
# HAPPY PATH TESTS - from_api_request()
# ============================================================================


def test_from_api_request_creates_command_from_dict():
    """Test from_api_request() creates command from API request dict."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    request = {
        "working_file": {
            "file_id": wf_id,
            "description_column": "C",
            "description_range": {"start": 2, "end": 100},
            "price_target_column": "F",
            "matching_report_column": "G",
        },
        "reference_file": {
            "file_id": rf_id,
            "description_column": "B",
            "description_range": {"start": 2, "end": 500},
            "price_source_column": "D",
        },
        "matching_threshold": 85.0,
        "matching_strategy": "best_match",
        "report_format": "detailed",
    }

    command = ProcessMatchingCommand.from_api_request(request)

    assert command.working_file.file_id == wf_id
    assert command.reference_file.file_id == rf_id
    assert command.matching_threshold == 85.0
    assert command.matching_strategy == MatchingStrategy.BEST_MATCH
    assert command.report_format == ReportFormat.DETAILED


def test_from_api_request_uses_defaults_when_missing():
    """Test from_api_request() uses defaults for missing fields."""
    wf_id = str(uuid4())
    rf_id = str(uuid4())

    request = {
        "working_file": {
            "file_id": wf_id,
            "description_column": "C",
            "description_range": {"start": 2, "end": 100},
            "price_target_column": "F",
        },
        "reference_file": {
            "file_id": rf_id,
            "description_column": "B",
            "description_range": {"start": 2, "end": 500},
            "price_source_column": "D",
        },
        # threshold, strategy, format not provided
    }

    command = ProcessMatchingCommand.from_api_request(request)

    assert command.matching_threshold == 75.0  # Default
    assert command.matching_strategy == MatchingStrategy.BEST_MATCH
    assert command.report_format == ReportFormat.SIMPLE


# ============================================================================
# HAPPY PATH TESTS - to_celery_dict()
# ============================================================================


def test_to_celery_dict_serializes_command(valid_command):
    """Test to_celery_dict() serializes command to dict for Celery."""
    celery_dict = valid_command.to_celery_dict()

    assert "working_file" in celery_dict
    assert "reference_file" in celery_dict
    assert celery_dict["matching_threshold"] == 80.0
    assert celery_dict["matching_strategy"] == "best_match"  # Enum value as string
    assert celery_dict["report_format"] == "simple"  # Enum value as string


def test_to_celery_dict_includes_nested_structures(valid_command):
    """Test to_celery_dict() includes nested Range and column data."""
    celery_dict = valid_command.to_celery_dict()

    # Check working file structure
    assert celery_dict["working_file"]["description_range"]["start"] == 2
    assert celery_dict["working_file"]["description_range"]["end"] == 100
    assert celery_dict["working_file"]["description_column"] == "C"

    # Check reference file structure
    assert celery_dict["reference_file"]["description_range"]["start"] == 2
    assert celery_dict["reference_file"]["description_range"]["end"] == 500
    assert celery_dict["reference_file"]["description_column"] == "B"


# ============================================================================
# HAPPY PATH TESTS - _is_valid_excel_column()
# ============================================================================


def test_is_valid_excel_column_accepts_single_letters():
    """Test _is_valid_excel_column() accepts A-Z."""
    assert ProcessMatchingCommand._is_valid_excel_column("A") is True
    assert ProcessMatchingCommand._is_valid_excel_column("B") is True
    assert ProcessMatchingCommand._is_valid_excel_column("Z") is True


def test_is_valid_excel_column_accepts_two_letters():
    """Test _is_valid_excel_column() accepts AA-ZZ."""
    assert ProcessMatchingCommand._is_valid_excel_column("AA") is True
    assert ProcessMatchingCommand._is_valid_excel_column("AB") is True
    assert ProcessMatchingCommand._is_valid_excel_column("ZZ") is True


def test_is_valid_excel_column_rejects_three_letters():
    """Test _is_valid_excel_column() rejects AAA+."""
    assert ProcessMatchingCommand._is_valid_excel_column("AAA") is False
    assert ProcessMatchingCommand._is_valid_excel_column("AAAA") is False


def test_is_valid_excel_column_rejects_non_letters():
    """Test _is_valid_excel_column() rejects non-letter characters."""
    assert ProcessMatchingCommand._is_valid_excel_column("A1") is False
    assert ProcessMatchingCommand._is_valid_excel_column("1A") is False
    assert ProcessMatchingCommand._is_valid_excel_column("A$") is False
    assert ProcessMatchingCommand._is_valid_excel_column("") is False


def test_is_valid_excel_column_rejects_lowercase():
    """Test _is_valid_excel_column() rejects lowercase letters."""
    assert ProcessMatchingCommand._is_valid_excel_column("a") is False
    assert ProcessMatchingCommand._is_valid_excel_column("ab") is False


# ============================================================================
# HAPPY PATH TESTS - column_to_index()
# ============================================================================


def test_column_to_index_converts_single_letters():
    """Test column_to_index() converts A-Z to 0-25."""
    assert ProcessMatchingCommand.column_to_index("A") == 0
    assert ProcessMatchingCommand.column_to_index("B") == 1
    assert ProcessMatchingCommand.column_to_index("Z") == 25


def test_column_to_index_converts_two_letters():
    """Test column_to_index() converts AA-ZZ correctly."""
    assert ProcessMatchingCommand.column_to_index("AA") == 26
    assert ProcessMatchingCommand.column_to_index("AB") == 27
    assert ProcessMatchingCommand.column_to_index("AZ") == 51
    assert ProcessMatchingCommand.column_to_index("BA") == 52
    assert ProcessMatchingCommand.column_to_index("ZZ") == 701


def test_column_to_index_raises_for_invalid_column():
    """Test column_to_index() raises ValueError for invalid column."""
    with pytest.raises(ValueError, match="Invalid Excel column format"):
        ProcessMatchingCommand.column_to_index("AAA")

    with pytest.raises(ValueError, match="Invalid Excel column format"):
        ProcessMatchingCommand.column_to_index("A1")

    with pytest.raises(ValueError, match="Invalid Excel column format"):
        ProcessMatchingCommand.column_to_index("")
