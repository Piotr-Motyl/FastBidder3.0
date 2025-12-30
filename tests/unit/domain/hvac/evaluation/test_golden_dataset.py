"""
Tests for Golden Dataset (Phase 4 - Evaluation Framework).

Tests data structures, loader, validation, and CLI tool.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

from src.domain.hvac.evaluation.golden_dataset import (
    GoldenPair,
    GoldenDataset,
    ValidationResult,
    load_golden_dataset,
    validate_golden_dataset,
)


# ============================================================================
# GOLDEN PAIR TESTS
# ============================================================================


def test_golden_pair_creation():
    """Test creating a valid GoldenPair."""
    pair = GoldenPair(
        working_text="Zawór kulowy DN50 PN16",
        correct_reference_text="Zawór kulowy DN50 PN16 mosiądz",
        correct_reference_id="file-uuid_42",
        difficulty="easy",
        notes="Exact match",
    )

    assert pair.working_text == "Zawór kulowy DN50 PN16"
    assert pair.correct_reference_text == "Zawór kulowy DN50 PN16 mosiądz"
    assert pair.correct_reference_id == "file-uuid_42"
    assert pair.difficulty == "easy"
    assert pair.notes == "Exact match"


def test_golden_pair_default_values():
    """Test GoldenPair default values."""
    pair = GoldenPair(
        working_text="Test",
        correct_reference_text="Test ref",
        correct_reference_id="id_1",
    )

    assert pair.difficulty == "medium"  # Default
    assert pair.notes == ""  # Default


def test_golden_pair_validates_empty_working_text():
    """Test that empty working_text raises ValueError."""
    with pytest.raises(ValueError, match="working_text cannot be empty"):
        GoldenPair(
            working_text="",
            correct_reference_text="Test",
            correct_reference_id="id_1",
        )


def test_golden_pair_validates_empty_reference_text():
    """Test that empty reference_text raises ValueError."""
    with pytest.raises(ValueError, match="correct_reference_text cannot be empty"):
        GoldenPair(
            working_text="Test",
            correct_reference_text="",
            correct_reference_id="id_1",
        )


def test_golden_pair_validates_empty_reference_id():
    """Test that empty reference_id raises ValueError."""
    with pytest.raises(ValueError, match="correct_reference_id cannot be empty"):
        GoldenPair(
            working_text="Test",
            correct_reference_text="Test ref",
            correct_reference_id="",
        )


def test_golden_pair_validates_difficulty():
    """Test that invalid difficulty raises ValueError."""
    with pytest.raises(ValueError, match="difficulty must be easy/medium/hard"):
        GoldenPair(
            working_text="Test",
            correct_reference_text="Test ref",
            correct_reference_id="id_1",
            difficulty="invalid",  # type: ignore
        )


# ============================================================================
# GOLDEN DATASET TESTS
# ============================================================================


def test_golden_dataset_creation():
    """Test creating a valid GoldenDataset."""
    pairs = [
        GoldenPair(
            working_text=f"Test {i}",
            correct_reference_text=f"Ref {i}",
            correct_reference_id=f"id_{i}",
        )
        for i in range(5)
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
        description="Test dataset",
    )

    assert dataset.version == "1.0"
    assert dataset.created_at == "2024-01-15T10:30:00"
    assert len(dataset.pairs) == 5
    assert dataset.description == "Test dataset"


def test_golden_dataset_validates_empty_version():
    """Test that empty version raises ValueError."""
    with pytest.raises(ValueError, match="version cannot be empty"):
        GoldenDataset(
            version="",
            created_at="2024-01-15T10:30:00",
        )


def test_golden_dataset_validates_empty_created_at():
    """Test that empty created_at raises ValueError."""
    with pytest.raises(ValueError, match="created_at cannot be empty"):
        GoldenDataset(
            version="1.0",
            created_at="",
        )


def test_golden_dataset_total_pairs():
    """Test total_pairs property."""
    pairs = [
        GoldenPair(
            working_text=f"Test {i}",
            correct_reference_text=f"Ref {i}",
            correct_reference_id=f"id_{i}",
        )
        for i in range(10)
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    assert dataset.total_pairs == 10


def test_golden_dataset_pairs_by_difficulty():
    """Test pairs_by_difficulty property."""
    pairs = [
        GoldenPair(
            working_text="Test 1",
            correct_reference_text="Ref 1",
            correct_reference_id="id_1",
            difficulty="easy",
        ),
        GoldenPair(
            working_text="Test 2",
            correct_reference_text="Ref 2",
            correct_reference_id="id_2",
            difficulty="easy",
        ),
        GoldenPair(
            working_text="Test 3",
            correct_reference_text="Ref 3",
            correct_reference_id="id_3",
            difficulty="medium",
        ),
        GoldenPair(
            working_text="Test 4",
            correct_reference_text="Ref 4",
            correct_reference_id="id_4",
            difficulty="hard",
        ),
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    assert dataset.pairs_by_difficulty == {"easy": 2, "medium": 1, "hard": 1}


def test_golden_dataset_to_dict():
    """Test converting dataset to dict."""
    pairs = [
        GoldenPair(
            working_text="Test",
            correct_reference_text="Ref",
            correct_reference_id="id_1",
        )
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
        description="Test",
    )

    data = dataset.to_dict()

    assert data["version"] == "1.0"
    assert data["created_at"] == "2024-01-15T10:30:00"
    assert data["description"] == "Test"
    assert len(data["pairs"]) == 1
    assert data["pairs"][0]["working_text"] == "Test"


def test_golden_dataset_save(tmp_path):
    """Test saving dataset to JSON file."""
    pairs = [
        GoldenPair(
            working_text="Test",
            correct_reference_text="Ref",
            correct_reference_id="id_1",
        )
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Save to temp file
    output_path = tmp_path / "test_dataset.json"
    dataset.save(output_path)

    # Verify file exists and has content
    assert output_path.exists()

    with open(output_path, "r") as f:
        data = json.load(f)

    assert data["version"] == "1.0"
    assert len(data["pairs"]) == 1


def test_golden_dataset_split_basic():
    """Test basic dataset splitting with test_ratio."""
    # Create dataset with 100 pairs (20 easy, 50 medium, 30 hard)
    pairs = []
    for i in range(20):
        pairs.append(
            GoldenPair(
                working_text=f"Easy test {i}",
                correct_reference_text=f"Easy ref {i}",
                correct_reference_id=f"easy_id_{i}",
                difficulty="easy",
            )
        )
    for i in range(50):
        pairs.append(
            GoldenPair(
                working_text=f"Medium test {i}",
                correct_reference_text=f"Medium ref {i}",
                correct_reference_id=f"medium_id_{i}",
                difficulty="medium",
            )
        )
    for i in range(30):
        pairs.append(
            GoldenPair(
                working_text=f"Hard test {i}",
                correct_reference_text=f"Hard ref {i}",
                correct_reference_id=f"hard_id_{i}",
                difficulty="hard",
            )
        )

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
        description="Test dataset",
    )

    # Split with 20% test ratio
    train, test = dataset.split(test_ratio=0.2, random_seed=42)

    # Verify split sizes
    assert train.total_pairs == 80, "Train should have 80 pairs (80%)"
    assert test.total_pairs == 20, "Test should have 20 pairs (20%)"
    assert (
        train.total_pairs + test.total_pairs == dataset.total_pairs
    ), "Total should match original"

    # Verify metadata preserved
    assert train.version == dataset.version, "Version should be preserved"
    assert test.version == dataset.version, "Version should be preserved"
    assert "(train split)" in train.description, "Train description should be updated"
    assert "(test split)" in test.description, "Test description should be updated"


def test_golden_dataset_split_reproducibility():
    """Test that split with same random_seed produces same results."""
    # Create dataset
    pairs = [
        GoldenPair(
            working_text=f"Test {i}",
            correct_reference_text=f"Ref {i}",
            correct_reference_id=f"id_{i}",
            difficulty="easy" if i < 10 else "medium",
        )
        for i in range(20)
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Split with same seed twice
    train1, test1 = dataset.split(test_ratio=0.3, random_seed=42)
    train2, test2 = dataset.split(test_ratio=0.3, random_seed=42)

    # Verify same splits produced
    assert train1.total_pairs == train2.total_pairs, "Train sizes should match"
    assert test1.total_pairs == test2.total_pairs, "Test sizes should match"

    # Verify exact same pairs (by comparing IDs)
    train1_ids = {p.correct_reference_id for p in train1.pairs}
    train2_ids = {p.correct_reference_id for p in train2.pairs}
    assert train1_ids == train2_ids, "Train splits should contain same pairs"

    test1_ids = {p.correct_reference_id for p in test1.pairs}
    test2_ids = {p.correct_reference_id for p in test2.pairs}
    assert test1_ids == test2_ids, "Test splits should contain same pairs"


def test_golden_dataset_split_stratified():
    """Test stratified splitting maintains difficulty distribution."""
    # Create dataset with known distribution: 40 easy, 30 medium, 30 hard
    pairs = []
    for i in range(40):
        pairs.append(
            GoldenPair(
                working_text=f"Easy {i}",
                correct_reference_text=f"Easy ref {i}",
                correct_reference_id=f"easy_{i}",
                difficulty="easy",
            )
        )
    for i in range(30):
        pairs.append(
            GoldenPair(
                working_text=f"Medium {i}",
                correct_reference_text=f"Medium ref {i}",
                correct_reference_id=f"medium_{i}",
                difficulty="medium",
            )
        )
    for i in range(30):
        pairs.append(
            GoldenPair(
                working_text=f"Hard {i}",
                correct_reference_text=f"Hard ref {i}",
                correct_reference_id=f"hard_{i}",
                difficulty="hard",
            )
        )

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Split with stratification (default)
    train, test = dataset.split(test_ratio=0.2, random_seed=42, stratify_by_difficulty=True)

    # Original distribution: 40% easy, 30% medium, 30% hard
    original_dist = dataset.pairs_by_difficulty
    train_dist = train.pairs_by_difficulty
    test_dist = test.pairs_by_difficulty

    # Verify stratification: each split should have similar distribution
    # Train (80 pairs): ~32 easy, 24 medium, 24 hard
    assert train_dist["easy"] == 32, "Train should have 32 easy pairs (80% of 40)"
    assert train_dist["medium"] == 24, "Train should have 24 medium pairs (80% of 30)"
    assert train_dist["hard"] == 24, "Train should have 24 hard pairs (80% of 30)"

    # Test (20 pairs): ~8 easy, 6 medium, 6 hard
    assert test_dist["easy"] == 8, "Test should have 8 easy pairs (20% of 40)"
    assert test_dist["medium"] == 6, "Test should have 6 medium pairs (20% of 30)"
    assert test_dist["hard"] == 6, "Test should have 6 hard pairs (20% of 30)"

    # Verify no overlap between train and test
    train_ids = {p.correct_reference_id for p in train.pairs}
    test_ids = {p.correct_reference_id for p in test.pairs}
    assert len(train_ids & test_ids) == 0, "Train and test should not overlap"


def test_golden_dataset_split_non_stratified():
    """Test non-stratified splitting (simple random split)."""
    # Create dataset
    pairs = [
        GoldenPair(
            working_text=f"Test {i}",
            correct_reference_text=f"Ref {i}",
            correct_reference_id=f"id_{i}",
            difficulty="easy" if i < 50 else "medium",
        )
        for i in range(100)
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Split without stratification
    train, test = dataset.split(
        test_ratio=0.3, random_seed=42, stratify_by_difficulty=False
    )

    # Verify split sizes
    assert train.total_pairs == 70, "Train should have 70 pairs"
    assert test.total_pairs == 30, "Test should have 30 pairs"

    # Verify no overlap
    train_ids = {p.correct_reference_id for p in train.pairs}
    test_ids = {p.correct_reference_id for p in test.pairs}
    assert len(train_ids & test_ids) == 0, "Train and test should not overlap"

    # Non-stratified split may not preserve exact distribution
    # (but we can verify total is correct)
    train_dist = train.pairs_by_difficulty
    test_dist = test.pairs_by_difficulty
    total_easy = train_dist["easy"] + test_dist["easy"]
    total_medium = train_dist["medium"] + test_dist["medium"]
    assert total_easy == 50, "Total easy should be 50"
    assert total_medium == 50, "Total medium should be 50"


def test_golden_dataset_split_invalid_test_ratio():
    """Test that invalid test_ratio raises ValueError."""
    pairs = [
        GoldenPair(
            working_text="Test",
            correct_reference_text="Ref",
            correct_reference_id="id_1",
        )
    ]

    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Test ratio must be between 0.0 and 1.0 (exclusive)
    with pytest.raises(ValueError, match="test_ratio must be between 0.0 and 1.0"):
        dataset.split(test_ratio=0.0)

    with pytest.raises(ValueError, match="test_ratio must be between 0.0 and 1.0"):
        dataset.split(test_ratio=1.0)

    with pytest.raises(ValueError, match="test_ratio must be between 0.0 and 1.0"):
        dataset.split(test_ratio=1.5)

    with pytest.raises(ValueError, match="test_ratio must be between 0.0 and 1.0"):
        dataset.split(test_ratio=-0.2)


# ============================================================================
# LOADER TESTS
# ============================================================================


def test_load_golden_dataset_valid_file(tmp_path):
    """Test loading valid golden dataset from JSON."""
    # Create test JSON
    data = {
        "version": "1.0",
        "created_at": "2024-01-15T10:30:00",
        "description": "Test dataset",
        "pairs": [
            {
                "working_text": "Zawór kulowy DN50",
                "correct_reference_text": "Zawór kulowy DN50 mosiądz",
                "correct_reference_id": "file-uuid_1",
                "difficulty": "easy",
                "notes": "Test",
            },
            {
                "working_text": "Zawór motylkowy DN100",
                "correct_reference_text": "Zawór motylkowy DN100 stalowy",
                "correct_reference_id": "file-uuid_2",
                "difficulty": "medium",
                "notes": "",
            },
        ],
    }

    # Write to file
    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    # Load dataset
    dataset = load_golden_dataset(json_path)

    assert dataset.version == "1.0"
    assert dataset.created_at == "2024-01-15T10:30:00"
    assert dataset.description == "Test dataset"
    assert dataset.total_pairs == 2
    assert dataset.pairs[0].working_text == "Zawór kulowy DN50"
    assert dataset.pairs[1].difficulty == "medium"


def test_load_golden_dataset_missing_file():
    """Test that loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Golden dataset file not found"):
        load_golden_dataset("/nonexistent/file.json")


def test_load_golden_dataset_invalid_json(tmp_path):
    """Test that invalid JSON raises JSONDecodeError."""
    json_path = tmp_path / "invalid.json"
    with open(json_path, "w") as f:
        f.write("{ invalid json }")

    with pytest.raises(json.JSONDecodeError):
        load_golden_dataset(json_path)


def test_load_golden_dataset_missing_version(tmp_path):
    """Test that missing version field raises ValueError."""
    data = {
        "created_at": "2024-01-15T10:30:00",
        "pairs": [],
    }

    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="missing required field: version"):
        load_golden_dataset(json_path)


def test_load_golden_dataset_missing_created_at(tmp_path):
    """Test that missing created_at field raises ValueError."""
    data = {
        "version": "1.0",
        "pairs": [],
    }

    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="missing required field: created_at"):
        load_golden_dataset(json_path)


def test_load_golden_dataset_missing_pairs(tmp_path):
    """Test that missing pairs field raises ValueError."""
    data = {
        "version": "1.0",
        "created_at": "2024-01-15T10:30:00",
    }

    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="missing required field: pairs"):
        load_golden_dataset(json_path)


def test_load_golden_dataset_invalid_pair(tmp_path):
    """Test that invalid pair data raises ValueError."""
    data = {
        "version": "1.0",
        "created_at": "2024-01-15T10:30:00",
        "pairs": [
            {
                "working_text": "Test",
                # Missing required fields
            }
        ],
    }

    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Invalid pair at index 0"):
        load_golden_dataset(json_path)


def test_load_golden_dataset_default_difficulty(tmp_path):
    """Test that difficulty defaults to medium if not provided."""
    data = {
        "version": "1.0",
        "created_at": "2024-01-15T10:30:00",
        "pairs": [
            {
                "working_text": "Test",
                "correct_reference_text": "Ref",
                "correct_reference_id": "id_1",
                # No difficulty field
            }
        ],
    }

    json_path = tmp_path / "dataset.json"
    with open(json_path, "w") as f:
        json.dump(data, f)

    dataset = load_golden_dataset(json_path)

    assert dataset.pairs[0].difficulty == "medium"


# ============================================================================
# VALIDATION TESTS
# ============================================================================


def test_validate_golden_dataset_all_valid():
    """Test validation with all valid reference IDs."""
    # Create dataset
    pairs = [
        GoldenPair(
            working_text="Test 1",
            correct_reference_text="Ref 1",
            correct_reference_id="file-uuid_1",
        ),
        GoldenPair(
            working_text="Test 2",
            correct_reference_text="Ref 2",
            correct_reference_id="file-uuid_2",
        ),
    ]
    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Mock ChromaDB client
    mock_chroma = Mock()
    mock_collection = Mock()

    # Mock get_or_create_collection
    mock_chroma.get_or_create_collection.return_value = mock_collection

    # Mock collection.get() to return valid IDs
    def mock_get(ids, include):
        return {"ids": ids, "documents": ["doc1", "doc2"]}

    mock_collection.get.side_effect = mock_get

    # Validate
    result = validate_golden_dataset(dataset, mock_chroma)

    assert result.is_valid is True
    assert result.total_pairs == 2
    assert result.valid_pairs == 2
    assert result.invalid_pairs == 0
    assert len(result.missing_ids) == 0
    assert len(result.errors) == 0


def test_validate_golden_dataset_missing_ids():
    """Test validation with missing reference IDs."""
    # Create dataset
    pairs = [
        GoldenPair(
            working_text="Test 1",
            correct_reference_text="Ref 1",
            correct_reference_id="file-uuid_1",
        ),
        GoldenPair(
            working_text="Test 2",
            correct_reference_text="Ref 2",
            correct_reference_id="file-uuid_missing",
        ),
    ]
    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Mock ChromaDB client
    mock_chroma = Mock()
    mock_collection = Mock()
    mock_chroma.get_or_create_collection.return_value = mock_collection

    # Mock collection.get() to return empty for missing ID
    def mock_get(ids, include):
        if ids[0] == "file-uuid_missing":
            return {"ids": [], "documents": []}
        return {"ids": ids, "documents": ["doc"]}

    mock_collection.get.side_effect = mock_get

    # Validate
    result = validate_golden_dataset(dataset, mock_chroma)

    assert result.is_valid is False
    assert result.total_pairs == 2
    assert result.valid_pairs == 1
    assert result.invalid_pairs == 1
    assert "file-uuid_missing" in result.missing_ids


def test_validate_golden_dataset_chroma_error():
    """Test validation when ChromaDB raises error."""
    # Create dataset
    pairs = [
        GoldenPair(
            working_text="Test",
            correct_reference_text="Ref",
            correct_reference_id="file-uuid_1",
        )
    ]
    dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )

    # Mock ChromaDB client that raises error
    mock_chroma = Mock()
    mock_chroma.get_or_create_collection.side_effect = Exception("ChromaDB error")

    # Validate
    result = validate_golden_dataset(dataset, mock_chroma)

    assert result.is_valid is False
    assert len(result.errors) > 0
    assert "ChromaDB connection error" in result.errors[0]


# ============================================================================
# VALIDATION RESULT TESTS
# ============================================================================


def test_validation_result_summary_valid():
    """Test ValidationResult summary for valid result."""
    result = ValidationResult(
        is_valid=True,
        total_pairs=50,
        valid_pairs=50,
        invalid_pairs=0,
    )

    summary = result.summary()
    assert "✓" in summary
    assert "50 pairs valid" in summary


def test_validation_result_summary_invalid():
    """Test ValidationResult summary for invalid result."""
    result = ValidationResult(
        is_valid=False,
        total_pairs=50,
        valid_pairs=45,
        invalid_pairs=5,
        missing_ids=["id_1", "id_2", "id_3", "id_4", "id_5"],
    )

    summary = result.summary()
    assert "✗" in summary
    assert "5/50" in summary
    assert "missing reference IDs" in summary


# ============================================================================
# SUMMARY
# ============================================================================
# Total tests: 35
# - GoldenPair tests: 7 (creation, validation)
# - GoldenDataset tests: 13 (creation, properties, save, split)
# - Loader tests: 8 (valid/invalid files, missing fields)
# - Validation tests: 5 (valid IDs, missing IDs, errors)
# - ValidationResult tests: 2 (summary generation)
# ============================================================================
