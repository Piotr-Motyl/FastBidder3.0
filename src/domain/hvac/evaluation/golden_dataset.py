"""
Golden Dataset for Matching Evaluation (Phase 4).

Provides data structures and tools for managing evaluation datasets:
- GoldenPair: Single evaluation pair (working_text → correct_reference)
- GoldenDataset: Collection of golden pairs with metadata
- Loader: Load dataset from JSON files
- Validation: Verify reference IDs exist in ChromaDB

Usage:
    >>> # Load golden dataset from JSON
    >>> dataset = load_golden_dataset("data/golden_dataset.json")
    >>> print(f"Loaded {len(dataset.pairs)} pairs, version {dataset.version}")

    >>> # Validate against ChromaDB
    >>> from src.infrastructure.ai.vector_store.chroma_client import ChromaClient
    >>> chroma = ChromaClient()
    >>> validation_result = validate_golden_dataset(dataset, chroma)
    >>> print(f"Valid: {validation_result.is_valid}")

CLI Usage:
    python -m src.infrastructure.evaluation.golden_dataset validate path/to/dataset.json
"""

import json
import logging
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Literal, Tuple
from uuid import UUID

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class GoldenPair:
    """
    Single evaluation pair in golden dataset.

    Represents a known correct match between working text and reference.
    Used for evaluating matching engine quality.

    Attributes:
        working_text: Input text from working file (to be matched)
        correct_reference_text: Expected reference text that should match
        correct_reference_id: Expected reference ID in ChromaDB (format: {file_id}_{row_number})
        difficulty: Difficulty level for analysis ("easy", "medium", "hard")
            - easy: Exact or near-exact match (e.g., identical DN, PN, material)
            - medium: Requires semantic understanding (synonyms, variations)
            - hard: Complex cases (abbreviations, missing params, ambiguous)
        notes: Optional notes explaining why this is the correct match

    Examples:
        >>> # Easy case: exact match
        >>> pair = GoldenPair(
        ...     working_text="Zawór kulowy DN50 PN16 mosiężny",
        ...     correct_reference_text="Zawór kulowy DN50 PN16 mosiądz",
        ...     correct_reference_id="a3bb189e-8bf9-3888-9912-ace4e6543002_42",
        ...     difficulty="easy",
        ...     notes="Exact parameter match, only material synonym"
        ... )

        >>> # Hard case: abbreviations and missing params
        >>> pair = GoldenPair(
        ...     working_text="ZK DN50 PN16",
        ...     correct_reference_text="Zawór kulowy DN50 PN16 mosiężny kompaktowy",
        ...     correct_reference_id="a3bb189e-8bf9-3888-9912-ace4e6543002_100",
        ...     difficulty="hard",
        ...     notes="Abbreviated valve type, missing material in working text"
        ... )
    """

    working_text: str
    correct_reference_text: str
    correct_reference_id: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate golden pair data."""
        if not self.working_text or not self.working_text.strip():
            raise ValueError("working_text cannot be empty")
        if not self.correct_reference_text or not self.correct_reference_text.strip():
            raise ValueError("correct_reference_text cannot be empty")
        if not self.correct_reference_id or not self.correct_reference_id.strip():
            raise ValueError("correct_reference_id cannot be empty")
        if self.difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"difficulty must be easy/medium/hard, got: {self.difficulty}")


@dataclass
class GoldenDataset:
    """
    Collection of golden pairs for evaluation.

    Represents a versioned evaluation dataset with metadata.
    Used for systematic testing of matching engine quality.

    Attributes:
        version: Dataset version (e.g., "1.0", "2024-01-15")
        created_at: ISO timestamp when dataset was created
        pairs: List of GoldenPair objects
        description: Optional dataset description

    Examples:
        >>> dataset = GoldenDataset(
        ...     version="1.0",
        ...     created_at="2024-01-15T10:30:00",
        ...     pairs=[
        ...         GoldenPair(...),
        ...         GoldenPair(...),
        ...     ],
        ...     description="Initial evaluation dataset for HVAC matching"
        ... )
        >>> print(f"Dataset v{dataset.version}: {len(dataset.pairs)} pairs")
        Dataset v1.0: 50 pairs
    """

    version: str
    created_at: str
    pairs: list[GoldenPair] = field(default_factory=list)
    description: str = ""

    def __post_init__(self) -> None:
        """Validate golden dataset."""
        if not self.version or not self.version.strip():
            raise ValueError("version cannot be empty")
        if not self.created_at or not self.created_at.strip():
            raise ValueError("created_at cannot be empty")

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dict with dataset data, ready for JSON serialization

        Examples:
            >>> dataset_dict = dataset.to_dict()
            >>> with open("dataset.json", "w") as f:
            ...     json.dump(dataset_dict, f, indent=2)
        """
        return asdict(self)

    def save(self, path: Path | str) -> None:
        """
        Save dataset to JSON file.

        Args:
            path: File path to save to (will create parent directories)

        Examples:
            >>> dataset.save("data/golden_dataset.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved golden dataset to {path} ({len(self.pairs)} pairs)")

    @property
    def total_pairs(self) -> int:
        """Total number of pairs in dataset."""
        return len(self.pairs)

    @property
    def pairs_by_difficulty(self) -> dict[str, int]:
        """
        Count pairs by difficulty level.

        Returns:
            Dict with counts: {"easy": 10, "medium": 25, "hard": 15}
        """
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for pair in self.pairs:
            counts[pair.difficulty] += 1
        return counts

    def split(
        self,
        test_ratio: float = 0.2,
        random_seed: int | None = None,
        stratify_by_difficulty: bool = True,
    ) -> Tuple["GoldenDataset", "GoldenDataset"]:
        """
        Split dataset into train and test sets.

        This method enables splitting the golden dataset for threshold tuning and evaluation,
        allowing you to tune thresholds on training data and validate on held-out test data.

        Args:
            test_ratio: Fraction of data for test set (0.0-1.0). Default is 0.2 (80% train, 20% test).
            random_seed: Random seed for reproducibility (None = random). Use a fixed seed for
                        reproducible splits across runs.
            stratify_by_difficulty: Maintain difficulty distribution in splits. If True, each split
                                   will have the same proportion of easy/medium/hard pairs as the
                                   original dataset. Default is True.

        Returns:
            Tuple of (train_dataset, test_dataset) where each is a GoldenDataset instance.

        Raises:
            ValueError: If test_ratio is not between 0.0 and 1.0 (exclusive).

        Example:
            >>> # Split dataset with reproducible random seed
            >>> dataset = load_golden_dataset("data/golden.json")
            >>> train, test = dataset.split(test_ratio=0.2, random_seed=42)
            >>> print(f"Train: {train.total_pairs}, Test: {test.total_pairs}")
            Train: 80, Test: 20

            >>> # Verify stratification maintained difficulty distribution
            >>> print(train.pairs_by_difficulty)
            {"easy": 40, "medium": 25, "hard": 15}
            >>> print(test.pairs_by_difficulty)
            {"easy": 10, "medium": 6, "hard": 4}

        Usage in threshold tuning:
            >>> dataset = load_golden_dataset("data/golden.json")
            >>> train, test = dataset.split(test_ratio=0.3, random_seed=42)
            >>>
            >>> # Tune threshold on training set
            >>> report = await threshold_tuner.tune(golden_dataset=train, thresholds=[60, 70, 80])
            >>> optimal_threshold = report.recommended_threshold
            >>>
            >>> # Validate on test set
            >>> test_report = await evaluation_runner.evaluate(
            ...     golden_dataset=test,
            ...     threshold=optimal_threshold
            ... )
            >>> print(f"Test Precision@1: {test_report.precision_at_1:.2%}")
        """
        # Validate test_ratio
        if not (0.0 < test_ratio < 1.0):
            raise ValueError(
                f"test_ratio must be between 0.0 and 1.0 (exclusive), got {test_ratio}"
            )

        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)

        if stratify_by_difficulty:
            # Split by difficulty to maintain distribution
            train_pairs = []
            test_pairs = []

            for difficulty in ["easy", "medium", "hard"]:
                # Get all pairs of this difficulty level
                difficulty_pairs = [
                    p for p in self.pairs if p.difficulty == difficulty
                ]

                if not difficulty_pairs:
                    continue

                # Shuffle pairs of this difficulty
                random.shuffle(difficulty_pairs)

                # Split at the calculated index
                split_idx = int(len(difficulty_pairs) * (1 - test_ratio))
                train_pairs.extend(difficulty_pairs[:split_idx])
                test_pairs.extend(difficulty_pairs[split_idx:])
        else:
            # Simple random split without stratification
            shuffled_pairs = self.pairs.copy()
            random.shuffle(shuffled_pairs)

            split_idx = int(len(shuffled_pairs) * (1 - test_ratio))
            train_pairs = shuffled_pairs[:split_idx]
            test_pairs = shuffled_pairs[split_idx:]

        # Create new GoldenDataset instances
        train_dataset = GoldenDataset(
            pairs=train_pairs,
            version=self.version,
            created_at=self.created_at,
            description=f"{self.description} (train split)",
        )

        test_dataset = GoldenDataset(
            pairs=test_pairs,
            version=self.version,
            created_at=self.created_at,
            description=f"{self.description} (test split)",
        )

        return train_dataset, test_dataset


# ============================================================================
# VALIDATION RESULT
# ============================================================================


@dataclass
class ValidationResult:
    """
    Result of golden dataset validation.

    Attributes:
        is_valid: True if all reference IDs exist in ChromaDB
        total_pairs: Total pairs checked
        valid_pairs: Pairs with valid reference IDs
        invalid_pairs: Pairs with missing reference IDs
        missing_ids: List of reference IDs not found in ChromaDB
        errors: List of error messages
    """

    is_valid: bool
    total_pairs: int
    valid_pairs: int
    invalid_pairs: int
    missing_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """
        Generate validation summary.

        Returns:
            Human-readable summary string
        """
        if self.is_valid:
            return f"✓ All {self.total_pairs} pairs valid"
        else:
            return (
                f"✗ Validation failed: {self.invalid_pairs}/{self.total_pairs} pairs have missing reference IDs\n"
                f"Missing IDs: {', '.join(self.missing_ids[:5])}"
                + (f" ... and {len(self.missing_ids) - 5} more" if len(self.missing_ids) > 5 else "")
            )


# ============================================================================
# LOADER
# ============================================================================


def load_golden_dataset(path: Path | str) -> GoldenDataset:
    """
    Load golden dataset from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        GoldenDataset object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid or missing required fields
        json.JSONDecodeError: If file is not valid JSON

    JSON Format:
        {
            "version": "1.0",
            "created_at": "2024-01-15T10:30:00",
            "description": "Initial evaluation dataset",
            "pairs": [
                {
                    "working_text": "Zawór kulowy DN50 PN16",
                    "correct_reference_text": "Zawór kulowy DN50 PN16 mosiądz",
                    "correct_reference_id": "file-uuid_42",
                    "difficulty": "easy",
                    "notes": "Exact match"
                },
                ...
            ]
        }

    Examples:
        >>> dataset = load_golden_dataset("data/golden_dataset.json")
        >>> print(f"Loaded {dataset.total_pairs} pairs")
        Loaded 50 pairs
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Golden dataset file not found: {path}")

    logger.info(f"Loading golden dataset from {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Validate required fields
    if "version" not in data:
        raise ValueError("Golden dataset missing required field: version")
    if "created_at" not in data:
        raise ValueError("Golden dataset missing required field: created_at")
    if "pairs" not in data:
        raise ValueError("Golden dataset missing required field: pairs")

    # Parse pairs
    pairs = []
    for idx, pair_data in enumerate(data["pairs"]):
        try:
            pair = GoldenPair(
                working_text=pair_data["working_text"],
                correct_reference_text=pair_data["correct_reference_text"],
                correct_reference_id=pair_data["correct_reference_id"],
                difficulty=pair_data.get("difficulty", "medium"),
                notes=pair_data.get("notes", ""),
            )
            pairs.append(pair)
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid pair at index {idx}: {e}") from e

    dataset = GoldenDataset(
        version=data["version"],
        created_at=data["created_at"],
        pairs=pairs,
        description=data.get("description", ""),
    )

    logger.info(
        f"Loaded golden dataset v{dataset.version}: {dataset.total_pairs} pairs "
        f"(easy: {dataset.pairs_by_difficulty['easy']}, "
        f"medium: {dataset.pairs_by_difficulty['medium']}, "
        f"hard: {dataset.pairs_by_difficulty['hard']})"
    )

    return dataset


# ============================================================================
# VALIDATION
# ============================================================================


def validate_golden_dataset(
    dataset: GoldenDataset, chroma_client: "ChromaClient"
) -> ValidationResult:
    """
    Validate golden dataset against ChromaDB.

    Checks if all correct_reference_id exist in ChromaDB collection.
    This ensures evaluation dataset references are valid.

    Args:
        dataset: GoldenDataset to validate
        chroma_client: ChromaClient instance

    Returns:
        ValidationResult with validation status

    Examples:
        >>> from src.infrastructure.ai.vector_store.chroma_client import ChromaClient
        >>> chroma = ChromaClient()
        >>> dataset = load_golden_dataset("data/golden.json")
        >>> result = validate_golden_dataset(dataset, chroma)
        >>> if not result.is_valid:
        ...     print(result.summary())
    """
    logger.info(f"Validating golden dataset ({dataset.total_pairs} pairs)")

    missing_ids = []
    errors = []

    try:
        # Get collection
        collection = chroma_client.get_or_create_collection()

        # Check each reference ID
        for pair in dataset.pairs:
            reference_id = pair.correct_reference_id

            try:
                # Query ChromaDB for this ID
                results = collection.get(ids=[reference_id], include=["documents"])

                # Check if ID exists
                if not results["ids"] or reference_id not in results["ids"]:
                    missing_ids.append(reference_id)
                    logger.warning(f"Reference ID not found in ChromaDB: {reference_id}")

            except Exception as e:
                errors.append(f"Error checking ID {reference_id}: {e}")
                logger.error(f"Error checking reference ID {reference_id}: {e}")

    except Exception as e:
        errors.append(f"ChromaDB connection error: {e}")
        logger.error(f"Failed to validate dataset: {e}")
        return ValidationResult(
            is_valid=False,
            total_pairs=dataset.total_pairs,
            valid_pairs=0,
            invalid_pairs=dataset.total_pairs,
            missing_ids=[],
            errors=errors,
        )

    # Calculate validation result
    valid_pairs = dataset.total_pairs - len(missing_ids)
    invalid_pairs = len(missing_ids)
    is_valid = len(missing_ids) == 0 and len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid,
        total_pairs=dataset.total_pairs,
        valid_pairs=valid_pairs,
        invalid_pairs=invalid_pairs,
        missing_ids=missing_ids,
        errors=errors,
    )

    if is_valid:
        logger.info(f"✓ Golden dataset valid: all {dataset.total_pairs} reference IDs found")
    else:
        logger.warning(f"✗ Validation failed: {invalid_pairs} missing IDs, {len(errors)} errors")

    return result


# ============================================================================
# CLI TOOL
# ============================================================================


if __name__ == "__main__":
    """
    CLI tool for golden dataset operations.

    Usage:
        python -m src.infrastructure.evaluation.golden_dataset validate path/to/dataset.json
        python -m src.infrastructure.evaluation.golden_dataset info path/to/dataset.json
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.infrastructure.evaluation.golden_dataset validate <dataset.json>")
        print("  python -m src.infrastructure.evaluation.golden_dataset info <dataset.json>")
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        if len(sys.argv) < 3:
            print("Error: Missing dataset path")
            sys.exit(1)

        dataset_path = sys.argv[2]

        try:
            dataset = load_golden_dataset(dataset_path)
            print(f"\n{'='*60}")
            print(f"Golden Dataset Info: {dataset_path}")
            print(f"{'='*60}")
            print(f"Version: {dataset.version}")
            print(f"Created: {dataset.created_at}")
            print(f"Description: {dataset.description or 'N/A'}")
            print(f"\nTotal pairs: {dataset.total_pairs}")
            print(f"Difficulty breakdown:")
            for difficulty, count in dataset.pairs_by_difficulty.items():
                pct = (count / dataset.total_pairs * 100) if dataset.total_pairs > 0 else 0
                print(f"  - {difficulty:8s}: {count:3d} ({pct:5.1f}%)")
            print(f"{'='*60}\n")
            sys.exit(0)

        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)

    elif command == "validate":
        if len(sys.argv) < 3:
            print("Error: Missing dataset path")
            sys.exit(1)

        dataset_path = sys.argv[2]

        try:
            # Load dataset
            dataset = load_golden_dataset(dataset_path)

            # Initialize ChromaDB
            from src.infrastructure.ai.vector_store.chroma_client import ChromaClient

            chroma = ChromaClient()

            # Validate
            result = validate_golden_dataset(dataset, chroma)

            # Print result
            print(f"\n{'='*60}")
            print(f"Validation Result: {dataset_path}")
            print(f"{'='*60}")
            print(result.summary())
            print(f"\nDetails:")
            print(f"  Total pairs: {result.total_pairs}")
            print(f"  Valid pairs: {result.valid_pairs}")
            print(f"  Invalid pairs: {result.invalid_pairs}")

            if result.errors:
                print(f"\nErrors:")
                for error in result.errors:
                    print(f"  - {error}")

            if result.missing_ids:
                print(f"\nMissing reference IDs (first 10):")
                for ref_id in result.missing_ids[:10]:
                    print(f"  - {ref_id}")
                if len(result.missing_ids) > 10:
                    print(f"  ... and {len(result.missing_ids) - 10} more")

            print(f"{'='*60}\n")

            # Exit code: 0 if valid, 1 if invalid
            sys.exit(0 if result.is_valid else 1)

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print("Available commands: validate, info")
        sys.exit(1)
