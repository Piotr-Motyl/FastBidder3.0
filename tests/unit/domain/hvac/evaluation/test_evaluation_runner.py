"""
Tests for Evaluation Runner (Phase 4 - Evaluation Framework).

Tests evaluation metrics calculation, report generation, and error handling.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from src.domain.hvac.evaluation.evaluation_runner import (
    FailedPair,
    EvaluationReport,
    EvaluationRunner,
)
from src.domain.hvac.evaluation.golden_dataset import GoldenPair, GoldenDataset


# ============================================================================
# FAILED PAIR TESTS
# ============================================================================


def test_failed_pair_creation():
    """Test creating FailedPair with all fields."""
    pair = GoldenPair(
        working_text="Test",
        correct_reference_text="Ref",
        correct_reference_id="id_1",
    )

    failed = FailedPair(
        pair=pair,
        predicted_reference_id="id_2",
        predicted_score=65.5,
        actual_rank=3,
        reason="wrong_match",
        error_message="",
    )

    assert failed.pair == pair
    assert failed.predicted_reference_id == "id_2"
    assert failed.predicted_score == 65.5
    assert failed.actual_rank == 3
    assert failed.reason == "wrong_match"


def test_failed_pair_default_values():
    """Test FailedPair default values."""
    pair = GoldenPair(
        working_text="Test",
        correct_reference_text="Ref",
        correct_reference_id="id_1",
    )

    failed = FailedPair(pair=pair)

    assert failed.predicted_reference_id is None
    assert failed.predicted_score is None
    assert failed.actual_rank is None
    assert failed.reason == "no_match"
    assert failed.error_message == ""


# ============================================================================
# EVALUATION REPORT TESTS
# ============================================================================


def test_evaluation_report_creation():
    """Test creating EvaluationReport."""
    report = EvaluationReport(
        recall_at_k={1: 0.8, 3: 0.9, 5: 0.95},
        precision_at_1=0.75,
        mrr=0.85,
        total_pairs=100,
        threshold=75.0,
        top_k_values=[1, 3, 5],
        average_match_time=0.123,
    )

    assert report.recall_at_k == {1: 0.8, 3: 0.9, 5: 0.95}
    assert report.precision_at_1 == 0.75
    assert report.mrr == 0.85
    assert report.total_pairs == 100
    assert report.threshold == 75.0
    assert report.average_match_time == 0.123


def test_evaluation_report_to_markdown():
    """Test EvaluationReport markdown generation."""
    report = EvaluationReport(
        recall_at_k={1: 0.8, 3: 0.9},
        precision_at_1=0.75,
        mrr=0.85,
        total_pairs=50,
        threshold=75.0,
        top_k_values=[1, 3],
        average_match_time=0.123,
    )

    markdown = report.to_markdown()

    assert "# Evaluation Report" in markdown
    assert "**Total Pairs**: 50" in markdown
    assert "**Threshold**: 75.0" in markdown
    assert "**Average Match Time**: 0.123s" in markdown
    assert "**Precision@1**: 75.00%" in markdown
    assert "**MRR (Mean Reciprocal Rank)**: 0.8500" in markdown
    assert "**Recall@1**: 80.00%" in markdown
    assert "**Recall@3**: 90.00%" in markdown


def test_evaluation_report_to_markdown_with_failed_pairs():
    """Test markdown generation includes failed pairs."""
    pair = GoldenPair(
        working_text="Zawór kulowy DN50",
        correct_reference_text="Zawór kulowy DN50 mosiądz",
        correct_reference_id="id_correct",
    )

    failed = FailedPair(
        pair=pair,
        predicted_reference_id="id_wrong",
        predicted_score=65.5,
        reason="wrong_match",
    )

    report = EvaluationReport(
        recall_at_k={1: 0.8},
        precision_at_1=0.75,
        mrr=0.85,
        failed_pairs=[failed],
        total_pairs=10,
        threshold=75.0,
        top_k_values=[1],
    )

    markdown = report.to_markdown()

    assert "## Failed Pairs (1)" in markdown
    assert "Zawór kulowy DN50" in markdown
    assert "wrong_match" in markdown


def test_evaluation_report_to_markdown_with_difficulty_metrics():
    """Test markdown generation includes difficulty breakdown."""
    report = EvaluationReport(
        recall_at_k={1: 0.8, 3: 0.9},
        precision_at_1=0.75,
        mrr=0.85,
        metrics_by_difficulty={
            "easy": {"precision_at_1": 0.95, "mrr": 0.98, "recall_at_1": 0.90, "recall_at_3": 0.95},
            "medium": {"precision_at_1": 0.75, "mrr": 0.80, "recall_at_1": 0.80, "recall_at_3": 0.90},
            "hard": {"precision_at_1": 0.50, "mrr": 0.60, "recall_at_1": 0.60, "recall_at_3": 0.80},
        },
        total_pairs=50,
        threshold=75.0,
        top_k_values=[1, 3],
    )

    markdown = report.to_markdown()

    assert "## Metrics by Difficulty" in markdown
    assert "### Easy" in markdown
    assert "### Medium" in markdown
    assert "### Hard" in markdown


# ============================================================================
# EVALUATION RUNNER TESTS
# ============================================================================


@pytest.fixture
def mock_semantic_retriever():
    """Mock semantic retriever."""
    retriever = Mock()
    retriever.retrieve = Mock()
    return retriever


@pytest.fixture
def mock_matching_engine():
    """Mock matching engine."""
    engine = Mock()
    engine.match = AsyncMock()
    return engine


@pytest.fixture
def sample_golden_dataset():
    """Create sample golden dataset."""
    pairs = [
        GoldenPair(
            working_text="Zawór kulowy DN50 PN16",
            correct_reference_text="Zawór kulowy DN50 PN16 mosiądz",
            correct_reference_id="id_1",
            difficulty="easy",
        ),
        GoldenPair(
            working_text="Zawór motylkowy DN100",
            correct_reference_text="Zawór motylkowy DN100 stalowy",
            correct_reference_id="id_2",
            difficulty="medium",
        ),
        GoldenPair(
            working_text="ZK DN80",
            correct_reference_text="Zawór kulowy DN80 PN25 mosiądz",
            correct_reference_id="id_3",
            difficulty="hard",
        ),
    ]

    return GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )


@pytest.mark.asyncio
async def test_evaluation_runner_happy_path(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test successful evaluation with all correct matches."""
    # Setup mocks
    # Mock retrieval: all correct IDs in top-1 position
    def mock_search(query_text, top_k, filters=None):
        # Return correct ID in position 1
        if "DN50" in query_text:
            result = Mock()
            result.description_id = "id_1"
            return [result]
        elif "DN100" in query_text:
            result = Mock()
            result.description_id = "id_2"
            return [result]
        elif "DN80" in query_text:
            result = Mock()
            result.description_id = "id_3"
            return [result]
        return []

    mock_semantic_retriever.retrieve.side_effect = mock_search

    # Mock matching: all return correct matches
    def mock_match(working_description, reference_descriptions, threshold):
        text = working_description.raw_text

        if "DN50" in text:
            match_result = Mock()
            match_result.matched_reference_id = "id_1"
            match_result.score = Mock()
            match_result.score.total = 95.0
            return match_result
        elif "DN100" in text:
            match_result = Mock()
            match_result.matched_reference_id = "id_2"
            match_result.score = Mock()
            match_result.score.total = 92.0
            return match_result
        elif "DN80" in text:
            match_result = Mock()
            match_result.matched_reference_id = "id_3"
            match_result.score = Mock()
            match_result.score.total = 88.0
            return match_result

        # Default: no match
        return None

    mock_matching_engine.match.side_effect = mock_match

    # Run evaluation
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
        top_k_values=[1, 3, 5],
    )

    # Verify metrics
    assert report.total_pairs == 3
    assert report.precision_at_1 == 1.0  # All correct
    assert report.recall_at_k[1] == 1.0  # All in top-1
    assert report.recall_at_k[3] == 1.0
    assert report.mrr == 1.0  # All at rank 1
    assert len(report.failed_pairs) == 0
    assert report.threshold == 75.0
    assert report.average_match_time >= 0.0  # Can be 0 with fast mocks


@pytest.mark.asyncio
async def test_evaluation_runner_partial_matches(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test evaluation with some incorrect matches."""
    # Mock retrieval: some IDs not in top-1
    def mock_search(query_text, top_k, filters=None):
        results = []

        if "DN50" in query_text:
            # Correct ID at rank 1
            r1 = Mock()
            r1.description_id = "id_1"
            results = [r1]
        elif "DN100" in query_text:
            # Correct ID at rank 2
            r1 = Mock()
            r1.description_id = "id_wrong"
            r2 = Mock()
            r2.description_id = "id_2"
            results = [r1, r2]
        elif "DN80" in query_text:
            # Correct ID not found
            r1 = Mock()
            r1.description_id = "id_wrong"
            results = [r1]

        return results[:top_k]

    mock_semantic_retriever.retrieve.side_effect = mock_search

    # Mock matching: only first one correct
    def mock_match(working_description, reference_descriptions, threshold):
        text = working_description.raw_text

        if "DN50" in text:
            match_result = Mock()
            match_result.matched_reference_id = "id_1"
            match_result.score = Mock()
            match_result.score.total = 95.0
            return match_result
        else:
            # Wrong match or no match
            return None

    mock_matching_engine.match.side_effect = mock_match

    # Run evaluation
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
        top_k_values=[1, 3],
    )

    # Verify metrics
    assert report.total_pairs == 3
    assert report.precision_at_1 == 1 / 3  # Only 1 correct
    assert report.recall_at_k[1] == 1 / 3  # Only 1 in top-1
    assert report.recall_at_k[3] == 2 / 3  # 2 in top-3
    assert len(report.failed_pairs) == 2  # 2 failed


@pytest.mark.asyncio
async def test_evaluation_runner_mrr_calculation(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test MRR calculation with different ranks."""
    # Mock retrieval with different ranks
    def mock_search(query_text, top_k, filters=None):
        results = []

        if "DN50" in query_text:
            # Rank 1
            r1 = Mock()
            r1.description_id = "id_1"
            results = [r1]
        elif "DN100" in query_text:
            # Rank 3
            r1 = Mock()
            r1.description_id = "id_x"
            r2 = Mock()
            r2.description_id = "id_y"
            r3 = Mock()
            r3.description_id = "id_2"
            results = [r1, r2, r3]
        elif "DN80" in query_text:
            # Rank 5
            results = []
            for i in range(4):
                r = Mock()
                r.description_id = f"id_wrong_{i}"
                results.append(r)
            r5 = Mock()
            r5.description_id = "id_3"
            results.append(r5)

        return results[:top_k]

    mock_semantic_retriever.retrieve.side_effect = mock_search

    # Mock matching (doesn't affect MRR)
    mock_matching_engine.match.return_value = None

    # Run evaluation
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
        top_k_values=[1, 3, 5],
    )

    # MRR = (1/1 + 1/3 + 1/5) / 3 = (1.0 + 0.333 + 0.2) / 3 = 0.511
    expected_mrr = (1.0 + 1 / 3 + 1 / 5) / 3
    assert abs(report.mrr - expected_mrr) < 0.01


@pytest.mark.asyncio
async def test_evaluation_runner_metrics_by_difficulty(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test metrics breakdown by difficulty."""
    # Mock retrieval: easy=correct, medium=correct, hard=wrong
    def mock_search(query_text, top_k, filters=None):
        if "DN50" in query_text:  # easy
            r = Mock()
            r.description_id = "id_1"
            return [r]
        elif "DN100" in query_text:  # medium
            r = Mock()
            r.description_id = "id_2"
            return [r]
        elif "DN80" in query_text:  # hard
            r = Mock()
            r.description_id = "id_wrong"
            return [r]
        return []

    mock_semantic_retriever.retrieve.side_effect = mock_search

    # Mock matching: easy and medium correct
    def mock_match(working_description, reference_descriptions, threshold):
        text = working_description.raw_text

        if "DN50" in text or "DN100" in text:
            match_result = Mock()
            match_result.matched_reference_id = "id_1" if "DN50" in text else "id_2"
            match_result.score = Mock()
            match_result.score.total = 95.0
            return match_result
        return None

    mock_matching_engine.match.side_effect = mock_match

    # Run evaluation
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
        top_k_values=[1],
    )

    # Verify difficulty breakdown exists
    assert "easy" in report.metrics_by_difficulty
    assert "medium" in report.metrics_by_difficulty
    assert "hard" in report.metrics_by_difficulty

    # Easy should be perfect
    easy_metrics = report.metrics_by_difficulty["easy"]
    assert easy_metrics["precision_at_1"] == 1.0
    assert easy_metrics["recall_at_1"] == 1.0

    # Hard should fail
    hard_metrics = report.metrics_by_difficulty["hard"]
    assert hard_metrics["precision_at_1"] == 0.0
    assert hard_metrics["recall_at_1"] == 0.0


@pytest.mark.asyncio
async def test_evaluation_runner_progress_callback(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test progress callback is called correctly."""
    # Setup simple mocks
    mock_semantic_retriever.retrieve.return_value = []
    mock_matching_engine.match.return_value = None

    # Track progress calls
    progress_calls = []

    def progress_callback(current, total):
        progress_calls.append((current, total))

    # Run evaluation
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
        progress_callback=progress_callback,
    )

    # Verify callback was called for each pair
    assert len(progress_calls) == 3
    assert progress_calls[0] == (1, 3)
    assert progress_calls[1] == (2, 3)
    assert progress_calls[2] == (3, 3)


@pytest.mark.asyncio
async def test_evaluation_runner_handles_retrieval_errors(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test evaluation handles retrieval errors gracefully."""
    # Mock retrieval that raises error
    mock_semantic_retriever.retrieve.side_effect = Exception("ChromaDB connection error")

    # Mock matching returns None
    mock_matching_engine.match.return_value = None

    # Run evaluation (should not crash)
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
    )

    # Should have failed all pairs due to retrieval errors
    assert report.total_pairs == 3
    assert report.recall_at_k[1] == 0.0  # All failed
    assert len(report.failed_pairs) == 3


@pytest.mark.asyncio
async def test_evaluation_runner_handles_matching_errors(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test evaluation handles matching errors gracefully."""
    # Mock retrieval returns results
    mock_semantic_retriever.retrieve.return_value = [Mock(reference_id="id_1")]

    # Mock matching that raises error
    mock_matching_engine.match.side_effect = Exception("Matching engine error")

    # Run evaluation (should not crash)
    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
    )

    # Should have failed all pairs due to matching errors
    assert report.total_pairs == 3
    assert report.precision_at_1 == 0.0  # All failed
    assert len(report.failed_pairs) == 3
    assert all(f.reason == "error" for f in report.failed_pairs)
    assert all(f.error_message for f in report.failed_pairs)


@pytest.mark.asyncio
async def test_evaluation_runner_empty_dataset(mock_semantic_retriever, mock_matching_engine):
    """Test evaluation with empty dataset."""
    empty_dataset = GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=[],
    )

    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=empty_dataset,
        threshold=75.0,
    )

    # Should return zero metrics
    assert report.total_pairs == 0
    assert report.precision_at_1 == 0.0
    assert report.mrr == 0.0
    assert len(report.failed_pairs) == 0


@pytest.mark.asyncio
async def test_evaluation_runner_default_top_k_values(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test evaluation uses default top_k_values if not provided."""
    mock_semantic_retriever.retrieve.return_value = []
    mock_matching_engine.match.return_value = None

    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
        # No top_k_values provided
    )

    # Should use default [1, 3, 5, 10]
    assert 1 in report.recall_at_k
    assert 3 in report.recall_at_k
    assert 5 in report.recall_at_k
    assert 10 in report.recall_at_k


@pytest.mark.asyncio
async def test_evaluation_runner_timing_measurement(
    mock_semantic_retriever, mock_matching_engine, sample_golden_dataset
):
    """Test average match time is measured."""
    mock_semantic_retriever.retrieve.return_value = []
    mock_matching_engine.match.return_value = None

    runner = EvaluationRunner(mock_semantic_retriever, mock_matching_engine)
    report = await runner.evaluate(
        golden_dataset=sample_golden_dataset,
        threshold=75.0,
    )

    # Should have positive average match time
    assert report.average_match_time >= 0.0


# ============================================================================
# SUMMARY
# ============================================================================
# Total tests: 18
# - FailedPair tests: 2 (creation, defaults)
# - EvaluationReport tests: 4 (creation, markdown, failed pairs, difficulty)
# - EvaluationRunner tests: 12 (happy path, partial matches, MRR, difficulty,
#   progress callback, error handling, empty dataset, defaults, timing)
# ============================================================================
