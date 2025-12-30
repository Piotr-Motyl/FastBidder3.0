"""
Tests for Threshold Tuner (Phase 4 - Evaluation Framework).

Tests threshold tuning, recommendation logic, and report generation.
"""

import json
import pytest
from unittest.mock import Mock, AsyncMock

from src.domain.hvac.evaluation.threshold_tuner import (
    ThresholdResult,
    ThresholdTuningReport,
    ThresholdTuner,
)
from src.domain.hvac.evaluation.golden_dataset import GoldenPair, GoldenDataset
from src.domain.hvac.evaluation.evaluation_runner import (
    EvaluationReport,
    FailedPair,
)


# ============================================================================
# THRESHOLD RESULT TESTS
# ============================================================================


def test_threshold_result_creation():
    """Test creating ThresholdResult."""
    result = ThresholdResult(
        threshold=75.0,
        precision=0.85,
        recall=0.90,
        f1_score=0.874,
        true_positives=45,
        false_positives=8,
        false_negatives=5,
    )

    assert result.threshold == 75.0
    assert result.precision == 0.85
    assert result.recall == 0.90
    assert result.f1_score == 0.874
    assert result.true_positives == 45
    assert result.false_positives == 8
    assert result.false_negatives == 5


# ============================================================================
# THRESHOLD TUNING REPORT TESTS
# ============================================================================


def test_threshold_tuning_report_creation():
    """Test creating ThresholdTuningReport."""
    results = [
        ThresholdResult(
            threshold=70.0,
            precision=0.80,
            recall=0.95,
            f1_score=0.870,
            true_positives=47,
            false_positives=12,
            false_negatives=3,
        ),
        ThresholdResult(
            threshold=80.0,
            precision=0.90,
            recall=0.85,
            f1_score=0.874,
            true_positives=42,
            false_positives=5,
            false_negatives=8,
        ),
    ]

    report = ThresholdTuningReport(
        results=results,
        recommended_threshold=80.0,
        recommendation_reason="Max precision with recall >= 0.7",
        best_precision_threshold=80.0,
        best_recall_threshold=70.0,
        best_f1_threshold=80.0,
        min_recall_constraint=0.7,
    )

    assert len(report.results) == 2
    assert report.recommended_threshold == 80.0
    assert report.best_precision_threshold == 80.0
    assert report.best_recall_threshold == 70.0


def test_threshold_tuning_report_to_dict():
    """Test converting report to dict."""
    results = [
        ThresholdResult(
            threshold=75.0,
            precision=0.85,
            recall=0.90,
            f1_score=0.874,
            true_positives=45,
            false_positives=8,
            false_negatives=5,
        )
    ]

    report = ThresholdTuningReport(
        results=results,
        recommended_threshold=75.0,
    )

    report_dict = report.to_dict()

    assert "results" in report_dict
    assert len(report_dict["results"]) == 1
    assert report_dict["recommended_threshold"] == 75.0
    assert report_dict["results"][0]["threshold"] == 75.0


def test_threshold_tuning_report_save_json(tmp_path):
    """Test saving report to JSON file."""
    results = [
        ThresholdResult(
            threshold=75.0,
            precision=0.85,
            recall=0.90,
            f1_score=0.874,
            true_positives=45,
            false_positives=8,
            false_negatives=5,
        )
    ]

    report = ThresholdTuningReport(
        results=results,
        recommended_threshold=75.0,
    )

    output_path = tmp_path / "report.json"
    report.save_json(output_path)

    # Verify file exists
    assert output_path.exists()

    # Verify content
    with open(output_path, "r") as f:
        data = json.load(f)

    assert data["recommended_threshold"] == 75.0
    assert len(data["results"]) == 1


def test_threshold_tuning_report_to_markdown():
    """Test markdown generation."""
    results = [
        ThresholdResult(
            threshold=70.0,
            precision=0.80,
            recall=0.95,
            f1_score=0.870,
            true_positives=47,
            false_positives=12,
            false_negatives=3,
        ),
        ThresholdResult(
            threshold=80.0,
            precision=0.90,
            recall=0.85,
            f1_score=0.874,
            true_positives=42,
            false_positives=5,
            false_negatives=8,
        ),
    ]

    report = ThresholdTuningReport(
        results=results,
        recommended_threshold=80.0,
        recommendation_reason="Max precision with recall >= 0.7",
        best_precision_threshold=80.0,
        best_recall_threshold=70.0,
        best_f1_threshold=80.0,
    )

    markdown = report.to_markdown()

    assert "# Threshold Tuning Report" in markdown
    assert "## Recommended Threshold" in markdown
    assert "**Threshold**: 80.0" in markdown
    assert "Max precision with recall >= 0.7" in markdown
    assert "## All Threshold Results" in markdown
    assert "70.0" in markdown
    assert "80.0" in markdown


def test_threshold_tuning_report_to_markdown_no_recommendation():
    """Test markdown generation when no recommendation."""
    results = [
        ThresholdResult(
            threshold=90.0,
            precision=0.95,
            recall=0.60,  # Below min_recall
            f1_score=0.735,
            true_positives=30,
            false_positives=2,
            false_negatives=20,
        )
    ]

    report = ThresholdTuningReport(
        results=results,
        recommended_threshold=None,
        recommendation_reason="No threshold achieves minimum recall of 70%",
        best_precision_threshold=90.0,
        best_recall_threshold=90.0,
        best_f1_threshold=90.0,
    )

    markdown = report.to_markdown()

    assert "⚠️ No Recommended Threshold" in markdown
    assert "No threshold achieves minimum recall" in markdown


# ============================================================================
# THRESHOLD TUNER TESTS
# ============================================================================


@pytest.fixture
def mock_evaluation_runner():
    """Mock evaluation runner."""
    runner = Mock()
    runner.evaluate = AsyncMock()
    return runner


@pytest.fixture
def sample_golden_dataset():
    """Create sample golden dataset."""
    pairs = [
        GoldenPair(
            working_text=f"Test {i}",
            correct_reference_text=f"Ref {i}",
            correct_reference_id=f"id_{i}",
        )
        for i in range(50)
    ]

    return GoldenDataset(
        version="1.0",
        created_at="2024-01-15T10:30:00",
        pairs=pairs,
    )


def create_mock_evaluation_report(
    threshold: float, precision: float, recall: float
) -> EvaluationReport:
    """Helper to create mock evaluation report."""
    total_pairs = 50
    correct_matches = int(recall * total_pairs)  # TP
    incorrect_pairs = total_pairs - correct_matches  # FN + (some FP if precision < 1.0)

    # Calculate failed pairs based on precision
    # precision = TP / (TP + FP)
    # If precision < 1.0, some matches are wrong (FP)
    if precision < 1.0 and precision > 0:
        # TP / (TP + FP) = precision
        # FP = TP/precision - TP = TP * (1/precision - 1)
        wrong_matches = int(correct_matches * (1 / precision - 1))
    else:
        wrong_matches = 0

    failed_pairs = []
    # Add wrong matches (FP)
    for i in range(wrong_matches):
        failed_pairs.append(
            FailedPair(
                pair=GoldenPair(
                    working_text=f"Test {i}",
                    correct_reference_text=f"Ref {i}",
                    correct_reference_id=f"id_{i}",
                ),
                reason="wrong_match",
            )
        )

    # Add no matches (FN)
    no_matches = incorrect_pairs - wrong_matches
    for i in range(max(0, no_matches)):
        failed_pairs.append(
            FailedPair(
                pair=GoldenPair(
                    working_text=f"Test {i+wrong_matches}",
                    correct_reference_text=f"Ref {i+wrong_matches}",
                    correct_reference_id=f"id_{i+wrong_matches}",
                ),
                reason="no_match",
            )
        )

    return EvaluationReport(
        recall_at_k={1: recall},
        precision_at_1=precision,
        mrr=0.85,
        total_pairs=total_pairs,
        threshold=threshold,
        top_k_values=[1],
        failed_pairs=failed_pairs,
    )


@pytest.mark.asyncio
async def test_threshold_tuner_happy_path(
    mock_evaluation_runner, sample_golden_dataset
):
    """Test successful threshold tuning."""
    # Mock evaluation reports for different thresholds
    # Lower threshold = higher recall, lower precision
    # Higher threshold = lower recall, higher precision
    def mock_evaluate(golden_dataset, threshold, top_k_values):
        if threshold == 70.0:
            return create_mock_evaluation_report(70.0, precision=0.80, recall=0.95)
        elif threshold == 75.0:
            return create_mock_evaluation_report(75.0, precision=0.85, recall=0.90)
        elif threshold == 80.0:
            return create_mock_evaluation_report(80.0, precision=0.90, recall=0.85)
        elif threshold == 85.0:
            return create_mock_evaluation_report(85.0, precision=0.92, recall=0.75)
        else:
            return create_mock_evaluation_report(threshold, precision=0.95, recall=0.65)

    mock_evaluation_runner.evaluate.side_effect = mock_evaluate

    # Run tuning
    tuner = ThresholdTuner(mock_evaluation_runner)
    report = await tuner.tune(
        golden_dataset=sample_golden_dataset,
        thresholds=[70.0, 75.0, 80.0, 85.0, 90.0],
        min_recall=0.7,
    )

    # Verify results
    assert len(report.results) == 5
    assert report.recommended_threshold == 85.0  # Max precision with recall >= 0.7
    assert report.best_precision_threshold == 90.0  # Highest precision overall
    assert report.best_recall_threshold == 70.0  # Highest recall
    assert "Maximum precision" in report.recommendation_reason


@pytest.mark.asyncio
async def test_threshold_tuner_no_valid_threshold(
    mock_evaluation_runner, sample_golden_dataset
):
    """Test tuning when no threshold meets min_recall."""
    # All thresholds have low recall
    def mock_evaluate(golden_dataset, threshold, top_k_values):
        return create_mock_evaluation_report(threshold, precision=0.95, recall=0.60)

    mock_evaluation_runner.evaluate.side_effect = mock_evaluate

    # Run tuning
    tuner = ThresholdTuner(mock_evaluation_runner)
    report = await tuner.tune(
        golden_dataset=sample_golden_dataset,
        thresholds=[80.0, 85.0, 90.0],
        min_recall=0.7,
    )

    # Verify no recommendation
    assert report.recommended_threshold is None
    assert "No threshold achieves minimum recall" in report.recommendation_reason


@pytest.mark.asyncio
async def test_threshold_tuner_default_thresholds(
    mock_evaluation_runner, sample_golden_dataset
):
    """Test tuner uses default thresholds if not provided."""

    def mock_evaluate(golden_dataset, threshold, top_k_values):
        return create_mock_evaluation_report(threshold, precision=0.85, recall=0.80)

    mock_evaluation_runner.evaluate.side_effect = mock_evaluate

    tuner = ThresholdTuner(mock_evaluation_runner)
    report = await tuner.tune(
        golden_dataset=sample_golden_dataset,
        # No thresholds provided - should use default
    )

    # Should have 8 default thresholds: [60, 65, 70, 75, 80, 85, 90, 95]
    assert len(report.results) == 8


@pytest.mark.asyncio
async def test_threshold_tuner_progress_callback(
    mock_evaluation_runner, sample_golden_dataset
):
    """Test progress callback is called."""

    def mock_evaluate(golden_dataset, threshold, top_k_values):
        return create_mock_evaluation_report(threshold, precision=0.85, recall=0.80)

    mock_evaluation_runner.evaluate.side_effect = mock_evaluate

    # Track progress calls
    progress_calls = []

    def progress_callback(current, total, threshold):
        progress_calls.append((current, total, threshold))

    tuner = ThresholdTuner(mock_evaluation_runner)
    await tuner.tune(
        golden_dataset=sample_golden_dataset,
        thresholds=[70.0, 75.0, 80.0],
        progress_callback=progress_callback,
    )

    # Verify callback was called for each threshold
    assert len(progress_calls) == 3
    assert progress_calls[0] == (1, 3, 70.0)
    assert progress_calls[1] == (2, 3, 75.0)
    assert progress_calls[2] == (3, 3, 80.0)


@pytest.mark.asyncio
async def test_calculate_threshold_metrics(mock_evaluation_runner):
    """Test threshold metrics calculation."""
    tuner = ThresholdTuner(mock_evaluation_runner)

    # Create eval report
    eval_report = create_mock_evaluation_report(75.0, precision=0.85, recall=0.90)

    # Calculate metrics
    result = tuner._calculate_threshold_metrics(
        threshold=75.0,
        eval_report=eval_report,
        total_pairs=50,
    )

    # Verify metrics
    assert result.threshold == 75.0
    assert result.precision == 0.85
    assert result.recall == 0.90
    # F1 = 2 * 0.85 * 0.90 / (0.85 + 0.90) = 0.874
    assert abs(result.f1_score - 0.874) < 0.01
    assert result.true_positives == 45  # 0.90 * 50
    assert result.false_positives > 0  # Some wrong matches
    assert result.false_negatives == 5  # 50 - 45


def test_find_recommended_threshold_happy_path(mock_evaluation_runner):
    """Test finding recommended threshold."""
    tuner = ThresholdTuner(mock_evaluation_runner)

    results = [
        ThresholdResult(
            threshold=70.0,
            precision=0.80,
            recall=0.95,
            f1_score=0.870,
            true_positives=47,
            false_positives=12,
            false_negatives=3,
        ),
        ThresholdResult(
            threshold=80.0,
            precision=0.90,
            recall=0.85,
            f1_score=0.874,
            true_positives=42,
            false_positives=5,
            false_negatives=8,
        ),
        ThresholdResult(
            threshold=90.0,
            precision=0.95,
            recall=0.75,
            f1_score=0.840,
            true_positives=37,
            false_positives=2,
            false_negatives=13,
        ),
    ]

    threshold, reason = tuner._find_recommended_threshold(
        results=results,
        min_recall=0.7,
    )

    # Should recommend 90.0 (max precision with recall >= 0.7)
    assert threshold == 90.0
    assert "Maximum precision" in reason


def test_find_recommended_threshold_no_valid(mock_evaluation_runner):
    """Test finding recommended when no threshold meets min_recall."""
    tuner = ThresholdTuner(mock_evaluation_runner)

    results = [
        ThresholdResult(
            threshold=90.0,
            precision=0.95,
            recall=0.60,
            f1_score=0.735,
            true_positives=30,
            false_positives=2,
            false_negatives=20,
        ),
        ThresholdResult(
            threshold=95.0,
            precision=0.98,
            recall=0.50,
            f1_score=0.660,
            true_positives=25,
            false_positives=1,
            false_negatives=25,
        ),
    ]

    threshold, reason = tuner._find_recommended_threshold(
        results=results,
        min_recall=0.7,
    )

    assert threshold is None
    assert "No threshold achieves minimum recall" in reason
    assert "60.00%" in reason  # Best recall as percentage


def test_find_recommended_threshold_tie_breaker(mock_evaluation_runner):
    """Test that max precision is used as tie breaker."""
    tuner = ThresholdTuner(mock_evaluation_runner)

    results = [
        ThresholdResult(
            threshold=80.0,
            precision=0.85,
            recall=0.80,
            f1_score=0.824,
            true_positives=40,
            false_positives=7,
            false_negatives=10,
        ),
        ThresholdResult(
            threshold=85.0,
            precision=0.90,  # Higher precision
            recall=0.80,  # Same recall
            f1_score=0.847,
            true_positives=40,
            false_positives=4,
            false_negatives=10,
        ),
    ]

    threshold, reason = tuner._find_recommended_threshold(
        results=results,
        min_recall=0.7,
    )

    # Should choose 85.0 (higher precision)
    assert threshold == 85.0


# ============================================================================
# SUMMARY
# ============================================================================
# Total tests: 14
# - ThresholdResult: 1 (creation)
# - ThresholdTuningReport: 5 (creation, to_dict, save_json, to_markdown x2)
# - ThresholdTuner: 8 (happy path, no valid, defaults, progress, calculate,
#   find_recommended x3)
# ============================================================================
