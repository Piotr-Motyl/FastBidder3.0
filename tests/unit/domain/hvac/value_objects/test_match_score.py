"""
Tests for MatchScore Value Object.
Covers: hybrid scoring, validation, factory method, immutability.
"""

import pytest
from pydantic import ValidationError

from src.domain.hvac.value_objects.match_score import MatchScore


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def strong_match_score():
    """Fixture for strong match (high param + high semantic)."""
    return MatchScore.create(
        parameter_score=100.0, semantic_score=92.0, threshold=75.0
    )


@pytest.fixture
def weak_match_score():
    """Fixture for weak match (low param + medium semantic)."""
    return MatchScore.create(
        parameter_score=50.0, semantic_score=76.0, threshold=75.0
    )


# ============================================================================
# HAPPY PATH TESTS - create() factory method
# ============================================================================


def test_create_calculates_final_score_correctly():
    """Test that create() correctly calculates weighted final score."""
    # Test case: param=100, semantic=90
    # Expected: 0.4 * 100 + 0.6 * 90 = 40 + 54 = 94.0
    score = MatchScore.create(parameter_score=100.0, semantic_score=90.0)
    assert score.parameter_score == 100.0
    assert score.semantic_score == 90.0
    assert score.final_score == pytest.approx(94.0, rel=1e-5)
    assert score.threshold == 75.0  # default


def test_create_with_custom_threshold():
    """Test create() with custom threshold."""
    score = MatchScore.create(
        parameter_score=80.0, semantic_score=85.0, threshold=80.0
    )
    assert score.threshold == 80.0
    # Final: 0.4 * 80 + 0.6 * 85 = 32 + 51 = 83.0
    assert score.final_score == pytest.approx(83.0, rel=1e-5)


def test_create_with_zero_scores():
    """Test create() with zero scores (no match)."""
    score = MatchScore.create(parameter_score=0.0, semantic_score=0.0)
    assert score.parameter_score == 0.0
    assert score.semantic_score == 0.0
    assert score.final_score == 0.0


def test_create_with_max_scores():
    """Test create() with maximum scores (perfect match)."""
    score = MatchScore.create(parameter_score=100.0, semantic_score=100.0)
    assert score.parameter_score == 100.0
    assert score.semantic_score == 100.0
    assert score.final_score == 100.0


# ============================================================================
# HAPPY PATH TESTS - weighted calculation
# ============================================================================


@pytest.mark.parametrize(
    "param_score,semantic_score,expected_final",
    [
        # Various combinations testing 40/60 weight distribution
        (100.0, 90.0, 94.0),  # 0.4*100 + 0.6*90 = 40 + 54 = 94
        (80.0, 70.0, 74.0),  # 0.4*80 + 0.6*70 = 32 + 42 = 74
        (50.0, 100.0, 80.0),  # 0.4*50 + 0.6*100 = 20 + 60 = 80
        (100.0, 50.0, 70.0),  # 0.4*100 + 0.6*50 = 40 + 30 = 70
        (75.0, 75.0, 75.0),  # 0.4*75 + 0.6*75 = 30 + 45 = 75
        (60.0, 80.0, 72.0),  # 0.4*60 + 0.6*80 = 24 + 48 = 72
    ],
)
def test_weighted_calculation_formula(param_score, semantic_score, expected_final):
    """Test that weighted formula (40/60) is correctly applied."""
    score = MatchScore.create(
        parameter_score=param_score, semantic_score=semantic_score
    )
    assert score.final_score == pytest.approx(expected_final, rel=1e-5)


def test_calculate_final_score_method():
    """Test calculate_final_score() helper method."""
    score = MatchScore.create(parameter_score=100.0, semantic_score=90.0)
    # Should match the already-calculated final_score
    calculated = score.calculate_final_score()
    assert calculated == pytest.approx(94.0, rel=1e-5)
    assert calculated == score.final_score


# ============================================================================
# HAPPY PATH TESTS - is_above_threshold()
# ============================================================================


def test_is_above_threshold_true(strong_match_score):
    """Test is_above_threshold() returns True for scores above threshold."""
    # strong_match_score: final=95.2, threshold=75.0
    assert strong_match_score.is_above_threshold() is True


def test_is_above_threshold_false(weak_match_score):
    """Test is_above_threshold() returns False for scores below threshold."""
    # weak_match_score: final=65.6, threshold=75.0
    assert weak_match_score.is_above_threshold() is False


def test_is_above_threshold_exactly_at_threshold():
    """Test is_above_threshold() returns True when score equals threshold."""
    # Final: 0.4*75 + 0.6*75 = 75.0
    score = MatchScore.create(
        parameter_score=75.0, semantic_score=75.0, threshold=75.0
    )
    assert score.final_score == 75.0
    assert score.is_above_threshold() is True


# ============================================================================
# HAPPY PATH TESTS - to_dict()
# ============================================================================


def test_to_dict_structure():
    """Test to_dict() returns correct structure."""
    score = MatchScore.create(parameter_score=100.0, semantic_score=92.0)
    result = score.to_dict()

    assert isinstance(result, dict)
    assert "parameter_score" in result
    assert "semantic_score" in result
    assert "final_score" in result
    assert "threshold" in result

    assert result["parameter_score"] == 100.0
    assert result["semantic_score"] == 92.0
    assert result["final_score"] == pytest.approx(95.2, rel=1e-5)
    assert result["threshold"] == 75.0


# ============================================================================
# EDGE CASES & ERROR HANDLING - Validation
# ============================================================================


def test_validation_parameter_score_above_range():
    """Test that parameter_score > 100 raises ValidationError."""
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        MatchScore.create(parameter_score=150.0, semantic_score=90.0)


def test_validation_parameter_score_below_range():
    """Test that parameter_score < 0 raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        MatchScore.create(parameter_score=-10.0, semantic_score=90.0)


def test_validation_semantic_score_above_range():
    """Test that semantic_score > 100 raises ValidationError."""
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        MatchScore.create(parameter_score=80.0, semantic_score=120.0)


def test_validation_semantic_score_below_range():
    """Test that semantic_score < 0 raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        MatchScore.create(parameter_score=80.0, semantic_score=-5.0)


def test_validation_threshold_above_range():
    """Test that threshold > 100 raises ValidationError."""
    with pytest.raises(ValidationError, match="less than or equal to 100"):
        MatchScore.create(parameter_score=80.0, semantic_score=90.0, threshold=150.0)


def test_validation_threshold_below_range():
    """Test that threshold < 0 raises ValidationError."""
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        MatchScore.create(parameter_score=80.0, semantic_score=90.0, threshold=-10.0)


def test_validation_final_score_mismatch():
    """Test that manually providing incorrect final_score raises ValidationError."""
    with pytest.raises(ValidationError, match="Invalid final_score"):
        # Trying to create with incorrect final_score
        # Expected: 0.4*100 + 0.6*90 = 94.0, but providing 80.0
        MatchScore(
            parameter_score=100.0,
            semantic_score=90.0,
            final_score=80.0,  # Wrong!
            threshold=75.0,
        )


# ============================================================================
# IMMUTABILITY TESTS
# ============================================================================


def test_immutability_cannot_modify_parameter_score(strong_match_score):
    """Test that MatchScore is immutable (frozen Pydantic model)."""
    with pytest.raises(ValidationError, match="Instance is frozen"):
        strong_match_score.parameter_score = 50.0  # type: ignore


def test_immutability_cannot_modify_final_score(strong_match_score):
    """Test that final_score cannot be modified."""
    with pytest.raises(ValidationError, match="Instance is frozen"):
        strong_match_score.final_score = 80.0  # type: ignore


# ============================================================================
# EQUALITY TESTS
# ============================================================================


def test_equality_same_values():
    """Test that two MatchScore instances with same values are equal."""
    score1 = MatchScore.create(parameter_score=100.0, semantic_score=90.0)
    score2 = MatchScore.create(parameter_score=100.0, semantic_score=90.0)
    assert score1 == score2


def test_inequality_different_values():
    """Test that MatchScore instances with different values are not equal."""
    score1 = MatchScore.create(parameter_score=100.0, semantic_score=90.0)
    score2 = MatchScore.create(parameter_score=80.0, semantic_score=90.0)
    assert score1 != score2


# ============================================================================
# BUSINESS LOGIC TESTS
# ============================================================================


def test_semantic_weight_dominates_when_param_low():
    """Test that semantic score (60% weight) can compensate for low param score."""
    # High semantic but low param
    score = MatchScore.create(parameter_score=0.0, semantic_score=100.0)
    # Final: 0.4*0 + 0.6*100 = 0 + 60 = 60.0
    assert score.final_score == 60.0
    # Still below default threshold (75.0)
    assert score.is_above_threshold() is False


def test_param_weight_cannot_alone_reach_threshold():
    """Test that param score alone (40% weight) cannot reach default threshold."""
    # Max param but zero semantic
    score = MatchScore.create(parameter_score=100.0, semantic_score=0.0)
    # Final: 0.4*100 + 0.6*0 = 40 + 0 = 40.0
    assert score.final_score == 40.0
    # Cannot reach default threshold (75.0) with param alone
    assert score.is_above_threshold() is False


def test_both_components_needed_for_high_score():
    """Test that both param and semantic scores contribute to final score."""
    # Balanced high scores
    score = MatchScore.create(parameter_score=85.0, semantic_score=90.0)
    # Final: 0.4*85 + 0.6*90 = 34 + 54 = 88.0
    assert score.final_score == pytest.approx(88.0, rel=1e-5)
    assert score.is_above_threshold() is True
