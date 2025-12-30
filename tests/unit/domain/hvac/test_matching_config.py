"""
Tests for MatchingConfig configuration objects.
Covers: ParameterWeights, MatchingConfig, validation, factories, immutability.
"""

import pytest

from src.domain.hvac.matching_config import (
    ParameterWeights,
    MatchingConfig,
    WEIGHT_PARAMETERS,
    WEIGHT_SEMANTIC,
    DEFAULT_THRESHOLD,
    MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE,
    ENABLE_FAST_FAIL,
    SEMANTIC_PLACEHOLDER_VALUE,
)


# ============================================================================
# TESTS - ParameterWeights
# ============================================================================


def test_parameter_weights_default():
    """Test ParameterWeights.default() creates valid instance."""
    weights = ParameterWeights.default()

    assert weights.dn == 0.35
    assert weights.pn == 0.10
    assert weights.valve_type == 0.15
    assert weights.material == 0.2
    assert weights.drive_type == 0.10
    assert weights.voltage == 0.05
    assert weights.manufacturer == 0.05


def test_parameter_weights_sum_to_one():
    """Test that default parameter weights sum to 1.0."""
    weights = ParameterWeights.default()
    total = (
        weights.dn
        + weights.pn
        + weights.valve_type
        + weights.material
        + weights.drive_type
        + weights.voltage
        + weights.manufacturer
    )

    assert 0.99 <= total <= 1.01  # Allow floating point errors


def test_parameter_weights_validation_fails_when_sum_incorrect():
    """Test ParameterWeights raises ValueError when weights don't sum to 1.0."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        ParameterWeights(
            dn=0.5,
            pn=0.3,
            valve_type=0.1,
            material=0.05,
            drive_type=0.03,
            voltage=0.01,
            manufacturer=0.05,  # Sum = 1.04 - outside tolerance range
        )


def test_parameter_weights_to_dict():
    """Test ParameterWeights.to_dict() returns correct structure."""
    weights = ParameterWeights.default()
    data = weights.to_dict()

    assert isinstance(data, dict)
    assert "dn" in data
    assert "pn" in data
    assert "valve_type" in data
    assert "material" in data
    assert "drive_type" in data
    assert "voltage" in data
    assert "manufacturer" in data
    assert data["dn"] == 0.35


def test_parameter_weights_immutability():
    """Test that ParameterWeights is immutable (frozen)."""
    weights = ParameterWeights.default()

    with pytest.raises(Exception):  # dataclass.FrozenInstanceError or AttributeError
        weights.dn = 0.5  # type: ignore


# ============================================================================
# TESTS - MatchingConfig
# ============================================================================


def test_matching_config_default():
    """Test MatchingConfig.default() creates valid instance with all defaults."""
    config = MatchingConfig.default()

    assert config.hybrid_param_weight == WEIGHT_PARAMETERS
    assert config.hybrid_semantic_weight == WEIGHT_SEMANTIC
    assert config.default_threshold == DEFAULT_THRESHOLD
    assert config.min_score_gap_for_high_confidence == MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE
    assert config.enable_fast_fail == ENABLE_FAST_FAIL
    assert config.semantic_placeholder == SEMANTIC_PLACEHOLDER_VALUE
    assert isinstance(config.parameter_weights, ParameterWeights)


def test_matching_config_hybrid_weights_sum_to_one():
    """Test that hybrid weights (param + semantic) sum to 1.0."""
    config = MatchingConfig.default()
    total = config.hybrid_param_weight + config.hybrid_semantic_weight

    assert 0.99 <= total <= 1.01


def test_matching_config_validation_fails_when_hybrid_weights_incorrect():
    """Test MatchingConfig raises ValueError when hybrid weights don't sum to 1.0."""
    with pytest.raises(ValueError, match="Hybrid weights must sum to 1.0"):
        MatchingConfig(
            hybrid_param_weight=0.5,
            hybrid_semantic_weight=0.3,  # Sum = 0.8, should fail
        )


def test_matching_config_for_testing_with_threshold_override():
    """Test for_testing() with threshold override."""
    config = MatchingConfig.for_testing(default_threshold=90.0)

    assert config.default_threshold == 90.0
    # Other values should be defaults
    assert config.hybrid_param_weight == WEIGHT_PARAMETERS
    assert config.enable_fast_fail == ENABLE_FAST_FAIL


def test_matching_config_for_testing_with_fast_fail_override():
    """Test for_testing() with fast_fail disabled."""
    config = MatchingConfig.for_testing(enable_fast_fail=False)

    assert config.enable_fast_fail is False
    # Other values should be defaults
    assert config.default_threshold == DEFAULT_THRESHOLD


def test_matching_config_for_testing_with_multiple_overrides():
    """Test for_testing() with multiple overrides."""
    config = MatchingConfig.for_testing(
        default_threshold=80.0,
        enable_fast_fail=False,
        semantic_placeholder=0.7,
    )

    assert config.default_threshold == 80.0
    assert config.enable_fast_fail is False
    assert config.semantic_placeholder == 0.7
    # Unchanged values should be defaults
    assert config.hybrid_param_weight == WEIGHT_PARAMETERS


def test_matching_config_for_testing_with_custom_parameter_weights():
    """Test for_testing() with custom ParameterWeights."""
    custom_weights = ParameterWeights(
        dn=0.5,
        pn=0.2,
        valve_type=0.1,
        material=0.1,
        drive_type=0.05,
        voltage=0.03,
        manufacturer=0.02,
    )

    config = MatchingConfig.for_testing(parameter_weights=custom_weights)

    assert config.parameter_weights.dn == 0.5
    assert config.parameter_weights.pn == 0.2


def test_matching_config_for_testing_with_min_score_gap_override():
    """Test for_testing() with min_score_gap_for_high_confidence override."""
    config = MatchingConfig.for_testing(min_score_gap_for_high_confidence=15.0)

    assert config.min_score_gap_for_high_confidence == 15.0
    # Other values should be defaults
    assert config.default_threshold == DEFAULT_THRESHOLD


def test_matching_config_to_dict():
    """Test MatchingConfig.to_dict() returns correct structure."""
    config = MatchingConfig.default()
    data = config.to_dict()

    assert isinstance(data, dict)
    assert "parameter_weights" in data
    assert "hybrid_param_weight" in data
    assert "hybrid_semantic_weight" in data
    assert "default_threshold" in data
    assert "min_score_gap_for_high_confidence" in data
    assert "enable_fast_fail" in data
    assert "semantic_placeholder" in data

    # Check nested parameter_weights is also dict
    assert isinstance(data["parameter_weights"], dict)
    assert data["hybrid_param_weight"] == 0.4
    assert data["min_score_gap_for_high_confidence"] == 10.0


def test_matching_config_immutability():
    """Test that MatchingConfig is immutable (frozen)."""
    config = MatchingConfig.default()

    with pytest.raises(Exception):  # dataclass.FrozenInstanceError or AttributeError
        config.default_threshold = 90.0  # type: ignore


def test_matching_config_contains_all_required_fields():
    """Test that MatchingConfig has all required fields from Task 3.3.2."""
    config = MatchingConfig.default()

    # Verify all required fields exist
    assert hasattr(config, "parameter_weights")
    assert hasattr(config, "hybrid_param_weight")
    assert hasattr(config, "hybrid_semantic_weight")
    assert hasattr(config, "default_threshold")
    assert hasattr(config, "min_score_gap_for_high_confidence")
    assert hasattr(config, "enable_fast_fail")
    assert hasattr(config, "semantic_placeholder")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_config_values_match_module_constants():
    """Integration test: Config default values match module-level constants."""
    config = MatchingConfig.default()

    assert config.hybrid_param_weight == WEIGHT_PARAMETERS
    assert config.hybrid_semantic_weight == WEIGHT_SEMANTIC
    assert config.default_threshold == DEFAULT_THRESHOLD
    assert config.min_score_gap_for_high_confidence == MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE
    assert config.enable_fast_fail == ENABLE_FAST_FAIL
    assert config.semantic_placeholder == SEMANTIC_PLACEHOLDER_VALUE


def test_config_can_be_used_for_dependency_injection():
    """Test that config can be passed as dependency (common DI pattern)."""
    config = MatchingConfig.for_testing(default_threshold=85.0)

    # Simulate passing to matching engine
    def mock_engine_init(cfg: MatchingConfig) -> float:
        return cfg.default_threshold

    result = mock_engine_init(config)
    assert result == 85.0


def test_for_testing_validation_still_enforced():
    """Test that for_testing() still validates hybrid weights."""
    with pytest.raises(ValueError, match="Hybrid weights must sum to 1.0"):
        MatchingConfig.for_testing(
            hybrid_param_weight=0.5,
            hybrid_semantic_weight=0.3,  # Invalid sum
        )


# ============================================================================
# TESTS - retrieval_top_k Validation (CRITICAL-3 fix)
# ============================================================================


def test_retrieval_top_k_validation_min():
    """Test MatchingConfig raises ValueError when retrieval_top_k < 1."""
    with pytest.raises(ValueError, match="retrieval_top_k must be at least 1"):
        MatchingConfig(retrieval_top_k=0)


def test_retrieval_top_k_validation_max():
    """Test MatchingConfig raises ValueError when retrieval_top_k > 500."""
    with pytest.raises(ValueError, match="exceeds maximum allowed value of 500"):
        MatchingConfig(retrieval_top_k=501)


def test_retrieval_top_k_default_value():
    """Test MatchingConfig has correct default for retrieval_top_k (backward compatibility)."""
    config = MatchingConfig.default()
    assert config.retrieval_top_k == 20


def test_retrieval_top_k_for_testing_override():
    """Test for_testing() allows retrieval_top_k override."""
    config = MatchingConfig.for_testing(retrieval_top_k=50)
    assert config.retrieval_top_k == 50
    # Other values should be defaults
    assert config.default_threshold == DEFAULT_THRESHOLD
