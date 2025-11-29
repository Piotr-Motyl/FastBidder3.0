"""
Matching Configuration

Configuration constants for SimpleMatchingEngine parameter weighting and thresholds.
Defines the business rules for hybrid matching (parameter-based + semantic).

Part of: Task 2.1.4 - SimpleMatchingEngine domain service
Phase: 2.1 - Domain Layer Details

Business Context:
    The hybrid matching algorithm combines parameter-based matching (regex/dictionary)
    with semantic similarity (AI embeddings). The weights reflect business priorities:
    - DN (30%) - Most critical parameter for HVAC equipment
    - Semantic (30%) - Captures intent and synonyms
    - Material (15%) - Important for compatibility
    - Valve Type (15%) - Functional requirement
    - PN (10%) - Less critical than DN but still important

Design Principles:
    - Configuration as code (not database)
    - Type-safe constants
    - Business rule documentation
    - Easy to modify for A/B testing
"""

from dataclasses import dataclass
from typing import Any, Final


# ============================================================================
# PARAMETER WEIGHTS - Business rules for scoring importance
# ============================================================================

# Individual parameter weights (must sum to 100% for parameter score)
WEIGHT_DN: Final[float] = 0.35  # Diameter Nominal - MOST CRITICAL
WEIGHT_PN: Final[float] = 0.10  # Pressure Nominal - less critical
WEIGHT_VALVE_TYPE: Final[float] = 0.15  # Valve type (kulowy, zwrotny, etc.)
WEIGHT_MATERIAL: Final[float] = 0.2  # Material (mosiądz, stal, etc.)
WEIGHT_DRIVE_TYPE: Final[float] = 0.10  # Drive type (ręczny, elektryczny)
WEIGHT_VOLTAGE: Final[float] = 0.05  # Voltage (for electric drives)
WEIGHT_MANUFACTURER: Final[float] = 0.05  # Manufacturer (bonus points)

# Sum validation for parameter weights (should equal 1.0)
PARAMETER_WEIGHTS_SUM: Final[float] = (
    WEIGHT_DN
    + WEIGHT_PN
    + WEIGHT_VALVE_TYPE
    + WEIGHT_MATERIAL
    + WEIGHT_DRIVE_TYPE
    + WEIGHT_VOLTAGE
    + WEIGHT_MANUFACTURER
)

# Hybrid scoring weights (parameter vs semantic)
WEIGHT_PARAMETERS: Final[float] = 0.40  # 40% from parameter matching
WEIGHT_SEMANTIC: Final[float] = 0.60  # 60% from semantic similarity

# Validation that hybrid weights sum to 1.0
HYBRID_WEIGHTS_SUM: Final[float] = WEIGHT_PARAMETERS + WEIGHT_SEMANTIC


# ============================================================================
# THRESHOLDS - Business rules for match acceptance
# ============================================================================

# Default threshold for accepting a match (0-100 scale)
DEFAULT_THRESHOLD: Final[float] = 75.0  # 75% minimum similarity

# Confidence level thresholds
HIGH_CONFIDENCE_THRESHOLD: Final[float] = 0.85  # Auto-approve above this
MEDIUM_CONFIDENCE_THRESHOLD: Final[float] = 0.70  # Review recommended
LOW_CONFIDENCE_THRESHOLD: Final[float] = 0.50  # Manual review required

# Score gap threshold for confidence calculation
MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE: Final[float] = 10.0  # points difference to 2nd best


# ============================================================================
# FAST-FAIL RULES - Performance optimization
# ============================================================================

# Critical parameters that must match for equipment compatibility
CRITICAL_PARAMETERS: Final[list[str]] = ["dn"]  # DN mismatch = incompatible equipment

# Enable/disable fast-fail optimization
ENABLE_FAST_FAIL: Final[bool] = True

# DN tolerance (optional - for future use)
# If DN difference > tolerance, fast-fail even if other params match
DN_TOLERANCE_PERCENT: Final[float] = 0.0  # 0% = exact match only (Phase 2)
# Future: 5% tolerance might allow DN50 to match DN48 or DN52


# ============================================================================
# SEMANTIC MATCHING CONFIGURATION
# ============================================================================

# Placeholder configuration for Phase 4 AI integration
# In Phase 2/3: semantic_score returns fixed neutral value
SEMANTIC_PLACEHOLDER_VALUE: Final[float] = 0.5  # Neutral (50%)

# Future Phase 4 configuration
SEMANTIC_MODEL_NAME: Final[str] = "paraphrase-multilingual-mpnet-base-v2"
SEMANTIC_CACHE_ENABLED: Final[bool] = True
SEMANTIC_BATCH_SIZE: Final[int] = 32


# ============================================================================
# VALIDATION AND HELPER DATACLASSES
# ============================================================================


@dataclass(frozen=True)
class ParameterWeights:
    """
    Structured representation of parameter weights for type safety.

    Provides named access to weights and validation that they sum correctly.
    Immutable to prevent accidental modification during matching.

    Usage:
        weights = ParameterWeights.default()
        score = weights.dn * dn_match + weights.pn * pn_match + ...
    """

    dn: float = WEIGHT_DN
    pn: float = WEIGHT_PN
    valve_type: float = WEIGHT_VALVE_TYPE
    material: float = WEIGHT_MATERIAL
    drive_type: float = WEIGHT_DRIVE_TYPE
    voltage: float = WEIGHT_VOLTAGE
    manufacturer: float = WEIGHT_MANUFACTURER

    def __post_init__(self) -> None:
        """Validate that weights sum to approximately 1.0"""
        total = (
            self.dn
            + self.pn
            + self.valve_type
            + self.material
            + self.drive_type
            + self.voltage
            + self.manufacturer
        )
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError(
                f"Parameter weights must sum to 1.0, got {total:.4f}. "
                f"Weights: DN={self.dn}, PN={self.pn}, type={self.valve_type}, "
                f"material={self.material}, drive={self.drive_type}, "
                f"voltage={self.voltage}, manufacturer={self.manufacturer}"
            )

    @classmethod
    def default(cls) -> "ParameterWeights":
        """Get default parameter weights from module constants"""
        return cls()

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for serialization/logging"""
        return {
            "dn": self.dn,
            "pn": self.pn,
            "valve_type": self.valve_type,
            "material": self.material,
            "drive_type": self.drive_type,
            "voltage": self.voltage,
            "manufacturer": self.manufacturer,
        }


@dataclass(frozen=True)
class MatchingConfig:
    """
    Complete configuration for SimpleMatchingEngine.

    Encapsulates all configuration values in a single immutable object.
    Can be passed to matching engine constructor for dependency injection.

    Attributes:
        parameter_weights: Weight distribution for individual parameters
        hybrid_param_weight: Weight for parameter score in final calculation (0.4)
        hybrid_semantic_weight: Weight for semantic score in final calculation (0.6)
        default_threshold: Minimum score for accepting a match (75.0)
        min_score_gap_for_high_confidence: Minimum score gap to second-best for high confidence (10.0)
        enable_fast_fail: Whether to use fast-fail optimization (True)
        semantic_placeholder: Placeholder value for semantic score in Phase 2/3 (0.5)

    Usage:
        config = MatchingConfig.default()
        engine = SimpleMatchingEngine(config)
    """

    parameter_weights: ParameterWeights = ParameterWeights.default()
    hybrid_param_weight: float = WEIGHT_PARAMETERS
    hybrid_semantic_weight: float = WEIGHT_SEMANTIC
    default_threshold: float = DEFAULT_THRESHOLD
    min_score_gap_for_high_confidence: float = MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE
    enable_fast_fail: bool = ENABLE_FAST_FAIL
    semantic_placeholder: float = SEMANTIC_PLACEHOLDER_VALUE

    def __post_init__(self) -> None:
        """Validate hybrid weights sum to 1.0"""
        total = self.hybrid_param_weight + self.hybrid_semantic_weight
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"Hybrid weights must sum to 1.0, got {total:.4f}. "
                f"param_weight={self.hybrid_param_weight}, "
                f"semantic_weight={self.hybrid_semantic_weight}"
            )

    @classmethod
    def default(cls) -> "MatchingConfig":
        """
        Get default configuration from module constants.

        Returns default configuration suitable for production use.

        Returns:
            MatchingConfig with all default values

        Examples:
            >>> config = MatchingConfig.default()
            >>> config.hybrid_param_weight
            0.4
            >>> config.default_threshold
            75.0
        """
        return cls()

    @classmethod
    def for_testing(cls, **overrides: Any) -> "MatchingConfig":
        """
        Create configuration with custom overrides for testing.

        Allows partial override of default values while keeping others at defaults.
        Useful for unit tests that need specific configuration scenarios.

        Args:
            **overrides: Keyword arguments to override default values
                Valid keys: parameter_weights, hybrid_param_weight, hybrid_semantic_weight,
                           default_threshold, min_score_gap_for_high_confidence,
                           enable_fast_fail, semantic_placeholder

        Returns:
            MatchingConfig with specified overrides applied

        Raises:
            ValueError: If hybrid weights don't sum to 1.0 after overrides

        Examples:
            >>> # Test with higher threshold
            >>> config = MatchingConfig.for_testing(default_threshold=90.0)
            >>> config.default_threshold
            90.0

            >>> # Test with disabled fast-fail
            >>> config = MatchingConfig.for_testing(enable_fast_fail=False)
            >>> config.enable_fast_fail
            False

            >>> # Test with custom parameter weights
            >>> custom_weights = ParameterWeights(
            ...     dn=0.5, pn=0.2, valve_type=0.1,
            ...     material=0.1, drive_type=0.05,
            ...     voltage=0.03, manufacturer=0.02
            ... )
            >>> config = MatchingConfig.for_testing(parameter_weights=custom_weights)
            >>> config.parameter_weights.dn
            0.5

        Business Logic:
            This factory is designed for test scenarios only. Production code
            should use `default()` to ensure consistent configuration.

        Architecture Note:
            Using kwargs allows forward compatibility - tests won't break
            if new config fields are added in future phases.
        """
        # Start with defaults
        defaults = {
            "parameter_weights": ParameterWeights.default(),
            "hybrid_param_weight": WEIGHT_PARAMETERS,
            "hybrid_semantic_weight": WEIGHT_SEMANTIC,
            "default_threshold": DEFAULT_THRESHOLD,
            "min_score_gap_for_high_confidence": MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE,
            "enable_fast_fail": ENABLE_FAST_FAIL,
            "semantic_placeholder": SEMANTIC_PLACEHOLDER_VALUE,
        }

        # Apply overrides
        defaults.update(overrides)

        # Create instance (will run __post_init__ validation)
        return cls(**defaults)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for serialization/logging.

        Returns:
            Dictionary representation of configuration

        Examples:
            >>> config = MatchingConfig.default()
            >>> data = config.to_dict()
            >>> data['hybrid_param_weight']
            0.4
            >>> 'parameter_weights' in data
            True
        """
        return {
            "parameter_weights": self.parameter_weights.to_dict(),
            "hybrid_param_weight": self.hybrid_param_weight,
            "hybrid_semantic_weight": self.hybrid_semantic_weight,
            "default_threshold": self.default_threshold,
            "min_score_gap_for_high_confidence": self.min_score_gap_for_high_confidence,
            "enable_fast_fail": self.enable_fast_fail,
            "semantic_placeholder": self.semantic_placeholder,
        }


# ============================================================================
# MODULE-LEVEL VALIDATION
# ============================================================================

# Ensure weights are configured correctly on module import
assert (
    0.99 <= PARAMETER_WEIGHTS_SUM <= 1.01
), f"Parameter weights must sum to 1.0, got {PARAMETER_WEIGHTS_SUM}"

assert (
    0.99 <= HYBRID_WEIGHTS_SUM <= 1.01
), f"Hybrid weights must sum to 1.0, got {HYBRID_WEIGHTS_SUM}"

assert (
    0.0 <= DEFAULT_THRESHOLD <= 100.0
), f"Threshold must be 0-100, got {DEFAULT_THRESHOLD}"
