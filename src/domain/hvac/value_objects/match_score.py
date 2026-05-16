"""
MatchScore — hybrid scoring result. final_score = 0.4 × parameter_score + 0.6 × semantic_score.
Use MatchScore.create() to auto-compute final_score; direct construction requires pre-calculated value.
"""

from pydantic import BaseModel, Field, model_validator


class MatchScore(BaseModel):
    """
    Hybrid matching score. final_score = 0.4 × parameter_score + 0.6 × semantic_score.

    semantic_score is real cosine similarity when using_ai=True (HybridMatchingEngine),
    or placeholder 50.0 when using_ai=False (SimpleMatchingEngine standalone).
    Check MatchResult.breakdown["using_ai"] to distinguish.
    """

    parameter_score: float = Field(
        ..., description="Score from parameter matching (0-100)", ge=0.0, le=100.0
    )

    semantic_score: float = Field(
        ..., description="Score from semantic similarity (0-100)", ge=0.0, le=100.0
    )

    final_score: float = Field(
        ...,
        description="Weighted average: 0.4 * parameter + 0.6 * semantic",
        ge=0.0,
        le=100.0,
    )

    threshold: float = Field(
        default=75.0, description="Minimum score for valid match", ge=0.0, le=100.0
    )

    model_config = {
        "frozen": True,  # Immutable value object
        "json_schema_extra": {
            "examples": [
                {
                    "parameter_score": 100.0,
                    "semantic_score": 92.0,
                    "final_score": 95.2,
                    "threshold": 75.0,
                }
            ]
        },
    }

    @model_validator(mode="after")
    def validate_final_score(self) -> "MatchScore":
        """
        Validate that final_score is correctly calculated from components.

        Business rule: final_score = 0.4 * parameter_score + 0.6 * semantic_score

        Raises:
            ValueError: If final_score doesn't match expected calculation

        Returns:
            Self for method chaining
        """
        expected_final = self.calculate_final_score()

        # Allow small floating point tolerance (0.01)
        if abs(self.final_score - expected_final) > 0.01:
            raise ValueError(
                f"Invalid final_score: expected {expected_final:.2f} "
                f"(0.4 * {self.parameter_score} + 0.6 * {self.semantic_score}), "
                f"got {self.final_score}"
            )

        return self

    def calculate_final_score(self) -> float:
        """Return 0.4 × parameter_score + 0.6 × semantic_score."""
        return 0.4 * self.parameter_score + 0.6 * self.semantic_score

    def is_above_threshold(self) -> bool:
        """True if final_score >= threshold. Matches below threshold are rejected."""
        return self.final_score >= self.threshold

    @classmethod
    def create(
        cls, parameter_score: float, semantic_score: float, threshold: float = 75.0
    ) -> "MatchScore":
        """Factory: auto-computes final_score. Preferred over direct construction."""
        # Calculate final score using business rule weights
        final = 0.4 * parameter_score + 0.6 * semantic_score

        return cls(
            parameter_score=parameter_score,
            semantic_score=semantic_score,
            final_score=final,
            threshold=threshold,
        )

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for JSON serialization and logging."""
        return {
            "parameter_score": self.parameter_score,
            "semantic_score": self.semantic_score,
            "final_score": self.final_score,
            "threshold": self.threshold,
        }
