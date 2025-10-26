"""
MatchScore Value Object

Represents the hybrid scoring result from parameter-based and semantic matching.
This is a core business concept in FastBidder's matching algorithm (40% params + 60% semantic).

Responsibility:
    - Encapsulate hybrid scoring logic (parameter + semantic)
    - Validate score ranges (0-100)
    - Provide threshold comparison
    - Immutable value object

Architecture Notes:
    - Value Object (immutable, defined by values)
    - Uses Pydantic for validation
    - Part of HVAC subdomain
    - No external dependencies
"""

from typing import Any

from pydantic import BaseModel, Field, model_validator


class MatchScore(BaseModel):
    """
    Immutable value object representing a hybrid matching score.

    FastBidder uses a hybrid matching approach:
    - 40% weight: Parameter matching (exact match for DN, PN, material, type)
    - 60% weight: Semantic matching (AI embeddings cosine similarity)

    This ensures that technical parameters (DN50 != DN100) are never confused,
    while semantic matching handles synonyms and variations.

    Attributes:
        parameter_score: Score from parameter matching (0-100)
            - 100 = all extracted parameters match exactly
            - 0 = no parameters match
            - Calculated by comparing DN, PN, material, valve type, etc.

        semantic_score: Score from semantic similarity (0-100)
            - Based on cosine similarity of sentence embeddings
            - Handles synonyms, variations, typos
            - Uses multilingual model (Polish + English)

        final_score: Weighted average (0-100)
            - Calculated as: 0.4 * parameter_score + 0.6 * semantic_score
            - This is the score used for threshold comparison

        threshold: Minimum score for a valid match (default 75.0)
            - Configurable per request
            - Business rule: matches below threshold are rejected

    Examples:
        >>> # Strong match (both high)
        >>> score = MatchScore(
        ...     parameter_score=100.0,
        ...     semantic_score=92.0,
        ...     threshold=75.0
        ... )
        >>> score.final_score
        95.2
        >>> score.is_above_threshold()
        True

        >>> # Weak match (DN mismatch)
        >>> score = MatchScore(
        ...     parameter_score=50.0,  # DN doesn't match
        ...     semantic_score=76.0,
        ...     threshold=75.0
        ... )
        >>> score.final_score
        65.6
        >>> score.is_above_threshold()
        False

    Validation Rules:
        - All scores must be in range [0, 100]
        - Threshold must be in range [0, 100]
        - final_score is auto-calculated (40/60 weights)
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
        """
        Calculate the weighted final score from components.

        Formula: 0.4 * parameter_score + 0.6 * semantic_score

        This method is used for:
        - Validation (ensuring final_score is correct)
        - Factory methods that compute final_score automatically

        Returns:
            Calculated final score (0-100)

        Examples:
            >>> score = MatchScore(
            ...     parameter_score=100.0,
            ...     semantic_score=90.0,
            ...     final_score=94.0,
            ...     threshold=75.0
            ... )
            >>> score.calculate_final_score()
            94.0
        """
        return 0.4 * self.parameter_score + 0.6 * self.semantic_score

    def is_above_threshold(self) -> bool:
        """
        Check if the match score exceeds the minimum threshold.

        Business rule: Only matches with final_score >= threshold are valid.
        Matches below threshold are rejected (user must review manually).

        Returns:
            True if final_score >= threshold, False otherwise

        Examples:
            >>> score = MatchScore(
            ...     parameter_score=100.0,
            ...     semantic_score=92.0,
            ...     final_score=95.2,
            ...     threshold=75.0
            ... )
            >>> score.is_above_threshold()
            True

            >>> weak_score = MatchScore(
            ...     parameter_score=50.0,
            ...     semantic_score=76.0,
            ...     final_score=65.6,
            ...     threshold=75.0
            ... )
            >>> weak_score.is_above_threshold()
            False
        """
        return self.final_score >= self.threshold

    @classmethod
    def create(
        cls, parameter_score: float, semantic_score: float, threshold: float = 75.0
    ) -> "MatchScore":
        """
        Factory method to create MatchScore with auto-calculated final_score.

        This is the preferred way to create MatchScore instances, as it ensures
        final_score is always correctly calculated from the component scores.

        Args:
            parameter_score: Score from parameter matching (0-100)
            semantic_score: Score from semantic similarity (0-100)
            threshold: Minimum score for valid match (default 75.0)

        Returns:
            New MatchScore instance with calculated final_score

        Raises:
            ValidationError: If any score is outside [0, 100] range

        Examples:
            >>> score = MatchScore.create(
            ...     parameter_score=100.0,
            ...     semantic_score=92.0,
            ...     threshold=75.0
            ... )
            >>> score.final_score
            95.2
            >>> score.is_above_threshold()
            True
        """
        # Calculate final score using business rule weights
        final = 0.4 * parameter_score + 0.6 * semantic_score

        return cls(
            parameter_score=parameter_score,
            semantic_score=semantic_score,
            final_score=final,
            threshold=threshold,
        )

    def to_dict(self) -> dict[str, float]:
        """
        Convert to dictionary representation.

        Useful for:
        - JSON serialization
        - Logging and debugging
        - API responses

        Returns:
            Dictionary with all score components

        Examples:
            >>> score = MatchScore.create(100.0, 92.0, 75.0)
            >>> score.to_dict()
            {
                'parameter_score': 100.0,
                'semantic_score': 92.0,
                'final_score': 95.2,
                'threshold': 75.0
            }
        """
        return {
            "parameter_score": self.parameter_score,
            "semantic_score": self.semantic_score,
            "final_score": self.final_score,
            "threshold": self.threshold,
        }
