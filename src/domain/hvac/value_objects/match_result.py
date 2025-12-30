"""
MatchResult Value Object

Represents the result of matching a working file description against a reference catalog.
Contains the matched item, scores, and justification for transparency and debugging.

Responsibility:
    - Encapsulate matching result (matched item + scores + reasoning)
    - Provide transparency (why this match was chosen)
    - Enable debugging and quality assurance
    - Immutable value object

Architecture Notes:
    - Value Object (immutable, defined by values)
    - Uses Pydantic for validation
    - Part of HVAC subdomain
    - References HVACDescription entity (forward reference for type hints)
"""

from typing import TYPE_CHECKING, Any
from uuid import UUID

from pydantic import BaseModel, Field
from .match_score import MatchScore


class MatchResult(BaseModel):
    """
    Immutable value object representing a successful match between two HVAC descriptions.

    This object is returned by the MatchingEngine after finding a suitable match
    in the reference catalog. It contains:
    - The matched reference item
    - Detailed scoring breakdown
    - Human-readable justification

    Attributes:
        matched_reference_id: ID of the matched reference description (UUID or ChromaDB ID string)
            - UUID: Domain entity ID for HVACDescription (when not from ChromaDB)
            - str: ChromaDB document ID format "{file_id}_{row_number}" (when from vector DB)
            - Used to retrieve full details (price, supplier, etc.) and for evaluation

        score: MatchScore value object with detailed scoring
            - parameter_score: 0-100 from exact parameter matching
            - semantic_score: 0-100 from AI embeddings similarity
            - final_score: Weighted average (40/60)
            - threshold: Minimum required score

        confidence: Confidence level of the match (0-1)
            - 0.0 = very uncertain (multiple similar matches)
            - 1.0 = very confident (clear best match)
            - Calculated based on score gap to second-best match

        message: Human-readable explanation (hybrid justification - Part 1)
            - For UI display to end users
            - Examples: "High confidence match", "Exact parameter match"
            - English language (can be translated in API layer)

        breakdown: Structured score details (hybrid justification - Part 2)
            - For debugging, logging, and analytics
            - Contains parameter match details and score components
            - Format: dict[str, Any] for flexibility

    Examples:
        >>> # Strong match with exact parameters
        >>> result = MatchResult(
        ...     matched_item_id=UUID("..."),
        ...     score=MatchScore.create(100.0, 92.0, 75.0),
        ...     confidence=0.95,
        ...     message="High confidence match - exact DN, PN, and valve type",
        ...     breakdown={
        ...         "parameter_matches": {"DN": True, "PN": True, "type": True},
        ...         "semantic_similarity": 0.92,
        ...         "score_gap_to_second": 15.3
        ...     }
        ... )
        >>> result.score.final_score
        95.2
        >>> result.confidence
        0.95

        >>> # Weaker match (only semantic, no exact params)
        >>> weak_result = MatchResult(
        ...     matched_item_id=UUID("..."),
        ...     score=MatchScore.create(50.0, 88.0, 75.0),
        ...     confidence=0.72,
        ...     message="Moderate confidence - semantic match only",
        ...     breakdown={
        ...         "parameter_matches": {"DN": False, "PN": True, "type": True},
        ...         "semantic_similarity": 0.88,
        ...         "score_gap_to_second": 8.1
        ...     }
        ... )
        >>> weak_result.score.final_score
        72.8

    Business Rules:
        - Only created when match score >= threshold
        - Confidence should correlate with score and uniqueness of match
        - message must be human-readable (for UI)
        - breakdown is flexible dict (for future expansion)
    """

    matched_reference_id: UUID | str = Field(
        description="ID of matched reference: UUID (domain entity) or str (ChromaDB format '{file_id}_{row_number}')"
    )

    score: MatchScore = Field(
        ..., description="Detailed scoring breakdown (param + semantic + final)"
    )

    confidence: float = Field(
        ..., description="Confidence level of the match (0-1)", ge=0.0, le=1.0
    )

    message: str = Field(
        ...,
        description="Human-readable justification for the match",
        min_length=1,
        max_length=500,
    )

    breakdown: dict[str, Any] = Field(
        ...,
        description="""Structured details for debugging and analytics.

        Phase 4 AI Metadata (populated by matching engine):

        When using HybridMatchingEngine (AI-enabled matching):
            - using_ai (bool): True - indicates AI embeddings were used
            - ai_model (str): Name of embedding model used for semantic matching
              (e.g., "paraphrase-multilingual-MiniLM-L12-v2")
              Model name is dynamically retrieved from EmbeddingService
            - stage1_candidates (int): Number of candidates retrieved in Stage 1
              (for debugging and performance analysis)
            - retrieval_top_k (int): Configuration value for top-K retrieval

        When using SimpleMatchingEngine (standalone, no AI):
            - using_ai (bool): False - indicates no AI was used (parameter-only matching)
            - ai_model (None): No AI model used

        Additional Fields (always present):
            - parameter_matches (dict): Details of DN, PN, material, type matching
            - semantic_similarity (float): Cosine similarity score (0-1)
            - score_gap_to_second (float): Score difference to second-best match

        Usage:
            To verify if AI was used in matching, check breakdown["using_ai"].
            This field determines whether score.semantic_score contains real AI data
            or a placeholder value.
        """,
    )

    model_config = {
        "frozen": True,  # Immutable value object
        "json_schema_extra": {
            "examples": [
                {
                    "matched_item_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
                    "score": {
                        "parameter_score": 100.0,
                        "semantic_score": 92.0,
                        "final_score": 95.2,
                        "threshold": 75.0,
                    },
                    "confidence": 0.95,
                    "message": "High confidence match - exact DN, PN, and valve type",
                    "breakdown": {
                        "parameter_matches": {"DN": True, "PN": True, "type": True},
                        "semantic_similarity": 0.92,
                        "score_gap_to_second": 15.3,
                    },
                }
            ]
        },
    }

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """
        Check if the match has high confidence.

        High confidence matches can be auto-approved without manual review.
        Low confidence matches should be flagged for user verification.

        Args:
            threshold: Minimum confidence for "high confidence" (default 0.8)

        Returns:
            True if confidence >= threshold

        Examples:
            >>> result = MatchResult(
            ...     matched_item_id=UUID("..."),
            ...     score=MatchScore.create(100.0, 92.0),
            ...     confidence=0.95,
            ...     message="High confidence match",
            ...     breakdown={}
            ... )
            >>> result.is_high_confidence()
            True
            >>> result.is_high_confidence(threshold=0.98)
            False
        """
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary representation.

        Useful for:
        - JSON serialization for API responses
        - Logging and debugging
        - Celery task results
        - Redis cache storage

        Returns:
            Dictionary with all match result data

        Examples:
            >>> result = MatchResult(
            ...     matched_item_id=UUID("3fa85f64-5717-4562-b3fc-2c963f66afa6"),
            ...     score=MatchScore.create(100.0, 92.0),
            ...     confidence=0.95,
            ...     message="High confidence match",
            ...     breakdown={"param_matches": {"DN": True}}
            ... )
            >>> result.to_dict()
            {
                'matched_item_id': '3fa85f64-5717-4562-b3fc-2c963f66afa6',
                'score': {
                    'parameter_score': 100.0,
                    'semantic_score': 92.0,
                    'final_score': 95.2,
                    'threshold': 75.0
                },
                'confidence': 0.95,
                'message': 'High confidence match',
                'breakdown': {'param_matches': {'DN': True}}
            }
        """
        return {
            "matched_item_id": str(self.matched_reference_id),
            "score": self.score.to_dict(),
            "confidence": self.confidence,
            "message": self.message,
            "breakdown": self.breakdown,
        }

    @classmethod
    def create(
        cls,
        matched_item_id: UUID,
        parameter_score: float,
        semantic_score: float,
        confidence: float,
        message: str,
        breakdown: dict[str, Any],
        threshold: float = 75.0,
    ) -> "MatchResult":
        """
        Factory method to create MatchResult with auto-created MatchScore.

        This is the preferred way to create MatchResult instances, as it handles
        the creation of the nested MatchScore value object.

        Args:
            matched_item_id: UUID of the matched reference description
            parameter_score: Score from parameter matching (0-100)
            semantic_score: Score from semantic similarity (0-100)
            confidence: Confidence level (0-1)
            message: Human-readable justification
            breakdown: Structured debugging details
            threshold: Minimum score for valid match (default 75.0)

        Returns:
            New MatchResult instance

        Raises:
            ValidationError: If any validation fails

        Examples:
            >>> result = MatchResult.create(
            ...     matched_item_id=UUID("..."),
            ...     parameter_score=100.0,
            ...     semantic_score=92.0,
            ...     confidence=0.95,
            ...     message="High confidence match",
            ...     breakdown={"param_matches": {"DN": True, "PN": True}},
            ...     threshold=75.0
            ... )
            >>> result.score.final_score
            95.2
        """
        score = MatchScore.create(
            parameter_score=parameter_score,
            semantic_score=semantic_score,
            threshold=threshold,
        )

        return cls(
            matched_reference_id=matched_item_id,
            score=score,
            confidence=confidence,
            message=message,
            breakdown=breakdown,
        )
