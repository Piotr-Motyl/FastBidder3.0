"""
MatchResult — immutable value object representing a successful HVAC match.
Contains the matched reference ID, hybrid score, confidence, and breakdown for debugging.
"""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field
from .match_score import MatchScore


class MatchResult(BaseModel):
    """
    Immutable value object returned by MatchingEngine when a match exceeds threshold.

    matched_reference_id: UUID (domain entity) or str ("{file_id}_{row_number}" from ChromaDB).
    breakdown["using_ai"]: True = HybridMatchingEngine with real embeddings; False = SimpleMatchingEngine placeholder.
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
        description="Structured details for debugging. using_ai=True → real embeddings; False → placeholder 50.0.",
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
        """True if confidence >= threshold (default 0.8). High-confidence matches can be auto-approved."""
        return self.confidence >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization, logging, and Celery results."""
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
        """Factory: creates MatchResult with auto-computed MatchScore (preferred over direct construction)."""
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
