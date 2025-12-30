"""
Evaluation framework for HVAC matching quality assessment (Phase 4).

This module provides tools for:
- Golden dataset management (curated test cases)
- Matching quality evaluation (Recall@K, Precision@1, MRR)
- Threshold tuning (precision-recall trade-off optimization)

Architecture Note:
    Located in domain layer (not infrastructure) because:
    - Golden dataset describes business logic (correct HVAC matches)
    - Evaluation metrics are domain concepts (matching quality)
    - No technical infrastructure concerns (ChromaDB, Redis, etc.)
"""

from src.domain.hvac.evaluation.golden_dataset import (
    GoldenPair,
    GoldenDataset,
    ValidationResult,
    load_golden_dataset,
)
from src.domain.hvac.evaluation.evaluation_runner import (
    EvaluationRunner,
    EvaluationReport,
    FailedPair,
)
from src.domain.hvac.evaluation.threshold_tuner import (
    ThresholdTuner,
    ThresholdResult,
    ThresholdTuningReport,
)

__all__ = [
    "GoldenPair",
    "GoldenDataset",
    "ValidationResult",
    "load_golden_dataset",
    "EvaluationRunner",
    "EvaluationReport",
    "FailedPair",
    "ThresholdTuner",
    "ThresholdResult",
    "ThresholdTuningReport",
]
