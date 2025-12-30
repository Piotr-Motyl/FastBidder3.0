"""
Threshold Tuner for Matching Quality Optimization (Phase 4).

Tunes matching threshold by testing multiple values on golden dataset.
Finds optimal threshold balancing precision and recall.

Usage:
    >>> from src.domain.hvac.evaluation.threshold_tuner import ThresholdTuner
    >>> from src.domain.hvac.evaluation.evaluation_runner import EvaluationRunner
    >>> from src.domain.hvac.evaluation.golden_dataset import load_golden_dataset
    >>>
    >>> dataset = load_golden_dataset("data/golden_dataset.json")
    >>> tuner = ThresholdTuner(evaluation_runner)
    >>> report = await tuner.tune(dataset, thresholds=[60, 65, 70, 75, 80, 85, 90, 95])
    >>> print(f"Recommended threshold: {report.recommended_threshold}")
    >>> print(report.to_markdown())

CLI Usage:
    python -m src.infrastructure.evaluation.threshold_tuner --dataset path/to/golden.json
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable

from src.domain.hvac.evaluation.golden_dataset import GoldenDataset
from src.domain.hvac.evaluation.evaluation_runner import (
    EvaluationRunner,
    EvaluationReport,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class ThresholdResult:
    """
    Evaluation results for a single threshold value.

    Attributes:
        threshold: Threshold value tested
        precision: Precision (TP / (TP + FP)) - % of matches that are correct
        recall: Recall (TP / (TP + FN)) - % of golden pairs that were matched
        f1_score: F1 Score (harmonic mean of precision and recall)
        true_positives: Number of correct matches
        false_positives: Number of incorrect matches
        false_negatives: Number of pairs that should match but didn't

    Metrics Explanation:
        - Precision: Of all matches made, what % are correct?
        - Recall: Of all correct matches possible, what % did we find?
        - F1: Balanced metric combining precision and recall
        - TP: Correct match found
        - FP: Wrong match made
        - FN: Should match but no match found or wrong match

    Examples:
        >>> result = ThresholdResult(
        ...     threshold=75.0,
        ...     precision=0.85,
        ...     recall=0.90,
        ...     f1_score=0.874,
        ...     true_positives=45,
        ...     false_positives=8,
        ...     false_negatives=5,
        ... )
        >>> print(f"At threshold {result.threshold}: P={result.precision:.2%}, R={result.recall:.2%}")
        At threshold 75.0: P=85.00%, R=90.00%
    """

    threshold: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass
class ThresholdTuningReport:
    """
    Complete threshold tuning report.

    Attributes:
        results: List of results for each threshold tested
        recommended_threshold: Recommended threshold (max precision where recall >= min_recall)
        recommendation_reason: Explanation of why this threshold was recommended
        best_precision_threshold: Threshold with highest precision
        best_recall_threshold: Threshold with highest recall
        best_f1_threshold: Threshold with highest F1 score
        min_recall_constraint: Minimum recall required for recommendation

    Examples:
        >>> report = ThresholdTuningReport(...)
        >>> print(f"Recommended: {report.recommended_threshold}")
        >>> print(report.to_markdown())
    """

    results: list[ThresholdResult] = field(default_factory=list)
    recommended_threshold: float | None = None
    recommendation_reason: str = ""
    best_precision_threshold: float | None = None
    best_recall_threshold: float | None = None
    best_f1_threshold: float | None = None
    min_recall_constraint: float = 0.7

    def to_dict(self) -> dict:
        """
        Convert to dictionary for JSON serialization.

        Returns:
            Dict with report data

        Examples:
            >>> report_dict = report.to_dict()
            >>> with open("tuning_report.json", "w") as f:
            ...     json.dump(report_dict, f, indent=2)
        """
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        """
        Save report to JSON file.

        Args:
            path: File path to save to

        Examples:
            >>> report.save_json("threshold_tuning_report.json")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Saved threshold tuning report to {path}")

    def to_markdown(self) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown-formatted tuning report

        Examples:
            >>> print(report.to_markdown())
            # Threshold Tuning Report
            ...
        """
        lines = []
        lines.append("# Threshold Tuning Report")
        lines.append("")

        # Recommendation
        if self.recommended_threshold is not None:
            lines.append("## Recommended Threshold")
            lines.append("")
            lines.append(f"**Threshold**: {self.recommended_threshold}")
            lines.append(f"**Reason**: {self.recommendation_reason}")
            lines.append("")

            # Find recommended result
            recommended_result = next(
                (r for r in self.results if r.threshold == self.recommended_threshold),
                None,
            )
            if recommended_result:
                lines.append(f"- **Precision**: {recommended_result.precision:.2%}")
                lines.append(f"- **Recall**: {recommended_result.recall:.2%}")
                lines.append(f"- **F1 Score**: {recommended_result.f1_score:.4f}")
                lines.append("")
        else:
            lines.append("## ⚠️ No Recommended Threshold")
            lines.append("")
            lines.append(f"**Reason**: {self.recommendation_reason}")
            lines.append("")

        # Best thresholds
        lines.append("## Best Thresholds by Metric")
        lines.append("")
        if self.best_precision_threshold is not None:
            lines.append(f"- **Best Precision**: {self.best_precision_threshold}")
        if self.best_recall_threshold is not None:
            lines.append(f"- **Best Recall**: {self.best_recall_threshold}")
        if self.best_f1_threshold is not None:
            lines.append(f"- **Best F1 Score**: {self.best_f1_threshold}")
        lines.append("")

        # All results table
        lines.append("## All Threshold Results")
        lines.append("")
        lines.append("| Threshold | Precision | Recall | F1 Score | TP | FP | FN |")
        lines.append("|-----------|-----------|--------|----------|----|----|-----|")

        for result in sorted(self.results, key=lambda r: r.threshold):
            marker = " ⭐" if result.threshold == self.recommended_threshold else ""
            lines.append(
                f"| {result.threshold}{marker} | "
                f"{result.precision:.2%} | "
                f"{result.recall:.2%} | "
                f"{result.f1_score:.4f} | "
                f"{result.true_positives} | "
                f"{result.false_positives} | "
                f"{result.false_negatives} |"
            )

        lines.append("")

        # Trade-off analysis
        lines.append("## Precision-Recall Trade-off")
        lines.append("")
        lines.append("As threshold increases:")
        lines.append("- **Precision** tends to increase (fewer false positives)")
        lines.append("- **Recall** tends to decrease (more false negatives)")
        lines.append("")
        lines.append(
            f"**Constraint**: Recall must be >= {self.min_recall_constraint:.0%} "
            f"for recommendation"
        )
        lines.append("")

        return "\n".join(lines)


# ============================================================================
# THRESHOLD TUNER
# ============================================================================


class ThresholdTuner:
    """
    Service for tuning matching threshold on golden dataset.

    Tests multiple threshold values and finds optimal balance between
    precision and recall.

    Strategy:
        - Test each threshold on golden dataset using EvaluationRunner
        - Calculate precision, recall, F1 for each threshold
        - Recommend threshold with max precision where recall >= min_recall

    Examples:
        >>> tuner = ThresholdTuner(evaluation_runner)
        >>> report = await tuner.tune(
        ...     golden_dataset=dataset,
        ...     thresholds=[60, 65, 70, 75, 80, 85, 90, 95],
        ...     min_recall=0.7,
        ... )
        >>> print(f"Recommended: {report.recommended_threshold}")
        >>> with open("report.md", "w") as f:
        ...     f.write(report.to_markdown())
    """

    def __init__(self, evaluation_runner: EvaluationRunner):
        """
        Initialize threshold tuner.

        Args:
            evaluation_runner: EvaluationRunner for testing thresholds
        """
        self.evaluation_runner = evaluation_runner

    async def tune(
        self,
        golden_dataset: GoldenDataset,
        thresholds: list[float] | None = None,
        min_recall: float = 0.7,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> ThresholdTuningReport:
        """
        Tune threshold on golden dataset.

        Args:
            golden_dataset: Dataset to evaluate on
            thresholds: List of threshold values to test (default: [60,65,70,75,80,85,90,95])
            min_recall: Minimum recall required for recommendation (default: 0.7)
            progress_callback: Optional callback(current, total, threshold) for progress

        Returns:
            ThresholdTuningReport with results and recommendation

        Examples:
            >>> report = await tuner.tune(dataset)
            >>> print(f"Tested {len(report.results)} thresholds")
        """
        if thresholds is None:
            thresholds = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0]

        logger.info(
            f"Starting threshold tuning with {len(thresholds)} thresholds "
            f"(min_recall={min_recall:.0%})"
        )

        results: list[ThresholdResult] = []

        # Test each threshold
        for idx, threshold in enumerate(thresholds):
            if progress_callback:
                progress_callback(idx + 1, len(thresholds), threshold)

            logger.info(f"Testing threshold {threshold} ({idx+1}/{len(thresholds)})")

            # Run evaluation at this threshold
            eval_report = await self.evaluation_runner.evaluate(
                golden_dataset=golden_dataset,
                threshold=threshold,
                top_k_values=[1],  # Only need top-1 for precision/recall
            )

            # Calculate threshold metrics
            threshold_result = self._calculate_threshold_metrics(
                threshold=threshold,
                eval_report=eval_report,
                total_pairs=golden_dataset.total_pairs,
            )

            results.append(threshold_result)

            logger.info(
                f"Threshold {threshold}: P={threshold_result.precision:.2%}, "
                f"R={threshold_result.recall:.2%}, F1={threshold_result.f1_score:.4f}"
            )

        # Find best thresholds by different metrics
        best_precision_threshold = max(results, key=lambda r: r.precision).threshold
        best_recall_threshold = max(results, key=lambda r: r.recall).threshold
        best_f1_threshold = max(results, key=lambda r: r.f1_score).threshold

        # Find recommended threshold
        recommended_threshold, recommendation_reason = self._find_recommended_threshold(
            results=results,
            min_recall=min_recall,
        )

        report = ThresholdTuningReport(
            results=results,
            recommended_threshold=recommended_threshold,
            recommendation_reason=recommendation_reason,
            best_precision_threshold=best_precision_threshold,
            best_recall_threshold=best_recall_threshold,
            best_f1_threshold=best_f1_threshold,
            min_recall_constraint=min_recall,
        )

        logger.info(
            f"Threshold tuning complete: Recommended threshold = {recommended_threshold}"
        )

        return report

    def _calculate_threshold_metrics(
        self,
        threshold: float,
        eval_report: EvaluationReport,
        total_pairs: int,
    ) -> ThresholdResult:
        """
        Calculate precision, recall, F1 from evaluation report.

        Args:
            threshold: Threshold value
            eval_report: Evaluation report from EvaluationRunner
            total_pairs: Total golden pairs

        Returns:
            ThresholdResult with calculated metrics

        Metrics calculation:
            - Precision@1 from eval_report is our precision (% of matches that are correct)
            - Recall@1 from eval_report is our recall (% of pairs that were matched correctly)
            - TP = correct matches = recall * total_pairs
            - FP = wrong matches (from failed_pairs with reason="wrong_match")
            - FN = pairs that should match but didn't = (1 - recall) * total_pairs
        """
        # Precision@1: % of matches that are correct
        precision = eval_report.precision_at_1

        # Recall@1: % of golden pairs that were correctly matched
        recall = eval_report.recall_at_k.get(1, 0.0)

        # Calculate F1 score
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0.0

        # Calculate TP, FP, FN
        # TP = pairs correctly matched = recall * total_pairs
        true_positives = int(recall * total_pairs)

        # FN = golden pairs not correctly matched = total_pairs - TP
        false_negatives = total_pairs - true_positives

        # FP = wrong matches (matches made but incorrect)
        # Count failed_pairs with reason="wrong_match"
        false_positives = sum(
            1 for fp in eval_report.failed_pairs if fp.reason == "wrong_match"
        )

        return ThresholdResult(
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )

    def _find_recommended_threshold(
        self,
        results: list[ThresholdResult],
        min_recall: float,
    ) -> tuple[float | None, str]:
        """
        Find recommended threshold.

        Strategy: Select threshold with maximum precision where recall >= min_recall

        Args:
            results: List of threshold results
            min_recall: Minimum recall constraint

        Returns:
            Tuple of (recommended_threshold, reason)
        """
        # Filter results with recall >= min_recall
        valid_results = [r for r in results if r.recall >= min_recall]

        if not valid_results:
            return (
                None,
                f"No threshold achieves minimum recall of {min_recall:.0%}. "
                f"Best recall: {max(results, key=lambda r: r.recall).recall:.2%} "
                f"at threshold {max(results, key=lambda r: r.recall).threshold}",
            )

        # Find threshold with max precision among valid results
        best_result = max(valid_results, key=lambda r: r.precision)

        reason = (
            f"Maximum precision ({best_result.precision:.2%}) "
            f"with recall >= {min_recall:.0%} ({best_result.recall:.2%})"
        )

        return (best_result.threshold, reason)


# ============================================================================
# CLI TOOL
# ============================================================================


if __name__ == "__main__":
    """
    CLI tool for threshold tuning.

    Usage:
        python -m src.infrastructure.evaluation.threshold_tuner --dataset path/to/golden.json
        python -m src.infrastructure.evaluation.threshold_tuner --dataset path/to/golden.json --min-recall 0.8
        python -m src.infrastructure.evaluation.threshold_tuner --dataset path/to/golden.json --output report.json
    """
    import sys
    import argparse
    import asyncio

    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Tune matching threshold on golden dataset")
    parser.add_argument("--dataset", required=True, help="Path to golden dataset JSON file")
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.7,
        help="Minimum recall constraint (default: 0.7)",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
        help="Thresholds to test (default: 60 65 70 75 80 85 90 95)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file path (optional)",
    )

    args = parser.parse_args()

    async def run_tuning():
        """Run threshold tuning."""
        try:
            # Load golden dataset
            from src.domain.hvac.evaluation.golden_dataset import load_golden_dataset

            dataset = load_golden_dataset(args.dataset)
            logger.info(f"Loaded golden dataset: {dataset.total_pairs} pairs")

            # Initialize components
            from src.infrastructure.ai.embeddings.embedding_service import (
                EmbeddingService,
            )
            from src.infrastructure.ai.vector_store.chroma_client import ChromaClient
            from src.infrastructure.ai.retrieval.semantic_retriever import (
                SemanticRetriever,
            )
            from src.infrastructure.matching.hybrid_matching_engine import (
                HybridMatchingEngine,
            )
            from src.domain.hvac.services.concrete_parameter_extractor import (
                ConcreteParameterExtractor,
            )
            from src.domain.hvac.services.simple_matching_engine import (
                SimpleMatchingEngine,
            )
            from src.domain.hvac.matching_config import MatchingConfig

            logger.info("Initializing AI components...")

            embedding_service = EmbeddingService()
            chroma_client = ChromaClient()
            semantic_retriever = SemanticRetriever(embedding_service, chroma_client)
            parameter_extractor = ConcreteParameterExtractor()
            config = MatchingConfig.default()
            simple_engine = SimpleMatchingEngine(
                parameter_extractor, config, embedding_service
            )
            matching_engine = HybridMatchingEngine(
                semantic_retriever=semantic_retriever,
                simple_matching_engine=simple_engine,
                config=config,
            )

            evaluation_runner = EvaluationRunner(semantic_retriever, matching_engine)
            tuner = ThresholdTuner(evaluation_runner)

            # Progress callback
            def progress(current, total, threshold):
                print(f"Progress: {current}/{total} - Testing threshold {threshold}")

            # Run tuning
            logger.info("Starting threshold tuning...")
            report = await tuner.tune(
                golden_dataset=dataset,
                thresholds=args.thresholds,
                min_recall=args.min_recall,
                progress_callback=progress,
            )

            # Print markdown report
            print("\n" + "=" * 60)
            print(report.to_markdown())
            print("=" * 60 + "\n")

            # Save JSON if requested
            if args.output:
                report.save_json(args.output)
                print(f"Saved JSON report to: {args.output}")

            # Exit code: 0 if recommendation found, 1 if not
            sys.exit(0 if report.recommended_threshold is not None else 1)

        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    # Run async tuning
    asyncio.run(run_tuning())
