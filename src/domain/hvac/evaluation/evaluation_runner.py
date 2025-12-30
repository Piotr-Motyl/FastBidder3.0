"""
Evaluation Runner for Matching Quality Metrics (Phase 4).

Runs matching engine on golden dataset and calculates quality metrics:
- Recall@K: Percentage of correct references found in top-K results
- Precision@1: Percentage where top-1 match is correct
- MRR (Mean Reciprocal Rank): Average inverse rank of correct match

Usage:
    >>> from src.domain.hvac.evaluation.golden_dataset import load_golden_dataset
    >>> from src.infrastructure.matching.hybrid_matching_engine import HybridMatchingEngine
    >>> from src.infrastructure.ai.retrieval.semantic_retriever import SemanticRetriever
    >>>
    >>> dataset = load_golden_dataset("data/golden_dataset.json")
    >>> runner = EvaluationRunner(semantic_retriever, matching_engine)
    >>> report = await runner.evaluate(dataset, threshold=75.0, top_k_values=[1, 3, 5])
    >>> print(report.to_markdown())
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Optional, Any

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.evaluation.golden_dataset import GoldenDataset, GoldenPair

logger = logging.getLogger(__name__)


# ============================================================================
# PROTOCOLS
# ============================================================================


class SemanticRetrieverProtocol(Protocol):
    """Protocol for semantic retriever (Stage 1 of matching pipeline)."""

    async def search_similar(
        self,
        query_text: str,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Search for similar documents in vector database.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            metadata_filters: Optional filters (e.g., {"dn": "50", "pn": "16"})

        Returns:
            List of RetrievalResult objects
        """
        ...


class MatchingEngineProtocol(Protocol):
    """Protocol for matching engine (full pipeline or simple)."""

    async def match(
        self,
        working_description: HVACDescription,
        reference_descriptions: list[HVACDescription],
        threshold: float = 75.0,
    ) -> Optional[Any]:
        """
        Find best match for working description.

        Args:
            working_description: Description to match
            reference_descriptions: Candidate descriptions (can be empty for hybrid)
            threshold: Minimum score threshold

        Returns:
            MatchResult or None if no match above threshold
        """
        ...


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class FailedPair:
    """
    Golden pair that failed to match correctly.

    Attributes:
        pair: Original golden pair
        predicted_reference_id: ID that was matched (None if no match)
        predicted_score: Score of the match (None if no match)
        actual_rank: Rank of correct reference in top-K (None if not in top-K)
        reason: Why it failed ("no_match", "wrong_match", "not_in_top_k", "error")
        error_message: Error message if reason is "error"
    """

    pair: GoldenPair
    predicted_reference_id: str | None = None
    predicted_score: float | None = None
    actual_rank: int | None = None
    reason: str = "no_match"
    error_message: str = ""


@dataclass
class EvaluationReport:
    """
    Evaluation report with quality metrics.

    Attributes:
        recall_at_k: Recall at different K values, e.g., {1: 0.80, 3: 0.92, 5: 0.96}
        precision_at_1: Precision at rank 1 (% where top-1 match is correct)
        mrr: Mean Reciprocal Rank (average of 1/rank for correct matches)
        metrics_by_difficulty: Metrics broken down by difficulty level
        failed_pairs: Pairs that didn't match correctly
        total_pairs: Total pairs evaluated
        threshold: Matching threshold used
        top_k_values: K values used for Recall@K
        average_match_time: Average time per match in seconds
    """

    recall_at_k: dict[int, float]
    precision_at_1: float
    mrr: float
    metrics_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    failed_pairs: list[FailedPair] = field(default_factory=list)
    total_pairs: int = 0
    threshold: float = 75.0
    top_k_values: list[int] = field(default_factory=list)
    average_match_time: float = 0.0

    def to_markdown(self) -> str:
        """
        Generate markdown report.

        Returns:
            Markdown-formatted evaluation report

        Examples:
            >>> report = EvaluationReport(...)
            >>> print(report.to_markdown())
            # Evaluation Report
            ...
        """
        lines = []
        lines.append("# Evaluation Report")
        lines.append("")
        lines.append(f"**Total Pairs**: {self.total_pairs}")
        lines.append(f"**Threshold**: {self.threshold}")
        lines.append(f"**Average Match Time**: {self.average_match_time:.3f}s")
        lines.append("")

        # Overall Metrics
        lines.append("## Overall Metrics")
        lines.append("")
        lines.append(f"- **Precision@1**: {self.precision_at_1:.2%}")
        lines.append(f"- **MRR (Mean Reciprocal Rank)**: {self.mrr:.4f}")
        lines.append("")

        # Recall@K
        lines.append("### Recall@K")
        lines.append("")
        for k in sorted(self.recall_at_k.keys()):
            recall = self.recall_at_k[k]
            lines.append(f"- **Recall@{k}**: {recall:.2%}")
        lines.append("")

        # Metrics by Difficulty
        if self.metrics_by_difficulty:
            lines.append("## Metrics by Difficulty")
            lines.append("")
            for difficulty in ["easy", "medium", "hard"]:
                if difficulty in self.metrics_by_difficulty:
                    metrics = self.metrics_by_difficulty[difficulty]
                    lines.append(f"### {difficulty.capitalize()}")
                    lines.append("")
                    lines.append(f"- **Precision@1**: {metrics.get('precision_at_1', 0.0):.2%}")
                    lines.append(f"- **MRR**: {metrics.get('mrr', 0.0):.4f}")
                    for k in sorted(self.top_k_values):
                        recall_key = f"recall_at_{k}"
                        if recall_key in metrics:
                            lines.append(f"- **Recall@{k}**: {metrics[recall_key]:.2%}")
                    lines.append("")

        # Failed Pairs
        if self.failed_pairs:
            lines.append(f"## Failed Pairs ({len(self.failed_pairs)})")
            lines.append("")
            lines.append("| Working Text | Correct Reference | Reason | Predicted ID | Score |")
            lines.append("|--------------|-------------------|--------|--------------|-------|")

            for failed in self.failed_pairs[:20]:  # Limit to first 20
                working = failed.pair.working_text[:50] + "..." if len(failed.pair.working_text) > 50 else failed.pair.working_text
                correct = failed.pair.correct_reference_text[:40] + "..." if len(failed.pair.correct_reference_text) > 40 else failed.pair.correct_reference_text
                reason = failed.reason
                pred_id = failed.predicted_reference_id[:20] + "..." if failed.predicted_reference_id and len(failed.predicted_reference_id) > 20 else (failed.predicted_reference_id or "N/A")
                score = f"{failed.predicted_score:.2f}" if failed.predicted_score is not None else "N/A"
                lines.append(f"| {working} | {correct} | {reason} | {pred_id} | {score} |")

            if len(self.failed_pairs) > 20:
                lines.append("")
                lines.append(f"*... and {len(self.failed_pairs) - 20} more*")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# EVALUATION RUNNER
# ============================================================================


class EvaluationRunner:
    """
    Service for running evaluation on golden dataset.

    Evaluates matching quality by running matching engine on golden dataset
    and calculating metrics (Recall@K, Precision@1, MRR).

    Examples:
        >>> runner = EvaluationRunner(semantic_retriever, matching_engine)
        >>> report = await runner.evaluate(
        ...     golden_dataset=dataset,
        ...     threshold=75.0,
        ...     top_k_values=[1, 3, 5, 10],
        ...     progress_callback=lambda current, total: print(f"{current}/{total}")
        ... )
        >>> print(f"Precision@1: {report.precision_at_1:.2%}")
        >>> print(f"Recall@5: {report.recall_at_k[5]:.2%}")
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetrieverProtocol,
        matching_engine: MatchingEngineProtocol,
    ):
        """
        Initialize evaluation runner.

        Args:
            semantic_retriever: Retriever for top-K candidate search (Stage 1)
            matching_engine: Full matching engine for final matching (Stage 1+2)
        """
        self.semantic_retriever = semantic_retriever
        self.matching_engine = matching_engine

    async def evaluate(
        self,
        golden_dataset: GoldenDataset,
        threshold: float = 75.0,
        top_k_values: list[int] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> EvaluationReport:
        """
        Evaluate matching quality on golden dataset.

        Args:
            golden_dataset: Dataset of golden pairs to evaluate
            threshold: Minimum matching score threshold
            top_k_values: K values for Recall@K (default: [1, 3, 5, 10])
            progress_callback: Optional callback(current, total) for progress tracking

        Returns:
            EvaluationReport with metrics and failed pairs

        Examples:
            >>> report = await runner.evaluate(dataset, threshold=75.0)
            >>> print(report.to_markdown())
        """
        if top_k_values is None:
            top_k_values = [1, 3, 5, 10]

        logger.info(
            f"Starting evaluation on {golden_dataset.total_pairs} pairs "
            f"(threshold={threshold}, top_k={max(top_k_values)})"
        )

        # Storage for results
        retrieval_results: list[dict[str, Any]] = []
        matching_results: list[dict[str, Any]] = []
        failed_pairs: list[FailedPair] = []
        total_match_time = 0.0

        # Evaluate each pair
        for idx, pair in enumerate(golden_dataset.pairs):
            # Progress callback
            if progress_callback:
                progress_callback(idx + 1, golden_dataset.total_pairs)

            # Create working description from golden pair
            working_desc = self._create_working_description(pair)

            # Stage 1: Retrieval (for Recall@K)
            try:
                top_k_candidates = self.semantic_retriever.retrieve(
                    query_text=pair.working_text,
                    top_k=max(top_k_values),
                    filters=None,
                )

                # Extract IDs from candidates (ChromaDB format: {file_id}_{row_number})
                candidate_ids = [c.description_id for c in top_k_candidates]

                # Find rank of correct reference
                correct_rank = None
                if pair.correct_reference_id in candidate_ids:
                    correct_rank = candidate_ids.index(pair.correct_reference_id) + 1  # 1-indexed

                retrieval_results.append(
                    {
                        "pair": pair,
                        "candidate_ids": candidate_ids,
                        "correct_rank": correct_rank,
                    }
                )

            except Exception as e:
                logger.error(f"Retrieval error for pair {idx}: {e}")
                retrieval_results.append(
                    {
                        "pair": pair,
                        "candidate_ids": [],
                        "correct_rank": None,
                        "error": str(e),
                    }
                )

            # Stage 2: Full matching (for Precision@1)
            matching_start = time.time()
            try:
                match_result = await self.matching_engine.match(
                    working_description=working_desc,
                    reference_descriptions=[],  # Hybrid engine doesn't need this
                    threshold=threshold,
                )

                match_time = time.time() - matching_start
                total_match_time += match_time

                # Check if match is correct
                is_correct = False
                predicted_id = None
                predicted_score = None

                if match_result is not None:
                    predicted_id = str(match_result.matched_reference_id)
                    predicted_score = match_result.score.final_score
                    is_correct = predicted_id == pair.correct_reference_id

                matching_results.append(
                    {
                        "pair": pair,
                        "predicted_id": predicted_id,
                        "predicted_score": predicted_score,
                        "is_correct": is_correct,
                    }
                )

                # Track failed pair
                if not is_correct:
                    # Determine reason
                    reason = "no_match" if match_result is None else "wrong_match"

                    # Get actual rank from retrieval results
                    actual_rank = retrieval_results[-1].get("correct_rank")

                    failed_pairs.append(
                        FailedPair(
                            pair=pair,
                            predicted_reference_id=predicted_id,
                            predicted_score=predicted_score,
                            actual_rank=actual_rank,
                            reason=reason,
                        )
                    )

            except Exception as e:
                logger.error(f"Matching error for pair {idx}: {e}")
                matching_results.append(
                    {
                        "pair": pair,
                        "predicted_id": None,
                        "predicted_score": None,
                        "is_correct": False,
                        "error": str(e),
                    }
                )
                failed_pairs.append(
                    FailedPair(
                        pair=pair,
                        reason="error",
                        error_message=str(e),
                    )
                )

        # Calculate metrics
        recall_at_k = self._calculate_recall_at_k(retrieval_results, top_k_values)
        precision_at_1 = self._calculate_precision_at_1(matching_results)
        mrr = self._calculate_mrr(retrieval_results)
        metrics_by_difficulty = self._calculate_metrics_by_difficulty(
            retrieval_results, matching_results, top_k_values
        )
        average_match_time = total_match_time / len(golden_dataset.pairs) if golden_dataset.pairs else 0.0

        report = EvaluationReport(
            recall_at_k=recall_at_k,
            precision_at_1=precision_at_1,
            mrr=mrr,
            metrics_by_difficulty=metrics_by_difficulty,
            failed_pairs=failed_pairs,
            total_pairs=golden_dataset.total_pairs,
            threshold=threshold,
            top_k_values=top_k_values,
            average_match_time=average_match_time,
        )

        logger.info(
            f"Evaluation complete: Precision@1={precision_at_1:.2%}, "
            f"Recall@{max(top_k_values)}={recall_at_k[max(top_k_values)]:.2%}, "
            f"MRR={mrr:.4f}"
        )

        return report

    def _create_working_description(self, pair: GoldenPair) -> HVACDescription:
        """
        Create HVACDescription from GoldenPair working text.

        Args:
            pair: Golden pair

        Returns:
            HVACDescription instance
        """
        # Create description with minimal required fields
        from uuid import uuid4

        desc = HVACDescription(
            raw_text=pair.working_text,
            source_row_number=0,
            file_id=uuid4(),
        )
        return desc

    def _calculate_recall_at_k(
        self, retrieval_results: list[dict[str, Any]], top_k_values: list[int]
    ) -> dict[int, float]:
        """
        Calculate Recall@K for different K values.

        Recall@K = % of cases where correct reference is in top-K results

        Args:
            retrieval_results: List of retrieval result dicts
            top_k_values: K values to calculate recall for

        Returns:
            Dict mapping K -> Recall@K
        """
        recall_at_k = {}

        for k in top_k_values:
            correct_in_top_k = 0

            for result in retrieval_results:
                correct_rank = result.get("correct_rank")
                if correct_rank is not None and correct_rank <= k:
                    correct_in_top_k += 1

            recall = correct_in_top_k / len(retrieval_results) if retrieval_results else 0.0
            recall_at_k[k] = recall

        return recall_at_k

    def _calculate_precision_at_1(self, matching_results: list[dict[str, Any]]) -> float:
        """
        Calculate Precision@1 (top-1 accuracy).

        Precision@1 = % of cases where top-1 match is correct

        Args:
            matching_results: List of matching result dicts

        Returns:
            Precision@1 value
        """
        if not matching_results:
            return 0.0

        correct_count = sum(1 for r in matching_results if r.get("is_correct", False))
        return correct_count / len(matching_results)

    def _calculate_mrr(self, retrieval_results: list[dict[str, Any]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = average of (1/rank) for correct matches
        If correct reference not found, contributes 0.

        Args:
            retrieval_results: List of retrieval result dicts

        Returns:
            MRR value
        """
        if not retrieval_results:
            return 0.0

        reciprocal_ranks = []

        for result in retrieval_results:
            correct_rank = result.get("correct_rank")
            if correct_rank is not None:
                reciprocal_ranks.append(1.0 / correct_rank)
            else:
                reciprocal_ranks.append(0.0)

        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def _calculate_metrics_by_difficulty(
        self,
        retrieval_results: list[dict[str, Any]],
        matching_results: list[dict[str, Any]],
        top_k_values: list[int],
    ) -> dict[str, dict[str, float]]:
        """
        Calculate metrics broken down by difficulty level.

        Args:
            retrieval_results: List of retrieval result dicts
            matching_results: List of matching result dicts
            top_k_values: K values for Recall@K

        Returns:
            Dict mapping difficulty -> metrics dict
        """
        metrics_by_difficulty = {}

        for difficulty in ["easy", "medium", "hard"]:
            # Filter results by difficulty
            retrieval_filtered = [
                r for r in retrieval_results if r["pair"].difficulty == difficulty
            ]
            matching_filtered = [
                r for r in matching_results if r["pair"].difficulty == difficulty
            ]

            if not retrieval_filtered:
                continue

            # Calculate metrics
            recall_at_k = self._calculate_recall_at_k(retrieval_filtered, top_k_values)
            precision_at_1 = self._calculate_precision_at_1(matching_filtered)
            mrr = self._calculate_mrr(retrieval_filtered)

            # Build metrics dict
            metrics = {
                "precision_at_1": precision_at_1,
                "mrr": mrr,
            }
            for k, recall in recall_at_k.items():
                metrics[f"recall_at_{k}"] = recall

            metrics_by_difficulty[difficulty] = metrics

        return metrics_by_difficulty
