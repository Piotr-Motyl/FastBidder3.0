#!/usr/bin/env python3
"""
CLI tool for tuning matching threshold using golden dataset.

This script finds the optimal matching threshold by testing multiple values
on a golden dataset and balancing precision vs recall.

Usage:
    python scripts/tune_threshold.py --dataset data/golden_dataset.json
    python scripts/tune_threshold.py --dataset data/golden.json --thresholds 60,70,80,90
    python scripts/tune_threshold.py --dataset data/golden.json --min-recall 0.8 --output tuning.md

Requirements:
    - Golden dataset JSON file with test cases
    - HybridMatchingEngine or SimpleMatchingEngine
    - ChromaDB indexed with reference descriptions (for AI matching)

Output:
    - Markdown or JSON report with threshold comparison
    - Recommended optimal threshold
    - Precision/Recall trade-off analysis
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.domain.hvac.evaluation.golden_dataset import load_golden_dataset
from src.domain.hvac.evaluation.evaluation_runner import EvaluationRunner
from src.domain.hvac.evaluation.threshold_tuner import ThresholdTuner
from src.infrastructure.ai.embeddings.embedding_service import EmbeddingService
from src.infrastructure.ai.vector_store.chroma_client import ChromaClientSingleton
from src.infrastructure.ai.retrieval.semantic_retriever import SemanticRetriever
from src.infrastructure.matching.hybrid_matching_engine import HybridMatchingEngine
from src.domain.hvac.services.simple_matching_engine import SimpleMatchingEngine
from src.domain.hvac.services.concrete_parameter_extractor import (
    ConcreteParameterExtractor,
)
from src.domain.hvac.matching_config import MatchingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tune matching threshold using golden dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune with default thresholds (60,65,70,75,80,85,90,95)
  python scripts/tune_threshold.py --dataset data/golden_dataset.json

  # Tune with custom threshold range
  python scripts/tune_threshold.py --dataset data/golden.json --thresholds 50,60,70,80,90

  # Tune with stricter recall constraint
  python scripts/tune_threshold.py --dataset data/golden.json --min-recall 0.8

  # Save report to file
  python scripts/tune_threshold.py --dataset data/golden.json --output tuning_report.md

  # Tune without AI (SimpleMatchingEngine only)
  python scripts/tune_threshold.py --dataset data/golden.json --no-ai

  # Output as JSON for further processing
  python scripts/tune_threshold.py --dataset data/golden.json --format json --output report.json
        """,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to golden dataset JSON file",
    )

    parser.add_argument(
        "--thresholds",
        type=str,
        default="60,65,70,75,80,85,90,95",
        help="Comma-separated threshold values to test (default: 60,65,70,75,80,85,90,95)",
    )

    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.7,
        help="Minimum recall required for recommendation (default: 0.7)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for report (default: stdout)",
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable AI matching (use SimpleMatchingEngine only)",
    )

    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=50,
        help="Number of candidates for Stage 1 retrieval (default: 50)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


async def main():
    """Main execution function."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Step 1: Load golden dataset
    logger.info(f"Loading golden dataset from {args.dataset}")
    try:
        golden_dataset = load_golden_dataset(args.dataset)
        logger.info(
            f"Loaded {golden_dataset.total_pairs} golden pairs "
            f"(version: {golden_dataset.version})"
        )
    except FileNotFoundError:
        logger.error(f"Golden dataset not found: {args.dataset}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load golden dataset: {e}")
        sys.exit(1)

    # Step 2: Parse threshold values
    try:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        logger.info(f"Testing {len(thresholds)} threshold values: {thresholds}")
    except ValueError:
        logger.error(
            f"Invalid --thresholds format: {args.thresholds}. Expected comma-separated floats."
        )
        sys.exit(1)

    # Validate min-recall
    if not (0.0 < args.min_recall <= 1.0):
        logger.error(
            f"Invalid --min-recall: {args.min_recall}. Must be between 0.0 and 1.0"
        )
        sys.exit(1)

    # Step 3: Initialize matching engine
    logger.info("Initializing matching engine...")
    config = MatchingConfig(
        default_threshold=75.0,  # Will be overridden during tuning
        retrieval_top_k=args.retrieval_top_k,
    )

    parameter_extractor = ConcreteParameterExtractor()

    semantic_retriever = None  # Initialize to None for SimpleMatchingEngine

    if args.no_ai:
        # Use SimpleMatchingEngine only
        logger.info("Using SimpleMatchingEngine (AI disabled)")
        matching_engine = SimpleMatchingEngine(
            parameter_extractor=parameter_extractor,
            config=config,
        )
    else:
        # Use HybridMatchingEngine with AI
        try:
            logger.info("Initializing AI components (this may take 10-30 seconds)...")
            embedding_service = EmbeddingService()
            chroma_client = ChromaClientSingleton.get_instance()
            semantic_retriever = SemanticRetriever(
                embedding_service=embedding_service,
                chroma_client=chroma_client,
            )

            simple_engine = SimpleMatchingEngine(
                parameter_extractor=parameter_extractor,
                config=config,
                embedding_service=embedding_service,
            )

            matching_engine = HybridMatchingEngine(
                semantic_retriever=semantic_retriever,
                simple_matching_engine=simple_engine,
                config=config,
            )
            logger.info("AI components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI components: {e}")
            logger.info("Falling back to SimpleMatchingEngine")
            matching_engine = SimpleMatchingEngine(
                parameter_extractor=parameter_extractor,
                config=config,
            )

    # Step 4: Initialize EvaluationRunner and ThresholdTuner
    evaluation_runner = EvaluationRunner(
        matching_engine=matching_engine,
        semantic_retriever=semantic_retriever,
    )

    threshold_tuner = ThresholdTuner(evaluation_runner)

    # Step 5: Run threshold tuning
    logger.info(
        f"Starting threshold tuning with min_recall={args.min_recall:.0%}"
    )

    # Progress callback
    def progress_callback(current: int, total: int, threshold: float):
        percentage = (current / total) * 100
        print(
            f"Progress: {current}/{total} ({percentage:.1f}%) - Testing threshold {threshold}",
            end="\r",
        )

    try:
        report = await threshold_tuner.tune(
            golden_dataset=golden_dataset,
            thresholds=thresholds,
            min_recall=args.min_recall,
            progress_callback=progress_callback,
        )
        print()  # New line after progress
        logger.info("Threshold tuning completed successfully")
    except Exception as e:
        logger.error(f"Threshold tuning failed: {e}", exc_info=True)
        sys.exit(1)

    # Step 6: Generate and output report
    if args.format == "markdown":
        output_content = report.to_markdown()
    else:  # json
        import json

        output_content = json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    if args.output:
        args.output.write_text(output_content, encoding="utf-8")
        logger.info(f"Report saved to {args.output}")
    else:
        print("\n" + "=" * 80)
        print(output_content)
        print("=" * 80)

    # Step 7: Log recommendation summary
    if report.recommended_threshold is not None:
        logger.info(
            f"✓ Recommended threshold: {report.recommended_threshold} "
            f"({report.recommendation_reason})"
        )
    else:
        logger.warning(
            "✗ No threshold meets the min_recall constraint. "
            f"Consider lowering --min-recall (current: {args.min_recall:.0%}) "
            "or improving golden dataset quality."
        )
        sys.exit(2)  # Warning exit code

    logger.info("Threshold tuning successful")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
