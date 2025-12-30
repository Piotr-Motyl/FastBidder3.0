#!/usr/bin/env python3
"""
CLI tool for evaluating HVAC matching quality using golden dataset.

This script evaluates matching engine performance using curated test cases
(golden dataset) and generates a detailed evaluation report.

Usage:
    python scripts/evaluate_matching.py --dataset data/golden_dataset.json --threshold 75.0
    python scripts/evaluate_matching.py --dataset data/golden.json --output report.md

Requirements:
    - Golden dataset JSON file with test cases
    - HybridMatchingEngine or SimpleMatchingEngine
    - ChromaDB indexed with reference descriptions (for AI matching)

Output:
    - Markdown report with metrics (Recall@K, Precision@1, MRR)
    - Failed pairs analysis
    - Performance statistics
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
        description="Evaluate HVAC matching quality using golden dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings
  python scripts/evaluate_matching.py --dataset data/golden_dataset.json

  # Evaluate with custom threshold and output
  python scripts/evaluate_matching.py --dataset data/golden.json --threshold 80 --output report.md

  # Evaluate without AI (SimpleMatchingEngine only)
  python scripts/evaluate_matching.py --dataset data/golden.json --no-ai

  # Evaluate with different top-K values
  python scripts/evaluate_matching.py --dataset data/golden.json --top-k 1,3,5,10
        """,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to golden dataset JSON file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=75.0,
        help="Matching threshold (default: 75.0)",
    )

    parser.add_argument(
        "--top-k",
        type=str,
        default="1,3,5",
        help="Comma-separated K values for Recall@K (default: 1,3,5)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for report (default: stdout)",
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

    # Step 2: Initialize matching engine
    logger.info("Initializing matching engine...")
    config = MatchingConfig(
        default_threshold=args.threshold,
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

    # Step 3: Parse top-K values
    try:
        top_k_values = [int(k.strip()) for k in args.top_k.split(",")]
    except ValueError:
        logger.error(
            f"Invalid --top-k format: {args.top_k}. Expected comma-separated integers."
        )
        sys.exit(1)

    # Step 4: Run evaluation
    logger.info(f"Starting evaluation with threshold={args.threshold}, top_k={top_k_values}")

    evaluation_runner = EvaluationRunner(
        matching_engine=matching_engine,
        semantic_retriever=semantic_retriever,
    )

    # Progress callback
    def progress_callback(current: int, total: int):
        percentage = (current / total) * 100
        print(f"Progress: {current}/{total} ({percentage:.1f}%)", end="\r")

    try:
        report = await evaluation_runner.evaluate(
            golden_dataset=golden_dataset,
            threshold=args.threshold,
            top_k_values=top_k_values,
            progress_callback=progress_callback,
        )
        print()  # New line after progress
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)

    # Step 5: Generate and output report
    markdown_report = report.to_markdown()

    if args.output:
        args.output.write_text(markdown_report)
        logger.info(f"Report saved to {args.output}")
    else:
        print("\n" + "=" * 80)
        print(markdown_report)
        print("=" * 80)

    # Step 6: Exit with status code based on results
    if report.precision_at_1 < 0.7:
        logger.warning(
            f"Precision@1 ({report.precision_at_1:.2%}) is below 70%. "
            f"Consider tuning threshold or improving golden dataset."
        )
        sys.exit(2)  # Warning exit code

    logger.info("Evaluation successful")
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
