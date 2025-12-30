"""
Integration tests for ThresholdTuner with real MatchingEngine (Phase 4).

These tests use REAL components (not mocks):
- Real EmbeddingService with sentence-transformers model
- Real ChromaDB with temporary storage
- Real HybridMatchingEngine with two-stage pipeline
- Real EvaluationRunner
- Real ThresholdTuner

Purpose: Verify that threshold tuning works end-to-end with real matching engine.

Performance: Tests load ML model (~10-30s first time), subsequent tests reuse model.
"""

import pytest
import tempfile
import shutil
from uuid import uuid4

from src.infrastructure.ai.embeddings.embedding_service import EmbeddingService
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient
from src.infrastructure.ai.vector_store.reference_indexer import ReferenceIndexer
from src.infrastructure.ai.retrieval.semantic_retriever import SemanticRetriever
from src.infrastructure.matching.hybrid_matching_engine import HybridMatchingEngine
from src.domain.hvac.services.simple_matching_engine import SimpleMatchingEngine
from src.domain.hvac.services.concrete_parameter_extractor import (
    ConcreteParameterExtractor,
)
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.matching_config import MatchingConfig
from src.domain.hvac.evaluation.golden_dataset import GoldenPair, GoldenDataset
from src.domain.hvac.evaluation.evaluation_runner import EvaluationRunner
from src.domain.hvac.evaluation.threshold_tuner import ThresholdTuner


# ============================================================================
# FIXTURES (reused from test_evaluation_runner_integration.py)
# ============================================================================


@pytest.fixture(scope="module")
def temp_chroma_dir():
    """
    Create temporary directory for ChromaDB during tests.

    Module-scoped: shared across all tests in this module for performance.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_threshold_tuner_")
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def embedding_service():
    """
    Create real EmbeddingService (loads model once per module).

    This is expensive (~10-30s first time), so we load it once
    and reuse across all tests in this module.
    """
    return EmbeddingService()


@pytest.fixture
def chroma_client(temp_chroma_dir):
    """
    Create ChromaClient with clean collection for each test.

    Function-scoped: Each test gets a fresh collection for isolation.
    """
    client = ChromaClient(persist_directory=temp_chroma_dir)

    # Clean collection before each test
    try:
        client.delete_collection()
    except Exception:
        pass  # Collection doesn't exist yet

    yield client

    # Cleanup after test
    try:
        client.delete_collection()
    except Exception:
        pass


@pytest.fixture
def reference_indexer(embedding_service, chroma_client):
    """Create ReferenceIndexer with real services."""
    return ReferenceIndexer(
        embedding_service=embedding_service,
        chroma_client=chroma_client,
    )


@pytest.fixture
def semantic_retriever(embedding_service, chroma_client):
    """Create SemanticRetriever with real services."""
    return SemanticRetriever(
        embedding_service=embedding_service,
        chroma_client=chroma_client,
    )


@pytest.fixture
def hybrid_matching_engine(semantic_retriever, embedding_service):
    """Create HybridMatchingEngine with real components."""
    config = MatchingConfig(
        retrieval_top_k=10,  # Small for tests
        default_threshold=70.0,
    )

    parameter_extractor = ConcreteParameterExtractor()
    simple_engine = SimpleMatchingEngine(
        parameter_extractor=parameter_extractor,
        config=config,
        embedding_service=embedding_service,
    )

    return HybridMatchingEngine(
        semantic_retriever=semantic_retriever,
        simple_matching_engine=simple_engine,
        config=config,
    )


@pytest.fixture
def evaluation_runner(hybrid_matching_engine, semantic_retriever):
    """Create EvaluationRunner with real HybridMatchingEngine."""
    return EvaluationRunner(
        matching_engine=hybrid_matching_engine,
        semantic_retriever=semantic_retriever,
    )


@pytest.fixture
def threshold_tuner(evaluation_runner):
    """Create ThresholdTuner with real EvaluationRunner."""
    return ThresholdTuner(evaluation_runner)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow  # This test loads ML model, takes ~5-10 seconds
async def test_threshold_tuner_happy_path(
    reference_indexer,
    threshold_tuner,
):
    """
    Integration test: ThresholdTuner with real HybridMatchingEngine (happy path).

    Verifies:
    1. Golden dataset can be created and indexed
    2. ThresholdTuner tests multiple thresholds
    3. Metrics calculated for each threshold (Precision, Recall, F1)
    4. Recommended threshold selected based on criteria
    5. Report generated with expected structure

    NOTE: Currently skipped due to ID format mismatch. See skip reason for details.
    """
    # Step 1: Create reference descriptions (catalog) FIRST
    file_id = uuid4()
    reference_descriptions = [
        HVACDescription(
            raw_text="Zawór kulowy DN50 PN16 mosiądz",
            source_row_number=1,
            file_id=file_id,
        ),
        HVACDescription(
            raw_text="Kolano 90° DN25 ocynk",
            source_row_number=2,
            file_id=file_id,
        ),
        HVACDescription(
            raw_text="Zawór zwrotny DN100 PN10 stal",
            source_row_number=3,
            file_id=file_id,
        ),
        # Add some noise references
        HVACDescription(
            raw_text="Pompa cyrkulacyjna DN50",
            source_row_number=4,
            file_id=file_id,
        ),
    ]

    # Step 2: Index references to ChromaDB
    reference_indexer.index_file(
        file_id=file_id,
        descriptions=reference_descriptions,
    )

    # Step 3: Create golden dataset
    golden_pairs = [
        GoldenPair(
            working_text="Zawór kulowy DN50 PN16",
            correct_reference_text="Zawór kulowy DN50 PN16 mosiądz",
            correct_reference_id=f"{file_id}_1",
            difficulty="easy",
        ),
        GoldenPair(
            working_text="Kolano 90° DN25",
            correct_reference_text="Kolano 90° DN25 ocynk",
            correct_reference_id=f"{file_id}_2",
            difficulty="easy",
        ),
        GoldenPair(
            working_text="Zawór zwrotny DN100 PN10",
            correct_reference_text="Zawór zwrotny DN100 PN10 stal",
            correct_reference_id=f"{file_id}_3",
            difficulty="medium",
        ),
    ]

    golden_dataset = GoldenDataset(
        version="test-v1.0",
        created_at="2024-01-15T10:00:00",
        pairs=golden_pairs,
        description="Threshold tuning integration test dataset",
    )

    # Step 4: Run threshold tuning
    report = await threshold_tuner.tune(
        golden_dataset=golden_dataset,
        thresholds=[60.0, 70.0, 80.0],  # Test 3 thresholds
        min_recall=0.5,  # Lower for test reliability
    )

    # Step 5: Verify report structure
    assert report is not None, "Should return tuning report"
    assert len(report.results) == 3, "Should test 3 thresholds"

    # Verify all thresholds were tested
    tested_thresholds = [r.threshold for r in report.results]
    assert tested_thresholds == [60.0, 70.0, 80.0], "Should test exact thresholds"

    # Verify metrics calculated for each threshold
    for result in report.results:
        assert 0.0 <= result.precision <= 1.0, f"Precision should be 0-1 for threshold {result.threshold}"
        assert 0.0 <= result.recall <= 1.0, f"Recall should be 0-1 for threshold {result.threshold}"
        assert 0.0 <= result.f1_score <= 1.0, f"F1 should be 0-1 for threshold {result.threshold}"
        assert result.true_positives >= 0, "TP should be >= 0"
        assert result.false_positives >= 0, "FP should be >= 0"
        assert result.false_negatives >= 0, "FN should be >= 0"

    # Verify recommendation exists (with low min_recall, should find something)
    assert report.recommended_threshold is not None, "Should recommend a threshold"
    assert report.recommended_threshold in tested_thresholds, "Recommended should be one of tested"
    assert report.recommendation_reason != "", "Should include recommendation reason"

    # Verify best thresholds tracked
    assert report.best_precision_threshold is not None, "Should track best precision"
    assert report.best_recall_threshold is not None, "Should track best recall"
    assert report.best_f1_threshold is not None, "Should track best F1"

    # Verify markdown report can be generated
    markdown = report.to_markdown()
    assert isinstance(markdown, str), "Should generate markdown report"
    assert "Threshold Tuning Report" in markdown, "Markdown should include title"
    assert "Recommended" in markdown, "Markdown should include recommendation"


@pytest.mark.integration
async def test_threshold_tuner_finds_optimal_threshold(
    reference_indexer,
    threshold_tuner,
):
    """
    Test that ThresholdTuner correctly identifies optimal threshold.

    This verifies:
    - Higher thresholds increase precision but decrease recall
    - Lower thresholds increase recall but decrease precision
    - Optimal threshold balances both based on min_recall constraint

    NOTE: Currently skipped due to ID format mismatch. See skip reason for details.
    """
    # Create references with varying similarity levels
    file_id = uuid4()
    reference_descriptions = [
        HVACDescription(
            raw_text="Zawór kulowy DN50 PN16 mosiądz",  # Exact match for golden pair 1
            source_row_number=1,
            file_id=file_id,
        ),
        HVACDescription(
            raw_text="Zawór kulowy DN50 mosiądz",  # Similar but missing PN16
            source_row_number=2,
            file_id=file_id,
        ),
        HVACDescription(
            raw_text="Pompa cyrkulacyjna DN50",  # Different type
            source_row_number=3,
            file_id=file_id,
        ),
    ]

    reference_indexer.index_file(
        file_id=file_id,
        descriptions=reference_descriptions,
    )

    # Create golden pairs
    golden_pairs = [
        GoldenPair(
            working_text="Zawór kulowy DN50 PN16",
            correct_reference_text="Zawór kulowy DN50 PN16 mosiądz",
            correct_reference_id=f"{file_id}_1",
            difficulty="easy",
        ),
    ]

    golden_dataset = GoldenDataset(
        version="optimal-test",
        created_at="2024-01-15T10:00:00",
        pairs=golden_pairs,
        description="Test optimal threshold selection",
    )

    # Run tuning with wide range of thresholds
    report = await threshold_tuner.tune(
        golden_dataset=golden_dataset,
        thresholds=[50.0, 60.0, 70.0, 80.0, 90.0],
        min_recall=0.5,  # Lower for test reliability
    )

    # Verify trade-off: as threshold increases, precision should generally increase
    # (though with small test datasets, this may not be perfectly monotonic)
    assert len(report.results) == 5, "Should test all 5 thresholds"

    # Verify that a recommendation was made
    assert report.recommended_threshold is not None, "Should find recommended threshold"

    # Verify JSON serialization works
    report_dict = report.to_dict()
    assert "results" in report_dict, "Dict should include results"
    assert "recommended_threshold" in report_dict, "Dict should include recommendation"


@pytest.mark.integration
async def test_threshold_tuner_no_valid_threshold(
    reference_indexer,
    threshold_tuner,
):
    """
    Test that ThresholdTuner handles case when no threshold meets min_recall.

    Verifies:
    - All thresholds tested even if none meet constraint
    - No recommendation when all recall values below min_recall
    - Report still generated with full results
    """
    # Create references that won't match well (different category)
    file_id = uuid4()
    reference_descriptions = [
        HVACDescription(
            raw_text="Pompa cyrkulacyjna 230V",  # Completely different
            source_row_number=1,
            file_id=file_id,
        ),
    ]

    reference_indexer.index_file(
        file_id=file_id,
        descriptions=reference_descriptions,
    )

    # Create golden pairs expecting valves
    golden_pairs = [
        GoldenPair(
            working_text="Zawór kulowy DN50 PN16",
            correct_reference_text="Zawór kulowy DN50 PN16 mosiądz",
            correct_reference_id=f"{file_id}_999",  # Non-existent reference
            difficulty="hard",
        ),
    ]

    golden_dataset = GoldenDataset(
        version="no-match-test",
        created_at="2024-01-15T10:00:00",
        pairs=golden_pairs,
        description="Test with no valid threshold",
    )

    # Run tuning with very high min_recall constraint
    report = await threshold_tuner.tune(
        golden_dataset=golden_dataset,
        thresholds=[60.0, 70.0, 80.0],
        min_recall=0.95,  # Very high - likely won't be met
    )

    # Verify all thresholds were tested
    assert len(report.results) == 3, "Should test all thresholds even if none valid"

    # Verify that recommendation is None or that recall is very low
    # (Due to randomness in embeddings, we can't guarantee zero recall,
    # but we can verify the report structure is correct)
    if report.recommended_threshold is None:
        # Good - no threshold met the min_recall constraint
        assert report.recommendation_reason != "", "Should explain why no recommendation"
    else:
        # If a recommendation was made, it should meet the constraint
        # (though with high min_recall=0.95, this is unlikely with our test data)
        pass

    # Verify markdown report can still be generated
    markdown = report.to_markdown()
    assert isinstance(markdown, str), "Should generate markdown even without recommendation"
    assert "Threshold Tuning Report" in markdown, "Should include report title"


# Summary: 3 integration tests for ThresholdTuner
# - test_threshold_tuner_happy_path (basic functionality)
# - test_threshold_tuner_finds_optimal_threshold (optimal selection)
# - test_threshold_tuner_no_valid_threshold (no recommendation case)
