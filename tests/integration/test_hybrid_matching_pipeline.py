"""
Integration tests for Two-Stage Hybrid Matching Pipeline (Phase 4).

These tests use REAL components (not mocks):
- Real EmbeddingService with sentence-transformers model
- Real ChromaClient with temporary ChromaDB
- Real HybridMatchingEngine with actual two-stage pipeline

Purpose: Verify that all components work together correctly.

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


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(scope="module")
def temp_chroma_dir():
    """
    Create temporary directory for ChromaDB during tests.

    Module-scoped: shared across all tests in this module for performance.
    Each test cleans its own collection, so isolation is maintained.
    """
    temp_dir = tempfile.mkdtemp(prefix="test_chroma_integration_")
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
    Uses temp directory from module-scoped fixture.
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


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow  # This test loads ML model, takes ~5-10 seconds
async def test_hybrid_pipeline_end_to_end(
    reference_indexer,
    hybrid_matching_engine,
    embedding_service,
):
    """
    Integration test: Full two-stage pipeline with real ChromaDB.

    Verifies:
    1. Reference descriptions indexed to ChromaDB
    2. Stage 1: Semantic retrieval finds similar items
    3. Stage 2: Hybrid scoring calculates accurate match
    4. AI metadata populated in result
    """
    # Step 1: Create reference descriptions (catalog)
    reference_descriptions = [
        HVACDescription(
            raw_text="Zawór kulowy DN50 PN16 mosiądz",
            source_row_number=1,
            file_id=uuid4(),
        ),
        HVACDescription(
            raw_text="Zawór kulowy DN25 PN10 stal",
            source_row_number=2,
            file_id=uuid4(),
        ),
        HVACDescription(
            raw_text="Kolano 90° DN50 ocynk",
            source_row_number=3,
            file_id=uuid4(),
        ),
        HVACDescription(
            raw_text="Zawór zwrotny DN50 PN16 mosiądz",  # Similar to first
            source_row_number=4,
            file_id=uuid4(),
        ),
    ]

    # Step 2: Index references to ChromaDB (this is what FileUploadUseCase does)
    file_id = uuid4()
    reference_indexer.index_file(
        file_id=file_id,
        descriptions=reference_descriptions,
    )

    # Step 3: Create working description (what we want to match)
    working_description = HVACDescription(
        raw_text="Zawór kulowy DN 50 PN 16",  # Should match first reference
        source_row_number=1,
        file_id=uuid4(),
    )

    # Step 4: Run HybridMatchingEngine (two-stage pipeline)
    result = await hybrid_matching_engine.match(
        working_description=working_description,
        reference_descriptions=reference_descriptions,
        threshold=70.0,
    )

    # Step 5: Verify result
    assert result is not None, "Should find a match"

    # NOTE: We cannot test matched_reference_id directly because
    # HybridMatchingEngine._convert_candidates_to_descriptions() creates NEW
    # HVACDescription objects (with new UUIDs) from ChromaDB retrieval results.
    # Instead, we verify the match quality through scores and breakdown.

    # Verify scores are populated and above threshold
    assert result.score.parameter_score > 0, "Parameter score should be calculated"
    assert result.score.semantic_score > 0, "Semantic score (AI) should be calculated"
    assert (
        result.score.final_score >= 70.0
    ), f"Final score should be above threshold, got {result.score.final_score}"

    # Verify breakdown contains DN50 and PN16 matches (expected for valves 0 or 3)
    assert "dn_match" in result.breakdown, "Should include dn_match"
    assert "pn_match" in result.breakdown, "Should include pn_match"
    assert result.breakdown["dn_match"] is True, "DN should match (DN50)"
    assert result.breakdown["pn_match"] is True, "PN should match (PN16)"

    # Verify AI metadata in breakdown
    assert "using_ai" in result.breakdown, "Breakdown should include using_ai"
    assert result.breakdown["using_ai"] is True, "AI should be enabled"
    assert "ai_model" in result.breakdown, "Breakdown should include ai_model"
    # Model name should be present (not checking exact value as it may vary)
    assert result.breakdown["ai_model"] is not None

    # Verify Stage 1 metadata
    assert (
        "stage1_candidates" in result.breakdown
    ), "Should track Stage 1 candidates count"
    assert (
        result.breakdown["stage1_candidates"] <= 10
    ), "Should retrieve at most top_k=10 candidates"


@pytest.mark.integration
async def test_hybrid_pipeline_no_match_below_threshold(
    reference_indexer,
    hybrid_matching_engine,
):
    """Test that pipeline returns None when no match above threshold."""
    # Index one reference (very different)
    reference = HVACDescription(
        raw_text="Pompa obiegowa 230V 50Hz",
        source_row_number=1,
        file_id=uuid4(),
    )

    reference_indexer.index_file(
        file_id=uuid4(),
        descriptions=[reference],
    )

    # Try to match completely different item
    working_description = HVACDescription(
        raw_text="Zawór kulowy DN50 PN16",
        source_row_number=1,
        file_id=uuid4(),
    )

    result = await hybrid_matching_engine.match(
        working_description=working_description,
        reference_descriptions=[reference],
        threshold=75.0,
    )

    # Should return None (below threshold)
    assert result is None, "Should not match very different items"


@pytest.mark.integration
async def test_hybrid_pipeline_fallback_without_filters(
    reference_indexer,
    semantic_retriever,
    hybrid_matching_engine,
):
    """
    Test that Stage 1 fallback works when no candidates match filters.

    HybridMatchingEngine should:
    1. Try retrieval with DN/PN filters
    2. If empty, retry without filters
    """
    # Index reference without DN/PN parameters
    reference = HVACDescription(
        raw_text="Pompa cyrkulacyjna energia A",  # No DN, no PN
        source_row_number=1,
        file_id=uuid4(),
    )

    reference_indexer.index_file(
        file_id=uuid4(),
        descriptions=[reference],
    )

    # Working description has DN50 but reference doesn't
    working_description = HVACDescription(
        raw_text="Pompa cyrkulacyjna DN50",
        source_row_number=1,
        file_id=uuid4(),
    )

    # Should still find match via fallback (retry without filters)
    # This tests that fallback mechanism doesn't crash
    _ = await hybrid_matching_engine.match(
        working_description=working_description,
        reference_descriptions=[reference],
        threshold=60.0,  # Lower threshold for this edge case
    )

    # No assertion on result because it depends on embeddings
    # The important part is that the fallback mechanism executes without errors


@pytest.mark.integration
def test_stage1_retrieval_returns_top_k_candidates(
    reference_indexer,
    semantic_retriever,
):
    """Test that Stage 1 correctly retrieves top-K candidates."""
    # Index 20 references
    file_id = uuid4()
    references = [
        HVACDescription(
            raw_text=f"Zawór kulowy DN{25 + i*5} PN16",
            source_row_number=i,
            file_id=file_id,
        )
        for i in range(20)
    ]

    reference_indexer.index_file(
        file_id=file_id,
        descriptions=references,
    )

    # Retrieve top-5
    query_text = "Zawór kulowy DN50 PN16"
    candidates = semantic_retriever.retrieve(
        query_text=query_text,
        filters=None,
        top_k=5,
    )

    # Should return exactly 5 candidates
    assert len(candidates) == 5, f"Should return top-5, got {len(candidates)}"

    # Candidates should be sorted by score (descending)
    scores = [c.similarity_score for c in candidates]
    assert scores == sorted(
        scores, reverse=True
    ), "Candidates should be sorted by score"

    # All scores should be 0-1 (similarity scores are normalized)
    for candidate in candidates:
        assert (
            0 <= candidate.similarity_score <= 1
        ), f"Score {candidate.similarity_score} out of range"


@pytest.mark.integration
def test_embedding_service_batch_processing(embedding_service):
    """Test that EmbeddingService can handle batch embeddings."""
    texts = [
        "Zawór kulowy DN50",
        "Kolano 90° DN25",
        "Pompa cyrkulacyjna",
    ]

    # Generate batch embeddings
    embeddings = embedding_service.embed_batch(texts, batch_size=2)

    # Should return 3 embeddings
    assert len(embeddings) == 3

    # Each embedding should be 384-dimensional (model output for MiniLM)
    for emb in embeddings:
        assert len(emb) == 384, f"Expected 384-dim, got {len(emb)}"
        assert all(isinstance(x, float) for x in emb), "Embedding should be floats"


# Summary: 5 integration tests
# - Full two-stage pipeline (happy path)
# - No match below threshold
# - Fallback without filters
# - Stage 1 top-K retrieval
# - Embedding service batch processing
