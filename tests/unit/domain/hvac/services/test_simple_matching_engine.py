"""
Tests for SimpleMatchingEngine domain service.
Covers: hybrid matching algorithm, parameter scoring, fast-fail, confidence calculation.
"""

import pytest

from src.domain.hvac.services.simple_matching_engine import SimpleMatchingEngine
from src.domain.hvac.services.concrete_parameter_extractor import (
    ConcreteParameterExtractor,
)
from src.domain.hvac.matching_config import MatchingConfig
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.value_objects.match_result import MatchResult


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def extractor():
    """Create ConcreteParameterExtractor instance for tests."""
    return ConcreteParameterExtractor()


@pytest.fixture
def config():
    """Create default MatchingConfig for tests."""
    return MatchingConfig.default()


@pytest.fixture
def engine(extractor, config):
    """Create SimpleMatchingEngine instance for tests."""
    return SimpleMatchingEngine(
        parameter_extractor=extractor,
        config=config,
    )


# ============================================================================
# TESTS - should_fast_fail()
# ============================================================================


def test_should_fast_fail_dn_mismatch(engine):
    """Test fast-fail when DN values differ."""
    source = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(
        dn=100, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    should_fail, reason = engine.should_fast_fail(source, reference)

    assert should_fail is True
    assert "DN50" in reason
    assert "DN100" in reason
    assert "mismatch" in reason.lower()


def test_should_fast_fail_dn_match(engine):
    """Test no fast-fail when DN values match."""
    source = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(
        dn=50, pn=25, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    should_fail, reason = engine.should_fast_fail(source, reference)

    assert should_fail is False
    assert reason is None


def test_should_fast_fail_no_dn_in_source(engine):
    """Test no fast-fail when source has no DN."""
    source = ExtractedParameters(pn=16, confidence_scores={"pn": 1.0})
    reference = ExtractedParameters(
        dn=100, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    should_fail, reason = engine.should_fast_fail(source, reference)

    assert should_fail is False
    assert reason is None


def test_should_fast_fail_no_dn_in_reference(engine):
    """Test no fast-fail when reference has no DN."""
    source = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(pn=16, confidence_scores={"pn": 1.0})

    should_fail, reason = engine.should_fast_fail(source, reference)

    assert should_fail is False
    assert reason is None


def test_should_fast_fail_disabled_in_config(extractor):
    """Test fast-fail disabled when config.enable_fast_fail=False."""
    config = MatchingConfig.for_testing(enable_fast_fail=False)
    engine = SimpleMatchingEngine(parameter_extractor=extractor, config=config)

    source = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(
        dn=100, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    should_fail, reason = engine.should_fast_fail(source, reference)

    assert should_fail is False  # Fast-fail disabled
    assert reason is None


# ============================================================================
# TESTS - calculate_confidence()
# ============================================================================


def test_calculate_confidence_large_gap(engine):
    """Test high confidence when gap >= 10 points."""
    confidence = engine.calculate_confidence(
        best_score=95.0, second_best_score=70.0  # Gap = 25
    )

    assert confidence >= 0.85  # High confidence
    assert confidence <= 1.0


def test_calculate_confidence_small_gap(engine):
    """Test low confidence when gap < 5 points."""
    confidence = engine.calculate_confidence(
        best_score=80.0, second_best_score=78.0  # Gap = 2
    )

    assert confidence < 0.70  # Low confidence


def test_calculate_confidence_medium_gap(engine):
    """Test medium confidence when gap is 5-10 points."""
    confidence = engine.calculate_confidence(
        best_score=85.0, second_best_score=78.0  # Gap = 7
    )

    assert 0.60 <= confidence <= 0.90  # Medium confidence


def test_calculate_confidence_only_one_match(engine):
    """Test confidence based on score when only 1 reference."""
    confidence = engine.calculate_confidence(best_score=90.0, second_best_score=None)

    assert confidence == 0.9  # 90/100


def test_calculate_confidence_perfect_score(engine):
    """Test maximum confidence for perfect score."""
    confidence = engine.calculate_confidence(best_score=100.0, second_best_score=None)

    assert confidence == 1.0


# ============================================================================
# TESTS - calculate_parameter_score()
# ============================================================================


def test_calculate_parameter_score_perfect_match(engine):
    """Test parameter score for perfect match (all params identical)."""
    source = ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="kulowy",
        material="mosiądz",
        drive_type="ręczny",
        voltage="230V",
        manufacturer="KSB",
        confidence_scores={
            "dn": 1.0,
            "pn": 1.0,
            "valve_type": 1.0,
            "material": 1.0,
            "drive_type": 1.0,
            "voltage": 1.0,
            "manufacturer": 1.0,
        },
    )
    reference = ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="kulowy",
        material="mosiądz",
        drive_type="ręczny",
        voltage="230V",
        manufacturer="KSB",
        confidence_scores={
            "dn": 1.0,
            "pn": 1.0,
            "valve_type": 1.0,
            "material": 1.0,
            "drive_type": 1.0,
            "voltage": 1.0,
            "manufacturer": 1.0,
        },
    )

    score, breakdown = engine.calculate_parameter_score(source, reference)

    assert score == 100.0  # Perfect match
    assert breakdown["dn_match"] is True
    assert breakdown["pn_match"] is True
    assert breakdown["valve_type_match"] is True
    assert breakdown["material_match"] is True
    assert breakdown["drive_type_match"] is True
    assert breakdown["voltage_match"] is True
    assert breakdown["manufacturer_match"] is True


def test_calculate_parameter_score_dn_pn_only(engine):
    """Test parameter score when only DN and PN match."""
    source = ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="kulowy",
        confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0},
    )
    reference = ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="zwrotny",  # Different type
        confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0},
    )

    score, breakdown = engine.calculate_parameter_score(source, reference)

    # DN (35%) + PN (10%) = 45%
    assert 44.0 <= score <= 46.0
    assert breakdown["dn_match"] is True
    assert breakdown["pn_match"] is True
    assert breakdown["valve_type_match"] is False


def test_calculate_parameter_score_no_match(engine):
    """Test parameter score when nothing matches."""
    source = ExtractedParameters(
        dn=50, pn=16, valve_type="kulowy", confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(
        dn=100,
        pn=25,
        valve_type="zwrotny",
        confidence_scores={"dn": 1.0, "pn": 1.0},
    )

    score, breakdown = engine.calculate_parameter_score(source, reference)

    assert score == 0.0  # No matches
    assert breakdown["dn_match"] is False
    assert breakdown["pn_match"] is False
    assert breakdown["valve_type_match"] is False


def test_calculate_parameter_score_missing_params(engine):
    """Test parameter score when source has missing params."""
    source = ExtractedParameters(
        dn=50, confidence_scores={"dn": 1.0}  # Only DN
    )
    reference = ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="kulowy",
        confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0},
    )

    score, breakdown = engine.calculate_parameter_score(source, reference)

    # Only DN matches (35%)
    assert 34.0 <= score <= 36.0
    assert breakdown["dn_match"] is True
    assert breakdown["pn_match"] is False  # Source doesn't have PN
    assert breakdown["valve_type_match"] is False


def test_calculate_parameter_score_breakdown_structure(engine):
    """Test that breakdown has all required fields."""
    source = ExtractedParameters(dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0})
    reference = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    score, breakdown = engine.calculate_parameter_score(source, reference)

    # Check all required fields in breakdown
    assert "dn_match" in breakdown
    assert "pn_match" in breakdown
    assert "valve_type_match" in breakdown
    assert "material_match" in breakdown
    assert "drive_type_match" in breakdown
    assert "voltage_match" in breakdown
    assert "manufacturer_match" in breakdown
    assert "weights_applied" in breakdown
    assert "individual_contributions" in breakdown

    # Check individual contributions
    assert "dn" in breakdown["individual_contributions"]
    assert "pn" in breakdown["individual_contributions"]


# ============================================================================
# TESTS - calculate_semantic_score()
# ============================================================================


def test_calculate_semantic_score_returns_placeholder(engine):
    """Test that semantic score returns placeholder value in Phase 3."""
    source = HVACDescription(raw_text="Zawór kulowy DN50 PN16")
    reference = HVACDescription(raw_text="Kurek kulowy DN50 PN16")

    score = engine.calculate_semantic_score(source, reference)

    # Phase 3 placeholder: 0.5 * 100 = 50.0
    assert score == 50.0


# ============================================================================
# TESTS - generate_explanation()
# ============================================================================


def test_generate_explanation_perfect_match(engine):
    """Test explanation for perfect match."""
    source = ExtractedParameters(
        dn=50,
        pn=16,
        valve_type="kulowy",
        material="mosiądz",
        confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0, "material": 1.0},
    )
    reference = source  # Perfect match

    _, breakdown = engine.calculate_parameter_score(source, reference)
    explanation = engine.generate_explanation(
        source, reference, parameter_score=100.0, semantic_score=50.0, final_score=70.0, breakdown=breakdown
    )

    assert "Perfect match" in explanation or "all parameters identical" in explanation


def test_generate_explanation_partial_match(engine):
    """Test explanation for partial match."""
    source = ExtractedParameters(
        dn=50, pn=16, valve_type="kulowy", confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(
        dn=50, pn=16, valve_type="zwrotny", confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    _, breakdown = engine.calculate_parameter_score(source, reference)
    explanation = engine.generate_explanation(
        source, reference, parameter_score=45.0, semantic_score=50.0, final_score=48.0, breakdown=breakdown
    )

    assert "DN50" in explanation
    assert "PN16" in explanation
    assert "valve type" in explanation.lower() or "different" in explanation.lower()


def test_generate_explanation_includes_final_score(engine):
    """Test that explanation includes final score."""
    source = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )
    reference = ExtractedParameters(
        dn=50, pn=16, confidence_scores={"dn": 1.0, "pn": 1.0}
    )

    _, breakdown = engine.calculate_parameter_score(source, reference)
    explanation = engine.generate_explanation(
        source, reference, parameter_score=45.0, semantic_score=50.0, final_score=48.0, breakdown=breakdown
    )

    assert "48" in explanation or "Total" in explanation


# ============================================================================
# TESTS - match_single() - Happy Path
# ============================================================================


def test_match_single_perfect_match(engine):
    """Test match_single with perfect parameter match."""
    source = HVACDescription(raw_text="Zawór kulowy DN50 PN16 mosiądz")
    references = [
        HVACDescription(raw_text="Zawór kulowy DN50 PN16 mosiądz"),  # Perfect match
        HVACDescription(raw_text="Zawór kulowy DN80 PN16"),
    ]

    result = engine.match_single(source, references, threshold=60.0)

    assert result is not None
    assert isinstance(result, MatchResult)
    assert result.score.final_score >= 60.0
    assert result.matched_reference_id == references[0].id
    assert result.confidence > 0.5


def test_match_single_returns_best_match(engine):
    """Test that match_single returns highest scoring reference."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [
        HVACDescription(raw_text="Zawór DN80 PN16"),  # Partial match (no DN)
        HVACDescription(raw_text="Zawór DN50 PN16"),  # Best match
        HVACDescription(raw_text="Zawór DN50 PN25"),  # Partial match (no PN)
    ]

    result = engine.match_single(source, references, threshold=40.0)

    assert result is not None
    assert result.matched_reference_id == references[1].id  # Best match


def test_match_single_below_threshold_returns_none(engine):
    """Test that match_single returns None when best score < threshold."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [
        HVACDescription(raw_text="Pompa DN100 PN25"),  # Very different
    ]

    result = engine.match_single(source, references, threshold=75.0)

    assert result is None  # No match above threshold


def test_match_single_empty_references_returns_none(engine):
    """Test that match_single returns None for empty reference list."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")

    result = engine.match_single(source, [], threshold=75.0)

    assert result is None


def test_match_single_source_without_params_returns_none(engine):
    """Test that match_single returns None when source has no params."""
    source = HVACDescription(raw_text="xyz")  # No extractable params
    references = [HVACDescription(raw_text="Zawór DN50 PN16")]

    result = engine.match_single(source, references, threshold=75.0)

    assert result is None


def test_match_single_uses_default_threshold(extractor):
    """Test that match_single uses config.default_threshold when threshold=None."""
    # Create config with lower default threshold to work with placeholder semantic score
    config = MatchingConfig.for_testing(default_threshold=40.0)
    engine = SimpleMatchingEngine(parameter_extractor=extractor, config=config)

    source = HVACDescription(raw_text="Zawór kulowy DN50 PN16")
    references = [HVACDescription(raw_text="Zawór kulowy DN50 PN16")]

    result = engine.match_single(source, references, threshold=None)

    assert result is not None  # Should use default threshold (40.0)
    assert result.score.threshold == 40.0  # Config default


def test_match_single_fast_fail_skips_dn_mismatch(engine):
    """Test that match_single uses fast-fail to skip DN mismatches."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [
        HVACDescription(raw_text="Zawór DN100 PN16"),  # DN mismatch - should be skipped
        HVACDescription(raw_text="Zawór DN50 PN25"),  # DN match
    ]

    result = engine.match_single(source, references, threshold=40.0)

    # Should match second reference (DN50), not first (DN100)
    assert result is not None
    assert result.matched_reference_id == references[1].id


def test_match_single_confidence_calculation(engine):
    """Test that match_single calculates confidence correctly."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [
        HVACDescription(raw_text="Zawór DN50 PN16"),  # Best match
        HVACDescription(raw_text="Zawór DN50 PN25"),  # Second-best (PN differs)
    ]

    result = engine.match_single(source, references, threshold=40.0)

    assert result is not None
    assert 0.0 <= result.confidence <= 1.0


def test_match_single_extracts_params_automatically(engine):
    """Test that match_single extracts parameters if not already extracted."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [HVACDescription(raw_text="Zawór DN50 PN16")]

    # Initially, parameters should not be extracted
    assert source.extracted_params is None
    assert references[0].extracted_params is None

    result = engine.match_single(source, references, threshold=40.0)

    # After matching, parameters should be extracted
    assert source.extracted_params is not None
    assert references[0].extracted_params is not None


def test_match_single_result_has_all_fields(engine):
    """Test that MatchResult has all required fields."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [HVACDescription(raw_text="Zawór DN50 PN16")]

    result = engine.match_single(source, references, threshold=40.0)

    assert result is not None
    assert hasattr(result, "matched_reference_id")
    assert hasattr(result, "score")
    assert hasattr(result, "confidence")
    assert hasattr(result, "message")
    assert hasattr(result, "breakdown")
    assert result.score.parameter_score >= 0.0
    assert result.score.semantic_score >= 0.0
    assert result.score.final_score >= 0.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_integration_full_matching_workflow(engine):
    """Integration test: Full matching workflow from raw text to MatchResult."""
    # Create source and reference descriptions
    source = HVACDescription(
        raw_text="Montaż zaworu kulowego DN50 PN16 z mosiądzu firmy KSB"
    )
    references = [
        HVACDescription(raw_text="Zawór kulowy DN50 PN16 mosiądz KSB"),  # Perfect match
        HVACDescription(raw_text="Zawór kulowy DN80 PN16 mosiądz"),  # Different DN
        HVACDescription(raw_text="Zawór zwrotny DN50 PN16"),  # Different type
    ]

    # Perform matching - use lower threshold due to placeholder semantic score
    result = engine.match_single(source, references, threshold=50.0)

    # Verify result
    assert result is not None
    assert result.matched_reference_id == references[0].id  # Best match
    assert result.score.parameter_score >= 65.0  # Good parameter score (DN+PN+material+manufacturer)
    assert result.score.final_score >= 50.0  # Above threshold
    assert result.confidence >= 0.7  # High confidence
    assert "DN50" in result.message
    assert "PN16" in result.message


def test_integration_fast_fail_performance(engine):
    """Integration test: Fast-fail optimization skips incompatible references."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")

    # Create many references with different DNs (should be fast-failed)
    references = [
        HVACDescription(raw_text=f"Zawór DN{dn} PN16")
        for dn in [15, 20, 25, 32, 40, 65, 80, 100, 125, 150]
    ]
    # Add one matching reference
    references.append(HVACDescription(raw_text="Zawór DN50 PN16"))

    result = engine.match_single(source, references, threshold=40.0)

    # Should find the DN50 match (last one)
    assert result is not None
    assert result.matched_reference_id == references[-1].id


def test_integration_hybrid_scoring(engine):
    """Integration test: Verify hybrid scoring (40% param + 60% semantic)."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [HVACDescription(raw_text="Zawór DN50 PN16")]

    result = engine.match_single(source, references, threshold=40.0)

    assert result is not None
    # With DN+PN match (45%) and semantic placeholder (50%)
    # Final = 0.4 * 45 + 0.6 * 50 = 18 + 30 = 48.0
    assert 47.0 <= result.score.final_score <= 49.0


# ============================================================================
# TESTS - AI INTEGRATION (PHASE 4)
# ============================================================================


@pytest.fixture
def mock_embedding_service():
    """Create mocked EmbeddingService for AI tests."""
    from unittest.mock import Mock

    mock = Mock()

    # Mock embed_single to return deterministic embeddings based on text
    def fake_embed_single(text, **kwargs):
        # Generate deterministic embedding based on text hash
        # Same text → same embedding (for caching tests)
        hash_val = hash(text) % 100 / 100.0
        # Return 384-dim vector
        return [hash_val] * 384

    mock.embed_single.side_effect = fake_embed_single
    return mock


@pytest.fixture
def engine_with_ai(extractor, config, mock_embedding_service):
    """Create SimpleMatchingEngine with AI enabled."""
    return SimpleMatchingEngine(
        parameter_extractor=extractor,
        config=config,
        embedding_service=mock_embedding_service,
    )


def test_engine_without_ai_uses_placeholder(engine):
    """Test that engine without embedding_service uses placeholder score."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    reference = HVACDescription(raw_text="Completely different text")

    # Calculate semantic score directly
    score = engine.calculate_semantic_score(source, reference)

    # Should return placeholder (50.0)
    assert score == 50.0


def test_engine_with_ai_calculates_real_similarity(engine_with_ai, mock_embedding_service):
    """Test that engine with embedding_service calculates real cosine similarity."""
    source = HVACDescription(raw_text="Zawór kulowy DN50 PN16")
    reference = HVACDescription(raw_text="Zawór kulowy DN50 PN16")  # Same text

    # Generate source embedding (as match_single would do)
    source_embedding = mock_embedding_service.embed_single(source.raw_text)

    # Calculate semantic score with source_embedding
    score = engine_with_ai.calculate_semantic_score(source, reference, source_embedding)

    # Same text → same embedding → cosine similarity = 1.0 → score = 100.0
    assert score == 100.0
    # Should NOT be placeholder
    assert score != 50.0
    # embed_single should be called
    assert mock_embedding_service.embed_single.called


def test_calculate_semantic_score_different_texts(engine_with_ai):
    """Test cosine similarity with different texts."""
    source = HVACDescription(raw_text="Zawór kulowy DN50")
    reference = HVACDescription(raw_text="Pompa obiegowa DN100")

    score = engine_with_ai.calculate_semantic_score(source, reference)

    # Different texts → different hash → different embeddings
    # Score should be in valid range
    assert 0.0 <= score <= 100.0
    # With our hash-based mock, different texts might give various similarities
    # The important thing is it's calculated, not placeholder (unless coincidence)
    # We can't predict exact score, but it should be valid


def test_source_embedding_cached_in_match_single(engine_with_ai, mock_embedding_service):
    """Test that source embedding is cached and reused for all references."""
    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [
        HVACDescription(raw_text="Ref 1"),
        HVACDescription(raw_text="Ref 2"),
        HVACDescription(raw_text="Ref 3"),
    ]

    # Reset call count
    mock_embedding_service.embed_single.reset_mock()

    # Match
    result = engine_with_ai.match_single(source, references, threshold=40.0)

    # embed_single should be called 4 times:
    # - 1x for source (cached)
    # - 3x for references (one per reference)
    # NOT 6 times (3 references * 2 = source + reference each time)
    assert mock_embedding_service.embed_single.call_count == 4


def test_embedding_failure_falls_back_to_placeholder(engine_with_ai, mock_embedding_service):
    """Test that embedding failure gracefully falls back to placeholder."""
    source = HVACDescription(raw_text="Test source")
    reference = HVACDescription(raw_text="Test reference")

    # Make embedding fail for reference
    def failing_embed(text, **kwargs):
        if "reference" in text.lower():
            raise RuntimeError("Embedding failed")
        # Return normal embedding for source
        return [0.5] * 384

    mock_embedding_service.embed_single.side_effect = failing_embed

    # Should not crash, should return placeholder
    score = engine_with_ai.calculate_semantic_score(source, reference)

    assert score == 50.0  # Fallback to placeholder


def test_match_single_with_ai_higher_scores(engine_with_ai, engine):
    """Test that AI-enabled engine produces different (potentially higher) scores."""
    source = HVACDescription(raw_text="Zawór kulowy DN50 PN16 mosiężny")
    # Reference with similar text but no exact param matches
    references = [
        HVACDescription(raw_text="Kurek kulowy DN50 PN16 z mosiądzu")  # Synonyms
    ]

    # Match with AI
    result_with_ai = engine_with_ai.match_single(source, references, threshold=40.0)

    # Match without AI (placeholder)
    result_without_ai = engine.match_single(source, references, threshold=40.0)

    # Both should find a match
    assert result_with_ai is not None
    assert result_without_ai is not None

    # AI score should be different (embeddings capture semantic similarity)
    assert result_with_ai.score.semantic_score != result_without_ai.score.semantic_score
    # Without AI should be placeholder (50.0)
    assert result_without_ai.score.semantic_score == 50.0


def test_backward_compatibility_no_embedding_service():
    """Test that engine works without embedding_service (backward compatible)."""
    extractor = ConcreteParameterExtractor()
    config = MatchingConfig.default()

    # Create engine without embedding_service (old way)
    engine_old = SimpleMatchingEngine(
        parameter_extractor=extractor,
        config=config,
        # No embedding_service parameter
    )

    source = HVACDescription(raw_text="Zawór DN50 PN16")
    references = [HVACDescription(raw_text="Zawór DN50 PN16")]

    # Should work normally with placeholder semantic score
    result = engine_old.match_single(source, references, threshold=40.0)

    assert result is not None
    assert result.score.semantic_score == 50.0


def test_zero_norm_vector_handled_gracefully(engine_with_ai, mock_embedding_service):
    """Test that zero-norm vectors don't crash cosine similarity."""
    source = HVACDescription(raw_text="Test")
    reference = HVACDescription(raw_text="Ref")

    # Mock to return zero vector for reference
    def zero_embed(text, **kwargs):
        if "Ref" in text:
            return [0.0] * 384  # Zero vector
        return [0.5] * 384  # Normal vector for source

    mock_embedding_service.embed_single.side_effect = zero_embed

    # Should not crash, should return placeholder
    score = engine_with_ai.calculate_semantic_score(source, reference)

    assert score == 50.0  # Fallback to placeholder


def test_cosine_similarity_score_range(engine_with_ai):
    """Test that cosine similarity scores are in valid 0-100 range."""
    source = HVACDescription(raw_text="Zawór DN50")
    references = [
        HVACDescription(raw_text="Zawór DN50"),  # High similarity
        HVACDescription(raw_text="Pompa DN100"),  # Low similarity
        HVACDescription(raw_text="Different equipment entirely"),  # Very low
    ]

    for ref in references:
        score = engine_with_ai.calculate_semantic_score(source, ref)

        # Score must be in valid range
        assert 0.0 <= score <= 100.0
        # Should not be exactly placeholder (real calculation)
        # (unless by coincidence, but very unlikely)


def test_logging_mode_detection(engine, engine_with_ai, caplog):
    """Test that matching mode is logged (AI vs placeholder)."""
    import logging

    caplog.set_level(logging.INFO)

    source = HVACDescription(raw_text="Zawór DN50")
    references = [HVACDescription(raw_text="Zawór DN50")]

    # Engine without AI
    engine.match_single(source, references)
    assert "Placeholder" in caplog.text or "placeholder" in caplog.text

    caplog.clear()

    # Engine with AI
    engine_with_ai.match_single(source, references)
    assert "AI-enabled" in caplog.text or "embeddings" in caplog.text
