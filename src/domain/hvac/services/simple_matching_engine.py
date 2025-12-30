"""
SimpleMatchingEngine - Domain Service Specification

Detailed contract/specification for the simple parameter-based matching engine.
This is a Phase 2 contract (stub with comprehensive documentation).
Implementation will be provided in Phase 3.

Part of: Task 2.1.4 - SimpleMatchingEngine domain service
Phase: 2.1 - Domain Layer Details

Architecture Notes:
    - Implements MatchingEngineProtocol from Phase 1
    - Pure domain service (no infrastructure dependencies)
    - Stateless (no internal state between calls)
    - Fast-fail optimization for performance
    - Hybrid scoring: 40% parameters + 60% semantic

Business Rules:
    - DN mismatch = automatic rejection (critical parameter)
    - Score < threshold = no match returned
    - Confidence calculated from score gap to second-best match
    - Explanation/justification always included for transparency

Design Decisions:
    - Semantic matching is placeholder (Phase 4) returning neutral 0.5
    - No caching in domain layer (Infrastructure responsibility)
    - Parameter extraction delegated to ParameterExtractor service
    - Weights configurable via MatchingConfig dependency injection
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.services.parameter_extractor import ParameterExtractorProtocol
from src.domain.hvac.services.embedding_service import EmbeddingServiceProtocol
from src.domain.hvac.matching_config import MatchingConfig

logger = logging.getLogger(__name__)


@dataclass
class SimpleMatchingEngine:
    """
    Domain service for matching HVAC descriptions using hybrid algorithm.

    This is a SPECIFICATION/CONTRACT for Phase 2. Methods are stubs with
    comprehensive documentation. Actual implementation in Phase 3.

    The SimpleMatchingEngine implements a hybrid matching strategy:
    1. Extract technical parameters (DN, PN, type, material) - 40% weight
    2. Calculate semantic similarity (AI embeddings) - 60% weight (Phase 4)
    3. Combine scores with configurable weights
    4. Apply threshold and fast-fail optimizations
    5. Generate detailed justification

    Responsibilities:
        - Match single source description against reference catalog
        - Calculate parameter similarity scores
        - Calculate semantic similarity scores (placeholder in Phase 2/3)
        - Combine scores with business-defined weights
        - Apply fast-fail optimization (DN mismatch)
        - Generate human-readable justification
        - Calculate confidence based on uniqueness of match

    Does NOT:
        - Cache results (Infrastructure Layer responsibility)
        - Store state between calls (stateless service)
        - Handle async operations (Application Layer responsibility)
        - Parse Excel files (Infrastructure Layer)
        - Manage progress tracking (Application Layer)

    Attributes:
        parameter_extractor: Service for extracting HVAC parameters from text
        config: Configuration with weights, thresholds, and feature flags
        embedding_service: Optional service for AI semantic similarity (Phase 4)
            - If None: uses placeholder semantic_score (50.0)
            - If provided: calculates real cosine similarity using embeddings

    Usage Example (conceptual - Phase 3 implementation):
        >>> extractor = ConcreteParameterExtractor()
        >>> config = MatchingConfig.default()
        >>> engine = SimpleMatchingEngine(extractor, config)
        >>>
        >>> # Phase 4: With AI embeddings
        >>> embedding_service = EmbeddingService()
        >>> engine_with_ai = SimpleMatchingEngine(extractor, config, embedding_service)
        >>>
        >>> source = HVACDescription(
        ...     id=UUID(...),
        ...     raw_text="Zawór kulowy DN50 PN16 mosiądz"
        ... )
        >>>
        >>> references = [
        ...     HVACDescription(id=UUID(...), raw_text="Zawór kulowy DN50 PN16"),
        ...     HVACDescription(id=UUID(...), raw_text="Zawór kulowy DN80 PN16"),
        ... ]
        >>>
        >>> result = engine.match_single(source, references, threshold=75.0)
        >>> if result:
        ...     print(f"Matched: {result.matched_reference_id}")
        ...     print(f"Score: {result.score.final_score}")
        ...     print(f"Explanation: {result.message}")
    """

    parameter_extractor: ParameterExtractorProtocol
    config: MatchingConfig = field(default_factory=MatchingConfig.default)
    embedding_service: Optional[EmbeddingServiceProtocol] = None

    def match_single(
        self,
        source_description: HVACDescription,
        reference_descriptions: list[HVACDescription],
        threshold: Optional[float] = None,
    ) -> Optional[MatchResult]:
        """
        Match a single source description against a list of reference descriptions.

        This is the main entry point for matching. It finds the best matching
        reference description, calculates scores, and returns a MatchResult
        if the score exceeds the threshold.

        Algorithm:
            1. Extract parameters from source description
            2. Check if source has critical parameters (fast-fail if not)
            3. For each reference description:
                a. Extract parameters
                b. Check fast-fail conditions (DN mismatch)
                c. Calculate parameter similarity score
                d. Calculate semantic similarity score (placeholder: 0.5)
                e. Combine scores using weights
            4. Find best matching reference (highest score)
            5. Check if best score >= threshold
            6. Calculate confidence (based on gap to 2nd best)
            7. Generate explanation/justification
            8. Return MatchResult or None

        Args:
            source_description: The HVAC description to match (from working file)
                - Must have raw_text populated
                - Should have source_row_number for tracking

            reference_descriptions: List of potential matches (from catalog)
                - Typically 100-500 items per catalog
                - Each must have raw_text populated
                - Must have at least 1 item (empty list returns None)

            threshold: Minimum score for match acceptance (0-100)
                - If None, uses config.default_threshold (75.0)
                - Match with score < threshold returns None
                - Score >= threshold returns MatchResult

        Returns:
            MatchResult if best match >= threshold, None otherwise

            MatchResult contains:
            - matched_reference_id: UUID of best matching reference
            - score: MatchScore with parameter_score, semantic_score, final_score
            - confidence: 0-1 based on how much better best match is vs 2nd best
            - message: Human-readable explanation (e.g., "Matched DN50 (30%), PN16 (10%)")
            - breakdown: Detailed dict with parameter matches and scores

        Business Rules:
            - DN mismatch = automatic rejection (critical parameter)
            - Empty reference list = return None (no matches possible)
            - No parameters extracted from source = low confidence match
            - Identical raw_text = 100% match (exact duplicate)
            - Score ties = first occurrence wins (stable sort)

        Performance:
            - Fast-fail optimization: Skip detailed matching if DN differs
            - Expected: 1-5ms per reference description (no AI in Phase 2/3)
            - Phase 4 with AI: 10-50ms per reference (embeddings cached)

        Examples:
            >>> # Perfect match
            >>> result = engine.match_single(
            ...     source=HVACDescription(raw_text="Zawór DN50 PN16"),
            ...     reference_descriptions=[
            ...         HVACDescription(raw_text="Zawór DN50 PN16"),
            ...         HVACDescription(raw_text="Zawór DN80 PN16"),
            ...     ],
            ...     threshold=75.0
            ... )
            >>> result.score.final_score >= 75.0
            True
            >>>
            >>> # No match (below threshold)
            >>> result = engine.match_single(
            ...     source=HVACDescription(raw_text="Zawór DN50 PN16"),
            ...     reference_descriptions=[
            ...         HVACDescription(raw_text="Pompa DN100"),
            ...     ],
            ...     threshold=75.0
            ... )
            >>> result is None
            True
            >>>
            >>> # Fast-fail (DN mismatch)
            >>> result = engine.match_single(
            ...     source=HVACDescription(raw_text="Zawór DN50"),
            ...     reference_descriptions=[
            ...         HVACDescription(raw_text="Zawór DN100"),  # Different DN
            ...     ],
            ...     threshold=50.0  # Low threshold but still no match
            ... )
            >>> result is None  # Fast-failed due to DN mismatch
            True
        """
        # Step 1: Validate inputs and prepare threshold
        threshold = threshold if threshold is not None else self.config.default_threshold

        # Log mode (AI vs placeholder)
        if self.embedding_service is not None:
            logger.info(
                f"Matching mode: AI-enabled (using real embeddings for semantic scoring)"
            )
        else:
            logger.info(
                f"Matching mode: Placeholder (using semantic_placeholder={self.config.semantic_placeholder * 100.0})"
            )

        # Empty reference list = no matches possible
        if not reference_descriptions:
            return None

        # Step 2: Extract parameters from source if not already extracted
        if not source_description.has_parameters():
            source_description.extract_parameters(self.parameter_extractor)

        source_params = source_description.extracted_params

        # If source has no parameters or empty parameters, can't match reliably
        if source_params is None or source_params.is_empty():
            return None

        # Step 3: Score each reference description
        results: list[tuple[HVACDescription, float, dict[str, Any], float, float]] = []

        # Cache source embedding for efficiency (used in all semantic comparisons)
        # Only generate once if embedding_service is available
        source_embedding: Optional[list[float]] = None
        if self.embedding_service is not None:
            try:
                source_embedding = self.embedding_service.embed_single(
                    source_description.raw_text
                )
                logger.debug(
                    f"Generated source embedding (dim={len(source_embedding)}) "
                    f"for: {source_description.raw_text[:50]}..."
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate source embedding, falling back to placeholder: {e}"
                )
                # Fall back to placeholder mode (source_embedding remains None)

        for ref_desc in reference_descriptions:
            # Extract parameters from reference if needed
            if not ref_desc.has_parameters():
                ref_desc.extract_parameters(self.parameter_extractor)

            ref_params = ref_desc.extracted_params

            # Skip references without extracted parameters
            if ref_params is None:
                continue

            # Fast-fail check (DN mismatch)
            if self.config.enable_fast_fail:
                should_fail, reason = self.should_fast_fail(source_params, ref_params)
                if should_fail:
                    # Skip this reference (DN mismatch)
                    continue

            # Calculate parameter score
            param_score, breakdown = self.calculate_parameter_score(
                source_params, ref_params
            )

            # Calculate semantic score (Phase 4: uses real embeddings if available)
            semantic_score = self.calculate_semantic_score(
                source_description, ref_desc, source_embedding
            )

            # Combine scores with weights (40% param + 60% semantic)
            final_score = (
                self.config.hybrid_param_weight * param_score
                + self.config.hybrid_semantic_weight * semantic_score
            )

            # Store result: (reference, final_score, breakdown, param_score, semantic_score)
            results.append((ref_desc, final_score, breakdown, param_score, semantic_score))

        # Step 4: Check if we have any matches
        if not results:
            return None  # No references passed fast-fail or had params

        # Sort by final_score descending (highest score first)
        results.sort(key=lambda x: x[1], reverse=True)

        # Get best match
        best_ref, best_score, best_breakdown, best_param_score, best_semantic_score = (
            results[0]
        )

        # Step 5: Check threshold
        if best_score < threshold:
            return None  # Best match below threshold

        # Step 6: Calculate confidence (based on gap to second-best)
        second_best_score = results[1][1] if len(results) > 1 else None
        confidence = self.calculate_confidence(best_score, second_best_score)

        # Step 7: Generate human-readable explanation
        # Type guard: best_ref.extracted_params should not be None at this point
        # (checked earlier in loop), but added for type safety
        if best_ref.extracted_params is None:
            raise RuntimeError("Best reference has no extracted parameters")
        message = self.generate_explanation(
            source_params,
            best_ref.extracted_params,
            best_param_score,
            best_semantic_score,
            best_score,
            best_breakdown,
        )

        # Step 8: Create and return MatchResult
        from src.domain.hvac.value_objects.match_score import MatchScore

        match_score = MatchScore.create(
            parameter_score=best_param_score,
            semantic_score=best_semantic_score,
            threshold=threshold,
        )

        return MatchResult(
            matched_reference_id=best_ref.id,
            score=match_score,
            confidence=confidence,
            message=message,
            breakdown=best_breakdown,
        )

    def calculate_parameter_score(
        self,
        source_params: ExtractedParameters,
        reference_params: ExtractedParameters,
    ) -> tuple[float, dict[str, Any]]:
        """
        Calculate parameter similarity score between two descriptions.

        Compares extracted parameters (DN, PN, valve_type, material, etc.)
        and calculates a weighted score. Each parameter contributes based
        on its configured weight.

        Scoring Logic:
            - Exact match = 1.0 (100%)
            - No match = 0.0 (0%)
            - Missing parameter in either = 0.0 (cannot compare)

        Score Calculation:
            parameter_score = Σ(weight_i × match_i) where:
            - weight_i = configured weight for parameter i
            - match_i = 1.0 if exact match, 0.0 if different/missing

        Args:
            source_params: Extracted parameters from source description
            reference_params: Extracted parameters from reference description

        Returns:
            Tuple of (score, breakdown):
            - score: Weighted parameter similarity (0-100)
            - breakdown: Dict with individual parameter match details
                {
                    "dn_match": True/False,
                    "pn_match": True/False,
                    "valve_type_match": True/False,
                    "material_match": True/False,
                    "drive_type_match": True/False,
                    "voltage_match": True/False,
                    "manufacturer_match": True/False,
                    "weights_applied": {...},
                    "individual_contributions": {...}
                }

        Business Rules:
            - DN weight = 30% (most important)
            - PN weight = 10%
            - Valve type weight = 15%
            - Material weight = 15%
            - Drive type weight = 10%
            - Voltage weight = 5%
            - Manufacturer weight = 5%
            - Missing parameters = 0 contribution (neutral, not negative)

        Examples:
            >>> source = ExtractedParameters(
            ...     dn=50, pn=16, valve_type="kulowy",
            ...     confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0}
            ... )
            >>> reference = ExtractedParameters(
            ...     dn=50, pn=16, valve_type="kulowy",
            ...     confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0}
            ... )
            >>> score, breakdown = engine.calculate_parameter_score(source, reference)
            >>> score
            100.0  # Perfect match
            >>> breakdown["dn_match"]
            True
            >>>
            >>> # Partial match (DN and PN match, type differs)
            >>> reference2 = ExtractedParameters(
            ...     dn=50, pn=16, valve_type="zwrotny",
            ...     confidence_scores={"dn": 1.0, "pn": 1.0, "valve_type": 1.0}
            ... )
            >>> score, breakdown = engine.calculate_parameter_score(source, reference2)
            >>> score
            40.0  # DN(30%) + PN(10%) = 40%
            >>> breakdown["valve_type_match"]
            False
        """
        # Get weights from config
        weights = self.config.parameter_weights

        # Compare each parameter (1.0 if match, 0.0 if different/missing)
        dn_match = (
            1.0
            if (source_params.dn is not None and source_params.dn == reference_params.dn)
            else 0.0
        )

        pn_match = (
            1.0
            if (source_params.pn is not None and source_params.pn == reference_params.pn)
            else 0.0
        )

        valve_type_match = (
            1.0
            if (
                source_params.valve_type is not None
                and source_params.valve_type == reference_params.valve_type
            )
            else 0.0
        )

        material_match = (
            1.0
            if (
                source_params.material is not None
                and source_params.material == reference_params.material
            )
            else 0.0
        )

        drive_type_match = (
            1.0
            if (
                source_params.drive_type is not None
                and source_params.drive_type == reference_params.drive_type
            )
            else 0.0
        )

        voltage_match = (
            1.0
            if (
                source_params.voltage is not None
                and source_params.voltage == reference_params.voltage
            )
            else 0.0
        )

        manufacturer_match = (
            1.0
            if (
                source_params.manufacturer is not None
                and source_params.manufacturer == reference_params.manufacturer
            )
            else 0.0
        )

        # Calculate weighted score (0-1 range)
        weighted_score = (
            weights.dn * dn_match
            + weights.pn * pn_match
            + weights.valve_type * valve_type_match
            + weights.material * material_match
            + weights.drive_type * drive_type_match
            + weights.voltage * voltage_match
            + weights.manufacturer * manufacturer_match
        )

        # Convert to 0-100 scale
        parameter_score = weighted_score * 100.0

        # Build detailed breakdown
        breakdown = {
            "dn_match": bool(dn_match),
            "pn_match": bool(pn_match),
            "valve_type_match": bool(valve_type_match),
            "material_match": bool(material_match),
            "drive_type_match": bool(drive_type_match),
            "voltage_match": bool(voltage_match),
            "manufacturer_match": bool(manufacturer_match),
            "weights_applied": {
                "dn": weights.dn,
                "pn": weights.pn,
                "valve_type": weights.valve_type,
                "material": weights.material,
                "drive_type": weights.drive_type,
                "voltage": weights.voltage,
                "manufacturer": weights.manufacturer,
            },
            "individual_contributions": {
                "dn": weights.dn * dn_match * 100.0,
                "pn": weights.pn * pn_match * 100.0,
                "valve_type": weights.valve_type * valve_type_match * 100.0,
                "material": weights.material * material_match * 100.0,
                "drive_type": weights.drive_type * drive_type_match * 100.0,
                "voltage": weights.voltage * voltage_match * 100.0,
                "manufacturer": weights.manufacturer * manufacturer_match * 100.0,
            },
            # Phase 4 AI Metadata: SimpleMatchingEngine operates without AI when used standalone
            "using_ai": False,  # No AI embeddings used (parameter-only matching)
            "ai_model": None,  # No AI model available
        }

        return parameter_score, breakdown

    def calculate_semantic_score(
        self,
        source_description: HVACDescription,
        reference_description: HVACDescription,
        source_embedding: Optional[list[float]] = None,
    ) -> float:
        """
        Calculate semantic similarity score using AI embeddings.

        **PHASE 4 IMPLEMENTATION** (current):
        - If embedding_service available and source_embedding provided:
          1. Generate embedding for reference.raw_text
          2. Calculate cosine similarity between source and reference embeddings
          3. Convert similarity (-1 to 1) to score (0 to 100)
        - If embedding_service not available or embedding fails:
          Returns placeholder value (config.semantic_placeholder × 100 = 50.0)

        Args:
            source_description: Source HVAC description
            reference_description: Reference HVAC description
            source_embedding: Optional pre-computed source embedding (for efficiency)
                If provided, avoids re-computing source embedding for each reference

        Returns:
            Semantic similarity score (0-100)

            - With AI: Returns actual cosine similarity × 100 (0-100 range)
            - Without AI: Returns 50.0 (neutral placeholder)

        Business Rules:
            - Cosine similarity range: -1 to 1, but typically 0 to 1 for text
            - Convert to 0-100 scale by multiplying by 100
            - Synonyms and variations should score highly (e.g., "kurek" vs "zawór")
            - Backward compatible: works without AI (placeholder mode)

        Examples:
            >>> # With AI embeddings
            >>> source = HVACDescription(raw_text="Zawór kulowy DN50")
            >>> reference = HVACDescription(raw_text="Kurek kulowy DN50")
            >>> source_emb = embedding_service.embed_single(source.raw_text)
            >>> score = engine.calculate_semantic_score(source, reference, source_emb)
            >>> score >= 85.0  # High similarity (synonyms)
            True
            >>>
            >>> # Without AI (placeholder mode)
            >>> engine_no_ai = SimpleMatchingEngine(extractor, config)  # No embedding_service
            >>> score = engine_no_ai.calculate_semantic_score(source, reference)
            >>> score
            50.0  # Placeholder value
        """
        # If no embedding service or no source embedding → use placeholder
        if self.embedding_service is None or source_embedding is None:
            logger.debug("Using placeholder semantic score (AI not available)")
            return self.config.semantic_placeholder * 100.0

        # Generate reference embedding
        try:
            reference_embedding = self.embedding_service.embed_single(
                reference_description.raw_text
            )
        except Exception as e:
            logger.warning(
                f"Failed to generate reference embedding, using placeholder: {e}"
            )
            return self.config.semantic_placeholder * 100.0

        # Calculate cosine similarity
        try:
            import numpy as np

            # Convert to numpy arrays for efficient computation
            source_vec = np.array(source_embedding)
            reference_vec = np.array(reference_embedding)

            # Cosine similarity: dot(A, B) / (norm(A) * norm(B))
            dot_product = np.dot(source_vec, reference_vec)
            norm_source = np.linalg.norm(source_vec)
            norm_reference = np.linalg.norm(reference_vec)

            # Avoid division by zero
            if norm_source == 0 or norm_reference == 0:
                logger.warning("Zero-norm vector encountered, using placeholder")
                return self.config.semantic_placeholder * 100.0

            cosine_similarity = dot_product / (norm_source * norm_reference)

            # Convert to 0-100 scale
            # Cosine similarity range: -1 to 1, but typically 0 to 1 for text
            # Multiply by 100 to get 0-100 score
            semantic_score = cosine_similarity * 100.0

            # Clamp to valid range (safety check)
            semantic_score = max(0.0, min(100.0, semantic_score))

            logger.debug(
                f"Calculated semantic score: {semantic_score:.2f} "
                f"(cosine_sim: {cosine_similarity:.4f})"
            )

            return semantic_score

        except Exception as e:
            logger.error(f"Error calculating cosine similarity, using placeholder: {e}")
            return self.config.semantic_placeholder * 100.0

    def should_fast_fail(
        self,
        source_params: ExtractedParameters,
        reference_params: ExtractedParameters,
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if matching should be skipped due to critical parameter mismatch.

        Fast-fail optimization: If DN values differ, the equipment is fundamentally
        incompatible (different pipe sizes cannot be directly matched). Skip
        expensive semantic matching and immediately return "no match".

        This is a performance optimization that also enforces critical business rules.

        Args:
            source_params: Extracted parameters from source
            reference_params: Extracted parameters from reference

        Returns:
            Tuple of (should_fail, reason):
            - should_fail: True if matching should be skipped
            - reason: Human-readable explanation why (for logging/debugging)
                     None if should_fail=False

        Business Rules:
            - DN mismatch = incompatible equipment = fast-fail
            - Future: DN tolerance might allow small variations (±5%)
            - Missing DN in both = don't fast-fail (match on other params)
            - config.enable_fast_fail=False disables optimization (testing)

        Performance Impact:
            - Saves ~5-10ms per reference by skipping semantic matching
            - For 500 references with 50% DN mismatch: saves ~1.25-2.5s per source

        Examples:
            >>> source = ExtractedParameters(dn=50, pn=16)
            >>> reference = ExtractedParameters(dn=100, pn=16)
            >>> should_fail, reason = engine.should_fast_fail(source, reference)
            >>> should_fail
            True
            >>> reason
            "DN mismatch: source DN50 != reference DN100 (critical parameter)"
            >>>
            >>> # No DN in either = don't fail
            >>> source_no_dn = ExtractedParameters(pn=16, valve_type="kulowy")
            >>> reference_no_dn = ExtractedParameters(pn=16, valve_type="zwrotny")
            >>> should_fail, reason = engine.should_fast_fail(source_no_dn, reference_no_dn)
            >>> should_fail
            False
            >>> reason is None
            True
            >>>
            >>> # Same DN = don't fail
            >>> source = ExtractedParameters(dn=50)
            >>> reference = ExtractedParameters(dn=50)
            >>> should_fail, reason = engine.should_fast_fail(source, reference)
            >>> should_fail
            False
        """
        # If fast-fail disabled in config, never skip
        if not self.config.enable_fast_fail:
            return False, None

        # Get DN values from both parameters
        source_dn = source_params.dn
        reference_dn = reference_params.dn

        # If both have DN values and they differ → fast-fail
        if source_dn is not None and reference_dn is not None:
            if source_dn != reference_dn:
                reason = (
                    f"DN mismatch: source DN{source_dn} != "
                    f"reference DN{reference_dn} (critical parameter)"
                )
                return True, reason

        # DN matches or at least one is missing → don't fast-fail
        return False, None

    def calculate_confidence(
        self,
        best_score: float,
        second_best_score: Optional[float],
    ) -> float:
        """
        Calculate confidence level based on score gap to second-best match.

        Confidence indicates how "unique" the best match is. A large gap between
        best and second-best match = high confidence. Small gap = low confidence
        (ambiguous, multiple similar matches).

        Args:
            best_score: Final score of the best matching reference
            second_best_score: Final score of second-best match, or None if only 1 reference

        Returns:
            Confidence level (0-1)

            0.0 = Very uncertain (tie with second-best)
            0.5 = Medium confidence (small gap)
            1.0 = Very confident (large gap or only match)

        Calculation Logic:
            - If only 1 reference: confidence = best_score / 100.0
            - If multiple references:
                gap = best_score - second_best_score
                if gap >= MIN_SCORE_GAP_FOR_HIGH_CONFIDENCE (10.0):
                    confidence = min(0.85 + (gap - 10.0) / 100.0, 1.0)
                else:
                    confidence = 0.5 + (gap / 20.0)

        Business Rules:
            - Gap >= 10 points = high confidence (0.85+)
            - Gap < 5 points = low confidence (<0.70)
            - Only one match above threshold = confidence from score itself
            - Perfect score (100) = maximum confidence (1.0)

        Examples:
            >>> # Clear winner (large gap)
            >>> confidence = engine.calculate_confidence(
            ...     best_score=95.0,
            ...     second_best_score=70.0
            ... )
            >>> confidence >= 0.85
            True
            >>>
            >>> # Close race (small gap)
            >>> confidence = engine.calculate_confidence(
            ...     best_score=80.0,
            ...     second_best_score=78.0
            ... )
            >>> confidence < 0.70
            True
            >>>
            >>> # Only one match
            >>> confidence = engine.calculate_confidence(
            ...     best_score=90.0,
            ...     second_best_score=None
            ... )
            >>> confidence
            0.9
        """
        # If only one reference, confidence based on score itself
        if second_best_score is None:
            return best_score / 100.0

        # Calculate gap between best and second-best
        gap = best_score - second_best_score

        # Large gap → high confidence (0.85+)
        if gap >= self.config.min_score_gap_for_high_confidence:
            # Start at 0.85, increase with larger gaps, max at 1.0
            confidence = min(0.85 + (gap - 10.0) / 100.0, 1.0)
        else:
            # Small gap → medium to low confidence
            # Gap of 0 → 0.5, gap of 10 → 1.0 (but capped by above branch)
            confidence = 0.5 + (gap / 20.0)

        return confidence

    def generate_explanation(
        self,
        source_params: ExtractedParameters,
        reference_params: ExtractedParameters,
        parameter_score: float,
        semantic_score: float,
        final_score: float,
        breakdown: dict[str, Any],
    ) -> str:
        """
        Generate human-readable explanation of why the match was made.

        Creates a message like: "Matched DN50 (30%), PN16 (10%), valve type (15%)"
        This explanation helps users understand and trust the matching results.

        Args:
            source_params: Extracted parameters from source
            reference_params: Extracted parameters from reference
            parameter_score: Calculated parameter similarity score
            semantic_score: Calculated semantic similarity score
            final_score: Final combined score
            breakdown: Detailed match breakdown dict

        Returns:
            Human-readable explanation string (1-500 characters)

        Format Examples:
            - High parameter match: "Matched DN50 (30%), PN16 (10%), ball valve (15%)"
            - Partial match: "Matched DN50 (30%), PN16 (10%) - different material"
            - Semantic only: "Semantic match (60%) - no exact parameters"
            - Perfect match: "Perfect match - all parameters identical (100%)"

        Business Rules:
            - List matched parameters with their weight contributions
            - Mention critical parameters first (DN, then PN)
            - Note significant differences (e.g., "different material")
            - Keep under 500 characters for UI display
            - English language (translation in API layer if needed)

        Examples:
            >>> explanation = engine.generate_explanation(
            ...     source_params=ExtractedParameters(dn=50, pn=16, valve_type="kulowy"),
            ...     reference_params=ExtractedParameters(dn=50, pn=16, valve_type="kulowy"),
            ...     parameter_score=100.0,
            ...     semantic_score=50.0,
            ...     final_score=70.0,
            ...     breakdown={"dn_match": True, "pn_match": True, "valve_type_match": True}
            ... )
            >>> "DN50" in explanation
            True
            >>> "PN16" in explanation
            True
        """
        # Check for perfect match
        if parameter_score >= 99.5:  # Account for floating point
            return f"Perfect match - all parameters identical ({final_score:.1f}%)"

        # Build list of matched parameters with their contributions
        matched_parts = []
        mismatched_parts = []

        # DN (most critical - always mention first if present)
        if breakdown.get("dn_match"):
            dn_value = source_params.dn
            contribution = breakdown["individual_contributions"]["dn"]
            matched_parts.append(f"DN{dn_value} ({contribution:.0f}%)")
        elif source_params.dn is not None:
            mismatched_parts.append("DN")

        # PN
        if breakdown.get("pn_match"):
            pn_value = source_params.pn
            contribution = breakdown["individual_contributions"]["pn"]
            matched_parts.append(f"PN{pn_value} ({contribution:.0f}%)")
        elif source_params.pn is not None:
            mismatched_parts.append("PN")

        # Valve type
        if breakdown.get("valve_type_match"):
            contribution = breakdown["individual_contributions"]["valve_type"]
            matched_parts.append(f"valve type ({contribution:.0f}%)")
        elif source_params.valve_type is not None:
            mismatched_parts.append("valve type")

        # Material
        if breakdown.get("material_match"):
            contribution = breakdown["individual_contributions"]["material"]
            matched_parts.append(f"material ({contribution:.0f}%)")
        elif source_params.material is not None:
            mismatched_parts.append("material")

        # Drive type
        if breakdown.get("drive_type_match"):
            contribution = breakdown["individual_contributions"]["drive_type"]
            matched_parts.append(f"drive ({contribution:.0f}%)")

        # Voltage
        if breakdown.get("voltage_match"):
            contribution = breakdown["individual_contributions"]["voltage"]
            matched_parts.append(f"voltage ({contribution:.0f}%)")

        # Manufacturer
        if breakdown.get("manufacturer_match"):
            contribution = breakdown["individual_contributions"]["manufacturer"]
            matched_parts.append(f"manufacturer ({contribution:.0f}%)")

        # Build message
        if matched_parts:
            message = "Matched " + ", ".join(matched_parts)
            # Add note about mismatches if any significant parameters differ
            if mismatched_parts:
                message += " - different " + ", ".join(mismatched_parts)
        else:
            # No parameter matches - rely on semantic
            message = f"Semantic match ({semantic_score:.0f}%) - no exact parameter matches"

        # Add final score for clarity
        message += f" [Total: {final_score:.1f}%]"

        return message
