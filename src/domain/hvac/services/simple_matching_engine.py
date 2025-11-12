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

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import UUID

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.value_objects.match_result import MatchResult
from src.domain.hvac.value_objects.extracted_parameters import ExtractedParameters
from src.domain.hvac.services.parameter_extractor import ParameterExtractorProtocol
from src.domain.hvac.matching_config import MatchingConfig


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

    Usage Example (conceptual - Phase 3 implementation):
        >>> extractor = ConcreteParameterExtractor()
        >>> config = MatchingConfig.default()
        >>> engine = SimpleMatchingEngine(extractor, config)
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
        # Phase 3 implementation placeholder
        ...

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
        # Phase 3 implementation placeholder
        ...

    def calculate_semantic_score(
        self,
        source_description: HVACDescription,
        reference_description: HVACDescription,
    ) -> float:
        """
        Calculate semantic similarity score using AI embeddings.

        **PHASE 2/3 PLACEHOLDER**: This method returns a neutral fixed value (0.5)
        because AI integration happens in Phase 4. The contract is defined now
        to prepare for future implementation.

        **PHASE 4 IMPLEMENTATION** (future):
        1. Generate embedding for source.raw_text
        2. Generate embedding for reference.raw_text
        3. Calculate cosine similarity between embeddings
        4. Convert similarity (-1 to 1) to score (0 to 100)

        Args:
            source_description: Source HVAC description
            reference_description: Reference HVAC description

        Returns:
            Semantic similarity score (0-100)

            Phase 2/3: Always returns 50.0 (neutral placeholder)
            Phase 4: Returns actual cosine similarity × 100

        Business Rules:
            - Phase 2/3: Return config.semantic_placeholder × 100 (= 50.0)
            - Phase 4: Use paraphrase-multilingual-mpnet-base-v2 model
            - Phase 4: Cache embeddings in Infrastructure Layer (Redis)
            - Synonyms and variations should score highly (e.g., "kurek" vs "zawór")

        Examples:
            >>> # Phase 2/3 behavior (placeholder)
            >>> source = HVACDescription(raw_text="Zawór kulowy DN50")
            >>> reference = HVACDescription(raw_text="Kurek kulowy DN50")
            >>> score = engine.calculate_semantic_score(source, reference)
            >>> score
            50.0  # Fixed placeholder value
            >>>
            >>> # Phase 4 behavior (future - not implemented yet)
            >>> # source = HVACDescription(raw_text="Zawór kulowy DN50")
            >>> # reference = HVACDescription(raw_text="Kurek kulowy DN50")
            >>> # score = engine.calculate_semantic_score(source, reference)
            >>> # score would be ~95.0 (synonyms, high similarity)
        """
        # Phase 2/3: Return placeholder
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
        # Phase 3 implementation placeholder
        ...

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
        # Phase 3 implementation placeholder
        ...

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
        # Phase 3 implementation placeholder
        ...
