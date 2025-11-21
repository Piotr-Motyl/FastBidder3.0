"""
MatchingEngine Protocol

Core business service protocol for matching HVAC descriptions.
Defines the contract for parameter + semantic matching orchestration.

Responsibility:
    - Define matching algorithm contract (Protocol-based interface)
    - Coordinate parameter and semantic matching
    - Calculate hybrid scores (40% param + 60% semantic)
    - Return best match above threshold or None

Architecture Notes:
    - Domain Service (business logic that doesn't fit in entities)
    - Protocol interface (structural typing for Dependency Injection)
    - Async interface (for future AI API calls, embeddings generation)
    - Implementation in Infrastructure layer (depends on external libraries)
    - Phase 1: High-level contract only (no implementation)

Why Protocol instead of ABC:
    - Better for Dependency Injection (structural typing)
    - Consistent with Repository pattern (also uses Protocol)
    - No need to inherit from base class in Infrastructure
    - More flexible for testing (easy mocking)
"""

from typing import Optional, Protocol

from ..entities.hvac_description import HVACDescription
from ..value_objects.match_result import MatchResult


class MatchingEngineProtocol(Protocol):
    """
    Protocol defining the contract for HVAC description matching.

    This is a Domain Service that encapsulates the core matching business logic.
    The actual implementation will be in the Infrastructure layer because it
    depends on external libraries (regex, sentence-transformers, etc.).

    Matching Algorithm (Hybrid Approach):
    1. Parameter Matching (40% weight):
       - Extract and compare DN, PN, material, valve type, etc.
       - Exact match scoring: DN50 = DN50 (100%), DN50 ≠ DN100 (0%)
       - Guarantees technical accuracy

    2. Semantic Matching (60% weight):
       - Generate embeddings using sentence-transformers
       - Calculate cosine similarity between embeddings
       - Handles synonyms, typos, variations
       - Example: "napęd" = "siłownik" (semantic similarity)

    3. Final Score:
       - Weighted average: 0.4 * param_score + 0.6 * semantic_score
       - Only return match if final_score >= threshold
       - Return None if no match above threshold

    Business Rules:
        - Threshold is configurable per request (default 75%)
        - Only DN, PN, material, type are used for parameter matching (Phase 1)
        - Semantic model: paraphrase-multilingual-mpnet-base-v2 (Phase 4)
        - If multiple matches above threshold, return the highest scoring one

    Examples:
        >>> # Usage in Application Layer (ProcessMatchingUseCase)
        >>> engine: MatchingEngineProtocol = get_matching_engine()  # DI from Infrastructure
        >>> result = await engine.match(
        ...     working_description=wf_desc,
        ...     reference_descriptions=ref_catalog,
        ...     threshold=75.0
        ... )
        >>> if result:
        ...     print(f"Match found: {result.score.final_score}%")
        ...     print(f"Matched item: {result.matched_item_id}")
        ... else:
        ...     print("No match found above threshold")

    Phase 1 Scope:
        - Protocol contract only (no implementation)
        - Two async methods: match() and calculate_confidence()
        - Returns MatchResult or None
        - Implementation deferred to Infrastructure layer (Phase 3)

    Future Enhancements (Phase 4+):
        - Batch matching for performance
        - Caching of embeddings (Redis)
        - Multiple matching strategies (strict, fuzzy, semantic-only)
        - Confidence scoring based on second-best match gap
    """

    async def match(
        self,
        working_description: HVACDescription,
        reference_descriptions: list[HVACDescription],
        threshold: float = 75.0,
    ) -> Optional[MatchResult]:
        """
        Find the best matching reference description for a working description.

        This method orchestrates the hybrid matching algorithm:
        1. For each reference description:
           a. Calculate parameter match score (0-100)
           b. Calculate semantic similarity score (0-100)
           c. Combine using weights: 0.4 * param + 0.6 * semantic
        2. Select reference with highest final score
        3. Calculate confidence using calculate_confidence() method
        4. If score >= threshold, return MatchResult (with confidence)
        5. If score < threshold, return None (no valid match)

        Implementation Requirements:
            The concrete implementation MUST:
            1. Find the best match using hybrid scoring algorithm
            2. Call calculate_confidence() to compute confidence level
            3. Return MatchResult with all fields properly populated:
               - matched_item_id (UUID of best match)
               - score (MatchScore with param/semantic/final)
               - confidence (from calculate_confidence() method)
               - message (human-readable explanation)
               - breakdown (structured debugging details)

        Args:
            working_description: Description to be matched (from WF file)
                - Must have extracted_params populated
                - Must have valid raw_text

            reference_descriptions: Catalog of potential matches (from REF file)
                - Each must have extracted_params populated
                - Typically 100-400 descriptions (Phase 1 limit)

            threshold: Minimum score for valid match (default 75.0)
                - Range: 0-100
                - Configurable per request
                - Business rule: matches below threshold are rejected

        Returns:
            MatchResult if best match >= threshold, None otherwise

            MatchResult contains:
            - matched_item_id: UUID of the best matching reference
            - score: MatchScore with parameter/semantic/final scores
            - confidence: 0-1 (calculated using calculate_confidence())
            - message: Human-readable explanation
            - breakdown: Structured debugging details

        Raises:
            DomainException: If business rules are violated
                - working_description has no parameters
                - reference_descriptions is empty
                - threshold out of range [0, 100]

        Examples:
            >>> # High confidence match
            >>> wf_desc = HVACDescription(
            ...     raw_text="Zawór kulowy DN50 PN16 mosiężny",
            ...     extracted_params=ExtractedParameters(dn=50, pn=16, material='brass')
            ... )
            >>> ref_catalog = [
            ...     HVACDescription(
            ...         raw_text="Zawór kulowy mosiężny DN50 PN16 z siłownikiem",
            ...         extracted_params=ExtractedParameters(dn=50, pn=16, material='brass')
            ...     ),
            ...     # ... more references
            ... ]
            >>>
            >>> engine: MatchingEngineProtocol = ConcreteMatchingEngine()  # Infrastructure
            >>> result = await engine.match(wf_desc, ref_catalog, threshold=75.0)
            >>>
            >>> if result:
            ...     print(f"Match found: {result.score.final_score}%")
            ...     print(f"Confidence: {result.confidence}")
            ...     print(f"Message: {result.message}")
            ...     # Output: "Match found: 95.2%"
            ...     #         "Confidence: 0.95"
            ...     #         "Message: High confidence match - exact DN, PN, material"
            ... else:
            ...     print("No match above threshold")

            >>> # No match scenario (DN mismatch)
            >>> wf_desc = HVACDescription(
            ...     raw_text="Zawór DN100",
            ...     extracted_params=ExtractedParameters(dn=100)
            ... )
            >>> ref_catalog = [
            ...     HVACDescription(
            ...         raw_text="Zawór DN50",
            ...         extracted_params=ExtractedParameters(dn=50)
            ...     )
            ... ]
            >>> result = await engine.match(wf_desc, ref_catalog, threshold=75.0)
            >>> result is None
            True  # No match because DN mismatch lowers score below threshold

        Implementation Notes (for Infrastructure layer):
            - Use ParameterExtractor to ensure parameters are extracted
            - Generate embeddings lazily (cache in Redis if possible)
            - Calculate all scores in parallel for performance
            - MUST call calculate_confidence() to populate confidence field
            - Log matching details for debugging
            - Handle edge cases (empty parameters, identical scores)
        """
        ...

    async def calculate_confidence(
        self, best_score: float, second_best_score: Optional[float]
    ) -> float:
        """
        Calculate confidence level based on score gap to second-best match.

        This method is called by match() to populate the confidence field in MatchResult.
        Confidence indicates how certain we are that the top match is correct:
        - High confidence (0.9-1.0): Clear winner, large gap to second place
        - Medium confidence (0.7-0.9): Good match, moderate gap
        - Low confidence (0.5-0.7): Weak match, small gap or only one candidate

        Formula (example - implementation can vary):
            if only one candidate: confidence = 0.5
            else: confidence = min(1.0, score_gap / 20.0)
            where score_gap = best_score - second_best_score

        Args:
            best_score: Final score of the best match (0-100)
            second_best_score: Final score of second-best match, or None if only one candidate

        Returns:
            Confidence level (0-1)

        Examples:
            >>> engine: MatchingEngineProtocol = ConcreteMatchingEngine()
            >>>
            >>> # Large gap → high confidence
            >>> confidence = await engine.calculate_confidence(95.0, 70.0)
            >>> confidence
            1.0  # Gap of 25 points → very confident

            >>> # Small gap → lower confidence
            >>> confidence = await engine.calculate_confidence(76.0, 74.5)
            >>> confidence
            0.075  # Gap of 1.5 points → not very confident

            >>> # Only one candidate → medium confidence
            >>> confidence = await engine.calculate_confidence(85.0, None)
            >>> confidence
            0.5  # No comparison possible

        Usage in match() implementation:
            >>> # Inside Infrastructure implementation of match()
            >>> scores = [calculate_final_score(wf, ref) for ref in references]
            >>> sorted_scores = sorted(scores, reverse=True)
            >>>
            >>> best_score = sorted_scores[0]
            >>> second_best = sorted_scores[1] if len(sorted_scores) > 1 else None
            >>>
            >>> confidence = await self.calculate_confidence(best_score, second_best)
            >>>
            >>> return MatchResult(
            ...     matched_item_id=best_match.id,
            ...     score=best_score,
            ...     confidence=confidence,  # ← Populated using this method
            ...     message="...",
            ...     breakdown={...}
            ... )
        """
        ...
