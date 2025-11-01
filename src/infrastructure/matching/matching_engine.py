"""
Concrete Matching Engine Implementation

Implements MatchingEngineProtocol from Domain Layer.
Provides hybrid matching: 40% parameter + 60% semantic similarity.

Responsibility:
    - Implement Domain matching interface
    - Parameter extraction and comparison (DN, PN, material, type)
    - Semantic similarity using embeddings (Phase 4)
    - Hybrid scoring algorithm (weighted average)
    - Confidence calculation

Architecture Notes:
    - Infrastructure Layer (depends on external libraries)
    - Implements Domain MatchingEngineProtocol (explicit inheritance)
    - Phase 1: Contract skeleton only
    - Phase 3: Parameter matching implementation
    - Phase 4: Semantic matching with AI embeddings
"""

from typing import Optional

from src.domain.hvac import (
    HVACDescription,
    MatchingEngineProtocol,
    MatchResult,
)


class ConcreteMatchingEngine(MatchingEngineProtocol):
    """
    Hybrid matching engine implementation.

    Explicitly implements MatchingEngineProtocol from Domain Layer.
    This provides better type checking and IDE support.

    This is the core business logic implementation that combines:
    1. Parameter Matching (40% weight):
       - Exact match for DN, PN, material, valve type
       - Regex-based parameter extraction
       - Dictionary-based normalization (synonyms)

    2. Semantic Matching (60% weight):
       - Sentence embeddings (paraphrase-multilingual-mpnet-base-v2)
       - Cosine similarity between embeddings
       - Handles typos, variations, synonyms

    3. Confidence Calculation:
       - Based on score gap to second-best match
       - Higher gap = higher confidence

    Phase 1 Scope:
        - Contract skeleton with NotImplementedError
        - Type hints and docstrings complete
        - Explicit Protocol inheritance

    Phase 3 Implementation:
        - Parameter extraction (regex patterns for DN, PN)
        - Parameter matching logic
        - Basic scoring (parameter-only, no semantics)

    Phase 4 Implementation:
        - Embedding generation (sentence-transformers)
        - Semantic similarity calculation
        - Full hybrid scoring (40/60 weights)
        - Confidence calculation with gap analysis

    Examples:
        >>> # Phase 1: Contract only
        >>> engine = ConcreteMatchingEngine()
        >>> # Phase 3: Will work with parameter matching
        >>> # Phase 4: Will work with full hybrid matching
    """

    def __init__(self) -> None:
        """
        Initialize matching engine with dependencies.

        Phase 1: No dependencies (skeleton only)
        Phase 3: Add parameter extractor, normalizers
        Phase 4: Add embedding model, vector store

        Future initialization (Phase 4):
            - Load sentence-transformer model
            - Initialize parameter extraction patterns
            - Load synonym dictionaries
            - Setup caching for embeddings
        """
        pass

    async def match(
        self,
        working_description: HVACDescription,
        reference_descriptions: list[HVACDescription],
        threshold: float = 75.0,
    ) -> Optional[MatchResult]:
        """
        Find best matching reference for working description (implements Protocol).

        Hybrid Algorithm:
        1. For each reference:
           a. Calculate parameter score (0-100)
              - Extract DN, PN, material, type from both descriptions
              - Compare: DN match = 100% or 0%, PN match = 100% or 0%, etc.
              - Average all parameter scores
           b. Calculate semantic score (0-100)
              - Generate embeddings for both descriptions
              - Calculate cosine similarity (0-1) → convert to 0-100
           c. Calculate final score:
              - final = 0.4 * param_score + 0.6 * semantic_score
        2. Sort by final score (descending)
        3. If best score >= threshold:
           a. Calculate confidence using score gap
           b. Return MatchResult with all details
        4. Else: Return None (no match above threshold)

        Args:
            working_description: Description to be matched (from WF file)
            reference_descriptions: Catalog of potential matches (from REF file)
            threshold: Minimum score for valid match (default 75.0)

        Returns:
            MatchResult if match found, None otherwise

        Raises:
            ValueError: If threshold out of range or inputs invalid

        Phase 1: NotImplementedError (contract only)
        Phase 3: Parameter matching only (semantic=0)
        Phase 4: Full hybrid matching
        """
        raise NotImplementedError(
            "match() to be implemented in Phase 3 (parameter matching) "
            "and Phase 4 (semantic matching). "
            "Will implement hybrid algorithm: "
            "1. Extract parameters from descriptions "
            "2. Calculate parameter similarity scores "
            "3. Generate embeddings and calculate semantic similarity "
            "4. Combine using 40/60 weights "
            "5. Select best match above threshold "
            "6. Calculate confidence and return MatchResult"
        )

    async def calculate_confidence(
        self, best_score: float, second_best_score: Optional[float]
    ) -> float:
        """
        Calculate confidence based on score gap (implements Protocol).

        Confidence Formula:
            if second_best is None:
                confidence = 0.5  # Only one candidate
            else:
                gap = best_score - second_best_score
                confidence = min(1.0, gap / 20.0)  # 20 point gap = 100% confidence

        Examples:
            - Gap of 25 points: confidence = 1.0 (very confident)
            - Gap of 10 points: confidence = 0.5 (moderate)
            - Gap of 1 point: confidence = 0.05 (low)
            - Only 1 candidate: confidence = 0.5 (uncertain)

        Args:
            best_score: Final score of best match (0-100)
            second_best_score: Final score of second match, or None

        Returns:
            Confidence level (0-1)

        Phase 1: NotImplementedError (contract only)
        Phase 3: Basic implementation
        """
        raise NotImplementedError(
            "calculate_confidence() to be implemented in Phase 3. "
            "Will calculate confidence based on score gap: "
            "confidence = min(1.0, (best - second_best) / 20.0)"
        )

    async def _extract_parameters(self, description: HVACDescription) -> dict:
        """
        Extract HVAC parameters from description text (helper method).

        Phase 3 Implementation:
            - Regex patterns for DN: DN\s?(\d+), (\d+)\s?mm
            - Regex patterns for PN: PN\s?(\d+), (\d+)\s?bar
            - Dictionary lookup for materials: brass, steel, stainless, etc.
            - Dictionary lookup for valve types: ball, check, butterfly, etc.

        Args:
            description: HVACDescription entity

        Returns:
            Dict with extracted parameters:
            {
                'dn': 50,
                'pn': 16,
                'material': 'brass',
                'valve_type': 'ball'
            }

        Phase 1: NotImplementedError
        """
        raise NotImplementedError(
            "_extract_parameters() to be implemented in Phase 3. "
            "Will use regex patterns to extract DN, PN, material, valve type."
        )

    async def _calculate_parameter_score(
        self, wf_params: dict, ref_params: dict
    ) -> float:
        """
        Calculate parameter matching score (helper method).

        Phase 3 Implementation:
            For each parameter (DN, PN, material, type):
            - If both have parameter and values match exactly: 100%
            - If both have parameter but values differ: 0%
            - If one missing: skip (don't penalize)
            Average all matched parameters

        Args:
            wf_params: Parameters from working description
            ref_params: Parameters from reference description

        Returns:
            Parameter match score (0-100)

        Phase 1: NotImplementedError
        """
        raise NotImplementedError(
            "_calculate_parameter_score() to be implemented in Phase 3. "
            "Will compare extracted parameters and calculate exact match score."
        )

    async def _calculate_semantic_score(self, wf_text: str, ref_text: str) -> float:
        """
        Calculate semantic similarity score (helper method).

        Phase 4 Implementation:
            1. Generate embeddings using sentence-transformers
               Model: paraphrase-multilingual-mpnet-base-v2
            2. Calculate cosine similarity (0-1)
            3. Convert to 0-100 scale

        Args:
            wf_text: Working description raw text
            ref_text: Reference description raw text

        Returns:
            Semantic similarity score (0-100)

        Phase 1: NotImplementedError
        Phase 3: Returns 0.0 (no semantic matching yet)
        Phase 4: Full implementation with embeddings
        """
        raise NotImplementedError(
            "_calculate_semantic_score() to be implemented in Phase 4. "
            "Will use sentence-transformers to generate embeddings and "
            "calculate cosine similarity."
        )
