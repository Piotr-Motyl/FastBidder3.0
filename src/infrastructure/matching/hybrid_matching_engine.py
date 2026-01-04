"""
HybridMatchingEngine - Two-Stage Matching Pipeline.

Combines Stage 1 (semantic retrieval) with Stage 2 (precise scoring) for
efficient and accurate HVAC description matching.

Architecture:
    Stage 1 (Retrieval): SemanticRetriever narrows down candidates
        - Input: Thousands of reference descriptions in vector database
        - Process: Semantic similarity search + metadata filters
        - Output: Top-K most similar candidates (~20)
        - Performance: ~100-200ms

    Stage 2 (Scoring): SimpleMatchingEngine scores candidates
        - Input: Top-K candidates from Stage 1
        - Process: Hybrid scoring (40% param + 60% semantic)
        - Output: Best match above threshold or None
        - Performance: ~50-100ms for 20 candidates

    Total: < 2s per match (meets performance target)

Responsibility:
    - Implement MatchingEngineProtocol (async interface)
    - Orchestrate SemanticRetriever + SimpleMatchingEngine
    - Build metadata filters from extracted parameters
    - Fallback to no-filter retrieval if needed
    - Convert between RetrievalResult and HVACDescription

Architecture Notes:
    - Part of Infrastructure Layer (depends on AI services)
    - Implements Domain Protocol (MatchingEngineProtocol)
    - Uses Dependency Injection for retriever and scoring engine
    - Async interface (future-proof for API calls)
"""

import logging
from typing import Any, Optional

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.hvac.services.semantic_retriever import (
    SemanticRetrieverProtocol,
    RetrievalResult,
)
from src.domain.hvac.services.simple_matching_engine import SimpleMatchingEngine
from src.domain.hvac.matching_config import MatchingConfig
from src.domain.hvac.value_objects.match_result import MatchResult

logger = logging.getLogger(__name__)


class HybridMatchingEngine:
    """
    Two-stage matching pipeline combining semantic retrieval with precise scoring.

    This engine implements the MatchingEngineProtocol using a two-stage approach:
    1. Stage 1 (Retrieval): Use SemanticRetriever to narrow down candidates
    2. Stage 2 (Scoring): Use SimpleMatchingEngine to score and select best match

    This architecture provides:
    - Performance: Only score top-K candidates (not all references)
    - Accuracy: Semantic search finds relevant candidates, hybrid scoring ensures correctness
    - Scalability: Handles thousands of references efficiently

    Algorithm (Two-Stage Matching):
        1. Extract parameters from source description
        2. Build metadata filters (DN, PN if available)
        3. Retrieve top-K candidates using semantic search + filters
        4. Fallback: If no candidates, retry without filters
        5. Convert candidates to HVACDescription objects
        6. Score candidates using SimpleMatchingEngine
        7. Return best match above threshold or None

    Dependencies:
        - semantic_retriever: SemanticRetrieverProtocol for Stage 1
        - simple_matching_engine: SimpleMatchingEngine for Stage 2
        - config: MatchingConfig for thresholds and weights

    Performance Target:
        - Stage 1 (retrieval): < 200ms
        - Stage 2 (scoring): < 100ms for 20 candidates
        - Total: < 2s per match

    Examples:
        >>> # Setup dependencies (typically via DI container)
        >>> from src.infrastructure.ai.embeddings import EmbeddingService
        >>> from src.infrastructure.ai.vector_store import ChromaClient
        >>> from src.infrastructure.ai.retrieval import SemanticRetriever
        >>> from src.domain.hvac.services import SimpleMatchingEngine
        >>>
        >>> embedding_service = EmbeddingService()
        >>> chroma_client = ChromaClient()
        >>> retriever = SemanticRetriever(embedding_service, chroma_client)
        >>> scorer = SimpleMatchingEngine(parameter_extractor, config, embedding_service)
        >>>
        >>> engine = HybridMatchingEngine(
        ...     semantic_retriever=retriever,
        ...     simple_matching_engine=scorer,
        ...     config=config
        ... )
        >>>
        >>> # Match working description against catalog
        >>> source = HVACDescription(raw_text="Zawór kulowy DN50 PN16")
        >>> result = await engine.match(
        ...     working_description=source,
        ...     reference_descriptions=[],  # Not used in hybrid mode
        ...     threshold=75.0
        ... )
        >>>
        >>> if result:
        ...     print(f"Match found: {result.score.final_score}%")
        ...     print(f"Confidence: {result.confidence}")
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetrieverProtocol,
        simple_matching_engine: SimpleMatchingEngine,
        config: MatchingConfig | None = None,
        reference_file_id: str | None = None,
    ) -> None:
        """
        Initialize hybrid matching engine with dependencies.

        Args:
            semantic_retriever: Retriever for Stage 1 (candidate retrieval)
            simple_matching_engine: Scorer for Stage 2 (precise matching)
            config: Optional configuration (defaults to MatchingConfig.default())
            reference_file_id: Optional file_id to filter ChromaDB queries.
                If provided, only searches within this specific reference file.
                This prevents matches from other files in the database (e.g., from previous test runs).

        Examples:
            >>> engine = HybridMatchingEngine(
            ...     semantic_retriever=retriever,
            ...     simple_matching_engine=scorer,
            ...     config=MatchingConfig.default(),
            ...     reference_file_id="3e5bc285-b710-4b5f-911e-58d2524bfce8"
            ... )
        """
        self.semantic_retriever = semantic_retriever
        self.simple_matching_engine = simple_matching_engine
        self.config = config or MatchingConfig.default()
        self.reference_file_id = reference_file_id

        logger.info(
            f"HybridMatchingEngine initialized (two-stage pipeline, "
            f"reference_file_id={'set' if reference_file_id else 'not set'})"
        )

    async def match(
        self,
        working_description: HVACDescription,
        reference_descriptions: list[HVACDescription],
        threshold: float = 75.0,
    ) -> Optional[MatchResult]:
        """
        Match working description using two-stage pipeline.

        Stage 1 (Retrieval):
            - Extract parameters from source
            - Build metadata filters (DN, PN if available)
            - Retrieve top-K candidates using semantic search
            - Fallback to no-filter search if no candidates found

        Stage 2 (Scoring):
            - Convert candidates to HVACDescription objects
            - Score each candidate using SimpleMatchingEngine
            - Select best match above threshold
            - Calculate confidence based on score gap

        Args:
            working_description: Source description to match
            reference_descriptions: NOT USED in hybrid mode (uses vector DB)
            threshold: Minimum score for valid match (0-100)

        Returns:
            MatchResult if best match >= threshold, None otherwise

        Raises:
            ValueError: If working_description has no text
            RuntimeError: If retrieval or scoring fails

        Examples:
            >>> source = HVACDescription(raw_text="Zawór kulowy DN50 PN16")
            >>> result = await engine.match(source, [], threshold=75.0)
            >>> if result:
            ...     print(f"Match: {result.score.final_score}%")

        Caching strategy:
            Source embeddings are NOT cached per-batch in HybridMatchingEngine.
            Instead, each call to match() generates a fresh embedding for the working description.

            Rationale:
            - Working descriptions are typically processed in sequential batches by matching_tasks
            - Caching would require state management (breaking stateless design)
            - EmbeddingService itself may implement internal caching (model-level)

            For batch processing optimization, consider using SimpleMatchingEngine.match_batch()
            which does implement source embedding caching.

        Performance:
            - Stage 1: ~100-200ms (semantic retrieval)
            - Stage 2: ~50-100ms (scoring 20 candidates)
            - Total: < 2s target
        """
        logger.info(
            f"Starting two-stage matching for: {working_description.raw_text[:50]}..."
        )

        # Validate input
        if not working_description.raw_text:
            raise ValueError("working_description must have raw_text")

        # Stage 1: Retrieval (narrow down candidates)
        logger.debug("Stage 1: Retrieving top-K candidates...")
        candidates = await self._retrieve_candidates(working_description)

        if not candidates:
            logger.warning("No candidates found in Stage 1 (retrieval)")
            return None

        logger.info(f"Stage 1 complete: {len(candidates)} candidates retrieved")

        # Convert candidates to HVACDescription objects for Stage 2
        candidate_descriptions = self._convert_candidates_to_descriptions(candidates)

        # Stage 2: Scoring (precise matching on candidates)
        logger.debug(f"Stage 2: Scoring {len(candidate_descriptions)} candidates...")
        result = self.simple_matching_engine.match_single(
            source_description=working_description,
            reference_descriptions=candidate_descriptions,
            threshold=threshold,
        )

        if result:
            logger.info(
                f"Stage 2 complete: Match found with score {result.score.final_score:.1f}%"
            )

            # Phase 4: Enhance breakdown with AI metadata (HybridMatchingEngine uses AI)
            # Get AI model name from EmbeddingService (dynamically retrieved)
            ai_model_name = None
            if self.simple_matching_engine.embedding_service is not None:
                # Model name is stored in embedding_service.model_name attribute
                ai_model_name = self.simple_matching_engine.embedding_service.model_name

            enhanced_breakdown = {
                **result.breakdown,  # Preserve existing breakdown from SimpleMatchingEngine
                "using_ai": True,  # HybridMatchingEngine uses AI embeddings for semantic matching
                "ai_model": ai_model_name,  # Model name retrieved from EmbeddingService
                "stage1_candidates": len(candidates),  # Debug: Number of candidates from Stage 1 retrieval
                "retrieval_top_k": self.config.retrieval_top_k,  # Config: top-K parameter
            }

            # Recreate MatchResult with enhanced breakdown
            from src.domain.hvac.value_objects.match_result import MatchResult
            result = MatchResult(
                matched_reference_id=result.matched_reference_id,
                score=result.score,
                confidence=result.confidence,
                message=result.message,
                breakdown=enhanced_breakdown,
            )
        else:
            logger.info("Stage 2 complete: No match above threshold")

        return result

    async def calculate_confidence(
        self, best_score: float, second_best_score: Optional[float]
    ) -> float:
        """
        Calculate confidence based on score gap.

        Delegates to SimpleMatchingEngine.calculate_confidence().

        Args:
            best_score: Final score of best match (0-100)
            second_best_score: Final score of second-best, or None

        Returns:
            Confidence level (0-1)

        Examples:
            >>> confidence = await engine.calculate_confidence(95.0, 70.0)
            >>> confidence
            1.0  # Large gap → high confidence
        """
        return self.simple_matching_engine.calculate_confidence(
            best_score, second_best_score
        )

    async def _retrieve_candidates(
        self, source_description: HVACDescription
    ) -> list[RetrievalResult]:
        """
        Stage 1: Retrieve top-K candidates using semantic search.

        Process:
            1. Extract parameters from source
            2. Build metadata filters (DN, PN if available)
            3. Retrieve candidates with filters
            4. Fallback: If no candidates, retry without filters

        Args:
            source_description: Source to retrieve candidates for

        Returns:
            List of RetrievalResult from semantic search

        Note:
            Uses config.retrieval_top_k for number of candidates (default: 20)
        """
        # Extract parameters if not already done
        if not source_description.has_parameters():
            source_description.extract_parameters(
                self.simple_matching_engine.parameter_extractor
            )

        # Build metadata filters from extracted parameters
        filters = self._build_metadata_filters(source_description)

        # Add file_id filter to restrict search to current reference file
        # This prevents matches from other files in ChromaDB (e.g., from previous test runs)
        if self.reference_file_id:
            if filters is None:
                filters = {}
            filters["file_id"] = self.reference_file_id
            logger.debug(f"Added file_id filter: {self.reference_file_id}")

        # Retrieve candidates with filters
        top_k = self.config.retrieval_top_k

        if filters:
            logger.debug(f"Retrieving with filters: {filters}")
            candidates = self.semantic_retriever.retrieve(
                query_text=source_description.raw_text,
                filters=filters,
                top_k=top_k,
            )

            # Fallback: If no candidates with filters, try without HVAC filters but keep file_id filter
            # This ensures we still only search within the current reference file
            if not candidates:
                logger.warning(
                    "No candidates with full filters, retrying with file_id-only filter (semantic-only)"
                )
                # Create file_id-only filter for fallback search
                fallback_filters = {"file_id": self.reference_file_id} if self.reference_file_id else None
                candidates = self.semantic_retriever.retrieve(
                    query_text=source_description.raw_text,
                    filters=fallback_filters,
                    top_k=top_k,
                )
        else:
            # No HVAC parameter filters, but may have file_id filter
            # Use file_id-only filter for semantic-only search if available
            fallback_filters = {"file_id": self.reference_file_id} if self.reference_file_id else None
            logger.debug(
                f"No HVAC parameter filters, using semantic-only search "
                f"(file_id filter: {'set' if self.reference_file_id else 'not set'})"
            )
            candidates = self.semantic_retriever.retrieve(
                query_text=source_description.raw_text,
                filters=fallback_filters,
                top_k=top_k,
            )

        return candidates

    def _build_metadata_filters(
        self, source_description: HVACDescription
    ) -> dict[str, Any] | None:
        """
        Build metadata filters from extracted parameters.

        Filter strategy:
            - Include DN if extracted (critical parameter)
            - Include PN if extracted (important parameter)
            - Skip other parameters (too restrictive)

        Args:
            source_description: Source with extracted parameters

        Returns:
            Dict of filters or None if no filters available

        Examples:
            >>> # With DN and PN
            >>> source = HVACDescription(raw_text="Zawór DN50 PN16")
            >>> source.extract_parameters(extractor)
            >>> filters = engine._build_metadata_filters(source)
            >>> filters
            {"dn": "50", "pn": "16"}

            >>> # No parameters
            >>> source = HVACDescription(raw_text="Zawór kulowy")
            >>> filters = engine._build_metadata_filters(source)
            >>> filters
            None
        """
        if not source_description.has_parameters():
            return None

        params = source_description.extracted_params
        if params is None:
            return None

        filters = {}

        # Add DN filter if available (critical parameter)
        if params.dn is not None:
            filters["dn"] = str(params.dn)

        # Add PN filter if available (important parameter)
        if params.pn is not None:
            filters["pn"] = str(params.pn)

        # Return None if no filters (avoid empty dict)
        return filters if filters else None

    def _convert_candidates_to_descriptions(
        self, candidates: list[RetrievalResult]
    ) -> list[HVACDescription]:
        """
        Convert RetrievalResult objects to HVACDescription for Stage 2.

        Creates HVACDescription entities from retrieval results with:
        - raw_text from reference_text
        - file_id and source_row_number from parsed description_id
        - metadata preserved

        Args:
            candidates: List of RetrievalResult from Stage 1

        Returns:
            List of HVACDescription ready for scoring

        Note:
            Parameters will be extracted during scoring if needed
        """
        descriptions = []

        for candidate in candidates:
            desc = HVACDescription(
                raw_text=candidate.reference_text,
                source_row_number=candidate.source_row_number,
                file_id=candidate.file_id,
                chromadb_id=candidate.description_id,  # Preserve ChromaDB ID for evaluation
            )
            descriptions.append(desc)

        return descriptions
