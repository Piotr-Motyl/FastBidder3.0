"""
HybridMatchingEngine — two-stage HVAC matching pipeline.

Stage 1: SemanticRetriever → top-K candidates (~20) from ChromaDB (~100-200ms).
Stage 2: SimpleMatchingEngine → hybrid score (40% param + 60% semantic) on candidates (~50-100ms).
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
    Implements MatchingEngineProtocol via two-stage pipeline.

    reference_file_id: if set, filters ChromaDB queries to a single file,
        preventing stale matches from previous indexing sessions.
    """

    def __init__(
        self,
        semantic_retriever: SemanticRetrieverProtocol,
        simple_matching_engine: SimpleMatchingEngine,
        config: MatchingConfig | None = None,
        reference_file_id: str | None = None,
    ) -> None:
        """
        Args:
            reference_file_id: If set, restricts ChromaDB search to this file only
                (prevents cross-file matches from previous indexing sessions).
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
        Run two-stage match. reference_descriptions is unused (candidates come from ChromaDB).

        Raises ValueError if working_description has no raw_text.
        Returns None if Stage 1 returns no candidates or Stage 2 finds no match above threshold.
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

            # Phase 4: enhance breakdown with AI metadata.
            # `model_name` isn't part of EmbeddingServiceProtocol — it's an
            # implementation detail of EmbeddingService. Read defensively so any
            # mock implementation just yields ai_model=None.
            ai_model_name = None
            if self.simple_matching_engine.embedding_service is not None:
                ai_model_name = getattr(
                    self.simple_matching_engine.embedding_service, "model_name", None
                )

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
        """Delegates to SimpleMatchingEngine.calculate_confidence()."""
        return self.simple_matching_engine.calculate_confidence(
            best_score, second_best_score
        )

    async def _retrieve_candidates(
        self, source_description: HVACDescription
    ) -> list[RetrievalResult]:
        """
        Stage 1: retrieve top-K candidates. Applies DN/PN + file_id filters;
        falls back to file_id-only filter when full-filter search returns nothing.
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
        """Build {"dn": "50", "pn": "16"} filters from extracted params. DN and PN only — other params are too restrictive."""
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
        """Convert RetrievalResult list to HVACDescription list for Stage 2 scoring."""
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
