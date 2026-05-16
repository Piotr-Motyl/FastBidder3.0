"""
SimpleMatchingEngine — domain service for hybrid HVAC matching.
Stateless. Scoring: 40% parameter (exact match) + 60% semantic (AI embeddings).
Falls back to semantic_placeholder (50%) when no embedding_service is provided.
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
    Hybrid HVAC matching: 40% parameter score + 60% semantic score.

    Does NOT cache, store state, or handle async — stateless domain service.

    Attributes:
        parameter_extractor: Extracts DN, PN, material, etc. from raw text.
        config: Weights, threshold, and fast-fail flag.
        embedding_service: If provided, uses real cosine similarity; if None, uses
            config.semantic_placeholder × 100 (50.0) as the semantic score.
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
        Match source against reference catalog. Returns MatchResult or None.

        Args:
            source_description: Working file description to match.
            reference_descriptions: Catalog items. Empty list returns None.
            threshold: Min final_score to accept (default: config.default_threshold = 75.0).

        Score ties: first occurrence wins (stable sort).
        """
        # Step 1: Validate inputs and prepare threshold
        threshold = threshold if threshold is not None else self.config.default_threshold

        # Log mode (AI vs placeholder)
        if self.embedding_service is not None:
            logger.info(
                "Matching mode: AI-enabled (using real embeddings for semantic scoring)"
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

        # Pre-compute ALL reference embeddings in a single batch call.
        # 6x faster than N individual embed_single() calls (GPU/CPU batch optimisation).
        # Falls back gracefully to per-ref embed_single() if batch fails.
        reference_embeddings: dict[int, list[float]] = {}
        if self.embedding_service is not None and source_embedding is not None:
            try:
                ref_texts = [ref.raw_text for ref in reference_descriptions]
                batch_results = self.embedding_service.embed_batch(ref_texts)
                reference_embeddings = dict(enumerate(batch_results))
                logger.debug(
                    f"Pre-computed {len(batch_results)} reference embeddings (batch)"
                )
            except Exception as e:
                logger.warning(
                    f"Batch reference embedding failed, will fall back to per-ref embed_single: {e}"
                )

        for i, ref_desc in enumerate(reference_descriptions):
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

            # Calculate semantic score — pass pre-computed batch embedding when available
            semantic_score = self.calculate_semantic_score(
                source_description,
                ref_desc,
                source_embedding,
                reference_embeddings.get(i),
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
            matched_reference_id=best_ref.chromadb_id or best_ref.id,  # Prefer ChromaDB ID for evaluation
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
        Calculate weighted parameter similarity (0-100). Missing params contribute 0 (neutral, not negative).

        Returns (score, breakdown) where breakdown includes per-param match bools,
        weights_applied, individual_contributions, and using_ai=False.
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
        reference_embedding: Optional[list[float]] = None,
    ) -> float:
        """
        Cosine similarity × 100 (0-100). Falls back to config.semantic_placeholder × 100 = 50.0 when
        embedding_service is None or embedding fails.

        source_embedding/reference_embedding: pre-computed by match_single() via embed_batch() for efficiency.
        """
        # If no embedding service or no source embedding → use placeholder
        if self.embedding_service is None or source_embedding is None:
            logger.debug("Using placeholder semantic score (AI not available)")
            return self.config.semantic_placeholder * 100.0

        # Use pre-computed reference embedding (batch path) or fall back to embed_single()
        if reference_embedding is None:
            try:
                reference_embedding = self.embedding_service.embed_single(
                    reference_description.raw_text
                )
            except Exception as e:
                logger.warning(
                    f"Failed to generate reference embedding, using placeholder: {e}"
                )
                return self.config.semantic_placeholder * 100.0

        # Delegate cosine computation to embedding_service (Infrastructure concern)
        try:
            cosine_sim = self.embedding_service.similarity(
                source_embedding, reference_embedding
            )
            semantic_score = max(0.0, min(100.0, cosine_sim * 100.0))
            logger.debug(f"Calculated semantic score: {semantic_score:.2f}")
            return semantic_score

        except Exception as e:
            logger.error(f"Error calculating similarity, using placeholder: {e}")
            return self.config.semantic_placeholder * 100.0

    def should_fast_fail(
        self,
        source_params: ExtractedParameters,
        reference_params: ExtractedParameters,
    ) -> tuple[bool, Optional[str]]:
        """
        Returns (True, reason) when DN values are both present and differ; (False, None) otherwise.

        Missing DN in either side = don't fast-fail (match on other params).
        config.enable_fast_fail=False disables this check entirely (useful for testing).
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
        Confidence (0-1) from score gap between best and second-best match.

        Formula:
            second_best is None  → confidence = best_score / 100.0
            gap >= 10            → confidence = min(0.85 + (gap - 10) / 100, 1.0)
            gap < 10             → confidence = 0.5 + (gap / 20)
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
        Build human-readable match explanation. DN and PN listed first; mismatched params noted.

        Format: "Matched DN50 (30%), PN16 (10%) - different material [Total: 72.5%]"
        Falls back to "Semantic match (N%) - no exact parameter matches [Total: N%]".
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
