"""
Semantic Retriever Protocol for Two-Stage Matching (Stage 1: Retrieval).

Responsibility:
    Defines interface for semantic similarity search using vector database.
    Stage 1 of Two-Stage Matching: retrieves top-K candidate descriptions
    from reference catalog based on embedding similarity and metadata filters.

Architecture Notes:
    - Part of Domain Layer (Services)
    - Protocol (interface) - implementation in Infrastructure Layer
    - Used by HybridMatchingEngine for candidate retrieval
    - Returns RetrievalResult DTOs with similarity scores

Contains:
    - RetrievalResult: DTO for search results
    - SemanticRetrieverProtocol: Interface for retrieval service

Does NOT contain:
    - Vector database operations (delegated to Infrastructure)
    - Embedding generation (uses EmbeddingService)
    - Scoring logic (Stage 2 responsibility)

Two-Stage Matching Flow:
    1. STAGE 1 (Retrieval): SemanticRetriever finds top-K candidates
       - Generate embedding for query text
       - Apply metadata filters (hard constraints)
       - Search vector database by similarity
       - Return top-K results with scores

    2. STAGE 2 (Scoring): MatchingEngine scores candidates
       - Detailed parameter matching
       - Confidence calculation
       - Final ranking and selection
"""

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID


# ============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# ============================================================================


@dataclass
class RetrievalResult:
    """
    Single result from semantic retrieval search.

    Represents a candidate reference description retrieved from vector database
    based on semantic similarity to query text and metadata filters.

    Attributes:
        description_id: Unique identifier of reference description (format: {file_id}_{row_number})
        reference_text: Original text of reference description
        similarity_score: Cosine similarity score (0.0-1.0, higher = more similar)
            Normalized from distance: similarity = 1 - distance
        metadata: Additional metadata from vector database (DN, PN, material, etc.)
        file_id: UUID of source file (extracted from description_id)
        source_row_number: Row number in source file (extracted from description_id)

    Examples:
        >>> result = RetrievalResult(
        ...     description_id="a3bb189e-8bf9-3888-9912-ace4e6543002_42",
        ...     reference_text="Zawór kulowy DN50 PN16 mosiężny",
        ...     similarity_score=0.87,
        ...     metadata={"dn": "50", "pn": "16", "material": "brass"},
        ...     file_id=UUID("a3bb189e-8bf9-3888-9912-ace4e6543002"),
        ...     source_row_number=42
        ... )
        >>> result.similarity_score
        0.87
        >>> result.metadata["dn"]
        '50'
    """

    description_id: str
    reference_text: str
    similarity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    file_id: UUID | None = None
    source_row_number: int | None = None

    def __post_init__(self) -> None:
        """
        Parse description_id to extract file_id and source_row_number.

        Description ID format: {file_id}_{row_number}
        Example: "a3bb189e-8bf9-3888-9912-ace4e6543002_42"
        """
        if "_" in self.description_id:
            parts = self.description_id.rsplit("_", 1)
            if len(parts) == 2:
                try:
                    self.file_id = UUID(parts[0])
                    self.source_row_number = int(parts[1])
                except (ValueError, AttributeError):
                    # Invalid format - leave as None
                    pass


# ============================================================================
# PROTOCOL (INTERFACE)
# ============================================================================


class SemanticRetrieverProtocol(Protocol):
    """
    Protocol for semantic similarity retrieval from vector database.

    Defines interface for Stage 1 of Two-Stage Matching:
    - Generate embedding for query text
    - Apply metadata filters (hard constraints: DN, PN, material, etc.)
    - Search vector database by embedding similarity
    - Return top-K most similar reference descriptions

    This is a protocol (interface) - implementation in Infrastructure Layer.
    Allows dependency injection and testing with mocks.

    Typical Implementation:
        - Uses EmbeddingService to generate query embedding
        - Uses ChromaClient to search vector database
        - Applies filters as ChromaDB where clauses
        - Normalizes distance to similarity score (1 - distance)

    Performance Target:
        - retrieve(top_k=20) should complete in < 200ms
        - Includes embedding generation + database query

    Examples:
        >>> # Dependency injection in HybridMatchingEngine
        >>> from src.infrastructure.ai.retrieval import SemanticRetriever
        >>> retriever = SemanticRetriever(embedding_service, chroma_client)
        >>>
        >>> # Basic retrieval without filters
        >>> results = retriever.retrieve(
        ...     query_text="Zawór kulowy DN50 PN16",
        ...     top_k=20
        ... )
        >>> len(results)
        20
        >>> results[0].similarity_score > results[-1].similarity_score
        True
        >>>
        >>> # Retrieval with metadata filters
        >>> results = retriever.retrieve(
        ...     query_text="Zawór kulowy mosiężny",
        ...     filters={"dn": "50", "material": "brass"},
        ...     top_k=10
        ... )
        >>> all(r.metadata.get("dn") == "50" for r in results)
        True
    """

    def retrieve(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-K most similar reference descriptions from vector database.

        Stage 1 of Two-Stage Matching:
        1. Generate embedding for query_text using EmbeddingService
        2. Build ChromaDB where clause from filters (if provided)
        3. Query vector database with embedding + filters
        4. Normalize distances to similarity scores (1 - distance)
        5. Return top-K results sorted by similarity (descending)

        Args:
            query_text: Text to search for (e.g., "Zawór kulowy DN50 PN16")
            filters: Optional metadata filters for hard constraints.
                Keys: "dn", "pn", "material", "valve_type", etc.
                Values: string or numeric values to match.
                None values are ignored (not applied as filters).
                Example: {"dn": "50", "pn": "16"}
            top_k: Number of top results to return (default: 20)
                Typical range: 10-50 for Stage 1 retrieval

        Returns:
            List of RetrievalResult objects sorted by similarity (descending).
            Length may be < top_k if fewer matches exist in database.
            Empty list if no matches found or database is empty.

        Raises:
            ValueError: If query_text is empty or top_k < 1
            RuntimeError: If vector database is unavailable
            EmbeddingError: If embedding generation fails

        Filter Mapping:
            Filters are mapped to ChromaDB where clauses:
            - {"dn": "50"} → {"dn": {"$eq": "50"}}
            - {"dn": "50", "pn": "16"} → {"$and": [{"dn": {"$eq": "50"}}, {"pn": {"$eq": "16"}}]}
            - None values in filters are skipped (e.g., {"dn": "50", "pn": None} → only filter by dn)

        Score Normalization:
            ChromaDB returns L2 distance (lower = more similar).
            Normalized to similarity: similarity = 1 - distance
            Ensures higher score = better match (0.0-1.0 range).

        Performance:
            - Target: < 200ms for top_k=20
            - Embedding generation: ~50-100ms
            - Database query: ~50-100ms
            - Use batch operations when processing multiple queries

        Examples:
            >>> # Basic retrieval
            >>> results = retriever.retrieve("Zawór kulowy DN50 PN16", top_k=20)
            >>> results[0].reference_text
            'Zawór kulowy DN50 PN16 mosiężny'
            >>> results[0].similarity_score
            0.95

            >>> # With filters (hard constraints)
            >>> results = retriever.retrieve(
            ...     query_text="Zawór kulowy",
            ...     filters={"dn": "50", "material": "brass"},
            ...     top_k=10
            ... )
            >>> all(r.metadata["dn"] == "50" for r in results)
            True

            >>> # Empty database returns empty list
            >>> results = retriever.retrieve("test", top_k=5)
            >>> results
            []
        """
        ...
