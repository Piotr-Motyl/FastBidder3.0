"""
SemanticRetrieverProtocol — Stage 1 of two-stage matching: retrieves top-K candidates
from ChromaDB using embedding similarity and metadata filters.
Stage 2 (scoring) is handled by HybridMatchingEngine.
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

    Attributes:
        description_id: ChromaDB document ID in format "{file_id}_{row_number}"
        similarity_score: Cosine similarity (0.0-1.0); normalized from L2 distance: 1 - distance
        metadata: ChromaDB document metadata (keys: "dn", "pn", "material", etc.)
        file_id: Parsed from description_id automatically in __post_init__
        source_row_number: Parsed from description_id automatically in __post_init__
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
    Protocol for Stage 1 retrieval: embeds query text, applies metadata filters,
    queries ChromaDB, and returns top-K candidates sorted by similarity (descending).
    Performance target: < 200ms for top_k=20 (embedding + DB query).
    """

    def retrieve(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-K candidates from ChromaDB by embedding similarity.

        Args:
            query_text: Text to embed and search for.
            filters: Metadata hard constraints mapped to ChromaDB where clauses.
                {"dn": "50"} → {"dn": {"$eq": "50"}}
                {"dn": "50", "pn": "16"} → {"$and": [...]}
                None values in the dict are silently ignored.
            top_k: Number of results (default 20). May return fewer if DB has fewer items.

        Returns:
            RetrievalResult list sorted by similarity descending. Empty if no matches.
        """
        ...
