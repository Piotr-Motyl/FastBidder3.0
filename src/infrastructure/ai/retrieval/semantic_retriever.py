"""
Semantic Retriever Implementation for Two-Stage Matching (Stage 1).

Provides semantic similarity search using embeddings and vector database.
Retrieves top-K candidate descriptions based on query text and metadata filters.
"""

import logging
from typing import Any
from uuid import UUID

from src.domain.hvac.services.embedding_service import EmbeddingServiceProtocol
from src.domain.hvac.services.semantic_retriever import RetrievalResult
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Semantic retriever using embedding similarity and vector database.

    Stage 1 of Two-Stage Matching: retrieves top-K candidate reference
    descriptions from vector database based on semantic similarity and
    metadata filters (hard constraints).

    Process Flow:
        1. Generate embedding for query text (via EmbeddingService)
        2. Build metadata filters as ChromaDB where clause
        3. Query vector database with embedding + filters
        4. Normalize distance to similarity score (1 - distance)
        5. Return top-K results sorted by similarity descending

    Performance:
        - Target: retrieve(top_k=20) < 200ms
        - Embedding generation: ~50-100ms
        - Database query: ~50-100ms

    Dependencies:
        - EmbeddingService: generates embeddings for query text
        - ChromaClient: provides access to vector database

    Examples:
        >>> from src.infrastructure.ai.embeddings import EmbeddingService
        >>> from src.infrastructure.ai.vector_store import ChromaClient
        >>>
        >>> embedding_service = EmbeddingService()
        >>> chroma_client = ChromaClient()
        >>> retriever = SemanticRetriever(embedding_service, chroma_client)
        >>>
        >>> # Basic retrieval
        >>> results = retriever.retrieve("Zawór kulowy DN50 PN16", top_k=20)
        >>> len(results) <= 20
        True
        >>>
        >>> # With metadata filters
        >>> results = retriever.retrieve(
        ...     query_text="Zawór kulowy",
        ...     filters={"dn": "50", "material": "brass"},
        ...     top_k=10
        ... )
        >>> all(r.metadata.get("dn") == "50" for r in results)
        True
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        chroma_client: ChromaClient,
        collection_name: str | None = None,
    ) -> None:
        """
        Initialize semantic retriever with dependencies.

        Args:
            embedding_service: Service for generating embeddings
            chroma_client: Client for vector database operations
            collection_name: Optional collection name (defaults to ChromaClient.COLLECTION_NAME)

        Examples:
            >>> retriever = SemanticRetriever(embedding_service, chroma_client)
            >>> retriever = SemanticRetriever(embedding_service, chroma_client, "custom_collection")
        """
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client
        self.collection_name = collection_name or ChromaClient.COLLECTION_NAME

        logger.info(f"SemanticRetriever initialized with collection: {self.collection_name}")

    def retrieve(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Retrieve top-K most similar reference descriptions from vector database.

        Stage 1 of Two-Stage Matching:
        1. Validate inputs
        2. Generate embedding for query_text
        3. Build ChromaDB where clause from filters
        4. Query vector database
        5. Normalize distances to similarity scores
        6. Return top-K results sorted by similarity

        Args:
            query_text: Text to search for (e.g., "Zawór kulowy DN50 PN16")
            filters: Optional metadata filters for hard constraints.
                Keys: "dn", "pn", "material", "valve_type", etc.
                Values: string or numeric values to match.
                None values are ignored.
                Example: {"dn": "50", "pn": "16"}
            top_k: Number of top results to return (default: 20)

        Returns:
            List of RetrievalResult objects sorted by similarity (descending).
            Empty list if no matches found.

        Raises:
            ValueError: If query_text is empty or top_k < 1
            RuntimeError: If vector database is unavailable

        Examples:
            >>> results = retriever.retrieve("Zawór kulowy DN50 PN16", top_k=20)
            >>> results[0].similarity_score >= results[-1].similarity_score
            True
            >>> results = retriever.retrieve("test", filters={"dn": "50"}, top_k=10)
            >>> all(r.metadata["dn"] == "50" for r in results)
            True
        """
        # 1. Validate inputs
        if not query_text or not query_text.strip():
            raise ValueError("query_text cannot be empty")

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        logger.debug(f"Retrieving top-{top_k} results for query: {query_text[:50]}...")

        # 2. Generate embedding for query text
        try:
            query_embedding = self.embedding_service.embed_single(query_text)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

        # 3. Build ChromaDB where clause from filters
        where_clause = self._build_where_clause(filters) if filters else None

        # 4. Query vector database
        collection = self.chroma_client.get_or_create_collection(self.collection_name)

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise RuntimeError(f"Vector database query failed: {e}") from e

        # 5. Normalize distances to similarity scores and build results
        retrieval_results = self._build_retrieval_results(results)

        logger.info(
            f"Retrieved {len(retrieval_results)} results "
            f"(filters: {filters is not None}, top_k: {top_k})"
        )

        return retrieval_results

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """
        Build ChromaDB where clause from filters dictionary.

        Filters are combined with $and operator. None values are skipped.

        Args:
            filters: Dictionary of metadata filters
                Example: {"dn": "50", "pn": "16", "material": None}

        Returns:
            ChromaDB where clause or None if no valid filters
            Example: {"$and": [{"dn": {"$eq": "50"}}, {"pn": {"$eq": "16"}}]}

        Examples:
            >>> retriever._build_where_clause({"dn": "50"})
            {"dn": {"$eq": "50"}}
            >>> retriever._build_where_clause({"dn": "50", "pn": "16"})
            {"$and": [{"dn": {"$eq": "50"}}, {"pn": {"$eq": "16"}}]}
            >>> retriever._build_where_clause({"dn": "50", "pn": None})
            {"dn": {"$eq": "50"}}
            >>> retriever._build_where_clause({"dn": None, "pn": None})
            None
        """
        # Filter out None values
        valid_filters = {k: v for k, v in filters.items() if v is not None}

        if not valid_filters:
            return None

        # Build list of conditions
        conditions = [{key: {"$eq": value}} for key, value in valid_filters.items()]

        # Single condition: return directly
        if len(conditions) == 1:
            return conditions[0]

        # Multiple conditions: combine with $and
        return {"$and": conditions}

    def _build_retrieval_results(
        self, chroma_results: dict[str, Any]
    ) -> list[RetrievalResult]:
        """
        Build RetrievalResult objects from ChromaDB query results.

        Normalizes distances to similarity scores (1 - distance) and
        creates RetrievalResult DTOs.

        Args:
            chroma_results: Results from ChromaDB query
                Keys: "ids", "documents", "metadatas", "distances"
                Each is a list of lists (one list per query)

        Returns:
            List of RetrievalResult objects sorted by similarity descending

        Note:
            ChromaDB returns distances (lower = more similar).
            We normalize to similarity: similarity = 1 - distance
            This ensures higher score = better match (0.0-1.0 range).

        Examples:
            >>> chroma_results = {
            ...     "ids": [["file_id_1", "file_id_2"]],
            ...     "documents": [["text1", "text2"]],
            ...     "metadatas": [[{"dn": "50"}, {"dn": "100"}]],
            ...     "distances": [[0.1, 0.3]]
            ... }
            >>> results = retriever._build_retrieval_results(chroma_results)
            >>> results[0].similarity_score
            0.9
            >>> results[1].similarity_score
            0.7
        """
        # ChromaDB returns list of lists (one per query)
        # We send single query, so take first element
        ids = chroma_results.get("ids", [[]])[0]
        documents = chroma_results.get("documents", [[]])[0]
        metadatas = chroma_results.get("metadatas", [[]])[0]
        distances = chroma_results.get("distances", [[]])[0]

        # Build RetrievalResult objects
        results: list[RetrievalResult] = []

        for idx, doc_id in enumerate(ids):
            # Normalize distance to similarity (1 - distance)
            # ChromaDB L2 distance: lower = more similar
            # Similarity: higher = more similar (0.0-1.0)
            distance = distances[idx]
            similarity = 1.0 - distance

            # Clamp to [0.0, 1.0] range (safety check)
            similarity = max(0.0, min(1.0, similarity))

            result = RetrievalResult(
                description_id=doc_id,
                reference_text=documents[idx],
                similarity_score=similarity,
                metadata=metadatas[idx],
            )

            results.append(result)

        # Results are already sorted by distance (ascending) from ChromaDB
        # Which means sorted by similarity (descending) after normalization
        # No need to re-sort

        return results
