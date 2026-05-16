"""
Semantic Retriever Implementation for Two-Stage Matching (Stage 1).

Provides semantic similarity search using embeddings and vector database.
Retrieves top-K candidate descriptions based on query text and metadata filters.
"""

import logging
from typing import Any, cast

from src.domain.hvac.services.embedding_service import EmbeddingServiceProtocol
from src.domain.hvac.services.semantic_retriever import RetrievalResult
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """
    Stage 1 of two-stage matching: embed query → ChromaDB similarity search → top-K results.

    Performance target: retrieve(top_k=20) < 200ms (embedding ~50-100ms + DB query ~50-100ms).
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceProtocol,
        chroma_client: ChromaClient,
        collection_name: str | None = None,
    ) -> None:
        """collection_name defaults to ChromaClient.COLLECTION_NAME if not provided."""
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client
        self.collection_name = collection_name or ChromaClient.COLLECTION_NAME

        logger.info(
            f"SemanticRetriever initialized with collection: {self.collection_name}"
        )

    def retrieve(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
    ) -> list[RetrievalResult]:
        """
        Embed query_text, apply filters as ChromaDB where clause, return top_k results sorted by similarity.

        Raises ValueError for empty query_text or top_k < 1.
        Raises RuntimeError if embedding or ChromaDB query fails.
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

        # 5. Normalize distances to similarity scores and build results.
        # ChromaDB returns QueryResult (TypedDict); _build_retrieval_results
        # only reads list-shaped fields, so casting to dict[str, Any] is safe.
        retrieval_results = self._build_retrieval_results(cast(dict[str, Any], results))

        logger.info(
            f"Retrieved {len(retrieval_results)} results "
            f"(filters: {filters is not None}, top_k: {top_k})"
        )

        return retrieval_results

    def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """
        Build ChromaDB where clause. None values skipped.

        Single filter: {"dn": {"$eq": "50"}}
        Multiple filters: {"$and": [{"dn": {"$eq": "50"}}, {"pn": {"$eq": "16"}}]}
        All None: returns None.
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
        Convert ChromaDB query results to RetrievalResult objects.

        ChromaDB returns L2 distances (lower = more similar); normalized to similarity = 1 - distance,
        clamped to [0.0, 1.0]. ChromaDB result structure is list-of-lists; takes index [0] (single query).
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
