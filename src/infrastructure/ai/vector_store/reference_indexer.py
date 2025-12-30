"""
Reference indexer for embedding and storing HVAC descriptions in ChromaDB.

Pre-embeds reference descriptions for fast semantic search during matching.
Flow: descriptions → embeddings → ChromaDB with metadata.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from uuid import UUID

from src.domain.hvac.entities.hvac_description import HVACDescription
from src.infrastructure.ai.embeddings.embedding_service import EmbeddingService
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient

logger = logging.getLogger(__name__)


@dataclass
class IndexingResult:
    """
    Result of indexing a reference file into ChromaDB.

    Tracks success, failures, and performance metrics for indexing operation.

    Attributes:
        file_id: UUID of the indexed file.
        total_descriptions: Total number of descriptions in file.
        indexed_count: Successfully indexed descriptions.
        failed_count: Failed to index (e.g., empty text, embedding errors).
        errors: List of error messages for failed items.
        indexing_time_seconds: Total time taken for indexing.

    Example:
        >>> result = IndexingResult(
        ...     file_id=UUID("..."),
        ...     total_descriptions=100,
        ...     indexed_count=98,
        ...     failed_count=2
        ... )
        >>> result.success_rate
        98.0
    """

    file_id: UUID
    total_descriptions: int
    indexed_count: int
    failed_count: int
    errors: list[str] = field(default_factory=list)
    indexing_time_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """
        Calculate success rate as percentage.

        Returns:
            Success rate (0-100%). Returns 0.0 if no descriptions.

        Example:
            >>> result = IndexingResult(..., total=100, indexed=95, failed=5)
            >>> result.success_rate
            95.0
        """
        if self.total_descriptions == 0:
            return 0.0
        return (self.indexed_count / self.total_descriptions) * 100


class ReferenceIndexer:
    """
    Service for indexing reference HVAC descriptions into vector database.

    Pre-embeds reference descriptions and stores them with metadata in ChromaDB.
    Enables fast semantic similarity search during working file matching.

    Key features:
    - Batch embedding (32 descriptions at once) for efficiency
    - Idempotent: can re-index existing files
    - Partial failure handling: indexes what it can, reports errors
    - Metadata storage: DN, PN, material, valve_type, source row

    Metadata per document:
    - file_id: Source file UUID
    - dn: Diameter nominal (if extracted)
    - pn: Pressure nominal (if extracted)
    - material: Material type (if extracted)
    - valve_type: Valve type (if extracted)
    - source_row_number: Excel row number

    ChromaDB document ID format: "{file_id}_{row_number}"

    Example:
        >>> indexer = ReferenceIndexer(embedding_service, chroma_client)
        >>> descriptions = [...]  # List of HVACDescription entities
        >>> result = indexer.index_file(file_id, descriptions)
        >>> result.success_rate
        98.5
        >>> indexer.is_file_indexed(file_id)
        True
    """

    # Batch size for embedding generation (memory efficiency)
    BATCH_SIZE = 32

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_client: ChromaClient,
    ) -> None:
        """
        Initialize reference indexer.

        Args:
            embedding_service: Service for generating embeddings.
            chroma_client: ChromaDB client for vector storage.

        Example:
            >>> embedding_svc = EmbeddingService()
            >>> chroma = ChromaClient()
            >>> indexer = ReferenceIndexer(embedding_svc, chroma)
        """
        self.embedding_service = embedding_service
        self.chroma_client = chroma_client
        logger.info("ReferenceIndexer initialized")

    def index_file(
        self,
        file_id: UUID,
        descriptions: list[HVACDescription],
        skip_if_indexed: bool = True,
    ) -> IndexingResult:
        """
        Index reference file descriptions into ChromaDB.

        Generates embeddings and stores descriptions with metadata.
        Processes in batches for memory efficiency.

        Args:
            file_id: UUID of the source file.
            descriptions: List of HVAC descriptions to index.
            skip_if_indexed: If True and file already indexed, skip.
                If False, re-index (replaces existing).

        Returns:
            IndexingResult with counts, errors, and timing.

        Example:
            >>> result = indexer.index_file(
            ...     file_id=UUID("..."),
            ...     descriptions=[desc1, desc2, desc3]
            ... )
            >>> print(f"Indexed {result.indexed_count}/{result.total_descriptions}")
            Indexed 3/3
        """
        start_time = time.time()

        # Check if already indexed
        if skip_if_indexed and self.is_file_indexed(file_id):
            logger.info(f"File {file_id} already indexed, skipping")
            return IndexingResult(
                file_id=file_id,
                total_descriptions=len(descriptions),
                indexed_count=0,
                failed_count=0,
                errors=["File already indexed (skip_if_indexed=True)"],
                indexing_time_seconds=time.time() - start_time,
            )

        # Remove existing index if re-indexing
        if not skip_if_indexed and self.is_file_indexed(file_id):
            logger.info(f"Re-indexing file {file_id}, removing old index")
            self.remove_file(file_id)

        # Initialize result tracking
        total = len(descriptions)
        indexed_count = 0
        failed_count = 0
        errors: list[str] = []

        # Get or create collection
        collection = self.chroma_client.get_or_create_collection()

        # Process in batches
        for batch_start in range(0, total, self.BATCH_SIZE):
            batch_end = min(batch_start + self.BATCH_SIZE, total)
            batch = descriptions[batch_start:batch_end]

            try:
                # Prepare batch data
                texts = []
                ids = []
                metadatas = []

                for desc in batch:
                    # Skip descriptions with empty text
                    if not desc.raw_text or not desc.raw_text.strip():
                        failed_count += 1
                        errors.append(
                            f"Row {desc.source_row_number}: Empty description"
                        )
                        continue

                    # Prepare document ID
                    doc_id = f"{file_id}_{desc.source_row_number}"
                    ids.append(doc_id)
                    texts.append(desc.raw_text.strip())

                    # Prepare metadata
                    metadata: dict = {
                        "file_id": str(file_id),
                        "source_row_number": desc.source_row_number or 0,
                    }

                    # Add extracted parameters if available
                    if desc.extracted_params:
                        if desc.extracted_params.dn:
                            metadata["dn"] = str(desc.extracted_params.dn.value)
                        if desc.extracted_params.pn:
                            metadata["pn"] = str(desc.extracted_params.pn.value)
                        if desc.extracted_params.material:
                            metadata["material"] = desc.extracted_params.material
                        if desc.extracted_params.valve_type:
                            metadata["valve_type"] = desc.extracted_params.valve_type

                    metadatas.append(metadata)

                # Skip empty batch (all descriptions invalid)
                if not texts:
                    continue

                # Generate embeddings for batch
                embeddings = self.embedding_service.embed_batch(texts)

                # Add to ChromaDB
                collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )

                indexed_count += len(texts)
                logger.debug(
                    f"Indexed batch {batch_start}-{batch_end}: {len(texts)} descriptions"
                )

            except Exception as e:
                # Partial failure: log error, continue with next batch
                batch_error = f"Batch {batch_start}-{batch_end} failed: {str(e)}"
                errors.append(batch_error)
                failed_count += len(batch)
                logger.error(batch_error)

        # Calculate timing
        indexing_time = time.time() - start_time

        result = IndexingResult(
            file_id=file_id,
            total_descriptions=total,
            indexed_count=indexed_count,
            failed_count=failed_count,
            errors=errors,
            indexing_time_seconds=indexing_time,
        )

        logger.info(
            f"Indexing complete: {indexed_count}/{total} indexed "
            f"({result.success_rate:.1f}%) in {indexing_time:.2f}s"
        )

        return result

    def remove_file(self, file_id: UUID) -> int:
        """
        Remove all descriptions for a file from ChromaDB.

        Deletes all documents matching the file_id metadata.

        Args:
            file_id: UUID of the file to remove.

        Returns:
            Number of documents removed.

        Example:
            >>> count = indexer.remove_file(UUID("..."))
            >>> print(f"Removed {count} descriptions")
        """
        collection = self.chroma_client.get_or_create_collection()

        # Query all documents for this file
        try:
            results = collection.get(where={"file_id": str(file_id)})
            doc_ids = results.get("ids", [])

            if not doc_ids:
                logger.info(f"No documents found for file {file_id}")
                return 0

            # Delete documents
            collection.delete(ids=doc_ids)
            logger.info(f"Removed {len(doc_ids)} documents for file {file_id}")
            return len(doc_ids)

        except Exception as e:
            logger.error(f"Failed to remove file {file_id}: {e}")
            return 0

    def is_file_indexed(self, file_id: UUID) -> bool:
        """
        Check if file has been indexed.

        Queries ChromaDB for any documents with matching file_id.

        Args:
            file_id: UUID of the file to check.

        Returns:
            True if file has indexed documents, False otherwise.

        Example:
            >>> if indexer.is_file_indexed(file_id):
            ...     print("Already indexed")
        """
        collection = self.chroma_client.get_or_create_collection()

        try:
            results = collection.get(where={"file_id": str(file_id)}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Failed to check if file {file_id} indexed: {e}")
            return False

    def get_indexed_count(self, file_id: UUID | None = None) -> int:
        """
        Get count of indexed descriptions.

        Args:
            file_id: If provided, count for specific file.
                If None, count all descriptions.

        Returns:
            Number of indexed descriptions.

        Example:
            >>> total = indexer.get_indexed_count()
            >>> file_count = indexer.get_indexed_count(file_id)
        """
        collection = self.chroma_client.get_or_create_collection()

        try:
            if file_id is None:
                # Count all documents
                return collection.count()
            else:
                # Count documents for specific file
                results = collection.get(where={"file_id": str(file_id)})
                return len(results.get("ids", []))
        except Exception as e:
            logger.error(f"Failed to get indexed count: {e}")
            return 0
