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
    """Result of indexing a reference file into ChromaDB: counts, errors, timing."""

    file_id: UUID
    total_descriptions: int
    indexed_count: int
    failed_count: int
    errors: list[str] = field(default_factory=list)
    indexing_time_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        """Success rate as percentage (0-100%). Returns 0.0 if no descriptions."""
        if self.total_descriptions == 0:
            return 0.0
        return (self.indexed_count / self.total_descriptions) * 100


class ReferenceIndexer:
    """
    Pre-embeds reference descriptions and stores them with metadata in ChromaDB.

    Batch size: 32. Idempotent (skip_if_indexed=True by default). Partial failure handling.
    Document ID: "{file_id}_{row_number}". Metadata stored: file_id, source_row_number, dn, pn, material, valve_type.
    """

    # Batch size for embedding generation (memory efficiency)
    BATCH_SIZE = 32

    def __init__(
        self,
        embedding_service: EmbeddingService,
        chroma_client: ChromaClient,
    ) -> None:
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
        Embed descriptions and store in ChromaDB with metadata.

        skip_if_indexed=False forces re-index (removes old data first).
        Returns IndexingResult with counts, errors, and timing.
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
        """Remove all ChromaDB documents for this file. Returns count removed."""
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
        """Return True if file has any indexed documents in ChromaDB."""
        collection = self.chroma_client.get_or_create_collection()

        try:
            results = collection.get(where={"file_id": str(file_id)}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception as e:
            logger.error(f"Failed to check if file {file_id} indexed: {e}")
            return False

    def get_indexed_count(self, file_id: UUID | None = None) -> int:
        """Return indexed count for specific file, or total count if file_id is None."""
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
