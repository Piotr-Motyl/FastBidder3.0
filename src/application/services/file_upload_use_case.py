"""
FileUploadUseCase — orchestrates file upload + metadata extraction + optional indexing.

Working files: upload only.
Reference files: upload + ChromaDB indexing (if indexer is injected).
Indexing failures don't block the upload — file is saved either way.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.application.ports.file_storage import FileStorageServiceProtocol
from src.infrastructure.ai.vector_store.reference_indexer import ReferenceIndexer
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.shared.exceptions import InvalidHVACDescriptionError

logger = logging.getLogger(__name__)


class FileUploadResult(BaseModel):
    """
    DTO returned by FileUploadUseCase.execute(). Consumed by API layer for HTTP response.

    indexing_status values:
      - 'skipped': working file, or reference file with no indexer wired up
      - 'success': all descriptions indexed
      - 'partial': some descriptions failed
      - 'failed':  no descriptions indexed
      - None:      not applicable
    """

    file_id: str = Field(description="Unique file identifier (UUID as string)")
    filename: str
    size_mb: float = Field(ge=0.0)
    sheets_count: int = Field(ge=1)
    rows_count: int = Field(ge=0, description="Rows in first sheet (incl. header)")
    columns_count: int = Field(ge=0)
    upload_time: str = Field(description="ISO 8601 timestamp")
    preview: list[dict[str, Any]] = Field(default_factory=list)
    file_type: str = Field(description="'working' or 'reference'")
    indexing_status: str | None = Field(default=None)
    indexed_count: int | None = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                "filename": "my_catalog_2024.xlsx",
                "size_mb": 1.23,
                "sheets_count": 2,
                "rows_count": 150,
                "columns_count": 8,
                "upload_time": "2024-01-15T10:30:00.123456",
                "preview": [
                    {"Description": "Zawór kulowy DN50 PN16", "Price": 123.45, "Quantity": 10},
                ],
                "file_type": "reference",
                "indexing_status": "success",
                "indexed_count": 148,
            }
        }
    )


class FileUploadUseCase:
    """Generates file_id, saves file, extracts metadata + preview, and for reference
    files attempts vector-DB indexing. Exceptions propagate to the API layer."""

    def __init__(
        self,
        file_storage: FileStorageServiceProtocol,
        reference_indexer: ReferenceIndexer | None = None,
    ) -> None:
        self.file_storage = file_storage
        self.reference_indexer = reference_indexer

    async def execute(
        self, file_data: bytes, filename: str, file_type: str = "working"
    ) -> FileUploadResult:
        """
        Save file, extract metadata, and (for reference files) index descriptions.

        Raises:
            ValueError: invalid file extension (not .xlsx/.xls).
            FileSizeExceededError: file > 10MB.
            ExcelParsingError: file can't be parsed as Excel.
            OSError: filesystem failure.
        """
        file_id = uuid4()
        upload_time = datetime.now().isoformat()

        file_path = await self.file_storage.save_uploaded_file(
            file_id=file_id, file_data=file_data, filename=filename
        )
        metadata = await self.file_storage.extract_file_metadata(file_path)
        preview = await self.file_storage.extract_file_preview(file_path, rows=5)

        indexing_status, indexed_count = await self._maybe_index(
            file_type=file_type, file_path=file_path, file_id=file_id
        )

        return FileUploadResult(
            file_id=str(file_id),
            filename=metadata["filename"],
            size_mb=metadata["size_mb"],
            sheets_count=metadata["sheets_count"],
            rows_count=metadata["rows_count"],
            columns_count=metadata["columns_count"],
            upload_time=upload_time,
            preview=preview,
            file_type=file_type,
            indexing_status=indexing_status,
            indexed_count=indexed_count,
        )

    async def _maybe_index(
        self, file_type: str, file_path: Path, file_id: UUID
    ) -> tuple[str, int | None]:
        """Index reference files into ChromaDB. Failure is non-fatal (file is still saved)."""
        if file_type != "reference":
            return ("skipped", None)

        if self.reference_indexer is None:
            return ("skipped", 0)

        try:
            descriptions = await self._extract_descriptions_from_file(file_path, file_id)
            indexing_result = self.reference_indexer.index_file(file_id, descriptions)

            if indexing_result.indexed_count == 0:
                status = "failed"
            elif indexing_result.failed_count > 0:
                status = "partial"
            else:
                status = "success"

            return (status, indexing_result.indexed_count)

        except Exception as e:
            # Indexing is best-effort: file is already on disk, so don't fail the upload.
            logger.error(f"Indexing failed for file {file_id}: {e}")
            return ("failed", 0)

    async def _extract_descriptions_from_file(
        self, file_path: Path, file_id: UUID
    ) -> list[HVACDescription]:
        """Read first column of first sheet → HVACDescription entities, skipping empty cells."""
        import polars as pl

        # openpyxl is the project-standard backend (matches FileStorageService).
        df = pl.read_excel(file_path, sheet_id=1, engine="openpyxl")

        descriptions: list[HVACDescription] = []
        first_column = df.columns[0]

        for idx, row in enumerate(df.iter_rows(named=True)):
            # +2: idx is 0-based and the header occupies Excel row 1.
            row_number = idx + 2
            description_text = str(row[first_column]).strip()

            if not description_text or description_text == "None":
                continue

            try:
                desc = HVACDescription(
                    raw_text=description_text,
                    source_row_number=row_number,
                    file_id=file_id,
                )
                descriptions.append(desc)
            except InvalidHVACDescriptionError:
                # Invalid description (too short, etc.) — skip but keep going.
                continue

        return descriptions
