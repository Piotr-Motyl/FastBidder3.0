"""
File Upload Use Case

Responsibility:
    Orchestrates file upload process with validation, storage, and metadata extraction.
    Coordinates between API Layer and Infrastructure Layer.

Architecture Notes:
    - Part of Application Layer (Services)
    - Uses FileStorageService from Infrastructure Layer
    - Called by API Layer (files.py router)
    - Returns FileUploadResult DTO
    - Generates file_id (UUID4) for uploaded files

Contains:
    - FileUploadUseCase: Main use case for file upload
    - FileUploadResult: DTO for upload result

Does NOT contain:
    - HTTP concerns (belongs to API Layer)
    - File system operations (delegated to Infrastructure Layer)
    - Business logic (simple orchestration only)

Phase 2 Note:
    Detailed contract with full process flow documentation.
    Implementation in Phase 3.
"""

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from src.application.ports.file_storage import FileStorageServiceProtocol
from src.infrastructure.ai.vector_store.reference_indexer import (
    ReferenceIndexer,
    IndexingResult,
)
from src.domain.hvac.entities.hvac_description import HVACDescription
from src.domain.shared.exceptions import InvalidHVACDescriptionError


# ============================================================================
# DATA TRANSFER OBJECTS (DTOs)
# ============================================================================


class FileUploadResult(BaseModel):
    """
    Result of file upload operation.

    Returned by FileUploadUseCase.execute() and used by API Layer
    to construct HTTP response.

    Attributes:
        file_id: Unique identifier for uploaded file (UUID as string)
        filename: Original filename from user
        size_mb: File size in megabytes (2 decimal places)
        sheets_count: Number of sheets in Excel file
        rows_count: Number of rows in first sheet
        columns_count: Number of columns in first sheet
        upload_time: Upload timestamp in ISO format
        preview: First 5 rows from first sheet as list of dicts
        file_type: Type of uploaded file ("working" or "reference")
        indexing_status: Status of reference indexing (if applicable):
            - "skipped": Working file (not indexed)
            - "success": All descriptions indexed successfully
            - "partial": Some descriptions failed to index
            - "failed": Indexing completely failed
            - None: No indexing attempted
        indexed_count: Number of successfully indexed descriptions (reference files only)

    Examples:
        >>> result = FileUploadResult(
        ...     file_id="a3bb189e-8bf9-3888-9912-ace4e6543002",
        ...     filename="catalog.xlsx",
        ...     size_mb=1.23,
        ...     sheets_count=2,
        ...     rows_count=150,
        ...     columns_count=8,
        ...     upload_time="2024-01-15T10:30:00",
        ...     file_type="reference",
        ...     indexing_status="success",
        ...     indexed_count=148,
        ...     preview=[
        ...         {"Description": "Zawór DN50", "Price": 123.45},
        ...         {"Description": "Rura DN100", "Price": 234.56}
        ...     ]
        ... )
    """

    file_id: str = Field(description="Unique file identifier (UUID as string)")

    filename: str = Field(description="Original filename from user")

    size_mb: float = Field(
        description="File size in megabytes (2 decimal places)", ge=0.0
    )

    sheets_count: int = Field(
        description="Number of sheets in Excel file", ge=1
    )

    rows_count: int = Field(
        description="Number of rows in first sheet (including header)", ge=0
    )

    columns_count: int = Field(description="Number of columns in first sheet", ge=0)

    upload_time: str = Field(description="Upload timestamp in ISO 8601 format")

    preview: list[dict[str, Any]] = Field(
        description="First 5 rows from first sheet as JSON",
        default_factory=list,
    )

    file_type: str = Field(
        description="File type: 'working' (for matching) or 'reference' (catalog)"
    )

    indexing_status: str | None = Field(
        default=None,
        description="Indexing status: 'skipped', 'success', 'partial', 'failed', or None",
    )

    indexed_count: int | None = Field(
        default=None, description="Number of indexed descriptions (reference files only)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
                "filename": "my_catalog_2024.xlsx",
                "size_mb": 1.23,
                "sheets_count": 2,
                "rows_count": 150,
                "columns_count": 8,
                "upload_time": "2024-01-15T10:30:00.123456",
                "preview": [
                    {
                        "Description": "Zawór kulowy DN50 PN16",
                        "Price": 123.45,
                        "Quantity": 10,
                    },
                    {
                        "Description": "Rura stalowa DN100",
                        "Price": 234.56,
                        "Quantity": 5,
                    },
                    {
                        "Description": "Kolano 90° DN50",
                        "Price": 45.67,
                        "Quantity": 20,
                    },
                ],
            }
        }


# ============================================================================
# USE CASE
# ============================================================================


class FileUploadUseCase:
    """
    Use case for uploading Excel files with metadata extraction.

    This use case orchestrates the complete file upload process:
    1. Generate unique file_id (UUID4)
    2. Save file to temporary upload storage
    3. Extract metadata (sheets, rows, columns, size)
    4. Extract preview (first 5 rows)
    5. Return FileUploadResult with all information

    The use case is called by API Layer and delegates all infrastructure
    operations to FileStorageService via dependency injection.

    Process Flow:
        User uploads file (multipart/form-data)
        → API Layer validates request
        → FileUploadUseCase.execute(file_data, filename)
        → Generate file_id (UUID4)
        → FileStorageService.save_uploaded_file()
        → FileStorageService.extract_file_metadata()
        → FileStorageService.extract_file_preview()
        → Return FileUploadResult
        → API Layer converts to HTTP 201 response

    Attributes:
        file_storage: FileStorageService for file operations (injected)

    Examples:
        >>> # Dependency injection in API Layer
        >>> from src.infrastructure.file_storage import FileStorageService
        >>> file_storage = FileStorageService()
        >>> use_case = FileUploadUseCase(file_storage=file_storage)
        >>>
        >>> # Execute upload
        >>> with open("catalog.xlsx", "rb") as f:
        ...     file_data = f.read()
        >>> result = await use_case.execute(
        ...     file_data=file_data,
        ...     filename="my_catalog.xlsx"
        ... )
        >>> print(result.file_id)
        a3bb189e-8bf9-3888-9912-ace4e6543002
        >>> print(result.filename)
        my_catalog.xlsx
        >>> print(f"{result.rows_count} rows, {result.columns_count} columns")
        150 rows, 8 columns

    Phase 2 Contract:
        This class defines the interface contract with detailed documentation.
        Actual implementation will be added in Phase 3.
    """

    def __init__(
        self,
        file_storage: FileStorageServiceProtocol,
        reference_indexer: ReferenceIndexer | None = None,
    ) -> None:
        """
        Initialize use case with dependencies.

        Args:
            file_storage: FileStorageService for file operations (dependency injection)
            reference_indexer: Optional ReferenceIndexer for indexing reference files.
                If None, reference files will be uploaded but not indexed.

        Examples:
            >>> from src.infrastructure.file_storage import FileStorageService
            >>> file_storage = FileStorageService()
            >>> use_case = FileUploadUseCase(file_storage=file_storage)
            >>>
            >>> # With indexing enabled
            >>> from src.infrastructure.ai.vector_store.reference_indexer import ReferenceIndexer
            >>> indexer = ReferenceIndexer(embedding_service, chroma_client)
            >>> use_case = FileUploadUseCase(file_storage=file_storage, reference_indexer=indexer)
        """
        self.file_storage = file_storage
        self.reference_indexer = reference_indexer

    async def execute(
        self, file_data: bytes, filename: str, file_type: str = "working"
    ) -> FileUploadResult:
        """
        Execute file upload with metadata extraction and optional indexing.

        This is the main entry point for file upload process. Orchestrates all steps
        from file storage to metadata extraction, and optionally indexes reference files.

        Process Flow:
        1. Generate unique file_id using uuid4()
        2. Get current timestamp for upload_time
        3. Call file_storage.save_uploaded_file() to save file to uploads/{file_id}/
        4. Call file_storage.extract_file_metadata() to get Excel metadata
        5. Call file_storage.extract_file_preview() to get first 5 rows
        6. If file_type == "reference" and reference_indexer is available:
           - Index descriptions into vector database
           - Record indexing status and count
        7. Construct FileUploadResult with all data
        8. Return result to API Layer

        Args:
            file_data: Raw file bytes from multipart upload
            filename: Original filename from user (e.g., "my_catalog.xlsx")
            file_type: Type of file being uploaded:
                - "working": File for matching (not indexed)
                - "reference": Reference catalog (indexed for semantic search)
                Defaults to "working".

        Returns:
            FileUploadResult with file_id, metadata, preview, and indexing status

        Raises:
            ValueError: If file extension is invalid (not .xlsx/.xls)
            FileSizeExceededError: If file size > 10MB
            ExcelParsingError: If file cannot be parsed as Excel
            FileNotFoundError: If file disappears after save (rare)
            OSError: If file cannot be written to disk

        Examples:
            >>> use_case = FileUploadUseCase(file_storage=file_storage)
            >>> with open("catalog.xlsx", "rb") as f:
            ...     file_data = f.read()
            >>> result = await use_case.execute(
            ...     file_data=file_data,
            ...     filename="catalog.xlsx"
            ... )
            >>> print(f"Uploaded: {result.file_id}")
            >>> print(f"Size: {result.size_mb:.2f} MB")
            >>> print(f"Structure: {result.sheets_count} sheets, {result.rows_count} rows")
            >>> print(f"Preview: {len(result.preview)} rows")

        Implementation Note (Phase 3):
            - Generate file_id:
                * file_id = uuid4()  # Random UUID4
            - Get timestamp:
                * upload_time = datetime.now().isoformat()
            - Save file:
                * file_path = await self.file_storage.save_uploaded_file(
                    file_id=file_id,
                    file_data=file_data,
                    filename=filename
                )
            - Extract metadata:
                * metadata = await self.file_storage.extract_file_metadata(file_path)
            - Extract preview:
                * preview = await self.file_storage.extract_file_preview(file_path, rows=5)
            - Construct result:
                * return FileUploadResult(
                    file_id=str(file_id),  # Convert UUID to string
                    filename=metadata['filename'],
                    size_mb=metadata['size_mb'],
                    sheets_count=metadata['sheets_count'],
                    rows_count=metadata['rows_count'],
                    columns_count=metadata['columns_count'],
                    upload_time=upload_time,
                    preview=preview
                )

        Error Handling:
            All exceptions from FileStorageService propagate to API Layer:
            - ValueError → HTTP 400 Bad Request
            - FileSizeExceededError → HTTP 413 Payload Too Large
            - ExcelParsingError → HTTP 422 Unprocessable Entity
            - OSError → HTTP 500 Internal Server Error

        Architecture Note:
            - Use case does NOT handle exceptions (let them propagate)
            - Use case does NOT perform validation (delegated to FileStorageService)
            - Use case is THIN orchestration layer only
            - API Layer handles exception → HTTP status mapping

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        # 1. Generate unique file_id
        file_id = uuid4()

        # 2. Get current timestamp for upload_time
        upload_time = datetime.now().isoformat()

        # 3. Save file to uploads/{file_id}/
        file_path = await self.file_storage.save_uploaded_file(
            file_id=file_id, file_data=file_data, filename=filename
        )

        # 4. Extract metadata (sheets, rows, columns, size)
        metadata = await self.file_storage.extract_file_metadata(file_path)

        # 5. Extract preview (first 5 rows)
        preview = await self.file_storage.extract_file_preview(file_path, rows=5)

        # 6. Index reference files (if file_type is "reference" and indexer is available)
        indexing_status: str | None = None
        indexed_count: int | None = None

        if file_type == "reference":
            if self.reference_indexer is not None:
                try:
                    # Extract descriptions from file for indexing
                    descriptions = await self._extract_descriptions_from_file(
                        file_path, file_id
                    )

                    # Index descriptions into vector database
                    indexing_result = self.reference_indexer.index_file(
                        file_id, descriptions
                    )

                    # Determine indexing status based on result
                    if indexing_result.indexed_count == 0:
                        indexing_status = "failed"
                    elif indexing_result.failed_count > 0:
                        indexing_status = "partial"
                    else:
                        indexing_status = "success"

                    indexed_count = indexing_result.indexed_count

                except Exception as e:
                    # Don't block upload if indexing fails
                    indexing_status = "failed"
                    indexed_count = 0
                    # Log error but continue (upload succeeded even if indexing failed)
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.error(f"Indexing failed for file {file_id}: {e}")
            else:
                # No indexer configured
                indexing_status = "skipped"
                indexed_count = 0
        else:
            # Working file - not indexed
            indexing_status = "skipped"
            indexed_count = None

        # 7. Construct and return result
        return FileUploadResult(
            file_id=str(file_id),  # Convert UUID to string for JSON serialization
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

    async def _extract_descriptions_from_file(
        self, file_path: Path, file_id: UUID
    ) -> list[HVACDescription]:
        """
        Extract HVAC descriptions from Excel file for indexing.

        Reads all rows from the first sheet (skipping header) and creates
        HVACDescription entities. Only valid descriptions (non-empty text)
        are included.

        Args:
            file_path: Path to uploaded Excel file
            file_id: UUID of the uploaded file

        Returns:
            List of HVACDescription entities ready for indexing

        Raises:
            ExcelParsingError: If file cannot be read as Excel
            ValueError: If description validation fails

        Note:
            - Uses pandas to read Excel file
            - Assumes first column contains descriptions
            - Skips empty rows and invalid descriptions
            - Row numbers are 1-indexed (Excel format)
        """
        import pandas as pd

        # Read Excel file (first sheet only)
        df = pd.read_excel(file_path, sheet_name=0)

        descriptions: list[HVACDescription] = []

        # Iterate through rows and create HVACDescription entities
        for idx, row in df.iterrows():
            # Get first column value (description text)
            # idx is 0-based, but we want 1-based row numbers (row 1 = first data row after header)
            row_number = int(idx) + 2  # +1 for 0-based to 1-based, +1 for header row

            # Get first column value (assuming it contains description)
            first_column = df.columns[0]
            description_text = str(row[first_column]).strip()

            # Skip empty or invalid descriptions
            if not description_text or description_text == "nan":
                continue

            try:
                # Create HVACDescription entity
                desc = HVACDescription(
                    raw_text=description_text,
                    source_row_number=row_number,
                    file_id=file_id,
                )
                descriptions.append(desc)
            except InvalidHVACDescriptionError:
                # Skip invalid descriptions (e.g., too short)
                continue

        return descriptions
