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
from typing import Any, Protocol
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ============================================================================
# PROTOCOLS (Dependency Inversion)
# ============================================================================


class FileStorageServiceProtocol(Protocol):
    """
    Protocol for FileStorageService dependency injection.

    This protocol defines the interface that FileStorageService must implement
    for the FileUploadUseCase to work. Enables dependency inversion and testing.

    Methods required from FileStorageService:
        - save_uploaded_file(): Save file to upload storage
        - extract_file_metadata(): Extract Excel metadata
        - extract_file_preview(): Extract preview rows
    """

    async def save_uploaded_file(
        self, file_id: UUID, file_data: bytes, filename: str
    ) -> Path:
        """
        Save uploaded file to temporary upload storage.

        Args:
            file_id: Unique identifier for uploaded file
            file_data: Raw file bytes from upload
            filename: Original filename from user

        Returns:
            Path to saved file

        Raises:
            ValueError: If extension is invalid
            FileSizeExceededError: If file too large
            OSError: If file cannot be written
        """
        ...

    async def extract_file_metadata(self, file_path: Path) -> dict:
        """
        Extract metadata from Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            Dict with metadata (filename, size, size_mb, sheets_count, rows_count, columns_count, created_at)

        Raises:
            FileNotFoundError: If file not found
            ExcelParsingError: If file cannot be parsed
        """
        ...

    async def extract_file_preview(
        self, file_path: Path, rows: int = 5
    ) -> list[dict]:
        """
        Extract preview of first N rows from Excel file.

        Args:
            file_path: Path to Excel file
            rows: Number of rows to extract (default 5)

        Returns:
            List of dicts (each dict = one row)

        Raises:
            FileNotFoundError: If file not found
            ExcelParsingError: If file cannot be parsed
        """
        ...


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

    Examples:
        >>> result = FileUploadResult(
        ...     file_id="a3bb189e-8bf9-3888-9912-ace4e6543002",
        ...     filename="catalog.xlsx",
        ...     size_mb=1.23,
        ...     sheets_count=2,
        ...     rows_count=150,
        ...     columns_count=8,
        ...     upload_time="2024-01-15T10:30:00",
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

    def __init__(self, file_storage: FileStorageServiceProtocol) -> None:
        """
        Initialize use case with dependencies.

        Args:
            file_storage: FileStorageService for file operations (dependency injection)

        Examples:
            >>> from src.infrastructure.file_storage import FileStorageService
            >>> file_storage = FileStorageService()
            >>> use_case = FileUploadUseCase(file_storage=file_storage)
        """
        self.file_storage = file_storage

    async def execute(self, file_data: bytes, filename: str) -> FileUploadResult:
        """
        Execute file upload with metadata extraction (Phase 2 - Detailed Contract).

        This is the main entry point for file upload process. Orchestrates all steps
        from file storage to metadata extraction and returns complete result.

        Process Flow:
        1. Generate unique file_id using uuid4()
        2. Get current timestamp for upload_time
        3. Call file_storage.save_uploaded_file() to save file to uploads/{file_id}/
        4. Call file_storage.extract_file_metadata() to get Excel metadata
        5. Call file_storage.extract_file_preview() to get first 5 rows
        6. Construct FileUploadResult with all data
        7. Return result to API Layer

        Args:
            file_data: Raw file bytes from multipart upload
            filename: Original filename from user (e.g., "my_catalog.xlsx")

        Returns:
            FileUploadResult with file_id, metadata, and preview

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
        raise NotImplementedError(
            "execute() to be implemented in Phase 3. "
            "Will orchestrate: generate file_id → save file → extract metadata → extract preview → return result."
        )
