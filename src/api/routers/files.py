"""
API Router for File Upload

Responsibility:
    HTTP interface for uploading Excel files with metadata extraction.
    Thin layer that delegates to Application Layer use case via dependency injection.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Depends on Application Layer (FileUploadUseCase - Task 2.4.1)
    - Returns 201 Created for successful uploads (REST best practice)
    - No business logic - pure HTTP concerns
    - Uses dependency injection pattern (NOT direct service calls!)

Contains:
    - POST /files/upload - Upload Excel file with metadata extraction

Does NOT contain:
    - Business logic (delegated to Domain Layer)
    - File processing (delegated to Infrastructure Layer)
    - Excel parsing (delegated to Infrastructure Layer)
    - Direct file system access (delegated to Infrastructure Layer)
    - Job status tracking (separate router: jobs.py)

Phase 2 Note:
    This is a Phase 2 DETAILED CONTRACT. Implementation will be added in Phase 3.
    All endpoints raise NotImplementedError.
"""

import logging
from typing import Any, Dict, Optional, Literal

from fastapi import APIRouter, status, HTTPException, Depends, UploadFile, File, Query
from pydantic import BaseModel, Field

# Import Application Layer use case
from src.application.services.file_upload_use_case import FileUploadUseCase
from src.domain.shared.exceptions import (
    FileSizeExceededError,
    ExcelParsingError,
)

# Import Infrastructure services for dependency injection
from src.infrastructure.file_storage.file_storage_service import FileStorageService

# Import shared API schemas
from src.api.schemas.common import ErrorResponse

# Configure logger
logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class UploadFileResponse(BaseModel):
    """
    Response model for successful file upload.

    Returned with HTTP 201 Created to indicate file was uploaded and processed.
    Contains complete file metadata and preview for user confirmation.

    Attributes:
        file_id: Unique identifier for uploaded file (use in matching requests)
        filename: Original filename from user
        size_mb: File size in megabytes (2 decimal places)
        sheets_count: Number of sheets in Excel file
        rows_count: Number of rows in first sheet (including header)
        columns_count: Number of columns in first sheet
        upload_time: Upload timestamp in ISO 8601 format
        preview: First 5 rows from first sheet as JSON array
        message: Human-readable success message

    Usage Note:
        User should use `file_id` when submitting matching requests:
        POST /matching/process with working_file.file_id or reference_file.file_id

    Phase 2 Extensions:
        - preview: Allows user to verify file structure before processing
        - sheets_count, rows_count, columns_count: Help user select correct ranges
    """

    file_id: str = Field(description="Unique file identifier (UUID as string)")

    filename: str = Field(description="Original filename from user")

    size_mb: float = Field(
        description="File size in megabytes (2 decimal places)", ge=0.0
    )

    sheets_count: int = Field(description="Number of sheets in Excel file", ge=1)

    rows_count: int = Field(
        description="Number of rows in first sheet (including header)", ge=0
    )

    columns_count: int = Field(description="Number of columns in first sheet", ge=0)

    upload_time: str = Field(description="Upload timestamp in ISO 8601 format")

    preview: list[Dict[str, Any]] = Field(
        description="First 5 rows from first sheet as JSON array (for user verification)",
        default_factory=list,
    )

    message: str = Field(
        default="File uploaded successfully. Use file_id for matching requests.",
        description="Human-readable success message",
    )

    # Phase 4: AI Matching fields for vector DB indexing
    file_type: Literal["working", "reference"] = Field(
        default="working",
        description="File type for AI indexing (working or reference)",
    )

    indexing_status: Optional[str] = Field(
        default=None,
        description="ChromaDB indexing status (pending/completed/failed) - Phase 4",
    )

    indexed_count: Optional[int] = Field(
        default=None,
        description="Number of items indexed in vector DB - Phase 4",
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
                "message": "File uploaded successfully. Use file_id in matching requests.",
                "file_type": "reference",
                "indexing_status": "completed",
                "indexed_count": 150,
            }
        }


# ============================================================================
# ROUTER CONFIGURATION
# ============================================================================


router = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Bad Request - Invalid file extension",
        },
        413: {
            "model": ErrorResponse,
            "description": "Payload Too Large - File size exceeds 10MB limit",
        },
        422: {
            "model": ErrorResponse,
            "description": "Unprocessable Entity - File cannot be parsed as Excel",
        },
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================


async def get_file_upload_use_case():
    """
    Dependency injection for FileUploadUseCase.

    Returns:
        FileUploadUseCase: Application Layer use case for file upload

    Note:
        Implementation in Phase 3 (Task 3.4.1).
        Will inject actual use case with all its dependencies:
        - FileStorageService for file operations
        - Configuration for max file size and allowed extensions

    The use case will:
    - Generate unique file_id (UUID4)
    - Validate file extension and size
    - Save file to /tmp/fastbidder/uploads/{file_id}/
    - Extract metadata (sheets, rows, columns)
    - Extract preview (first 5 rows)
    - Return FileUploadResult
    """
    # Create FileStorageService instance
    file_storage = FileStorageService()

    # Create and return FileUploadUseCase with injected dependencies
    return FileUploadUseCase(file_storage=file_storage)


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadFileResponse,
    summary="Upload Excel file with metadata extraction",
    description=(
        "Upload Excel file (.xlsx or .xls) and receive file_id for matching requests. "
        "Returns metadata (sheets, rows, columns, size) and preview of first 5 rows. "
        "Max file size: 10MB. "
        "Use returned file_id in POST /matching/process requests."
    ),
    responses={
        201: {
            "description": "Success - File uploaded and metadata extracted",
            "model": UploadFileResponse,
        },
        400: {
            "description": "Bad Request - Invalid file extension (must be .xlsx or .xls)",
        },
        413: {"description": "Payload Too Large - File exceeds 10MB limit"},
        422: {
            "description": "Unprocessable Entity - File cannot be parsed as valid Excel"
        },
    },
)
async def upload_file(
    file: UploadFile = File(
        ...,
        description="Excel file to upload (.xlsx or .xls, max 10MB)",
    ),
    file_type: Literal["working", "reference"] = Query(
        default="working",
        description="File type for AI indexing: 'working' (to be matched) or 'reference' (catalog for matching against)",
    ),
    use_case=Depends(get_file_upload_use_case),
) -> UploadFileResponse:
    """
    Upload Excel file with metadata extraction and preview.

    Process Flow:
        1. Receive multipart/form-data file upload from user
        2. Read file data into memory (FastAPI handles this)
        3. Delegate to Application Layer use case
        4. Use case generates file_id and saves file to uploads/{file_id}/
        5. Use case extracts metadata (sheets, rows, columns, size)
        6. Use case extracts preview (first 5 rows from first sheet)
        7. Return 201 Created with file_id, metadata, and preview

    Args:
        file: Uploaded file from multipart/form-data (FastAPI UploadFile)
        use_case: Injected FileUploadUseCase from Application Layer

    Returns:
        UploadFileResponse with file_id, metadata, preview, and success message

    Raises:
        HTTPException 400: If file extension is not .xlsx or .xls
        HTTPException 413: If file size exceeds 10MB limit
        HTTPException 422: If file cannot be parsed as valid Excel
        HTTPException 500: If unexpected error during processing

    Architecture Note:
        This endpoint is a thin wrapper around Application Layer.
        All business logic is delegated to FileUploadUseCase.
        No direct file system access - follows dependency inversion principle.

    Examples:
        >>> # Using curl
        >>> curl -X POST "http://localhost:8000/api/files/upload" \\
        ...      -H "Content-Type: multipart/form-data" \\
        ...      -F "file=@catalog.xlsx"
        {
            "file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002",
            "filename": "catalog.xlsx",
            "size_mb": 1.23,
            "sheets_count": 2,
            "rows_count": 150,
            "columns_count": 8,
            "upload_time": "2024-01-15T10:30:00",
            "preview": [
                {"Description": "Zawór DN50", "Price": 123.45},
                {"Description": "Rura DN100", "Price": 234.56}
            ],
            "message": "File uploaded successfully. Use file_id in matching requests."
        }

        >>> # Using Python requests
        >>> import requests
        >>> with open("catalog.xlsx", "rb") as f:
        ...     response = requests.post(
        ...         "http://localhost:8000/api/files/upload",
        ...         files={"file": f}
        ...     )
        >>> data = response.json()
        >>> print(data["file_id"])
        a3bb189e-8bf9-3888-9912-ace4e6543002

        >>> # Then use file_id in matching request
        >>> matching_request = {
        ...     "working_file": {
        ...         "file_id": data["file_id"],
        ...         "description_column": "B",
        ...         ...
        ...     }
        ... }
    """
    # CONTRACT ONLY - Implementation in Phase 3
    #
    # Implementation will:
    # 1. Read file data from UploadFile
    # 2. Convert to bytes
    # 3. Call use_case.execute(file_data, filename)
    # 4. Convert FileUploadResult to UploadFileResponse
    # 5. Return response with 201 Created
    #
    # Example implementation:
    # from src.domain.shared.exceptions import FileSizeExceededError, ExcelParsingError
    #
    # try:
    #     # Read file data
    #     file_data = await file.read()
    #
    #     # Execute use case
    #     result = await use_case.execute(
    #         file_data=file_data,
    #         filename=file.filename
    #     )
    #
    #     # Convert to response
    #     return UploadFileResponse(
    #         file_id=result.file_id,
    #         filename=result.filename,
    #         size_mb=result.size_mb,
    #         sheets_count=result.sheets_count,
    #         rows_count=result.rows_count,
    #         columns_count=result.columns_count,
    #         upload_time=result.upload_time,
    #         preview=result.preview,
    #         message="File uploaded successfully. Use file_id in matching requests."
    #     )
    #
    # except ValueError as e:
    #     # Invalid file extension
    #     raise HTTPException(
    #         status_code=400,
    #         detail=ErrorResponse(
    #             code="INVALID_FILE_EXTENSION",
    #             message=str(e),
    #             details={"filename": file.filename}
    #         ).dict()
    #     )
    #
    # except FileSizeExceededError as e:
    #     # File too large
    #     raise HTTPException(
    #         status_code=413,
    #         detail=ErrorResponse(
    #             code="FILE_TOO_LARGE",
    #             message=str(e),
    #             details={"filename": file.filename, "max_size_mb": 10}
    #         ).dict()
    #     )
    #
    # except ExcelParsingError as e:
    #     # Cannot parse Excel file
    #     raise HTTPException(
    #         status_code=422,
    #         detail=ErrorResponse(
    #             code="EXCEL_PARSING_ERROR",
    #             message=str(e),
    #             details={"filename": file.filename}
    #         ).dict()
    #     )
    #
    # except Exception as e:
    #     # Unexpected error
    #     logger.error(f"Unexpected error during file upload: {e}")
    #     raise HTTPException(
    #         status_code=500,
    #         detail=ErrorResponse(
    #             code="INTERNAL_SERVER_ERROR",
    #             message="An unexpected error occurred during file upload",
    #             details={"error": str(e)}
    #         ).dict()
    #     )

    # Implementation based on Phase 2 contract
    try:
        # Read file data
        file_data = await file.read()

        # Execute use case
        result = await use_case.execute(file_data=file_data, filename=file.filename)

        # Convert to response (Phase 4: include file_type for AI indexing)
        return UploadFileResponse(
            file_id=result.file_id,
            filename=result.filename,
            size_mb=result.size_mb,
            sheets_count=result.sheets_count,
            rows_count=result.rows_count,
            columns_count=result.columns_count,
            upload_time=result.upload_time,
            preview=result.preview,
            message="File uploaded successfully. Use file_id in matching requests.",
            file_type=file_type,  # Phase 4: AI indexing support
            indexing_status=None,  # Phase 4: Will be updated by async indexing task
            indexed_count=None,  # Phase 4: Will be populated after indexing completes
        )

    except ValueError as e:
        # Invalid file extension
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE_EXTENSION",
                "message": str(e),
                "details": {"filename": file.filename},
            },
        )

    except FileSizeExceededError as e:
        # File too large
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": str(e),
                "details": {"filename": file.filename, "max_size_mb": 10},
            },
        )

    except ExcelParsingError as e:
        # Cannot parse Excel file
        raise HTTPException(
            status_code=422,
            detail={
                "code": "EXCEL_PARSING_ERROR",
                "message": str(e),
                "details": {"filename": file.filename},
            },
        )

    except Exception as e:
        # Unexpected error
        logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during file upload",
                "details": {"error": str(e)},
            },
        )
