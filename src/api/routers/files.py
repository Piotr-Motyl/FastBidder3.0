"""HTTP interface for file upload (POST /files/upload).

Working-file uploads skip AI indexing entirely. Reference-file uploads always
attempt ChromaDB indexing — the USE_AI_MATCHING env-var controls the matching
engine at process time, not whether the upload is indexed.
"""

import logging
from typing import Any, Dict, Optional, Literal

from fastapi import APIRouter, status, HTTPException, Depends, UploadFile, File, Query
from pydantic import BaseModel, ConfigDict, Field

from src.application.services.file_upload_use_case import FileUploadUseCase
from src.domain.shared.exceptions import (
    FileSizeExceededError,
    ExcelParsingError,
)
from src.infrastructure.file_storage.file_storage_service import FileStorageService
from src.api.schemas.common import ErrorResponse

logger = logging.getLogger(__name__)


class UploadFileResponse(BaseModel):
    """Returned with 201 Created. Includes file_id (use in /matching/process)
    and metadata + 5-row preview for the user to verify file structure."""

    file_id: str = Field(description="Unique file identifier (UUID as string)")
    filename: str = Field(description="Original filename from user")
    size_mb: float = Field(ge=0.0, description="File size in megabytes")
    sheets_count: int = Field(ge=1, description="Number of sheets in Excel file")
    rows_count: int = Field(ge=0, description="Rows in first sheet (incl. header)")
    columns_count: int = Field(ge=0, description="Columns in first sheet")
    upload_time: str = Field(description="Upload timestamp (ISO 8601)")
    preview: list[Dict[str, Any]] = Field(
        default_factory=list,
        description="First 5 rows from first sheet (for user verification)",
    )
    message: str = Field(
        default="File uploaded successfully. Use file_id for matching requests."
    )
    file_type: Literal["working", "reference"] = Field(default="working")
    indexing_status: Optional[str] = Field(
        default=None,
        description="ChromaDB indexing status: success/partial/failed/skipped",
    )
    indexed_count: Optional[int] = Field(
        default=None, description="Number of items indexed in vector DB"
    )

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
                    {"Description": "Rura stalowa DN100", "Price": 234.56, "Quantity": 5},
                    {"Description": "Kolano 90° DN50", "Price": 45.67, "Quantity": 20},
                ],
                "message": "File uploaded successfully. Use file_id in matching requests.",
                "file_type": "reference",
                "indexing_status": "completed",
                "indexed_count": 150,
            }
        }
    )


router = APIRouter(
    prefix="/files",
    tags=["files"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file extension"},
        413: {"model": ErrorResponse, "description": "File exceeds 10MB limit"},
        422: {"model": ErrorResponse, "description": "File cannot be parsed as Excel"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)


async def get_file_upload_use_case(
    file_type: Literal["working", "reference"] = Query(default="working"),
):
    """
    Read file_type from query string to decide whether to wire up ReferenceIndexer.
    Skipping it for working-file uploads avoids loading the embedding model on
    every upload.
    """
    file_storage = FileStorageService()
    reference_indexer = None

    if file_type == "reference":
        try:
            from src.infrastructure.ai.embeddings.embedding_service import (
                EmbeddingService,
            )
            from src.infrastructure.ai.vector_store.chroma_client import (
                ChromaClientSingleton,
            )
            from src.infrastructure.ai.vector_store.reference_indexer import (
                ReferenceIndexer,
            )

            embedding_service = EmbeddingService()
            chroma_client = ChromaClientSingleton.get_instance()
            reference_indexer = ReferenceIndexer(embedding_service, chroma_client)
            logger.info("ReferenceIndexer initialised for reference file upload")

        except Exception as e:
            # Don't block upload on indexer init failure — file will be saved
            # but not indexed; matching against it will fall back to non-AI engine.
            logger.warning(
                f"Failed to initialise ReferenceIndexer — file will be uploaded "
                f"but not indexed for AI matching: {e}"
            )
            reference_indexer = None

    return FileUploadUseCase(file_storage=file_storage, reference_indexer=reference_indexer)


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    response_model=UploadFileResponse,
    summary="Upload Excel file with metadata extraction",
    description=(
        "Upload Excel file (.xlsx or .xls) and receive file_id for matching requests. "
        "Returns metadata (sheets, rows, columns, size) and preview of first 5 rows. "
        "Max file size: 10MB."
    ),
    responses={
        201: {"description": "File uploaded and metadata extracted", "model": UploadFileResponse},
        400: {"description": "Invalid file extension (must be .xlsx or .xls)"},
        413: {"description": "File exceeds 10MB limit"},
        422: {"description": "File cannot be parsed as valid Excel"},
    },
)
async def upload_file(
    file: UploadFile = File(..., description="Excel file (.xlsx or .xls, max 10MB)"),
    file_type: Literal["working", "reference"] = Query(
        default="working",
        description=(
            "'working' = to be matched (no indexing); "
            "'reference' = catalog (indexed into ChromaDB for AI semantic search)"
        ),
    ),
    use_case=Depends(get_file_upload_use_case),
) -> UploadFileResponse:
    """Thin HTTP wrapper around FileUploadUseCase. Maps domain exceptions → HTTP codes."""
    try:
        file_data = await file.read()

        result = await use_case.execute(
            file_data=file_data, filename=file.filename, file_type=file_type
        )

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
            file_type=file_type,
            indexing_status=None,  # populated by async indexing task once it completes
            indexed_count=None,
        )

    except ValueError as e:
        # Raised on invalid file extension by FileStorageService.
        raise HTTPException(
            status_code=400,
            detail={
                "code": "INVALID_FILE_EXTENSION",
                "message": str(e),
                "details": {"filename": file.filename},
            },
        )

    except FileSizeExceededError as e:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "FILE_TOO_LARGE",
                "message": str(e),
                "details": {"filename": file.filename, "max_size_mb": 10},
            },
        )

    except ExcelParsingError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "EXCEL_PARSING_ERROR",
                "message": str(e),
                "details": {"filename": file.filename},
            },
        )

    except Exception as e:
        logger.error(f"Unexpected error during file upload: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during file upload",
                "details": {"error": str(e)},
            },
        )
