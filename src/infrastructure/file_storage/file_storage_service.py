"""
File Storage Service

Manages uploaded files and temporary file storage during job processing.
Handles file validation, upload, download, and cleanup.

Responsibility:
    - Upload/download Excel files
    - File validation (extension, size, basic structure)
    - Temporary file management (/tmp/fastbidder/{job_id}/)
    - Manual cleanup (called by Application Layer)
    - Implements FileStorageServiceProtocol from Application Layer

Architecture Notes:
    - Infrastructure Layer (file system operations)
    - Implements Application Layer Protocol interface
    - Used by Application Layer (ProcessMatchingUseCase)
    - Phase 1: Local file system only
    - Phase 5: Cloud storage (S3/GCS)
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from uuid import UUID


class FileStorageService:
    """
    Service for managing file uploads and temporary storage.

    Implements FileStorageServiceProtocol from Application Layer.
    This service handles all file system operations for FastBidder:
    - Validating uploaded Excel files
    - Storing files in job-specific directories
    - Retrieving files for processing
    - Cleaning up after job completion
    - Checking file existence and metadata (Protocol methods)

    Storage Structure:
        Base directory: /tmp/fastbidder/ (from env: TEMP_DIR)
        Job directories: /tmp/fastbidder/{job_id}/
        Files: /tmp/fastbidder/{job_id}/working_file.xlsx
               /tmp/fastbidder/{job_id}/reference_file.xlsx
               /tmp/fastbidder/{job_id}/result.xlsx

    Business Rules:
        - Max file size: 10MB (from env: MAX_FILE_SIZE_MB)
        - Allowed extensions: .xlsx, .xls (from env: ALLOWED_EXTENSIONS)
        - Each job has isolated directory
        - Manual cleanup (no automatic deletion)

    Phase 1 Scope:
        - Local file system only
        - Validation methods in this class (not separate module)
        - Manual cleanup (no TTL-based cleanup)
        - Implements FileStorageServiceProtocol for DI

    Examples:
        >>> service = FileStorageService()
        >>>
        >>> # Check if file exists (Protocol method)
        >>> exists = await service.file_exists(UUID("..."))
        >>>
        >>> # Upload file
        >>> file_path = await service.upload_file(
        ...     job_id=UUID("..."),
        ...     file_data=bytes_data,
        ...     filename="working_file.xlsx",
        ...     file_type="working"
        ... )
        >>>
        >>> # Get metadata (Protocol method)
        >>> metadata = await service.get_file_metadata(UUID("..."))
        >>> print(metadata['size'], metadata['format'])
        >>>
        >>> # Cleanup after job
        >>> await service.cleanup_job(UUID("..."))
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        max_size_mb: Optional[int] = None,
        allowed_extensions: Optional[list[str]] = None,
    ) -> None:
        """
        Initialize file storage service with configuration.

        Args:
            base_dir: Base directory for temp files (default from env: TEMP_DIR)
            max_size_mb: Max file size in MB (default from env: MAX_FILE_SIZE_MB)
            allowed_extensions: List of allowed file extensions (default from env)

        Raises:
            OSError: If base directory cannot be created
        """
        self.base_dir = Path(base_dir or os.getenv("TEMP_DIR", "/tmp/fastbidder"))
        self.max_size_bytes = (
            (max_size_mb or int(os.getenv("MAX_FILE_SIZE_MB", "10"))) * 1024 * 1024
        )  # Convert MB to bytes

        # Parse allowed extensions from env (comma-separated)
        extensions_str = os.getenv("ALLOWED_EXTENSIONS", ".xlsx,.xls")
        self.allowed_extensions = allowed_extensions or extensions_str.split(",")

        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def file_exists(self, file_id: UUID) -> bool:
        """
        Check if file exists in storage (implements Protocol method).

        This method is required by FileStorageServiceProtocol from Application Layer.
        Used by ProcessMatchingUseCase for business rule validation.

        Args:
            file_id: UUID of the file to check

        Returns:
            True if file exists, False otherwise

        Examples:
            >>> service = FileStorageService()
            >>> exists = await service.file_exists(UUID("3fa85f64-..."))
            >>> if exists:
            ...     print("File found in storage")

        Implementation Note:
            Phase 1: Check if job directory exists
            Phase 3: Check specific file types (working/reference)
            Phase 5: Check cloud storage
        """
        raise NotImplementedError(
            "file_exists() to be implemented in Phase 3. "
            "Will check if job directory or specific file exists in storage."
        )

    async def get_file_metadata(self, file_id: UUID) -> dict:
        """
        Get file metadata like size and format (implements Protocol method).

        This method is required by FileStorageServiceProtocol from Application Layer.
        Used by ProcessMatchingUseCase for estimating processing time.

        Args:
            file_id: UUID of the file

        Returns:
            Dict with file metadata:
            {
                'size': 1024000,        # Size in bytes
                'format': 'xlsx',       # File extension
                'exists': True,         # Whether file exists
                'created_at': '...'    # ISO timestamp
            }

        Raises:
            FileNotFoundError: If file does not exist

        Examples:
            >>> service = FileStorageService()
            >>> metadata = await service.get_file_metadata(UUID("3fa85f64-..."))
            >>> print(f"File size: {metadata['size']} bytes")
            >>> print(f"Format: {metadata['format']}")

        Implementation Note:
            Phase 1: Basic metadata (size, format, exists)
            Phase 3: Add timestamps, validation status
            Phase 5: Cloud storage metadata
        """
        raise NotImplementedError(
            "get_file_metadata() to be implemented in Phase 3. "
            "Will retrieve file size, format, and other metadata."
        )

    def _validate_extension(self, filename: str) -> bool:
        """
        Validate file extension (internal validation method).

        Business rule: Only .xlsx and .xls files allowed (Excel).

        Args:
            filename: Name of the file to validate

        Returns:
            True if extension is allowed, False otherwise

        Examples:
            >>> service = FileStorageService()
            >>> service._validate_extension("file.xlsx")
            True
            >>> service._validate_extension("file.pdf")
            False
        """
        return any(filename.lower().endswith(ext) for ext in self.allowed_extensions)

    def _validate_size(self, file_data: bytes) -> bool:
        """
        Validate file size (internal validation method).

        Business rule: Max 10MB per file (configurable).

        Args:
            file_data: Raw file bytes

        Returns:
            True if size is within limit, False otherwise

        Examples:
            >>> service = FileStorageService()
            >>> data = b"x" * (5 * 1024 * 1024)  # 5MB
            >>> service._validate_size(data)
            True
            >>> data = b"x" * (15 * 1024 * 1024)  # 15MB
            >>> service._validate_size(data)
            False
        """
        return len(file_data) <= self.max_size_bytes

    def _get_job_dir(self, job_id: UUID) -> Path:
        """
        Get job-specific directory path.

        Args:
            job_id: Unique job identifier

        Returns:
            Path to job directory

        Examples:
            >>> service = FileStorageService()
            >>> job_dir = service._get_job_dir(UUID("abc-123"))
            >>> str(job_dir)
            '/tmp/fastbidder/abc-123'
        """
        return self.base_dir / str(job_id)

    async def upload_file(
        self,
        job_id: UUID,
        file_data: bytes,
        filename: str,
        file_type: str,  # "working" or "reference"
    ) -> Path:
        """
        Upload and validate Excel file for job processing.

        Steps:
        1. Validate extension (.xlsx/.xls only)
        2. Validate size (max 10MB)
        3. Create job directory if not exists
        4. Save file to disk
        5. Return file path

        Args:
            job_id: Unique job identifier
            file_data: Raw file bytes
            filename: Original filename (for extension validation)
            file_type: Type of file ("working" or "reference")

        Returns:
            Path to saved file

        Raises:
            ValueError: If validation fails (extension or size)
            OSError: If file cannot be written

        Examples:
            >>> service = FileStorageService()
            >>> with open("test.xlsx", "rb") as f:
            ...     data = f.read()
            >>> path = await service.upload_file(
            ...     job_id=UUID("abc-123"),
            ...     file_data=data,
            ...     filename="working_file.xlsx",
            ...     file_type="working"
            ... )
            >>> print(path)
            /tmp/fastbidder/abc-123/working_file.xlsx
        """
        raise NotImplementedError(
            "upload_file() to be implemented in Phase 3. "
            "Will validate extension/size and save file to job directory."
        )

    def get_file_path(
        self, job_id: UUID, file_type: str  # "working" or "reference" or "result"
    ) -> Path:
        """
        Get path to existing file in job directory.

        Used by Application Layer to retrieve files for processing.

        Args:
            job_id: Unique job identifier
            file_type: Type of file to retrieve

        Returns:
            Path to file

        Raises:
            FileNotFoundError: If file does not exist

        Examples:
            >>> service = FileStorageService()
            >>> path = service.get_file_path(UUID("abc-123"), "working")
            >>> print(path.exists())
            True
        """
        raise NotImplementedError(
            "get_file_path() to be implemented in Phase 3. "
            "Will construct path to file based on job_id and file_type."
        )

    async def cleanup_job(self, job_id: UUID) -> None:
        """
        Delete all files for a job (manual cleanup).

        Called by Application Layer after:
        - Job completion and result download
        - Job failure and error handling
        - Manual cleanup requests

        Args:
            job_id: Unique job identifier

        Raises:
            OSError: If directory cannot be deleted

        Examples:
            >>> service = FileStorageService()
            >>> await service.cleanup_job(UUID("abc-123"))
            # Directory /tmp/fastbidder/abc-123 deleted
        """
        raise NotImplementedError(
            "cleanup_job() to be implemented in Phase 3. "
            "Will delete entire job directory and all files within."
        )

    async def cleanup_old_jobs(self, hours: int = 24) -> int:
        """
        Cleanup jobs older than specified hours (batch cleanup).

        Used for maintenance - removes abandoned jobs.
        Phase 1: Manual trigger only
        Phase 2: Celery periodic task

        Args:
            hours: Age threshold in hours (default 24)

        Returns:
            Number of jobs cleaned up

        Examples:
            >>> service = FileStorageService()
            >>> count = await service.cleanup_old_jobs(hours=24)
            >>> print(f"Cleaned up {count} old jobs")
        """
        raise NotImplementedError(
            "cleanup_old_jobs() to be implemented in Phase 3. "
            "Will scan job directories and delete those older than threshold."
        )
