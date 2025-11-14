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

import logging
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from src.domain.shared.exceptions import (
    ColumnNotFoundError,
    ExcelParsingError,
    FileSizeExceededError,
)

# Configure logger for file storage operations
logger = logging.getLogger(__name__)


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

    Storage Structure (Phase 2):
        Base directory: /tmp/fastbidder/ (from env: TEMP_DIR)
        Job structure: /tmp/fastbidder/{job_id}/input/
                      /tmp/fastbidder/{job_id}/output/
        Input files: /tmp/fastbidder/{job_id}/input/working_file.xlsx
                    /tmp/fastbidder/{job_id}/input/reference_file.xlsx
        Output files: /tmp/fastbidder/{job_id}/output/result.xlsx

    Business Rules:
        - Max file size: 10MB (from env: MAX_FILE_SIZE_MB)
        - Allowed extensions: .xlsx, .xls (from env: ALLOWED_EXTENSIONS)
        - Each job has isolated directory
        - Manual cleanup (no automatic deletion)

    Phase 2 Scope:
        - Local file system with input/output subdirectories
        - Atomic writes for output files (write to .tmp, then rename)
        - Cross-platform permissions (try/except for Windows compatibility)
        - Hard delete cleanup (no soft delete/archive)
        - Basic logging for all CRUD operations
        - Validation methods in this class (not separate module)
        - Manual cleanup (no TTL-based cleanup)
        - Implements FileStorageServiceProtocol for DI

    Phase 3+ Deferred:
        - Checksums, symlinks, lock files
        - Soft delete / archive system
        - Monitoring and alerts
        - Cloud storage (S3/GCS)

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

    async def file_exists(
        self, job_id: UUID, file_type: str  # "working" or "reference" or "result"
    ) -> bool:
        """
        Check if specific file exists in job storage (Phase 2 - Detailed Contract).

        This method is required by FileStorageServiceProtocol from Application Layer.
        Used by ProcessMatchingUseCase for business rule validation before processing.

        Process Flow:
        1. Get subdirectory path (input or output) based on file_type
        2. Get filename for file_type (working_file.xlsx, reference_file.xlsx, result.xlsx)
        3. Construct full path: {base_dir}/{job_id}/{subdir}/{filename}
        4. Check if file exists using Path.exists()
        5. Log result (DEBUG level)

        Args:
            job_id: UUID of the job to check
            file_type: Type of file to check ("working", "reference", "result")

        Returns:
            True if file exists in expected location, False otherwise

        Examples:
            >>> service = FileStorageService()
            >>> # Check if working file was uploaded
            >>> exists = await service.file_exists(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_type="working"
            ... )
            >>> if exists:
            ...     print("Working file found in storage")

            >>> # Check if result was generated
            >>> result_exists = await service.file_exists(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_type="result"
            ... )

        Implementation Note (Phase 3):
            - Get subdirectory using _get_subdirectory(file_type)
            - Get filename using _get_filename_for_type(file_type)
            - Construct path: base_dir / str(job_id) / subdir / filename
            - Return path.exists()
            - Log: "Checking file existence: {path} -> {exists}"

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "file_exists() to be implemented in Phase 3. "
            "Will check if specific file exists in job's input or output directory."
        )

    async def get_file_metadata(
        self, job_id: UUID, file_type: str  # "working" or "reference" or "result"
    ) -> dict:
        """
        Get file metadata like size and format (Phase 2 - Detailed Contract).

        This method is required by FileStorageServiceProtocol from Application Layer.
        Used by ProcessMatchingUseCase for estimating processing time and validation.

        Process Flow:
        1. Check if file exists using file_exists()
        2. If not exists, raise FileNotFoundError
        3. Get file path using get_file_path()
        4. Read file stats using Path.stat()
        5. Extract metadata: size, format (extension), timestamps
        6. Return metadata dictionary
        7. Log metadata retrieval (DEBUG level)

        Args:
            job_id: UUID of the job
            file_type: Type of file ("working", "reference", "result")

        Returns:
            Dict with file metadata:
            {
                'size': 1024000,              # Size in bytes (int)
                'size_mb': 1.02,              # Size in MB (float, 2 decimals)
                'format': 'xlsx',             # File extension without dot
                'exists': True,               # Always True if no exception
                'created_at': '2024-01-15...' # ISO timestamp (file creation)
                'modified_at': '2024-01-15...' # ISO timestamp (last modification)
                'file_type': 'working',       # Echo of input parameter
                'file_path': '/tmp/...'       # Full path as string
            }

        Raises:
            FileNotFoundError: If file does not exist in storage

        Examples:
            >>> service = FileStorageService()
            >>> metadata = await service.get_file_metadata(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_type="working"
            ... )
            >>> print(f"File size: {metadata['size_mb']:.2f} MB")
            >>> print(f"Format: {metadata['format']}")
            >>> print(f"Created: {metadata['created_at']}")

            >>> # Check metadata before processing
            >>> if metadata['size'] > 10_000_000:
            ...     raise FileSizeExceededError("File too large")

        Implementation Note (Phase 3):
            - First check exists with file_exists(), raise if False
            - Get path with get_file_path(job_id, file_type)
            - Use path.stat() to get os.stat_result
            - Extract: st_size, st_ctime, st_mtime
            - Calculate size_mb = st_size / (1024 * 1024)
            - Extract format from path.suffix (remove leading dot)
            - Convert timestamps to ISO format using datetime.fromtimestamp().isoformat()
            - Log: "Retrieved metadata for {file_type}: {size_mb:.2f}MB"

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "get_file_metadata() to be implemented in Phase 3. "
            "Will retrieve file size, format, timestamps, and other metadata."
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
        Upload and validate Excel file for job processing (Phase 2 - Detailed Contract).

        This method handles file upload to appropriate subdirectory with full validation.
        Input files (working, reference) are saved to input/ subdirectory.

        Process Flow:
        1. Validate file extension using _validate_extension(filename)
        2. Validate file size using _validate_size(file_data)
        3. Get subdirectory path using _get_subdirectory(file_type)
        4. Get target filename using _get_filename_for_type(file_type)
        5. Construct full directory path: {base_dir}/{job_id}/{subdir}
        6. Ensure directory exists using _ensure_directory_exists()
        7. Write file data to disk (direct write, NOT atomic for input files)
        8. Set file permissions using _set_permissions() (644)
        9. Log successful upload (INFO level)
        10. Return full file path

        Args:
            job_id: Unique job identifier (UUID)
            file_data: Raw file bytes from upload
            filename: Original filename from user (for extension validation only)
            file_type: Type of file ("working" or "reference")

        Returns:
            Path to saved file (absolute path)

        Raises:
            ValueError: If extension is not .xlsx/.xls
            FileSizeExceededError: If file_data size > max_size_bytes (10MB)
            OSError: If file cannot be written to disk

        Examples:
            >>> service = FileStorageService()
            >>> with open("test.xlsx", "rb") as f:
            ...     data = f.read()
            >>> path = await service.upload_file(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_data=data,
            ...     filename="my_working_file.xlsx",
            ...     file_type="working"
            ... )
            >>> print(path)
            /tmp/fastbidder/3fa85f64-.../input/working_file.xlsx

            >>> # Reference file upload
            >>> ref_path = await service.upload_file(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_data=ref_data,
            ...     filename="catalog.xlsx",
            ...     file_type="reference"
            ... )
            >>> print(ref_path)
            /tmp/fastbidder/3fa85f64-.../input/reference_file.xlsx

        Implementation Note (Phase 3):
            - Validation:
                * if not _validate_extension(filename):
                    raise ValueError(f"Invalid extension: {filename}. Allowed: {allowed_extensions}")
                * if not _validate_size(file_data):
                    raise FileSizeExceededError(len(file_data), max_size_bytes)
            - Path construction:
                * subdir = _get_subdirectory(file_type)  # "input"
                * target_filename = _get_filename_for_type(file_type)  # "working_file.xlsx"
                * dir_path = base_dir / str(job_id) / subdir
                * file_path = dir_path / target_filename
            - Directory creation:
                * _ensure_directory_exists(dir_path)
            - File write:
                * file_path.write_bytes(file_data)  # Direct write for input
            - Permissions:
                * _set_permissions(file_path, mode=0o644)  # rw-r--r--
            - Logging:
                * logger.info(f"Uploaded {file_type} file: {file_path} ({len(file_data)} bytes)")

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "upload_file() to be implemented in Phase 3. "
            "Will validate extension/size and save file to input subdirectory."
        )

    def get_file_path(
        self, job_id: UUID, file_type: str  # "working" or "reference" or "result"
    ) -> Path:
        """
        Get path to file in job storage (Phase 2 - Detailed Contract).

        Used by Application Layer (ProcessMatchingUseCase, ExcelReader/Writer) to retrieve
        file paths for processing. Does NOT check if file exists - use file_exists() first
        if existence check is needed.

        Process Flow:
        1. Get subdirectory using _get_subdirectory(file_type)
        2. Get filename using _get_filename_for_type(file_type)
        3. Construct full path: {base_dir}/{job_id}/{subdir}/{filename}
        4. Return Path object (no existence check)

        Args:
            job_id: Unique job identifier (UUID)
            file_type: Type of file ("working", "reference", "result")

        Returns:
            Path object pointing to file location (may or may not exist)

        Examples:
            >>> service = FileStorageService()
            >>> # Get path for working file
            >>> path = service.get_file_path(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_type="working"
            ... )
            >>> print(path)
            /tmp/fastbidder/3fa85f64-.../input/working_file.xlsx

            >>> # Get path for result file
            >>> result_path = service.get_file_path(
            ...     job_id=UUID("3fa85f64-..."),
            ...     file_type="result"
            ... )
            >>> print(result_path)
            /tmp/fastbidder/3fa85f64-.../output/result.xlsx

            >>> # Typical usage pattern
            >>> if await service.file_exists(job_id, "working"):
            ...     path = service.get_file_path(job_id, "working")
            ...     data = path.read_bytes()

        Implementation Note (Phase 3):
            - Get subdirectory:
                * subdir = _get_subdirectory(file_type)
                * Returns "input" for working/reference
                * Returns "output" for result
            - Get filename:
                * filename = _get_filename_for_type(file_type)
                * Returns "working_file.xlsx", "reference_file.xlsx", or "result.xlsx"
            - Construct path:
                * return self.base_dir / str(job_id) / subdir / filename
            - No logging needed (this is simple getter)
            - No validation or existence check

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "get_file_path() to be implemented in Phase 3. "
            "Will construct path to file based on job_id, file_type, and subdirectory structure."
        )

    async def cleanup_job(self, job_id: UUID) -> None:
        """
        Delete all files and directories for a job - hard delete (Phase 2 - Detailed Contract).

        Called by Application Layer after:
        - Job completion and result download by user
        - Job failure and error handling
        - Manual cleanup requests from admin

        This is a HARD DELETE operation - no soft delete, no archive, no recovery.
        The entire job directory ({job_id}/) including input/ and output/ subdirectories
        and all files within are permanently removed.

        Process Flow:
        1. Get job directory path using _get_job_dir(job_id)
        2. Check if directory exists
        3. If exists, delete recursively using shutil.rmtree()
        4. Log deletion (INFO level)
        5. If not exists, log warning (WARNING level) but don't raise exception

        Args:
            job_id: Unique job identifier (UUID)

        Raises:
            OSError: If directory cannot be deleted (permission denied, disk error)

        Examples:
            >>> service = FileStorageService()
            >>> # Cleanup after successful job
            >>> await service.cleanup_job(UUID("3fa85f64-..."))
            # Deletes: /tmp/fastbidder/3fa85f64-.../
            #   ├── input/
            #   │   ├── working_file.xlsx
            #   │   └── reference_file.xlsx
            #   └── output/
            #       └── result.xlsx

            >>> # Cleanup non-existent job (no error)
            >>> await service.cleanup_job(UUID("non-existent"))
            # Logs warning but doesn't raise exception

        Implementation Note (Phase 3):
            - Get job directory:
                * job_dir = _get_job_dir(job_id)  # base_dir / str(job_id)
            - Check existence:
                * if not job_dir.exists():
                    logger.warning(f"Job directory not found for cleanup: {job_id}")
                    return
            - Hard delete:
                * shutil.rmtree(job_dir)
                * Deletes directory and ALL contents recursively
            - Logging:
                * logger.info(f"Cleaned up job directory: {job_id}")
            - Error handling:
                * Let OSError propagate to caller (API Layer will convert to 500)
                * Possible errors: PermissionError, FileNotFoundError (race condition)

        Phase 2 Scope (Hard Delete):
            - Immediate permanent deletion
            - No soft delete (marking as deleted)
            - No archiving to backup location
            - No retention period
            - Phase 3+ may add soft delete / archive features

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "cleanup_job() to be implemented in Phase 3. "
            "Will delete entire job directory and all files within using shutil.rmtree()."
        )

    async def cleanup_old_jobs(self, hours: int = 24) -> int:
        """
        Cleanup jobs older than specified hours - batch maintenance (Phase 2 - Detailed Contract).

        Used for maintenance to remove abandoned or stale jobs. Scans all job directories
        in base_dir and deletes those with modification time older than threshold.

        Process Flow:
        1. Calculate cutoff timestamp (now - hours)
        2. Scan base_dir for all subdirectories (each is a job_id)
        3. For each job directory:
            a. Check if it's old using _is_directory_old(dir_path, hours)
            b. If old, delete using cleanup_job(job_id)
            c. Count successful deletions
        4. Log summary (INFO level)
        5. Return count of deleted jobs

        Args:
            hours: Age threshold in hours (default 24)
                   Jobs with mtime older than (now - hours) will be deleted

        Returns:
            Number of job directories successfully cleaned up (int)

        Examples:
            >>> service = FileStorageService()
            >>> # Daily cleanup (remove jobs older than 24 hours)
            >>> count = await service.cleanup_old_jobs(hours=24)
            >>> print(f"Cleaned up {count} old jobs")

            >>> # Weekly cleanup (remove jobs older than 7 days)
            >>> count = await service.cleanup_old_jobs(hours=168)

            >>> # Manual trigger for testing
            >>> count = await service.cleanup_old_jobs(hours=1)
            >>> # Removes jobs older than 1 hour

        Implementation Note (Phase 3):
            - Calculate cutoff:
                * cutoff_timestamp = time.time() - (hours * 3600)
            - Scan directories:
                * for job_dir in base_dir.iterdir():
                    if not job_dir.is_dir():
                        continue
                    if _is_directory_old(job_dir, hours):
                        try:
                            job_id = UUID(job_dir.name)
                            await cleanup_job(job_id)
                            cleaned_count += 1
                        except ValueError:
                            # Not a valid UUID directory name, skip
                            logger.warning(f"Skipping non-UUID directory: {job_dir.name}")
                        except OSError as e:
                            # Deletion failed, log and continue
                            logger.error(f"Failed to cleanup {job_dir.name}: {e}")
            - Logging:
                * logger.info(f"Cleaned up {cleaned_count} jobs older than {hours} hours")
            - Return count

        Usage Pattern:
            Phase 2: Manual trigger via API endpoint (admin only)
            Phase 3: Celery periodic task (e.g., every 6 hours)
            Phase 4: Configurable retention policy per user

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "cleanup_old_jobs() to be implemented in Phase 3. "
            "Will scan job directories and delete those older than threshold."
        )

    # ========================================
    # Helper Methods (Phase 2 - Detailed Contracts)
    # ========================================

    def _get_subdirectory(self, file_type: str) -> str:
        """
        Map file_type to subdirectory name (Phase 2 - Detailed Contract).

        This helper method implements the directory structure mapping:
        - Input files (working, reference) -> "input" subdirectory
        - Output files (result) -> "output" subdirectory

        Args:
            file_type: Type of file ("working", "reference", "result")

        Returns:
            Subdirectory name ("input" or "output")

        Raises:
            ValueError: If file_type is not recognized

        Examples:
            >>> service = FileStorageService()
            >>> service._get_subdirectory("working")
            'input'
            >>> service._get_subdirectory("reference")
            'input'
            >>> service._get_subdirectory("result")
            'output'

        Implementation Note (Phase 3):
            - Mapping logic:
                * if file_type in ("working", "reference"):
                    return "input"
                * elif file_type == "result":
                    return "output"
                * else:
                    raise ValueError(f"Unknown file_type: {file_type}")

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "_get_subdirectory() to be implemented in Phase 3. "
            "Will map file_type to subdirectory (input/output)."
        )

    def _get_filename_for_type(self, file_type: str) -> str:
        """
        Map file_type to standardized filename (Phase 2 - Detailed Contract).

        This helper method implements filename standardization:
        - working -> "working_file.xlsx"
        - reference -> "reference_file.xlsx"
        - result -> "result.xlsx"

        Original user filenames are NOT preserved. This ensures consistent
        file paths and simplifies file operations.

        Args:
            file_type: Type of file ("working", "reference", "result")

        Returns:
            Standardized filename (str)

        Raises:
            ValueError: If file_type is not recognized

        Examples:
            >>> service = FileStorageService()
            >>> service._get_filename_for_type("working")
            'working_file.xlsx'
            >>> service._get_filename_for_type("reference")
            'reference_file.xlsx'
            >>> service._get_filename_for_type("result")
            'result.xlsx'

        Implementation Note (Phase 3):
            - Mapping logic:
                * mapping = {
                    "working": "working_file.xlsx",
                    "reference": "reference_file.xlsx",
                    "result": "result.xlsx",
                }
                * if file_type not in mapping:
                    raise ValueError(f"Unknown file_type: {file_type}")
                * return mapping[file_type]

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "_get_filename_for_type() to be implemented in Phase 3. "
            "Will map file_type to standardized filename."
        )

    def _ensure_directory_exists(self, dir_path: Path) -> None:
        """
        Create directory if it doesn't exist (Phase 2 - Detailed Contract).

        Creates directory and all parent directories as needed (mkdir -p behavior).
        Sets permissions to 755 (drwxr-xr-x) for new directories using _set_permissions().

        Args:
            dir_path: Path object for directory to create

        Raises:
            OSError: If directory cannot be created (permission denied, disk full)

        Examples:
            >>> service = FileStorageService()
            >>> job_dir = Path("/tmp/fastbidder/3fa85f64-.../input")
            >>> service._ensure_directory_exists(job_dir)
            # Creates: /tmp/fastbidder/
            #          /tmp/fastbidder/3fa85f64-.../
            #          /tmp/fastbidder/3fa85f64-.../input/

        Implementation Note (Phase 3):
            - Check if exists:
                * if dir_path.exists():
                    return  # Already exists, nothing to do
            - Create directory:
                * dir_path.mkdir(parents=True, exist_ok=True)
                * parents=True: create parent directories as needed
                * exist_ok=True: no error if directory already exists (race condition)
            - Set permissions:
                * _set_permissions(dir_path, mode=0o755)  # drwxr-xr-x
            - Logging:
                * logger.debug(f"Created directory: {dir_path}")

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "_ensure_directory_exists() to be implemented in Phase 3. "
            "Will create directory with mkdir -p behavior and set permissions."
        )

    def _set_permissions(self, path: Path, mode: int) -> None:
        """
        Set file/directory permissions (cross-platform) (Phase 2 - Detailed Contract).

        Sets Unix-style permissions using chmod. Handles Windows gracefully with
        try/except pattern (Windows doesn't support chmod for most modes).

        Args:
            path: Path to file or directory
            mode: Permission mode (e.g., 0o644 for files, 0o755 for directories)

        Examples:
            >>> service = FileStorageService()
            >>> # Set file permissions: rw-r--r-- (644)
            >>> service._set_permissions(Path("/tmp/file.xlsx"), 0o644)
            >>> # Set directory permissions: rwxr-xr-x (755)
            >>> service._set_permissions(Path("/tmp/mydir"), 0o755)

        Implementation Note (Phase 3):
            - Cross-platform handling:
                * try:
                    os.chmod(path, mode)
                    logger.debug(f"Set permissions {oct(mode)} on {path}")
                except (OSError, NotImplementedError):
                    # Windows may not support chmod for all modes
                    # Log but don't fail
                    logger.debug(f"Could not set permissions on {path} (Windows?)")
            - Common modes:
                * 0o644 (rw-r--r--): Files readable by all, writable by owner
                * 0o755 (rwxr-xr-x): Directories executable/searchable by all
            - Why try/except:
                * Windows NTFS doesn't support Unix permissions
                * Some filesystems (FAT32) don't support chmod
                * Better to log and continue than fail the entire operation

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "_set_permissions() to be implemented in Phase 3. "
            "Will set file permissions with cross-platform try/except handling."
        )

    def _atomic_write_file(self, file_path: Path, data: bytes) -> None:
        """
        Write file atomically using temporary file (Phase 2 - Detailed Contract).

        Atomic write pattern prevents corruption if write is interrupted:
        1. Write to temporary file ({file_path}.tmp)
        2. Rename temporary file to final path (atomic operation)

        This ensures the file is either fully written or not present at all.
        Used ONLY for output files (result.xlsx), not for input files.

        Args:
            file_path: Final destination path for file
            data: File content as bytes

        Raises:
            OSError: If temporary file cannot be written or renamed

        Examples:
            >>> service = FileStorageService()
            >>> result_path = Path("/tmp/fastbidder/.../output/result.xlsx")
            >>> result_data = b"Excel file content..."
            >>> service._atomic_write_file(result_path, result_data)
            # Writes to: /tmp/fastbidder/.../output/result.xlsx.tmp
            # Then renames to: /tmp/fastbidder/.../output/result.xlsx

        Implementation Note (Phase 3):
            - Construct temp path:
                * tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
                * Example: result.xlsx -> result.xlsx.tmp
            - Write to temp file:
                * tmp_path.write_bytes(data)
            - Set permissions on temp file:
                * _set_permissions(tmp_path, mode=0o644)
            - Atomic rename:
                * tmp_path.rename(file_path)
                * On Unix: rename() is atomic operation
                * On Windows: may need tmp_path.replace(file_path) for overwrite
            - Logging:
                * logger.debug(f"Atomic write: {len(data)} bytes to {file_path}")
            - Why atomic:
                * If write interrupted (crash, power loss), tmp file exists but final doesn't
                * Final file is never in partial state
                * Critical for result files that user downloads

        Usage:
            - Use for output files (result.xlsx) in Phase 2
            - Use for input files in Phase 3+ if needed
            - Current: upload_file() uses direct write for input (acceptable risk)

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "_atomic_write_file() to be implemented in Phase 3. "
            "Will write file atomically using .tmp and rename pattern."
        )

    def _is_directory_old(self, dir_path: Path, hours: int) -> bool:
        """
        Check if directory is older than threshold (Phase 2 - Detailed Contract).

        Used by cleanup_old_jobs() to determine which job directories should be deleted.
        Compares directory modification time (mtime) with threshold.

        Args:
            dir_path: Path to directory to check
            hours: Age threshold in hours

        Returns:
            True if directory is older than threshold, False otherwise

        Examples:
            >>> service = FileStorageService()
            >>> job_dir = Path("/tmp/fastbidder/3fa85f64-...")
            >>> # Check if older than 24 hours
            >>> is_old = service._is_directory_old(job_dir, hours=24)
            >>> if is_old:
            ...     print("Directory should be cleaned up")

        Implementation Note (Phase 3):
            - Get directory stats:
                * if not dir_path.exists():
                    return False  # Doesn't exist, can't be old
                * stat_result = dir_path.stat()
                * mtime = stat_result.st_mtime  # Last modification time (Unix timestamp)
            - Calculate cutoff:
                * cutoff_timestamp = time.time() - (hours * 3600)
                * hours * 3600 converts hours to seconds
            - Compare:
                * return mtime < cutoff_timestamp
            - Why mtime:
                * mtime updates when files inside are modified
                * ctime (creation time) doesn't change with file modifications
                * mtime is better indicator of "last activity"

        Phase 2 Contract:
            This method defines the interface contract with detailed documentation.
            Actual implementation will be added in Phase 3.
        """
        raise NotImplementedError(
            "_is_directory_old() to be implemented in Phase 3. "
            "Will check directory mtime against threshold."
        )
