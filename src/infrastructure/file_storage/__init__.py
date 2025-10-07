"""
File Storage Infrastructure

Responsibility:
    File system operations for uploaded and generated files.
    Handles file upload, download, and cleanup.
    Uses Polars library for fast Excel operations (10x faster than Pandas).

Contains:
    To be implemented in Phase 3:
    - Local file storage service
    - File validation (size, extension)
    - Temporary file management
    - File cleanup utilities
    - Excel reader/writer using Polars

Does NOT contain:
    - Excel parsing logic (use separate Excel service)
    - Business validation (use Domain layer)
"""
