"""
Infrastructure Layer - External Dependencies

Implements technical capabilities that support the Domain Layer.
Handles all external dependencies: databases, file systems, external services.

Architecture:
    - Implements Domain repository interfaces (Dependency Inversion)
    - Implements Application Layer protocols (FileStorageServiceProtocol)
    - Depends on external libraries (Redis, Polars, sentence-transformers)
    - No Domain business logic (only technical implementations)

Modules:
    - persistence: Redis and repository implementations
    - file_storage: File system and Excel operations (Polars)
    - matching: Matching engine implementation

This module exports the most commonly used infrastructure services.

Exports:
    From persistence:
        - RedisProgressTracker: Track Celery job progress
        - HVACDescriptionRepository: Redis-based entity storage

    From file_storage:
        - FileStorageService: File upload/download/cleanup (implements Protocol)
        - ExcelReaderService: Read Excel with Polars
        - ExcelWriterService: Write results to Excel

    From matching:
        - ConcreteMatchingEngine: Hybrid matching implementation (implements Protocol)

Usage:
    >>> # Import from infrastructure module
    >>> from src.infrastructure import (
    ...     RedisProgressTracker,
    ...     FileStorageService,
    ...     ConcreteMatchingEngine
    ... )
    >>>
    >>> # Or import from specific submodules
    >>> from src.infrastructure.persistence import RedisProgressTracker
    >>> from src.infrastructure.file_storage import ExcelReaderService
    >>> from src.infrastructure.matching import ConcreteMatchingEngine
"""

# Persistence
from .persistence import HVACDescriptionRepository, RedisProgressTracker

# File Storage
from .file_storage import (
    ExcelReaderService,
    ExcelWriterService,
    FileStorageService,
)

# Matching
from .matching import ConcreteMatchingEngine

__all__ = [
    # Persistence
    "RedisProgressTracker",
    "HVACDescriptionRepository",
    # File Storage
    "FileStorageService",
    "ExcelReaderService",
    "ExcelWriterService",
    # Matching
    "ConcreteMatchingEngine",
]
