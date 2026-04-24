"""
Application Layer Ports (Interfaces)

Contains Protocol definitions for dependency inversion.
Infrastructure Layer implements these protocols.
"""

from src.application.ports.file_storage import FileStorageServiceProtocol
from src.application.ports.progress_tracker import ProgressTrackerProtocol

__all__ = ["FileStorageServiceProtocol", "ProgressTrackerProtocol"]
