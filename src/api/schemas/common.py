"""
Common API Schemas

Shared Pydantic models used across all API routers.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """
    Standard error response model for all API errors.

    Provides consistent error structure across all endpoints.

    Attributes:
        code: Machine-readable error code (e.g., "JOB_NOT_FOUND", "INVALID_FILE_EXTENSION")
        message: Human-readable error message
        details: Optional additional error details (validation errors, debug info)
    """

    code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "code": "FILE_NOT_FOUND",
                "message": "Working file not found in storage",
                "details": {"file_id": "a3bb189e-8bf9-3888-9912-ace4e6543002"},
            }
        }
