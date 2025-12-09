"""
FastAPI Application Setup

Main entry point for the FastBidder API application.

Responsibility:
    - FastAPI app initialization
    - Router registration (files, matching, jobs, results)
    - CORS middleware configuration
    - Global exception handler
    - Request logging middleware
    - Health check endpoint

Architecture Notes:
    - Part of API Layer (Presentation)
    - Entry point for HTTP server (uvicorn)
    - Centralizes cross-cutting concerns (logging, CORS, error handling)
    - No business logic - pure HTTP orchestration

Contains:
    - create_app() factory function
    - Global exception handlers
    - Request logging middleware
    - Health check endpoint: GET /health

Does NOT contain:
    - Business logic (delegated to Application Layer)
    - Direct database access (uses Infrastructure Layer)
    - Celery configuration (separate module)
"""

import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import routers
from src.api.routers import files, matching, jobs, results

# Import shared schemas
from src.api.schemas.common import ErrorResponse

# Import domain exceptions for global handling
from src.domain.shared.exceptions import (
    DomainException,
    InvalidDNValueError,
    InvalidPNValueError,
    IncompatibleDNPNError,
    InvalidHVACDescriptionError,
    InvalidProcessMatchingCommandError,
    FileSizeExceededError,
    ExcelParsingError,
    ColumnNotFoundError,
)

# Import application exceptions
from src.application.queries.get_job_status import JobNotFoundException

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class HealthCheckResponse(BaseModel):
    """
    Health check response model.

    Simple status indicator for monitoring and load balancers.

    Attributes:
        status: Health status (always "ok" if endpoint responds)
        version: API version (Phase 2: hardcoded)
        timestamp: Unix timestamp of health check
    """

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: float


# ============================================================================
# MIDDLEWARE
# ============================================================================


async def request_logging_middleware(request: Request, call_next):
    """
    Request logging middleware.

    Logs all incoming requests with method, path, status code, and duration.
    Provides observability for API usage and performance monitoring.

    Process Flow:
        1. Log incoming request (method, path)
        2. Record start time
        3. Call next middleware/endpoint
        4. Record end time
        5. Log response (status code, duration)
        6. Return response to client

    Args:
        request: FastAPI Request object
        call_next: Next middleware/endpoint in chain

    Returns:
        Response from endpoint

    Logging Format:
        INFO: "Incoming request: GET /api/files/upload"
        INFO: "Request completed: GET /api/files/upload - 201 - 0.123s"

    Examples:
        >>> # Successful request
        >>> # Logs:
        >>> # INFO: Incoming request: POST /api/matching/process
        >>> # INFO: Request completed: POST /api/matching/process - 201 - 0.456s

        >>> # Failed request
        >>> # Logs:
        >>> # INFO: Incoming request: GET /api/jobs/invalid-uuid/status
        >>> # INFO: Request completed: GET /api/jobs/invalid-uuid/status - 422 - 0.012s
    """
    # Log incoming request
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    # Record start time
    start_time = time.time()

    # Call next middleware/endpoint
    response = await call_next(request)

    # Calculate duration
    duration = time.time() - start_time

    # Log response
    logger.info(
        f"Request completed: {request.method} {request.url.path} - "
        f"{response.status_code} - {duration:.3f}s"
    )

    return response


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================


async def domain_exception_handler(request: Request, exc: DomainException):
    """
    Global exception handler for domain layer exceptions.

    Catches all DomainException subclasses and converts them to
    appropriate HTTP error responses with consistent ErrorResponse format.

    Mapping:
        - InvalidDNValueError -> 400 Bad Request
        - InvalidPNValueError -> 400 Bad Request
        - IncompatibleDNPNError -> 400 Bad Request
        - InvalidHVACDescriptionError -> 400 Bad Request
        - InvalidProcessMatchingCommandError -> 400 Bad Request
        - FileSizeExceededError -> 413 Payload Too Large
        - ExcelParsingError -> 422 Unprocessable Entity
        - ColumnNotFoundError -> 400 Bad Request
        - Other DomainException -> 400 Bad Request

    Args:
        request: FastAPI Request object
        exc: DomainException or subclass

    Returns:
        JSONResponse with ErrorResponse format and appropriate status code

    Examples:
        >>> # InvalidDNValueError
        >>> raise InvalidDNValueError("DN must be 15-1000, got 10")
        >>> # Returns: 400 {"code": "INVALID_DN_VALUE", "message": "...", "details": {}}

        >>> # FileSizeExceededError
        >>> raise FileSizeExceededError("File too large", 15728640, 10485760)
        >>> # Returns: 413 {"code": "FILE_TOO_LARGE", "message": "...", "details": {...}}
    """
    # Determine status code based on exception type
    if isinstance(exc, FileSizeExceededError):
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        error_code = "FILE_TOO_LARGE"
    elif isinstance(exc, ExcelParsingError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_code = "EXCEL_PARSING_ERROR"
    else:
        # All other domain exceptions -> 400 Bad Request
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = exc.__class__.__name__.replace("Error", "").upper()

    # Build ErrorResponse
    error_response = ErrorResponse(
        code=error_code,
        message=str(exc),
        details={"exception_type": exc.__class__.__name__},
    )

    # Log error
    logger.warning(
        f"Domain exception: {exc.__class__.__name__} - {str(exc)} - "
        f"Request: {request.method} {request.url.path}"
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.dict(),
    )


async def job_not_found_exception_handler(request: Request, exc: JobNotFoundException):
    """
    Global exception handler for JobNotFoundException.

    Converts JobNotFoundException to 404 Not Found response.

    Args:
        request: FastAPI Request object
        exc: JobNotFoundException

    Returns:
        JSONResponse with ErrorResponse format and 404 status code
    """
    error_response = ErrorResponse(
        code="JOB_NOT_FOUND",
        message=str(exc),
        details={"job_id": str(exc.job_id)},
    )

    logger.warning(
        f"Job not found: {exc.job_id} - Request: {request.method} {request.url.path}"
    )

    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error_response.dict(),
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unexpected exceptions.

    Catches all unhandled exceptions and converts to 500 Internal Server Error.
    Logs full stack trace for debugging.

    Args:
        request: FastAPI Request object
        exc: Any unhandled exception

    Returns:
        JSONResponse with ErrorResponse format and 500 status code

    Note:
        This is a catch-all handler. Should only trigger for truly unexpected errors.
    """
    error_response = ErrorResponse(
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        details={"error": str(exc), "type": exc.__class__.__name__},
    )

    # Log with full traceback
    logger.error(
        f"Unexpected error: {exc.__class__.__name__} - {str(exc)} - "
        f"Request: {request.method} {request.url.path}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict(),
    )


# ============================================================================
# APP FACTORY
# ============================================================================


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Creates and configures FastAPI app with all middleware, routers,
    and exception handlers.

    Configuration:
        - Title: FastBidder API
        - Version: 0.1.0
        - Description: HVAC matching API for bid preparation
        - CORS: Allow all origins (development mode)
        - Logging: INFO level with structured format
        - Routers: /api/files, /api/matching, /api/jobs, /api/results
        - Health: GET /health

    Returns:
        Configured FastAPI application instance

    Usage:
        >>> app = create_app()
        >>> # Run with uvicorn:
        >>> # uvicorn src.api.main:app --reload

    Architecture Note:
        Factory pattern allows easy testing with dependency overrides
        and configuration injection.
    """
    # Create FastAPI app
    app = FastAPI(
        title="FastBidder API",
        version="0.1.0",
        description=(
            "HVAC description matching API for automated bid preparation. "
            "Upload Excel files, match descriptions, and download results."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware (allow all origins for development)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Production: restrict to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.middleware("http")(request_logging_middleware)

    # Register global exception handlers
    app.add_exception_handler(DomainException, domain_exception_handler)
    app.add_exception_handler(JobNotFoundException, job_not_found_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # Register routers with /api prefix
    app.include_router(files.router, prefix="/api")
    app.include_router(matching.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(results.router, prefix="/api")

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        status_code=status.HTTP_200_OK,
        summary="Health check endpoint",
        description="Simple health check for monitoring and load balancers",
        tags=["health"],
    )
    async def health_check() -> HealthCheckResponse:
        """
        Health check endpoint.

        Returns simple status indicator for monitoring systems.

        Returns:
            HealthCheckResponse with status="ok", version, and timestamp

        Examples:
            >>> curl http://localhost:8000/health
            {
              "status": "ok",
              "version": "0.1.0",
              "timestamp": 1704976800.123
            }
        """
        return HealthCheckResponse(
            status="ok",
            version="0.1.0",
            timestamp=time.time(),
        )

    logger.info("FastAPI application created successfully")
    logger.info("Registered routers: /api/files, /api/matching, /api/jobs, /api/results")
    logger.info("Health check available at: GET /health")

    return app


# ============================================================================
# APP INSTANCE (for uvicorn)
# ============================================================================

# Create app instance for uvicorn
# Usage: uvicorn src.api.main:app --reload
app = create_app()
