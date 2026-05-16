"""
API layer entry point: FastAPI app factory, CORS, request logging, global exception handlers,
and health check. No business logic — delegates entirely to Application/Domain layers.
"""

import logging
import time
from typing import Any, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    FileSizeExceededError,
    ExcelParsingError,
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
    """Response model for GET /health — used by monitoring and load balancers."""

    status: str = "ok"
    version: str = "0.1.0"
    timestamp: float


# ============================================================================
# MIDDLEWARE
# ============================================================================


async def request_logging_middleware(request: Request, call_next):
    """Logs method, path, status code, and duration for every request."""
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

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
    Converts DomainException subclasses to HTTP error responses.

    Exception → HTTP status mapping:
        FileSizeExceededError  → 413 Payload Too Large
        ExcelParsingError      → 422 Unprocessable Entity
        (all other)            → 400 Bad Request
    """
    if isinstance(exc, FileSizeExceededError):
        status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        error_code = "FILE_TOO_LARGE"
    elif isinstance(exc, ExcelParsingError):
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_code = "EXCEL_PARSING_ERROR"
    else:
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = exc.__class__.__name__.replace("Error", "").upper()

    error_response = ErrorResponse(
        code=error_code,
        message=str(exc),
        details={"exception_type": exc.__class__.__name__},
    )

    logger.warning(
        f"Domain exception: {exc.__class__.__name__} - {str(exc)} - "
        f"Request: {request.method} {request.url.path}"
    )

    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(),
    )


async def job_not_found_exception_handler(request: Request, exc: JobNotFoundException):
    """Converts JobNotFoundException to 404 Not Found."""
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
        content=error_response.model_dump(),
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled exceptions — returns 500 with full traceback in logs."""
    error_response = ErrorResponse(
        code="INTERNAL_SERVER_ERROR",
        message="An unexpected error occurred",
        details={"error": str(exc), "type": exc.__class__.__name__},
    )

    logger.error(
        f"Unexpected error: {exc.__class__.__name__} - {str(exc)} - "
        f"Request: {request.method} {request.url.path}",
        exc_info=True,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
    )


# ============================================================================
# APP FACTORY
# ============================================================================


def create_app() -> FastAPI:
    """
    FastAPI application factory.

    Registers CORS, request logging, exception handlers, and all routers under /api.
    Factory pattern keeps the app testable — call this in tests with dependency overrides.
    Run with: uvicorn src.api.main:app --reload
    """
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Production: restrict to specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.middleware("http")(request_logging_middleware)

    app.add_exception_handler(DomainException, domain_exception_handler)
    app.add_exception_handler(JobNotFoundException, job_not_found_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.include_router(files.router, prefix="/api")
    app.include_router(matching.router, prefix="/api")
    app.include_router(jobs.router, prefix="/api")
    app.include_router(results.router, prefix="/api")

    @app.get(
        "/health",
        response_model=HealthCheckResponse,
        status_code=status.HTTP_200_OK,
        summary="Health check endpoint",
        description="Simple health check for monitoring and load balancers",
        tags=["health"],
    )
    async def health_check() -> HealthCheckResponse:
        return HealthCheckResponse(
            status="ok",
            version="0.1.0",
            timestamp=time.time(),
        )

    logger.info(
        "FastAPI application created — routers: /api/files, /api/matching, /api/jobs, /api/results"
    )

    return app


# ============================================================================
# APP INSTANCE (for uvicorn)
# ============================================================================

app = create_app()
