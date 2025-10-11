# API Layer

## Responsibility
FastAPI-based HTTP interface for the FastBidder application. Handles incoming requests, validates input, triggers asynchronous Celery tasks, and returns responses.

## Contains
- **Routers**: Endpoint definitions grouped by functionality
  - `upload.py` - File upload endpoints
  - `matching.py` - Matching process endpoints
  - `jobs.py` - Celery job status endpoints
  - `results.py` - Results retrieval endpoints
- **Models**: Pydantic Request/Response models
- **Dependencies**: FastAPI dependency injection
- **Middleware**: CORS, logging, error handling

## Does NOT contain
- Business logic (delegated to Domain layer)
- Data processing (delegated to Application layer)
- Database operations (delegated to Infrastructure layer)
- File parsing (delegated to Infrastructure layer)

## Architecture Notes
Follows Clean Architecture principles:
- Depends on Application layer (use cases)
- Independent of Infrastructure layer (dependency inversion)
- Thin layer focused only on HTTP concerns