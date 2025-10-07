"""
API Layer - FastAPI Presentation Layer

Responsibility:
    HTTP interface for the application. Handles requests, responses,
    and triggers asynchronous Celery tasks. No business logic.

Contains:
    - FastAPI routers (upload, matching, jobs, results)
    - Request/Response models (Pydantic)
    - Dependency injection setup
    - Middleware configuration (CORS, logging)

Does NOT contain:
    - Business logic (belongs to Domain layer)
    - Data processing (belongs to Application layer)
    - Database operations (belongs to Infrastructure layer)
"""
