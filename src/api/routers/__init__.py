"""
API Routers Package

Contains all FastAPI routers grouped by functionality.
Implements REST API endpoints for the FastBidder application.

Architecture Notes:
    - Part of API Layer (Presentation)
    - Each router handles a specific domain or concern
    - Routers are thin wrappers around Application Layer use cases
    - All routers follow dependency injection pattern

Available Routers:
    - matching_router: HVAC matching process endpoints
    - jobs_router: Job status tracking endpoints (cross-cutting concern)

Phase 1 Note:
    Currently only contracts (Task 1.1.1). Implementation in Phase 3.
"""

from .matching import router as matching_router
from .jobs import router as jobs_router

__all__ = ["matching_router", "jobs_router"]
