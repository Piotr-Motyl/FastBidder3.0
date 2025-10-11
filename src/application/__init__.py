"""
Application Layer - Use Cases and Orchestration

Responsibility:
    Coordinates the flow of data between API and Domain layers.
    Handles asynchronous task processing with Celery.

Contains:
    - Celery tasks (async processing)
    - Commands (CQRS write operations)
    - Queries (CQRS read operations)
    - Application services (orchestration)

Does NOT contain:
    - Domain business rules (belongs to Domain layer)
    - HTTP handling (belongs to API layer)
    - Infrastructure details (belongs to Infrastructure layer)
"""
