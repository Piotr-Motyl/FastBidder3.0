"""
Domain Layer - Core Business Logic

Responsibility:
    Heart of the application. Contains all business rules, entities,
    and domain services. Framework-independent and testable.

Contains:
    - HVAC subdomain (parameters, matching, extraction)
    - Domain entities and value objects
    - Domain services (matching engine, parameter extractor)
    - Repository interfaces (no implementations)

Does NOT contain:
    - Infrastructure details (database, file system, external APIs)
    - Application orchestration (belongs to Application layer)
    - HTTP concerns (belongs to API layer)
"""
