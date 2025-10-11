# Domain Layer

## Responsibility
Core business logic of the FastBidder application. Contains all business rules, entities, value objects, and domain services. Framework-independent and highly testable.

## Contains
- **HVAC subdomain**: HVAC-specific business logic
  - Entities: HVACDescription, MatchResult
  - Value Objects: DN, PN, Material, ValveType
  - Services: ParameterExtractor, MatchingEngine
  - Repository interfaces
- **Shared domain**: Cross-subdomain concepts

## Does NOT contain
- Infrastructure details (database, file system, external APIs)
- Application orchestration (use Application layer)
- HTTP concerns (use API layer)
- Framework-specific code (pure Python)

## Architecture Notes
- Heart of the application
- Framework-independent
- Contains all business rules
- Defines repository interfaces (implemented in Infrastructure)
- Uses Domain-Driven Design patterns