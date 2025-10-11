# Infrastructure Layer

## Responsibility
Implements technical capabilities that support the domain. Handles all external dependencies including databases, file systems, and external services.

## Contains
- **Persistence**: Data storage implementations
  - Redis client and utilities
  - Repository implementations (from Domain interfaces)
- **File Storage**: File system operations
  - Upload/download management
  - Temporary file handling
  - File validation
  - Polars-based Excel operations

## Does NOT contain
- Business logic (use Domain layer)
- Application orchestration (use Application layer)
- HTTP handling (use API layer)
- Domain entities (defined in Domain layer)

## Architecture Notes
- Implements Domain repository interfaces
- Dependency Inversion Principle (depends on Domain abstractions)
- Easily replaceable (e.g., Redis → PostgreSQL)
- All external dependencies isolated here