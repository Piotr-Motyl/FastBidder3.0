# Shared Utilities

## Responsibility
Cross-cutting concerns and generic utilities used across all layers. Provides common helpers that don't belong to any specific layer.

## Contains
- **Utils**: Generic utility functions
  - String utilities
  - Date/time helpers
  - Validation helpers
  - Common decorators

## Does NOT contain
- Layer-specific code
- Business logic (use Domain layer)
- Infrastructure implementations (use Infrastructure layer)

## Architecture Notes
- Pure utility functions
- No dependencies on other layers
- Reusable across the entire application