# Application Layer

## Responsibility
Orchestrates the flow of data between API and Domain layers. Coordinates use cases, manages asynchronous task processing with Celery, and implements CQRS pattern.

## Contains
- **Tasks**: Celery task definitions for async processing
  - `celery_app.py` - Celery configuration
  - `matching_tasks.py` - Matching process tasks
  - `progress.py` - Progress tracking in Redis
- **Commands**: CQRS write operations (ProcessMatchingCommand, etc.)
- **Queries**: CQRS read operations (GetResultsQuery, etc.)
- **Services**: Application orchestration services

## Does NOT contain
- Domain business rules (delegated to Domain layer)
- HTTP handling (delegated to API layer)
- Infrastructure details (delegated to Infrastructure layer)
- Data validation rules (delegated to Domain layer)

## Architecture Notes
- Uses CQRS pattern for separation of reads and writes
- Celery tasks for long-running operations
- Progress tracking via Redis
- Coordinates multiple domain services