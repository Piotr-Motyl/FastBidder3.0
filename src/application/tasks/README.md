# Celery Tasks

## Responsibility
Asynchronous task definitions for long-running operations. Manages background job execution, progress tracking, and result storage.

## Contains
- **celery_app.py**: Celery application configuration
- **Async tasks**: Task definitions for background processing
  - Matching tasks
  - File processing tasks
  - Embedding generation tasks (future)
- **Progress utilities**: Redis-based progress tracking

## Does NOT contain
- Business logic (delegates to Domain services)
- Synchronous operations (use Application services)
- HTTP endpoints (use API layer)

## Architecture Notes
- Tasks are thin orchestrators
- Business logic in Domain services
- Results stored in Redis
- Progress updates via Redis pub/sub