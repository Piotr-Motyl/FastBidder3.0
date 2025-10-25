# FastBidder 3.0

> AI-powered HVAC product matching system with hybrid parameter + semantic search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
[![Celery](https://img.shields.io/badge/Celery-5.3+-red.svg)

## 📖 About The Project

FastBidder automates the tedious process of matching HVAC and plumbing product descriptions with supplier catalogs to find accurate pricing. Built for companies in the Mechanical installations industry who need to quickly generate cost estimates from technical specifications. The system uses **hybrid matching** (40% parameter-based regex + 60% AI semantic similarity) to achieve 85%+ accuracy, reducing manual matching time from 8 working hours to 30 minutes instead.

This project demonstrates production-grade architecture principles: **Clean Architecture**, **CQRS pattern**, **async task processing with Celery**, and **domain-driven design**. Built with scalability and maintainability in mind, following Test-Driven Development with contract-first implementation approach.

**Tech Stack:** Python 3.10, FastAPI, Celery, Redis, Polars (instead of Pandas), Pydantic v2, Docker, Poetry

---

## 🎯 Current Implementation Status

**Phase 0: Infrastructure Setup** ✅ **COMPLETED**
- [x] Project structure (Clean Architecture)
- [x] Docker Compose (Redis + Celery worker)
- [x] Poetry dependencies
- [x] Makefile commands (14 commands)
- [x] Git repository configured

**Phase 1: High-Level Contracts** ✅ **COMPLETED**
- [x] Task 1.1.1: API Layer contracts (matching + jobs endpoints)
- [x] Task 1.1.2: Application Layer contracts (Commands, Queries, Use Cases, Celery tasks)
- [ ] Task 1.1.3: Domain Layer contracts (Entities, Services, Value Objects)
- [ ] Task 1.1.4: Infrastructure Layer contracts (FileStorage, Redis, Excel services)

**Next Steps:** Task 1.1.3 - Domain Layer contracts

**Phases Overview:**
```
Phase 0: Setup          ✅ Done
Phase 1: Contracts      🔄 In Progress (50%)
Phase 2: Detailed       ⏳ Pending
Phase 3: Implementation ⏳ Pending
Phase 4: AI Integration ⏳ Pending
Phase 5: Advanced       ⏳ Pending
Phase 6: Testing & Docs ⏳ Pending
```

---

## 🏗️ Architecture Overview

### Clean Architecture Layers

FastBidder follows **Clean Architecture** with strict dependency rules:

```
┌─────────────────────────────────────────┐
│         API Layer (Presentation)        │  ← HTTP endpoints
│  - FastAPI routers                      │
│  - Request/Response models              │
└──────────────┬──────────────────────────┘
               │ depends on
┌──────────────▼──────────────────────────┐
│       Application Layer (Use Cases)     │  ← Orchestration
│  - Commands (CQRS Write)                │
│  - Queries (CQRS Read)                  │
│  - Use Cases (orchestration)            │
│  - Celery Tasks (async processing)      │
└──────────────┬──────────────────────────┘
               │ depends on
┌──────────────▼──────────────────────────┐
│       Domain Layer (Business Logic)     │  ← Core business rules
│  - Entities (HVACDescription, Match)    │
│  - Value Objects (DN, PN, Material)     │
│  - Domain Services (MatchingEngine)     │
│  - Repository Interfaces                │
└──────────────▲──────────────────────────┘
               │ implemented by
┌──────────────┴──────────────────────────┐
│    Infrastructure Layer (External)      │  ← Technical capabilities
│  - Redis (progress tracking, cache)     │
│  - File Storage (Excel files)           │
│  - Repository Implementations           │
└─────────────────────────────────────────┘
```

**Key Principles:**
- **Dependency Inversion**: Outer layers depend on inner layers (never reverse)
- **CQRS Pattern**: Separate Commands (write) from Queries (read)
- **Contract-First**: Define interfaces before implementation
- **Async by Design**: Long-running operations via Celery

---

## 📁 Project Structure

```
fastbidder/
├── src/
│   ├── api/                     # API Layer (Presentation)
│   │   └── routers/
│   │       ├── matching.py      # ✅ POST /matching/process (contract)
│   │       ├── jobs.py          # ✅ GET /jobs/{job_id}/status (contract)
│   │       └── __init__.py      # ✅ Router exports
│   │
│   ├── application/             # Application Layer (Orchestration)
│   │   ├── commands/            # CQRS Write Operations
│   │   │   ├── process_matching.py  # ✅ ProcessMatchingCommand (contract)
│   │   │   └── __init__.py
│   │   ├── queries/             # CQRS Read Operations
│   │   │   ├── get_job_status.py    # ✅ GetJobStatusQuery + Handler (contract)
│   │   │   └── __init__.py
│   │   ├── services/            # Use Cases
│   │   │   ├── process_matching_use_case.py  # ✅ ProcessMatchingUseCase (contract)
│   │   │   └── __init__.py
│   │   ├── tasks/               # Celery Async Tasks
│   │   │   ├── celery_app.py   # ✅ Celery config (working)
│   │   │   ├── matching_tasks.py    # ✅ process_matching_task (contract)
│   │   │   └── __init__.py
│   │   └── models.py            # ✅ Shared models (JobStatus enum)
│   │
│   ├── domain/                  # Domain Layer (Business Logic)
│   │   ├── hvac/                # HVAC Bounded Context
│   │   │   ├── entities/        # ⏳ HVACDescription, MatchResult (pending)
│   │   │   ├── value_objects/   # ⏳ DN, PN, Material (pending)
│   │   │   ├── services/        # ⏳ MatchingEngine, ParameterExtractor (pending)
│   │   │   └── repositories/    # ⏳ Repository interfaces (pending)
│   │   └── shared/              # ⏳ Base classes (pending)
│   │
│   ├── infrastructure/          # Infrastructure Layer (External)
│   │   ├── persistence/
│   │   │   └── redis/           # ⏳ Redis client, progress tracker (pending)
│   │   └── file_storage/        # ⏳ Excel reader/writer with Polars (pending)
│   │
│   └── shared/                  # ⏳ Cross-cutting utilities (pending)
│
├── docker/
│   ├── Dockerfile               # ✅ Poetry + Celery worker
│   └── .dockerignore            # ✅
│
├── docker-compose.yml           # ✅ Redis + Celery + Flower
├── Makefile                     # ✅ 14 development commands
├── pyproject.toml               # ✅ Poetry dependencies
├── poetry.lock                  # ✅
├── .env                         # ✅ Environment variables
├── .env.example                 # ✅ Template
└── README.md                    # ✅ This file

Legend:
✅ Implemented/Working
📝 Contract defined (no implementation yet)
⏳ Pending (not started)
```

---

## 🔄 Happy Path Workflow

**User Journey:** Upload 2 Excel files → Get matched descriptions with prices

### Request Flow (End-to-End)

```
1. User uploads files
   POST /matching/process
   {
     "wf_file_id": "uuid-working-file",
     "ref_file_id": "uuid-reference-file",
     "threshold": 75.0
   }
        ↓
2. API Layer (matching.py)
   - Validates request (Pydantic)
   - Creates ProcessMatchingCommand
   - Injects ProcessMatchingUseCase
        ↓
3. Application Layer (ProcessMatchingUseCase)
   - Validates business rules (files exist, valid format)
   - Estimates processing time
   - Triggers Celery task
        ↓
4. Celery Task (process_matching_task)
   - Queued in Redis
   - Returns job_id immediately
   
   [ASYNC EXECUTION STARTS]
        ↓
5. Celery Worker (background)
   - Loads Excel files (Polars)
   - Extracts descriptions
   - Extracts parameters (DN, PN, etc.)
   - Matches descriptions (MatchingEngine)
   - Generates results with prices
   - Updates progress in Redis (0% → 100%)
        ↓
6. User polls status
   GET /jobs/{job_id}/status
        ↓
7. API Layer (jobs.py)
   - Creates GetJobStatusQuery
   - Injects GetJobStatusQueryHandler
        ↓
8. Query Handler
   - Retrieves status from Redis
   - Returns JobStatusResult
        ↓
9. User receives status
   {
     "status": "processing",
     "progress": 45,
     "message": "Matching descriptions (45/100)"
   }
        ↓
10. When complete (status: "completed")
    User downloads results
    GET /results/{job_id}/download
```

### Data Flow Diagram

```
┌──────────┐    HTTP     ┌─────────────┐
│  Client  │────────────▶│  API Layer  │
│          │◀────────────│  (FastAPI)  │
└──────────┘   202/200   └───────┬─────┘
                                 │
                          ┌──────▼──────────────┐
                          │ Application Layer   │
                          │ (Use Cases/CQRS)    │
                          └──────┬──────────────┘
                                 │
                    ┌────────────┴───────────┐
                    │                        │
            ┌───────▼────────┐      ┌────────▼────────┐
            │  Celery Task   │      │  Query Handler  │
            │  (async work)  │      │  (read status)  │
            └───────┬────────┘      └────────┬────────┘
                    │                        │
            ┌───────▼────────────────────────▼───────┐
            │         Infrastructure Layer           │
            │  - Redis (progress, cache)             │
            │  - FileStorage (Excel files)           │
            └────────────────────────────────────────┘
```

---

## 📋 Module Responsibilities

### API Layer
| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `api/routers/matching.py` | Trigger async matching process | 📝 Contract | `POST /matching/process` |
| `api/routers/jobs.py` | Query job status | 📝 Contract | `GET /jobs/{job_id}/status` |

### Application Layer
| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `application/commands/process_matching.py` | CQRS Write command | 📝 Contract | `ProcessMatchingCommand` |
| `application/queries/get_job_status.py` | CQRS Read query + handler | 📝 Contract | `GetJobStatusQuery`, `GetJobStatusQueryHandler` |
| `application/services/process_matching_use_case.py` | Orchestrates matching flow | 📝 Contract | `ProcessMatchingUseCase` |
| `application/tasks/celery_app.py` | Celery configuration | ✅ Working | `celery_app`, `health_check` task |
| `application/tasks/matching_tasks.py` | Async matching task | 📝 Contract | `process_matching_task` |
| `application/models.py` | Shared models | ✅ Working | `JobStatus` enum |

### Domain Layer
| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `domain/hvac/entities/` | Business entities | ⏳ Pending | `HVACDescription`, `MatchResult` |
| `domain/hvac/value_objects/` | Validated values | ⏳ Pending | `DN`, `PN`, `Material` |
| `domain/hvac/services/` | Business logic | ⏳ Pending | `MatchingEngine`, `ParameterExtractor` |

### Infrastructure Layer
| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `infrastructure/persistence/redis/` | Redis operations | ⏳ Pending | `RedisProgressTracker` |
| `infrastructure/file_storage/` | Excel file operations | ⏳ Pending | `ExcelReaderService`, `ExcelWriterService` |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Poetry 1.5+

### Installation

```bash
# Clone repository
git clone https://github.com/Piotr-Motyl/FastBidder3.0.git
cd FastBidder3.0

# Install dependencies
make install

# Copy environment template
cp .env.example .env

# Start Docker services (Redis + Celery + Flower)
make docker-up

# Verify services
make docker-health
```

### Verification

```bash
# Check Redis
docker exec fastbidder_redis redis-cli PING
# Expected: PONG

# Check Celery worker
docker exec fastbidder_celery_worker celery -A src.application.tasks inspect ping
# Expected: celery@hostname: OK

# Check Flower UI
open http://localhost:5555
```

---

## 🛠️ Development Commands

All commands available via Makefile:

### Local Development
```bash
make install        # Install dependencies with Poetry
make run            # Run FastAPI locally (with reload)
make celery-worker  # Run Celery worker locally
make celery-flower  # Run Flower UI locally
make lint           # Run linters (flake8 + mypy)
make format         # Format code (black + isort)
make test           # Run tests (when implemented)
```

### Docker Commands
```bash
make docker-up      # Start all services (Redis + Celery + Flower)
make docker-down    # Stop all services
make docker-logs    # Show logs
make docker-restart # Restart services
make docker-health  # Health check
make docker-test    # Run tests in Docker (when implemented)
```

---

## 🔧 Configuration

### Environment Variables

Key variables from `.env`:

```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Celery
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Matching Configuration (Happy Path)
DEFAULT_THRESHOLD=75
PARAM_WEIGHT=0.4              # 40% parameter matching
SEMANTIC_WEIGHT=0.6           # 60% semantic matching
MAX_DESCRIPTIONS_PER_REQUEST=100  # Phase 1 limit

# File Processing
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=.xlsx,.xls
```

### Docker Services

- **Redis**: `localhost:6379`
- **Flower UI**: `http://localhost:5555`
- **API**: `http://localhost:8000` (when running)

---

## 📊 Monitoring & Debugging

### Flower UI (Celery Monitoring)
```
URL: http://localhost:5555
Features:
- View active workers
- Monitor task progress
- Check task history
- Revoke/restart tasks
```

### Redis CLI
```bash
# Connect to Redis
docker exec -it fastbidder_redis redis-cli

# Check job status
GET job:3fa85f64-5717-4562-b3fc-2c963f66afa6:status

# List all job keys
KEYS job:*
```

### Logs
```bash
# All services
make docker-logs

# Specific service
docker logs fastbidder_celery_worker -f
```

---

## 🧪 Testing

**Status:** Tests will be implemented in Phase 2.

Planned test structure:
```
tests/
├── unit/           # Unit tests for each layer
├── integration/    # Integration tests (Redis, Celery)
└── e2e/           # End-to-end workflow tests
```

Run tests (when implemented):
```bash
make test           # Local
make docker-test    # Docker
```

---

## 📚 Key Concepts

### Clean Architecture
- **Inner layers** (Domain) contain business logic
- **Outer layers** (API, Infrastructure) handle technical details
- **Dependency rule**: Dependencies point inward only
- **Testability**: Inner layers have no external dependencies

### CQRS Pattern
- **Commands**: Write operations that modify state (`ProcessMatchingCommand`)
- **Queries**: Read operations that return data (`GetJobStatusQuery`)
- **Separation**: Different models for reads and writes
- **Benefits**: Scalability, clarity, optimized for each operation

### Contract-First Development (Phase 1)
1. Define interfaces and type signatures
2. Document expected behavior
3. Validate architecture
4. Implement in Phase 3

### Dependency Injection
- Dependencies passed via constructor (not created internally)
- Enables testing (mock dependencies)
- Follows Dependency Inversion Principle (SOLID)

---

## 🤝 Contributing

This is a personal portfolio project, but contributions are welcome!

### Workflow
1. Branch from `main`
2. Follow phase-by-phase implementation
3. Write contracts before implementation
4. Use conventional commits
5. Update this README when adding features

### Commit Format
```
Phase X Task Y.Z: Brief description

Detailed description (2 sentences max).

- Key change 1
- Key change 2
- Key change 3
```

---

## 📝 License

This project is for portfolio purposes. All rights reserved.

---

## 📞 Contact

**Author:** Piotr Motyl  
**LinkedIn:** [https://www.linkedin.com/in/piotr-motyl-634491257/](https://www.linkedin.com/in/piotr-motyl-634491257/)  
**GitHub:** [@Piotr-Motyl](https://github.com/Piotr-Motyl)

**Project Repository:** [FastBidder3.0](https://github.com/Piotr-Motyl/FastBidder3.0)

---

**Last Updated:** Phase 1 - Task 1.1.2 completed (2025-01-11)