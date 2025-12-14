# FastBidder 3.0

> 🚀 AI-powered HVAC product matching system with hybrid parameter + semantic search

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Celery](https://img.shields.io/badge/Celery-5.3+-red.svg)](https://docs.celeryq.dev/)
[![Redis](https://img.shields.io/badge/Redis-7.0+-DC382D.svg)](https://redis.io/)
[![Polars](https://img.shields.io/badge/Polars-0.20+-CD792C.svg)](https://www.pola.rs/)
[![Docker](https://img.shields.io/badge/Docker-24.0+-2496ED.svg)](https://www.docker.com/)
[![Clean Architecture](https://img.shields.io/badge/Architecture-Clean-brightgreen.svg)](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
[![License](https://img.shields.io/badge/license-Portfolio-blue.svg)](LICENSE)

---

## 📑 Table of Contents

- [About The Project](#-about-the-project)
- [Current Implementation Status](#-current-implementation-status)
- [Architecture Overview](#%EF%B8%8F-architecture-overview)
- [Domain Model](#-domain-model)
- [Matching Algorithm](#-matching-algorithm)
- [Project Structure](#-project-structure)
- [Happy Path Workflow](#-happy-path-workflow)
- [Module Responsibilities](#-module-responsibilities)
- [Quick Start](#-quick-start)
- [Development Commands](#%EF%B8%8F-development-commands)
- [Configuration](#-configuration)
- [Monitoring & Debugging](#-monitoring--debugging)
- [Testing](#-testing)
- [Key Concepts](#-key-concepts)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📖 About The Project

FastBidder automates the tedious process of matching HVAC and plumbing product descriptions with supplier catalogs to find accurate pricing. Built for companies in the Mechanical installations industry who need to quickly generate cost estimates from technical specifications. The system uses **hybrid matching** (40% parameter-based regex + 60% AI semantic similarity) to achieve 85%+ accuracy, reducing manual matching time from 8 working hours to 30 minutes instead.

This project demonstrates production-grade architecture principles: **Clean Architecture**, **CQRS pattern**, **async task processing with Celery**, and **domain-driven design**. Built with scalability and maintainability in mind, following Test-Driven Development with contract-first implementation approach.

**Tech Stack:** Python 3.10, FastAPI, Celery, Redis, Polars (instead of Pandas), Pydantic v2, Docker, Poetry

---

## 🎯 Current Implementation Status

```
Phase 0: Setup                ✅ Done
Phase 1: High-Level Contracts ✅ Done
Phase 2: Detailed Contracts   ✅ Done
Phase 3: Implementation       ✅ Done (All Sprints 3.1-3.10: Domain + Infra + App + API + E2E)
Phase 4: AI Integration       ⏳ Pending (Semantic matching)
Phase 5: Advanced Features    ⏳ Pending (Batch, optimization)
Phase 6: Testing & Docs       🚧 Partial (E2E ✅, Unit for API/App ⏳ Deferred)
```

**Next Steps:** Phase 4 - AI Integration (Semantic matching with sentence-transformers)

---

## 🏗️ Architecture Overview

### Clean Architecture Layers

FastBidder follows **Clean Architecture** with strict dependency rules and **Protocol-based dependency inversion**:

```
┌─────────────────────────────────────────┐
│         API Layer (Presentation)        │  ← HTTP endpoints (FastAPI)
│  - Routers: matching, jobs, files       │
│  - Request/Response schemas (Pydantic)  │
│  - Shared ErrorResponse schema          │
└──────────────┬──────────────────────────┘
               │ depends on
┌──────────────▼──────────────────────────┐
│       Application Layer (Use Cases)     │  ← Orchestration (CQRS)
│  - Commands (CQRS Write)                │
│  - Queries (CQRS Read)                  │
│  - Use Cases (business flow)            │
│  - Celery Tasks (async processing)      │
│  - Ports (Protocol interfaces)          │
└──────────────┬──────────────────────────┘
               │ depends on
┌──────────────▼──────────────────────────┐
│       Domain Layer (Business Logic)     │  ← Core business rules (DDD)
│  - Entities (HVACDescription)           │
│  - Value Objects (MatchScore, etc.)     │
│  - Domain Services (Protocols)          │
│  - Repository Interfaces (Protocols)    │
└──────────────▲──────────────────────────┘
               │ implemented by (Dependency Inversion)
┌──────────────┴──────────────────────────┐
│    Infrastructure Layer (External)      │  ← Technical capabilities
│  - Redis (progress tracking, cache)     │
│  - File Storage (Excel: Polars/openpyxl)│
│  - Repository Implementations           │
│  - Matching Engine Implementation       │
└─────────────────────────────────────────┘
```

### 🎯 Key Principles

- **Dependency Inversion**: Outer layers depend on inner layers (never reverse)
  - Infrastructure implements Domain Protocols
  - Application defines Ports, Infrastructure implements them
- **CQRS Pattern**: Separate Commands (write) from Queries (read)
- **Contract-First**: Define interfaces before implementation (Protocols)
- **Async by Design**: Long-running operations via Celery + Redis

---

## 🧬 Domain Model

FastBidder uses **Domain-Driven Design** with clear separation between Entities, Value Objects, and Domain Services.

### 📦 Entities (Mutable, with Identity)

**HVACDescription** - Core domain entity representing HVAC equipment description
```python
@dataclass
class HVACDescription:
    id: UUID                                    # Unique identifier
    raw_text: str                               # Original description text
    source_row_number: int                      # Excel row (1-based)
    file_id: UUID                               # Source file identifier
    extracted_params: ExtractedParameters | None  # Extracted DN, PN, etc.
    match_score: MatchScore | None              # Hybrid match score
    matched_price: Decimal | None               # Price from reference catalog
    matched_description: str | None             # Matched reference text
    state: EntityState                          # CREATED → PARAMETERS_EXTRACTED → MATCHED → PRICED

```

### 💎 Value Objects (Immutable, no Identity)

**ExtractedParameters** - Technical parameters extracted from description
```python
@dataclass(frozen=True)
class ExtractedParameters:
    dn: int | None                    # Diameter Nominal (DN50, DN100)
    pn: int | None                    # Pressure Nominal (PN16, PN10)
    material: str | None              # Material (brass, steel, etc.)
    valve_type: str | None            # Type (ball valve, check valve)
    confidence_scores: dict[str, float]  # Extraction confidence (0.0-1.0)
```

**MatchScore** - Hybrid matching score breakdown
```python
@dataclass(frozen=True)
class MatchScore:
    final_score: float              # Combined score (0-100)
    parameter_score: float          # Parameter matching (0-100)
    semantic_score: float           # Semantic similarity (0-100)

    @staticmethod
    def create(final: float, param: float, semantic: float) -> MatchScore:
        """Factory with validation (40% param + 60% semantic)"""
```

**MatchResult** - Complete match result with justification
```python
@dataclass(frozen=True)
class MatchResult:
    matched_reference_id: UUID           # Reference item UUID
    score: MatchScore                    # Score breakdown
    justification: str                   # Human-readable explanation
    parameter_scores: dict[str, float]   # Individual param scores
    semantic_score: float                # Raw semantic similarity
```

### 🔧 Domain Services (Protocols)

**MatchingEngineProtocol** - Hybrid matching service interface
```python
class MatchingEngineProtocol(Protocol):
    def match(
        self,
        working_item: HVACDescription,
        reference_catalog: list[HVACDescription],
        threshold: float = 75.0
    ) -> MatchResult | None:
        """Find best match using hybrid algorithm"""
```

**ParameterExtractorProtocol** - Parameter extraction service
```python
class ParameterExtractorProtocol(Protocol):
    def extract(self, text: str) -> ExtractedParameters:
        """Extract DN, PN, material from text using regex"""
```

---

## 🎲 Matching Algorithm

FastBidder uses a **hybrid matching algorithm** combining parameter-based and semantic similarity matching.

### Algorithm Overview

```
┌─────────────────────┐
│  Working Item       │
│  "Zawór DN50 PN16"  │
└──────────┬──────────┘
           │
    ┌──────▼────────────────────────┐
    │  1. Parameter Extraction      │
    │  DN=50, PN=16, type=valve     │
    └──────┬────────────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │  2. For each reference item:        │
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │ A. Parameter Matching (40%)    │ │
    │  │    - DN: Exact match → 100%    │ │
    │  │    - PN: Exact match → 100%    │ │
    │  │    - Material: Fuzzy → 0-100%  │ │
    │  │    Average → param_score       │ │
    │  └────────────────────────────────┘ │
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │ B. Semantic Matching (60%)     │ │
    │  │    - Embeddings (transformers) │ │
    │  │    - Cosine similarity         │ │
    │  │    → semantic_score            │ │
    │  └────────────────────────────────┘ │
    │                                      │
    │  ┌────────────────────────────────┐ │
    │  │ C. Combine Scores              │ │
    │  │ final = 0.4×param + 0.6×sem    │ │
    │  └────────────────────────────────┘ │
    └─────────────────────────────────────┘
           │
    ┌──────▼────────────────────┐
    │  3. Filter & Sort         │
    │  - Keep score >= threshold│
    │  - Return best match      │
    └───────────────────────────┘
```

### 📊 Scoring Details

**High-Level Scoring Model**
The algorithm uses a **two-component hybrid approach**:
- **Parameter Matching**: 40% weight (technical parameters)
- **Semantic Matching**: 60% weight (AI embeddings)

**Final Score Formula**
```
final_score = (0.4 × parameter_score) + (0.6 × semantic_score)
```

**Detailed Parameter Scoring (within the 40% parameter weight)**
**More detailed and proper calculation pattern will be adjusted after delivery  infrastructure and happy path.**
The parameter_score is calculated from individual parameter weights:
- **DN (Diameter)**: 30% - Exact match only
  - DN50 = DN50 → 100%
  - DN50 ≠ DN100 → 0%
- **PN (Pressure)**: 10% - Exact match only
  - PN16 = PN16 → 100%
  - PN16 ≠ PN10 → 0%
- **Material**: 15% - Fuzzy matching with synonyms
  - "brass" ~= "mosiądz" → 90%
  - "steel" ~= "stainless steel" → 80%
- **Valve Type**: 15% - Semantic similarity
  - "ball valve" ~= "zawór kulowy" → 95%
- **Other Parameters**: Other parameters will be added systematically as the program develops.


**Note**: The granular weights (DN=30%, PN=10%, Material=15%, Type=15%) are normalized within the parameter_score component, which then contributes 40% to the final score.

**Semantic Matching (60% weight)**
- Uses **sentence-transformers** (multilingual model)
- Converts descriptions to embeddings (768-dim vectors)
- Calculates **cosine similarity** (0.0 to 1.0)
- Scaled to 0-100 range

**Threshold Logic**
- Default threshold: **75.0** (configurable)
- Return match only if `final_score >= threshold`
- If multiple matches above threshold → return highest score
- If no matches above threshold → return `None`

### 🎯 Matching Engines

**HybridMatchingEngine** (Phase 4 - AI Integration)
- Full hybrid matching (40% param + 60% semantic)
- Uses sentence-transformers for embeddings
- ChromaDB for vector similarity search
- Production-ready implementation with all features

**SimpleMatchingEngine** (Phase 3 - Happy Path)
- Fallback engine for initial implementation
- Parameter-based exact matching only
- No AI/embeddings (faster, simpler)
- Used for testing and as fallback when AI unavailable

---

## 📁 Project Structure

```
fastbidder/
├── src/
│   ├── api/                          # 🌐 API Layer (Presentation)
│   │   ├── routers/
│   │   │   ├── matching.py           # POST /matching/process
│   │   │   ├── jobs.py               # GET /jobs/{job_id}/status
│   │   │   ├── files.py              # POST /files/upload
│   │   │   ├── results.py            # GET /results/{job_id}/download
│   │   │   └── __init__.py
│   │   ├── schemas/
│   │   │   ├── common.py             # ErrorResponse (shared)
│   │   │   └── __init__.py
│   │   ├── graphql/                  # ⏳ GraphQL API (Phase 5)
│   │   │   ├── schema.py             # Strawberry schema
│   │   │   ├── queries.py            # GraphQL queries
│   │   │   └── mutations.py          # GraphQL mutations
│   │   ├── websockets/               # ⏳ Real-time (Phase 5)
│   │   │   ├── sse.py                # Server-Sent Events
│   │   │   └── handlers.py           # WebSocket handlers
│   │   └── main.py                   # FastAPI app
│   │
│   ├── application/                  # 🎯 Application Layer (Use Cases)
│   │   ├── commands/
│   │   │   ├── process_matching.py   # ProcessMatchingCommand
│   │   │   └── __init__.py
│   │   ├── queries/
│   │   │   ├── get_job_status.py     # GetJobStatusQuery + Handler
│   │   │   └── __init__.py
│   │   ├── services/
│   │   │   ├── process_matching_use_case.py  # Main orchestration
│   │   │   ├── file_upload_use_case.py       # File validation
│   │   │   └── __init__.py
│   │   ├── tasks/
│   │   │   ├── celery_app.py         # ✅ Celery config
│   │   │   ├── matching_tasks.py     # Async matching task
│   │   │   └── __init__.py
│   │   ├── ports/
│   │   │   ├── file_storage.py       # FileStorageServiceProtocol
│   │   │   └── __init__.py
│   │   └── models.py                 # JobStatus, MatchingStrategy, ReportFormat
│   │
│   ├── domain/                       # 🧬 Domain Layer (Business Logic)
│   │   ├── hvac/
│   │   │   ├── entities/
│   │   │   │   ├── hvac_description.py       # HVACDescription entity
│   │   │   │   └── __init__.py
│   │   │   ├── value_objects/
│   │   │   │   ├── extracted_parameters.py   # DN, PN, material
│   │   │   │   ├── match_score.py            # Hybrid score
│   │   │   │   ├── match_result.py           # Match result
│   │   │   │   └── __init__.py
│   │   │   ├── services/
│   │   │   │   ├── matching_engine.py        # MatchingEngineProtocol
│   │   │   │   ├── parameter_extractor.py    # ParameterExtractorProtocol
│   │   │   │   ├── simple_matching_engine.py # SimpleMatchingEngine (fallback)
│   │   │   │   └── __init__.py
│   │   │   └── repositories/
│   │   │       ├── hvac_description_repository.py  # Protocol
│   │   │       └── __init__.py
│   │   └── shared/
│   │       ├── exceptions.py         # DomainException hierarchy
│   │       └── __init__.py
│   │
│   ├── infrastructure/               # ⚙️ Infrastructure Layer (External)
│   │   ├── persistence/
│   │   │   ├── redis/
│   │   │   │   ├── progress_tracker.py       # RedisProgressTracker
│   │   │   │   └── __init__.py
│   │   │   ├── repositories/
│   │   │   │   ├── hvac_description_repository.py  # Redis impl
│   │   │   │   └── __init__.py
│   │   │   └── __init__.py
│   │   ├── file_storage/
│   │   │   ├── file_storage_service.py       # FileStorageService
│   │   │   ├── excel_reader.py               # Polars-based reader
│   │   │   ├── excel_writer.py               # openpyxl-based writer
│   │   │   └── __init__.py
│   │   ├── matching/
│   │   │   ├── matching_engine.py            # HybridMatchingEngine (Phase 4)
│   │   │   └── __init__.py
│   │   ├── ai/                       # ⏳ AI/ML Infrastructure (Phase 4)
│   │   │   ├── embeddings/
│   │   │   │   ├── sentence_transformer.py   # Model wrapper
│   │   │   │   └── cache.py                 # Embedding cache
│   │   │   └── nlp/
│   │   │       ├── spacy_pipeline.py        # spaCy NER
│   │   │       └── patterns.py              # HVAC patterns
│   │   ├── monitoring/               # ⏳ Observability (Phase 5)
│   │   │   ├── logging.py            # Structured logging
│   │   │   ├── metrics.py            # Prometheus metrics
│   │   │   └── tracing.py            # OpenTelemetry
│   │   └── __init__.py
│   │
│   └── shared/                       # 🔧 Cross-cutting concerns (Phase 4+)
│       └── __init__.py
│
├── docker/
│   ├── Dockerfile
│   └── .dockerignore
│
├── docker-compose.yml                # Redis + Celery + Flower
├── Makefile                          # 14 development commands
├── pyproject.toml                    # Poetry dependencies
├── poetry.lock
├── .env                              # Environment variables
├── .env.example                      # Config template (safe for repo)
├── ROADMAP.md                        # High-level roadmap
├── IMPL_PLAN.md                      # Detailed sprint-by-sprint plan
└── README.md

Legend:
✅ Implemented/Working
📝 Contract defined (Phase 2 - ready for Phase 3 implementation)
⏳ Placeholder (Phase 3+)
```

---

## 🔄 Happy Path Workflow

**User Journey:** Upload 2 Excel files → Get matched descriptions with prices

### Request Flow (End-to-End)

```
1. 📤 User uploads files
   POST /files/upload (2x: working + reference)
   Returns: { file_id: UUID }
        ↓

2. 🚀 User triggers matching
   POST /matching/process
   {
     "wf_file_id": "uuid-working-file",
     "ref_file_id": "uuid-reference-file",
     "threshold": 75.0,
     "matching_strategy": "HYBRID",
     "report_format": "DETAILED"
   }
        ↓

3. 🌐 API Layer (matching.py)
   - Validates request (Pydantic)
   - Creates ProcessMatchingCommand
   - Injects ProcessMatchingUseCase
        ↓

4. 🎯 Application Layer (ProcessMatchingUseCase)
   - Validates business rules (files exist, valid format)
   - Estimates processing time
   - Triggers Celery task
        ↓

5. ⚡ Celery Task (process_matching_task)
   - Queued in Redis
   - Returns: { job_id: UUID, status: "queued" }

   [ASYNC EXECUTION STARTS IN BACKGROUND]
        ↓

6. 🔄 Celery Worker (background processing)
   a. Load Excel files (Polars for speed)
   b. Parse descriptions → HVACDescription entities
   c. Extract parameters → ExtractedParameters (DN, PN, etc.)
   d. Match descriptions → MatchingEngine.match()
   e. Calculate hybrid scores → MatchScore
   f. Generate results with prices
   g. Write output Excel (openpyxl)
   h. Update progress in Redis (0% → 100%)
        ↓

7. 📊 User polls status
   GET /jobs/{job_id}/status
   Returns: {
     "status": "processing",
     "progress": 45,
     "message": "Matching descriptions (45/100)"
   }
        ↓

8. ✅ When complete (status: "completed")
   GET /results/{job_id}/download
   Returns: Excel file with:
     - Original columns
     - Matched prices (colored by score)
     - Match reports (DN, PN, score)
```

### 📊 Data Flow Diagram

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
            │  - FileStorage (Polars/openpyxl)       │
            │  - MatchingEngine (hybrid algorithm)   │
            └────────────────────────────────────────┘
```

---

## 📋 Module Responsibilities

### 🌐 API Layer (Presentation)

| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `api/routers/matching.py` | Trigger async matching process | ✅ Implemented | `POST /matching/process` |
| `api/routers/jobs.py` | Query job status | ✅ Implemented | `GET /jobs/{job_id}/status` |
| `api/routers/files.py` | File upload endpoints | ✅ Implemented | `POST /files/upload` |
| `api/routers/results.py` | Result download | ✅ Implemented | `GET /results/{job_id}/download` |
| `api/schemas/common.py` | Shared response schemas | ✅ Implemented | `ErrorResponse` |

### 🎯 Application Layer (Use Cases)

| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `commands/process_matching.py` | CQRS Write command | ✅ Implemented | `ProcessMatchingCommand` |
| `queries/get_job_status.py` | CQRS Read query + handler | ✅ Implemented | `GetJobStatusQuery`, `JobStatusResult` |
| `services/process_matching_use_case.py` | Orchestrates matching flow | ✅ Implemented | `ProcessMatchingUseCase` |
| `services/file_upload_use_case.py` | File validation & storage | ✅ Implemented | `FileUploadUseCase` |
| `tasks/celery_app.py` | Celery configuration | ✅ Implemented | `celery_app`, `health_check` |
| `tasks/matching_tasks.py` | Async matching task | ✅ Implemented | `process_matching_task` |
| `ports/file_storage.py` | File storage Protocol | ✅ Implemented | `FileStorageServiceProtocol` |
| `models.py` | Shared models & enums | ✅ Implemented | `JobStatus`, `MatchingStrategy`, `ReportFormat` |

### 🧬 Domain Layer (Business Logic)

| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `entities/hvac_description.py` | Core domain entity | ✅ Implemented | `HVACDescription` (99% cov, 42 tests) |
| `matching_config.py` | Configuration dataclass | ✅ Implemented | `MatchingConfig` (100% cov, 19 tests) |
| `value_objects/diameter_nominal.py` | DN Value Object | ✅ Implemented | `DiameterNominal` (98% coverage) |
| `value_objects/pressure_nominal.py` | PN Value Object | ✅ Implemented | `PressureNominal` (90% coverage) |
| `value_objects/extracted_parameters.py` | Technical parameters | ✅ Implemented | `ExtractedParameters` (100% coverage) |
| `value_objects/match_score.py` | Hybrid scoring | ✅ Implemented | `MatchScore` (100% coverage) |
| `value_objects/match_result.py` | Match result | ✅ Implemented | `MatchResult` (100% coverage) |
| `services/matching_engine.py` | Matching service Protocol | 📝 Contract | `MatchingEngineProtocol` |
| `services/parameter_extractor.py` | Parameter extraction Protocol | 📝 Contract | `ParameterExtractorProtocol` |
| `services/simple_matching_engine.py` | Fallback matching engine | ✅ Implemented | `SimpleMatchingEngine` (92% coverage) |
| `repositories/hvac_description_repository.py` | Repository Protocol | 📝 Contract | `HVACDescriptionRepositoryProtocol` |
| `patterns.py` | Regex patterns & text helpers | ✅ Implemented | `normalize_text()`, `find_canonical_form()` (95% coverage) |
| `constants.py` | Domain dictionaries & constants | ✅ Implemented | `VALVE_TYPES`, `MATERIALS`, `DRIVE_TYPES`, `MANUFACTURERS` |
| `shared/exceptions.py` | Domain exceptions | 📝 Contract | `DomainException`, `ValidationError` |

### ⚙️ Infrastructure Layer (External)

| File | Responsibility | Status | Key Components |
|------|---------------|--------|----------------|
| `persistence/redis/progress_tracker.py` | Job progress tracking | ✅ Implemented | `RedisProgressTracker` |
| `persistence/repositories/hvac_description_repository.py` | Redis-based storage | 📝 Contract | `HVACDescriptionRepository` |
| `file_storage/file_storage_service.py` | File management | ✅ Implemented | `FileStorageService` (90% coverage) |
| `file_storage/excel_reader.py` | Excel parsing (Polars) | ✅ Implemented | `ExcelReaderService` (97% coverage) |
| `file_storage/excel_writer.py` | Excel generation (openpyxl) | ✅ Implemented | `ExcelWriterService` (96% coverage) |
| `matching/matching_engine.py` | Hybrid matching implementation | 📝 Contract | `HybridMatchingEngine` |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Docker & Docker Compose**
- **Poetry 1.5+**
- **Git**

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Piotr-Motyl/FastBidder3.0.git
cd FastBidder3.0/source_code/fastbidder

# 2. Install dependencies
make install

# 3. Copy environment template
cp .env.example .env

# 4. Start Docker services (Redis + Celery + Flower)
make docker-up

# 5. Verify services
make docker-health
```

### ✅ Verification

```bash
# Check Redis connection
docker exec fastbidder_redis redis-cli PING
# Expected: PONG

# Check Celery worker (in hybrid mode: run locally, not in Docker)
celery -A src.application.tasks inspect ping
# Expected: pong from celery@<hostname>

# Start Flower UI for monitoring (optional)
make celery-flower
# Then open: http://localhost:5555
```

**Note:** In hybrid development mode, Celery worker runs **locally** (not in Docker). Start with: `make celery-worker`

---

## 🛠️ Development Commands

All commands available via **Makefile** (14 commands):

### 💻 Local Development

```bash
make install        # Install dependencies with Poetry
make run            # Run FastAPI locally (with hot reload)
make celery-worker  # Run Celery worker locally
make celery-flower  # Run Flower UI locally (monitoring)
make lint           # Run linters (flake8 + mypy)
make format         # Format code (black + isort)
make test           # Run tests (Phase 6)
make clean          # Clean temp files and caches
```

### 🐳 Docker Commands

```bash
make docker-up      # Start all services (Redis + Celery + Flower)
make docker-down    # Stop all services
make docker-logs    # Show logs (all services)
make docker-restart # Restart services
make docker-health  # Health check (Redis + Celery)
make docker-test    # Run tests in Docker (Phase 6)
```

---

## 🔧 Configuration

### Environment Variables

Key variables from `.env` (see `.env.example` for full list):

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Celery Configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/1

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_CACHE_TTL=3600              # 1 hour cache TTL

# Matching Algorithm Configuration (Happy Path)
DEFAULT_THRESHOLD=75.0             # Minimum match score (0-100)
PARAM_WEIGHT=0.4                   # 40% parameter matching
SEMANTIC_WEIGHT=0.6                # 60% semantic matching
MAX_DESCRIPTIONS_PER_REQUEST=100   # Phase 3 initial limit
# Note: Phase 3 starts with 100 for happy path testing
#       Phase 5 will increase to 400 after batch processing optimization

# File Processing
MAX_FILE_SIZE_MB=10
ALLOWED_EXTENSIONS=.xlsx,.xls
TEMP_DIR=/tmp/fastbidder
```

### 🐳 Docker Services (Hybrid Development Mode)

**Running in Docker:**
- **Redis**: `localhost:6379` (DB 0: Celery broker, DB 1: Results, DB 2+: Progress tracking)

**Running Locally** (for faster development):
- **Celery Worker**: `make celery-worker`
- **Flower UI**: `make celery-flower` → `http://localhost:5555`
- **FastAPI**: `make run` → `http://localhost:8000`

**Why Hybrid?** No container rebuild after code changes, hot-reload works, easier debugging.

---

## 📊 Monitoring & Debugging

### 🌸 Flower UI (Celery Monitoring)

```
URL: http://localhost:5555

Features:
✅ View active workers
✅ Monitor task progress in real-time
✅ Check task history (success/failure)
✅ Revoke/restart tasks
✅ View worker statistics
```

### 🔍 Redis CLI (Direct Database Access)

```bash
# Connect to Redis
docker exec -it fastbidder_redis redis-cli

# Check job status
GET progress:3fa85f64-5717-4562-b3fc-2c963f66afa6

# List all job keys
KEYS progress:*

# Check result
GET result:3fa85f64-5717-4562-b3fc-2c963f66afa6
```

### 📋 Logs

```bash
# All services logs
make docker-logs

# Specific service logs (follow)
docker logs fastbidder_celery_worker -f
docker logs fastbidder_redis -f
docker logs fastbidder_flower -f
```

---

## 🧪 Testing

**Status:** Tests will be implemented in **Phase 6: Testing & Documentation**.

### Planned Test Structure

```
tests/
├── unit/               # Unit tests for each layer
│   ├── test_domain/    # Entities, Value Objects, Services
│   ├── test_application/  # Use Cases, Commands, Queries
│   └── test_infrastructure/  # Repository implementations
├── integration/        # Integration tests
│   ├── test_redis/     # Redis persistence
│   ├── test_celery/    # Celery task execution
│   └── test_excel/     # Excel parsing/writing
└── e2e/               # End-to-end workflow tests
    └── test_matching_workflow.py
```

### Run Tests (Phase 6)

```bash
# Local environment
make test

# Docker environment
make docker-test

# Coverage report
make test-coverage
```

---

## 📚 Key Concepts

### 🏛️ Clean Architecture

- **Inner layers** (Domain) contain pure business logic
- **Outer layers** (API, Infrastructure) handle technical details
- **Dependency rule**: Dependencies point **inward** only
- **Testability**: Inner layers have zero external dependencies
- **Benefit**: Easy to test, maintain, and replace components

### 📝 CQRS Pattern (Command Query Responsibility Segregation)

- **Commands**: Write operations that modify state (`ProcessMatchingCommand`)
- **Queries**: Read operations that return data (`GetJobStatusQuery`)
- **Separation**: Different models for reads and writes
- **Benefits**: Scalability, code clarity, optimized for each operation

### 🔌 Protocol-based Dependency Inversion

- **Domain defines Protocols** (interfaces)
- **Infrastructure implements Protocols** (concrete classes)
- **Application uses Ports** (Protocol interfaces for external services)
- **Benefit**: Loose coupling, easy mocking for tests

### 📋 Contract-First Development (Phase 1 & 2)

1. ✅ Define interfaces and type signatures (Protocols)
2. ✅ Document expected behavior (detailed docstrings)
3. ✅ Validate architecture (code review)
4. ⏳ Implement in Phase 3 (happy path)

### 💉 Dependency Injection

- Dependencies passed via constructor (not created internally)
- Enables testing (mock dependencies)
- Follows **Dependency Inversion Principle** (SOLID)

### 🧬 Domain-Driven Design (DDD)

- **Entities**: Mutable objects with identity (HVACDescription)
- **Value Objects**: Immutable objects without identity (MatchScore)
- **Domain Services**: Business logic that doesn't fit entities (MatchingEngine)
- **Repositories**: Data access abstraction (Protocols)

---

## 🤝 Contributing

This is a **personal portfolio project**, but contributions are welcome!

---

## 📝 License

This project is for **portfolio purposes**. All rights reserved.

---

## 📞 Contact

**Author:** Piotr Motyl
**Role:** Junior Python Developer
**LinkedIn:** [linkedin.com/in/piotr-motyl-634491257](https://www.linkedin.com/in/piotr-motyl-634491257/)
**GitHub:** [@Piotr-Motyl](https://github.com/Piotr-Motyl)

**Project Repository:** [github.com/Piotr-Motyl/FastBidder3.0](https://github.com/Piotr-Motyl/FastBidder3.0)

---

<div align="center">

**⭐ If you find this project interesting, please consider giving it a star! ⭐**

</div>
