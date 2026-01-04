# FastBidder - Developer Workflow Guide

This guide explains the development and testing workflow for FastBidder project.

---

## Table of Contents

1. [Initial Setup (One-time)](#initial-setup-one-time)
2. [Daily Development Workflow](#daily-development-workflow)
3. [Testing Workflow](#testing-workflow)
4. [Understanding Poetry Virtual Environment](#understanding-poetry-virtual-environment)
5. [Common Issues & Solutions](#common-issues--solutions)

---

## Initial Setup (One-time)

**You only need to do this ONCE when you first clone the project.**

### Option 1: Automated Setup (Recommended)

```bash
# In WSL2 terminal
cd /mnt/d/DOK_PIOTREK/Programowanie/PYTHON/FastBidder/source_code/fastbidder
make setup-dev
```

This single command will:
- Install system dependencies (gcc, g++, make, python3-dev)
- Install all Python dependencies via Poetry
- Generate test fixture files

### Option 2: Manual Setup

```bash
# 1. Install system dependencies (needed for chromadb compilation)
sudo apt update
sudo apt install -y python3-dev gcc g++ make

# 2. Install Python dependencies
poetry install

# 3. Generate test fixtures
poetry run python tests/fixtures/generate_fixtures.py
poetry run python tests/fixtures/generate_fixtures.py --performance
```

### After Initial Setup

Poetry creates a **single virtual environment** (`.venv` directory) that persists across terminal sessions.

**Important**: You do NOT need to run `poetry install` again unless:
- You add/remove dependencies in `pyproject.toml`
- You delete the `.venv` directory
- You switch to a different machine

---

## Daily Development Workflow

### Starting a New Terminal Session

**Good news**: You don't need to reinstall anything! Just use Poetry commands:

```bash
# Option 1: Run commands with poetry run prefix
poetry run pytest tests/unit/

# Option 2: Activate Poetry shell (recommended for long sessions)
poetry shell
# Now you can use pytest directly:
pytest tests/unit/
```

### Typical Development Session

```bash
# Terminal 1 (WSL2): Start services
cd /mnt/d/DOK_PIOTREK/Programowanie/PYTHON/FastBidder/source_code/fastbidder
make docker-up              # Start Redis
make celery-worker          # Start Celery worker (keep this running)

# Terminal 2 (WSL2): Development and testing
cd /mnt/d/DOK_PIOTREK/Programowanie/PYTHON/FastBidder/source_code/fastbidder
poetry shell                # Activate environment (optional, but convenient)

# Make your code changes...
# Run tests as needed:
make test-unit              # Fast unit tests
make test-integration       # Integration tests
make test-e2e              # E2E tests (requires Terminal 1 services)
```

---

## Testing Workflow

### Test Types & Requirements

| Test Type | Command | Docker Required? | Celery Required? | Speed |
|-----------|---------|------------------|------------------|-------|
| **Unit** | `make test-unit` | ❌ No | ❌ No | ⚡ Fast (~10s) |
| **Integration** | `make test-integration` | ✅ Yes (Redis) | ❌ No | 🐢 Medium (~30s) |
| **E2E** | `make test-e2e` | ✅ Yes (Redis) | ✅ Yes | 🐌 Slow (~2min) |
| **All** | `make test-all` | ✅ Yes | ✅ Yes | 🐌 Slowest (~3min) |

### Recommended Testing Strategy

#### During Development (Rapid Feedback Loop)

```bash
# 1. Run unit tests frequently (no setup needed)
make test-unit

# 2. Run integration tests when you change infrastructure code
make test-integration
```

#### Before Committing Code

```bash
# Terminal 1: Start services
make docker-up
make celery-worker

# Terminal 2: Run full test suite
make test-all
```
### Checking Service Status

Before running E2E tests, verify services are running:

```bash
make check-services
```

Output example:
```
Checking Docker services...
Redis: ✅ Running
Celery Worker: ❌ Not running (run: make celery-worker in separate terminal)
```

---

## Understanding Poetry Virtual Environment

### How Poetry Works

```
┌─────────────────────────────────────────┐
│  Your Project Directory                 │
│  /mnt/d/.../fastbidder/                 │
│                                          │
│  ┌────────────────────────────────┐    │
│  │  .venv/  (Virtual Environment)  │    │
│  │  ├── bin/python                 │    │
│  │  ├── lib/python3.10/            │    │
│  │  │   ├── chromadb/              │    │
│  │  │   ├── sentence-transformers/ │    │
│  │  │   ├── fastapi/               │    │
│  │  │   └── ... (all dependencies) │    │
│  └────────────────────────────────┘    │
│                                          │
│  pyproject.toml  (dependency list)      │
│  poetry.lock     (exact versions)       │
└─────────────────────────────────────────┘
```

### Key Points

1. **Single Environment**: Poetry creates ONE `.venv` directory
2. **Persistent**: The `.venv` exists between terminal sessions
3. **No Reinstall Needed**: Dependencies stay installed until you remove them
4. **Large AI Dependencies**: chromadb, sentence-transformers, NVIDIA CUDA packages are installed ONCE and stay forever

### Terminal Sessions

```bash
# Terminal 1 - Today
cd /mnt/d/.../fastbidder
poetry run pytest         # ✅ Uses .venv

# You close Terminal 1...
# You open Terminal 2 - Tomorrow

cd /mnt/d/.../fastbidder
poetry run pytest         # ✅ Uses SAME .venv (no reinstall!)
```

### When to Run `poetry install`

✅ **Run when**:
- First time setting up project: `make setup-dev`
- After adding dependency: `poetry add requests` → automatically installs
- After pulling changes with new dependencies: `poetry install`
- After deleting `.venv`: `poetry install`

❌ **Don't run when**:
- Starting a new terminal (just use `poetry run` or `poetry shell`)
- Switching between terminals
- After closing/reopening VSCode

---

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError" when running tests

**Symptom**: `ModuleNotFoundError: No module named 'chromadb'`

**Cause**: Not using Poetry's virtual environment

**Solution**:
```bash
# Option 1: Use poetry run
poetry run pytest

# Option 2: Activate poetry shell first
poetry shell
pytest
```

### Issue 2: Celery worker can't find files

**Symptom**: `FileNotFoundError: No working file found in upload directory`

**Cause**: Tests running in Windows, Celery in WSL2 (separate filesystems)

**Solution**: Always run tests in WSL2 terminal:
```bash
# ✅ Correct - WSL2 terminal
piotrmotyl@MOTYL-Lenovo:/mnt/d/.../fastbidder$ make test-e2e

# ❌ Wrong - Windows PowerShell/CMD
PS D:\...\fastbidder> make test-e2e
```

### Issue 3: E2E tests fail with "Redis connection refused"

**Symptom**: `redis.exceptions.ConnectionError: Error connecting to Redis`

**Cause**: Docker services not running

**Solution**:
```bash
# Check service status
make check-services

# Start services
make docker-up
make celery-worker  # In separate terminal
```

### Issue 4: Tests pass locally but fail in CI

**Symptom**: Different behavior in GitHub Actions vs local

**Cause**: Environment differences (file paths, service availability)

**Solution**:
```bash
# Run CI test mode locally (strict)
make test-ci
```

### Issue 5: "python3-dev not found" during poetry install

**Symptom**: `fatal error: Python.h: No such file or directory`

**Cause**: Missing system dependencies for C++ compilation (chromadb)

**Solution**:
```bash
make setup-system
# Or manually:
sudo apt install -y python3-dev gcc g++ make
```

---

## Summary: Quick Reference

### First Time Setup
```bash
make setup-dev    # Run once, installs everything
```

### Every Day Development
```bash
# Terminal 1: Services (keep running)
make docker-up
make celery-worker

# Terminal 2: Development
poetry shell       # Optional but convenient
make test-unit     # Run frequently during development
make test-e2e      # Before committing
```

### Remember
- ✅ Poetry environment persists across terminals - no need to reinstall
- ✅ Use WSL2 for all development and testing
- ✅ Run `make check-services` before E2E tests
- ✅ Keep Celery worker running in separate terminal during E2E tests
- ❌ Don't run `poetry install` for every terminal
- ❌ Don't run tests in Windows terminal when Celery runs in WSL2

---

## Additional Resources

- Poetry documentation: https://python-poetry.org/docs/
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/
- Celery documentation: https://docs.celeryproject.org/
- pytest documentation: https://docs.pytest.org/

---

**Questions?** Check [CLAUDE.md](CLAUDE.md) for project-specific instructions or consult the team.
