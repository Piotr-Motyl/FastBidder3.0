# FastBidder - Makefile
# Development and Docker commands

.PHONY: help setup-dev setup-system setup-temp-dirs install check-services run test test-all test-unit test-integration test-e2e test-e2e-debug test-ci evaluate lint format clean-chromadb inspect-chromadb docker-up docker-down docker-logs docker-restart docker-test docker-health celery-worker celery-flower

# Display help
help:
	@echo "FastBidder - Available commands:"
	@echo ""
	@echo "Initial Setup (run once):"
	@echo "  make setup-dev         - Complete dev environment setup (system deps + poetry install)"
	@echo "  make setup-system      - Install system dependencies (requires sudo)"
	@echo "  make setup-temp-dirs   - Create /tmp/fastbidder directories (auto-run by test-e2e)"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all          - Run all tests (unit + integration + E2E)"
	@echo "  make test-unit         - Run only unit tests (fast, no Docker needed)"
	@echo "  make test-integration  - Run integration tests (requires Docker)"
	@echo "  make test-e2e          - Run E2E tests (cleans ChromaDB first)"
	@echo "  make test-e2e-debug    - Run E2E tests WITHOUT cleaning ChromaDB (for debugging)"
	@echo "  make test-ci           - CI/CD test run (strict mode, coverage threshold)"
	@echo "  make evaluate          - Run matching quality evaluation"
	@echo "  make check-services    - Check if Docker services are running"
	@echo ""
	@echo "Debugging & Cleanup:"
	@echo "  make clean-chromadb    - Clean ChromaDB vector database"
	@echo "  make inspect-chromadb  - Inspect ChromaDB contents (after failed test)"
	@echo ""
	@echo "Local Development:"
	@echo "  make install           - Install dependencies with Poetry"
	@echo "  make run               - Run FastAPI app locally (with reload)"
	@echo "  make test              - Run all tests (alias for test-all)"
	@echo "  make lint              - Run linters (flake8, mypy)"
	@echo "  make format            - Format code (black, isort)"
	@echo "  make celery-worker     - Run Celery worker locally"
	@echo "  make celery-flower     - Run Flower monitoring locally"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-up         - Start all Docker services"
	@echo "  make docker-down       - Stop all Docker services"
	@echo "  make docker-logs       - Show Docker logs"
	@echo "  make docker-restart    - Restart Docker services"
	@echo "  make docker-test       - Run tests in Docker"
	@echo "  make docker-health     - Check services health"

# Setup Commands (run once for initial project setup)
setup-dev:
	@echo "=== FastBidder Development Environment Setup ==="
	@echo ""
	@echo "Step 1/4: Installing system dependencies (requires sudo)..."
	@$(MAKE) setup-system
	@echo ""
	@echo "Step 2/4: Installing Python dependencies with Poetry..."
	@$(MAKE) install
	@echo ""
	@echo "Step 3/4: Setting up temporary directories..."
	@$(MAKE) setup-temp-dirs
	@echo ""
	@echo "Step 4/4: Generating test fixtures..."
	@poetry run python tests/fixtures/generate_fixtures.py
	@poetry run python tests/fixtures/generate_fixtures.py --performance
	@echo ""
	@echo "✅ Development environment setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Start Docker services: make docker-up"
	@echo "  2. In separate terminal, start Celery worker: make celery-worker"
	@echo "  3. Run tests: make test-all"

setup-system:
	@echo "Installing system dependencies (gcc, g++, make, python3-dev)..."
	@command -v gcc >/dev/null 2>&1 || { \
		echo "Installing build tools..."; \
		sudo apt update && sudo apt install -y python3-dev gcc g++ make; \
	}
	@echo "✅ System dependencies OK"

setup-temp-dirs:
	@echo "Setting up temporary directories for file storage..."
	@mkdir -p /tmp/fastbidder/uploads 2>/dev/null || sudo mkdir -p /tmp/fastbidder/uploads
	@mkdir -p /tmp/fastbidder/jobs 2>/dev/null || sudo mkdir -p /tmp/fastbidder/jobs
	@chmod -R 777 /tmp/fastbidder 2>/dev/null || sudo chmod -R 777 /tmp/fastbidder
	@echo "✅ Temp directories ready: /tmp/fastbidder/"

# Local Development Commands
install:
	poetry install

check-services:
	@echo "Checking services status..."
	@echo ""
	@echo -n "Redis (Docker): "
	@docker compose exec -T redis redis-cli ping 2>/dev/null | grep -q PONG && echo "✅ Running" || echo "❌ Not running (run: make docker-up)"
	@echo -n "Celery Worker (Local): "
	@poetry run celery -A src.application.tasks.celery_app inspect ping -t 2 2>/dev/null | grep -q "pong" && echo "✅ Running" || echo "❌ Not running (run: make celery-worker in separate terminal)"

run:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests (alias for test-all)
test:
	poetry run pytest -v --cov=src --cov-report=term-missing

# Run all tests (unit + integration + E2E)
test-all:
	poetry run pytest -v --cov=src --cov-report=term-missing

# Run only unit tests (fast, no Docker needed)
test-unit:
	poetry run pytest tests/unit/ -v --cov=src --cov-report=term-missing

# Run only integration tests (requires Docker: Redis + ChromaDB)
test-integration:
	poetry run pytest tests/integration/ -v -m integration

# Run only E2E tests (requires Docker: Redis + Celery worker)
# Cleans ChromaDB BEFORE tests to ensure deterministic state
test-e2e: clean-chromadb
	@echo "Preparing environment for E2E tests..."
	@$(MAKE) setup-temp-dirs
	@echo ""
	@echo "Checking if required services are running..."
	@$(MAKE) check-services
	@echo ""
	@echo "Running E2E tests..."
	poetry run pytest tests/e2e/ -v -m e2e

# Run E2E tests WITHOUT cleaning ChromaDB (for debugging failed tests)
# Allows inspection of ChromaDB data after test failure
test-e2e-debug:
	@echo "🐛 Running E2E tests in DEBUG mode (keeping ChromaDB data)..."
	@echo "Note: ChromaDB will NOT be cleaned before or after tests"
	@echo "Use 'make inspect-chromadb' after test to inspect data"
	@echo ""
	@$(MAKE) setup-temp-dirs
	@echo ""
	@$(MAKE) check-services
	@echo ""
	poetry run pytest tests/e2e/ -v -m e2e -s

# CI/CD: Run all tests with strict mode (coverage threshold, skip slow tests)
test-ci:
	poetry run pytest -v --cov=src --cov-report=xml --cov-fail-under=80 -m "not slow"

# Run matching quality evaluation with golden dataset
evaluate:
	@echo "Running matching evaluation..."
	@echo "Note: Ensure golden dataset exists at tests/fixtures/golden_dataset.json"
	poetry run python scripts/evaluate_matching.py \
		--dataset tests/fixtures/golden_dataset.json \
		--threshold 75.0 \
		--output evaluation_report.md

lint:
	poetry run flake8 src tests
	poetry run mypy src

format:
	poetry run black src tests
	poetry run isort src tests

# Clean ChromaDB vector database (removes all indexed embeddings)
# Use before E2E tests to ensure deterministic state
clean-chromadb:
	@echo "🧹 Cleaning ChromaDB vector database..."
	@rm -rf data/chroma_db 2>/dev/null || true
	@echo "✓ ChromaDB cleaned (data/chroma_db/ removed)"

# Inspect ChromaDB contents after failed test
# Shows statistics and sample documents from vector database
inspect-chromadb:
	@echo "Inspecting ChromaDB vector database..."
	@poetry run python -c "\
from pathlib import Path; \
import sys; \
if not Path('data/chroma_db').exists(): \
    print('ERROR: ChromaDB directory not found (data/chroma_db/)'); \
    print('       Run a test first or check if it was cleaned'); \
    sys.exit(1); \
from src.infrastructure.ai.vector_store.chroma_client import ChromaClient; \
client = ChromaClient(); \
collection = client.get_or_create_collection(); \
count = collection.count(); \
print(f'\\nChromaDB initialized at: {client.persist_directory}'); \
print(f'Collection: {collection.name}'); \
print(f'Total documents: {count}'); \
if count > 0: \
    docs = collection.get(limit=3); \
    print(f'\\nSample documents (first 3):'); \
    for i, doc_id in enumerate(docs['ids'][:3]): \
        meta = docs['metadatas'][i]; \
        text = docs['documents'][i][:60] + '...' if len(docs['documents'][i]) > 60 else docs['documents'][i]; \
        print(f'  {i+1}. ID: {doc_id}'); \
        print(f'     Text: {text}'); \
        print(f'     Metadata: file_id={meta.get(\"file_id\")}, dn={meta.get(\"dn\")}, pn={meta.get(\"pn\")}'); \
else: \
    print('\\nWARNING: No documents in database'); \
"

celery-worker:
	poetry run celery -A src.application.tasks.celery_app worker --loglevel=info

celery-flower:
	poetry run celery -A src.application.tasks.celery_app flower --port=5555

# Docker Commands
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose restart

docker-test:
	docker compose exec celery_worker celery -A src.application.tasks.celery_app inspect ping

docker-health:
	@echo "Redis health:"
	@docker compose exec redis redis-cli ping || echo "Redis not responding"
	@echo "\nCelery worker health:"
	@docker compose exec celery_worker celery -A src.application.tasks.celery_app inspect active || echo "Celery not responding"