# FastBidder - Makefile
# Development and Docker commands

.PHONY: help install run test test-all test-unit test-integration test-e2e test-ci evaluate lint format docker-up docker-down docker-logs docker-restart docker-test docker-health celery-worker celery-flower

# Display help
help:
	@echo "FastBidder - Available commands:"
	@echo ""
	@echo "Testing:"
	@echo "  make test-all          - Run all tests (unit + integration + E2E)"
	@echo "  make test-unit         - Run only unit tests (fast, no Docker needed)"
	@echo "  make test-integration  - Run integration tests (requires Docker)"
	@echo "  make test-e2e          - Run E2E tests (requires Docker + Celery)"
	@echo "  make test-ci           - CI/CD test run (strict mode, coverage threshold)"
	@echo "  make evaluate          - Run matching quality evaluation"
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

# Local Development Commands
install:
	poetry install

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
test-e2e:
	poetry run pytest tests/e2e/ -v -m e2e

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