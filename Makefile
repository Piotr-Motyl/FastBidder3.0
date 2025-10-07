# FastBidder - Makefile
# Development and Docker commands

.PHONY: help install run test lint format docker-up docker-down docker-logs docker-restart docker-test docker-health celery-worker celery-flower

# Display help
help:
	@echo "FastBidder - Available commands:"
	@echo ""
	@echo "Local Development:"
	@echo "  make install        - Install dependencies with Poetry"
	@echo "  make run            - Run FastAPI app locally (with reload)"
	@echo "  make test           - Run tests locally with pytest"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make format         - Format code (black, isort)"
	@echo "  make celery-worker  - Run Celery worker locally"
	@echo "  make celery-flower  - Run Flower monitoring locally"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-up      - Start all Docker services"
	@echo "  make docker-down    - Stop all Docker services"
	@echo "  make docker-logs    - Show Docker logs"
	@echo "  make docker-restart - Restart Docker services"
	@echo "  make docker-test    - Run tests in Docker"
	@echo "  make docker-health  - Check services health"

# Local Development Commands
install:
	poetry install

run:
	poetry run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

test:
	poetry run pytest -v --cov=src --cov-report=term-missing

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