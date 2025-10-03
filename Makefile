# ===================================
# FastBidder Development Commands
# ===================================

.PHONY: help docker-up docker-down docker-logs docker-restart docker-test docker-health

help:
	@echo "╔════════════════════════════════════════╗"
	@echo "║   FastBidder Development Commands      ║"
	@echo "╚════════════════════════════════════════╝"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-up      - Start all services (Redis, Celery, Flower)"
	@echo "  make docker-down    - Stop all services"
	@echo "  make docker-logs    - Show logs from all services"
	@echo "  make docker-restart - Restart all services"
	@echo "  make docker-test    - Test Celery health check task"
	@echo "  make docker-health  - Check service health status"
	@echo ""

# Start all Docker services
docker-up:
	@echo "🚀 Starting FastBidder services..."
	docker compose up -d
	@echo "✅ Services started!"
	@echo "📊 Flower UI: http://localhost:5555"
	@echo "🔴 Redis: localhost:6379"

# Stop all Docker services
docker-down:
	@echo "🛑 Stopping FastBidder services..."
	docker compose down
	@echo "✅ Services stopped!"

# Show logs from all services
docker-logs:
	docker compose logs -f

# Restart all services
docker-restart:
	@echo "🔄 Restarting services..."
	docker compose restart
	@echo "✅ Services restarted!"

# Test Celery health check task
docker-test:
	@echo "🧪 Testing Celery health check..."
	@echo ""
	@docker compose exec -T celery_worker python -c "from src.application.tasks.celery_app import health_check; result = health_check.delay(); import time; time.sleep(1); print(result.get(timeout=5))"
	@echo ""
	@echo "✅ If you see task result above, Celery is working correctly!"

# Check health status of services
docker-health:
	@echo "🏥 Checking service health..."
	@echo ""
	@echo "Redis:"
	@docker compose exec -T redis redis-cli ping || echo "❌ Redis not responding"
	@echo ""
	@echo "Celery Worker:"
	@docker compose exec -T celery_worker celery -A src.application.tasks inspect ping || echo "❌ Celery not responding"
	@echo ""
	@echo "✅ Health check complete!"