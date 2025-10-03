FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Copy source code
COPY src/ ./src/

# Set Python path
ENV PYTHONPATH=/app

# Health check (for future use)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "from src.application.tasks.celery_app import celery_app; celery_app.control.inspect().ping()" || exit 1

# Default command (will be overridden in docker compose)
CMD ["celery", "-A", "src.application.tasks", "worker", "--loglevel=info"]