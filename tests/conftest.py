"""
Pytest Configuration and Shared Fixtures

This module contains pytest configuration and shared fixtures used across
all test suites (unit, integration, e2e).

Fixtures:
    - redis_client: Redis client for tests
    - clean_redis: Cleans Redis database before/after tests
    - test_client: FastAPI TestClient for API testing
    - sample_files: Paths to sample Excel fixtures
    - docker_services: Ensures Docker services are running

Architecture Notes:
    - Fixtures follow pytest best practices
    - Redis cleanup ensures test isolation
    - TestClient doesn't require running server
    - Docker services check prevents cryptic errors

Usage:
    Tests automatically have access to these fixtures by name:

    def test_something(test_client, clean_redis):
        response = test_client.get("/health")
        assert response.status_code == 200
"""

import logging
import time
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

# Import Redis connection
from src.infrastructure.persistence.redis.connection import (
    get_redis_client,
    health_check,
    close_connections,
)

# Import FastAPI app
from src.api.main import create_app

# Configure logger for tests
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# REDIS FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def redis_client():
    """
    Provide Redis client for tests.

    Scope: session (shared across all tests in session)

    Yields:
        redis.Redis: Redis client instance

    Cleanup:
        Closes all connections after test session

    Note:
        Uses singleton connection pool from infrastructure layer.
    """
    client = get_redis_client()
    yield client
    # Cleanup after all tests
    close_connections()


@pytest.fixture(scope="function")
def clean_redis(redis_client):
    """
    Clean Redis database before and after each test.

    Ensures test isolation by flushing Redis database.
    Critical for E2E tests that use Redis for progress tracking.

    Scope: function (runs before/after each test)

    Args:
        redis_client: Redis client from redis_client fixture

    Yields:
        redis.Redis: Clean Redis client

    Examples:
        >>> def test_job_status(clean_redis):
        ...     # Redis is clean at start
        ...     progress_tracker.start_job(job_id, 100)
        ...     # Test runs...
        ...     # Redis is cleaned after test
    """
    # Clean before test
    redis_client.flushdb()
    logger.info("Redis database flushed (before test)")

    yield redis_client

    # Clean after test
    redis_client.flushdb()
    logger.info("Redis database flushed (after test)")


@pytest.fixture(scope="session")
def docker_services():
    """
    Ensure Docker services (Redis, Celery) are running.

    Checks if Redis is available before running E2E tests.
    Prevents cryptic errors when Docker services are not running.

    Scope: session (runs once at start of test session)

    Raises:
        RuntimeError: If Redis is not available

    Usage:
        >>> def test_e2e_workflow(docker_services, test_client):
        ...     # This test requires Redis to be running
        ...     # docker_services fixture ensures it's available
        ...     pass

    Note:
        Run `docker-compose up -d` before running E2E tests.
    """
    # Check if Redis is available
    if not health_check():
        raise RuntimeError(
            "Redis is not available. "
            "Please run 'docker-compose up -d' to start Redis and Celery services."
        )

    logger.info("Docker services check: Redis is available ✓")
    yield


# ============================================================================
# FASTAPI FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    """
    Provide FastAPI TestClient for API testing.

    TestClient doesn't require running server - it calls app directly.
    All API tests should use this fixture instead of real HTTP requests.

    Scope: session (shared across all tests)

    Yields:
        TestClient: FastAPI test client

    Examples:
        >>> def test_health_endpoint(test_client):
        ...     response = test_client.get("/health")
        ...     assert response.status_code == 200
        ...     assert response.json()["status"] == "ok"

    Note:
        TestClient automatically handles:
        - Request/response serialization
        - Exception handling
        - Middleware execution
        - Dependency injection (can be overridden)
    """
    app = create_app()
    with TestClient(app) as client:
        logger.info("FastAPI TestClient created")
        yield client
    logger.info("FastAPI TestClient closed")


# ============================================================================
# FILE FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def sample_files() -> dict[str, Path]:
    """
    Provide paths to sample Excel fixture files.

    Returns paths to sample_working_file.xlsx and sample_reference_file.xlsx
    generated by tests/fixtures/generate_fixtures.py (Task 3.10.1).

    Scope: session (shared across all tests)

    Returns:
        dict: Paths to fixture files
            - "working": Path to sample_working_file.xlsx (20 rows)
            - "reference": Path to sample_reference_file.xlsx (50 rows)

    Raises:
        FileNotFoundError: If fixture files don't exist

    Examples:
        >>> def test_upload(test_client, sample_files):
        ...     with open(sample_files["working"], "rb") as f:
        ...         response = test_client.post(
        ...             "/api/files/upload",
        ...             files={"file": ("working.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        ...         )
        ...     assert response.status_code == 201

    Note:
        If fixture files don't exist, run:
        python tests/fixtures/generate_fixtures.py
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    working_file = fixtures_dir / "sample_working_file.xlsx"
    reference_file = fixtures_dir / "sample_reference_file.xlsx"

    # Check if files exist
    if not working_file.exists():
        raise FileNotFoundError(
            f"Sample working file not found: {working_file}\n"
            "Run: python tests/fixtures/generate_fixtures.py"
        )

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Sample reference file not found: {reference_file}\n"
            "Run: python tests/fixtures/generate_fixtures.py"
        )

    logger.info(f"Sample files loaded: {working_file}, {reference_file}")

    return {
        "working": working_file,
        "reference": reference_file,
    }


@pytest.fixture(scope="session")
def performance_files() -> dict[str, Path]:
    """
    Provide paths to performance test Excel fixture files.

    Returns paths to performance_working_file.xlsx and performance_reference_file.xlsx
    generated by tests/fixtures/generate_fixtures.py --performance (Task 3.10.3).

    Scope: session (shared across all tests)

    Returns:
        dict: Paths to fixture files
            - "working": Path to performance_working_file.xlsx (100 rows)
            - "reference": Path to performance_reference_file.xlsx (200 rows)

    Raises:
        FileNotFoundError: If fixture files don't exist

    Examples:
        >>> def test_performance(test_client, performance_files):
        ...     with open(performance_files["working"], "rb") as f:
        ...         response = test_client.post(
        ...             "/api/files/upload",
        ...             files={"file": ("working.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        ...         )
        ...     assert response.status_code == 201

    Note:
        If fixture files don't exist, run:
        python tests/fixtures/generate_fixtures.py --performance
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    working_file = fixtures_dir / "performance_working_file.xlsx"
    reference_file = fixtures_dir / "performance_reference_file.xlsx"

    # Check if files exist
    if not working_file.exists():
        raise FileNotFoundError(
            f"Performance working file not found: {working_file}\n"
            "Run: python tests/fixtures/generate_fixtures.py --performance"
        )

    if not reference_file.exists():
        raise FileNotFoundError(
            f"Performance reference file not found: {reference_file}\n"
            "Run: python tests/fixtures/generate_fixtures.py --performance"
        )

    logger.info(f"Performance files loaded: {working_file}, {reference_file}")

    return {
        "working": working_file,
        "reference": reference_file,
    }


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """
    Pytest configuration hook.

    Registers custom markers for test categorization.

    Markers:
        - e2e: End-to-end tests (require Docker services)
        - integration: Integration tests (may require external services)
        - unit: Unit tests (no external dependencies)
        - slow: Slow tests (>1s execution time)

    Usage:
        @pytest.mark.e2e
        def test_full_workflow():
            pass

        # Run only E2E tests:
        # pytest -m e2e

        # Run all except E2E:
        # pytest -m "not e2e"
    """
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests (require Docker services)"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests (may require external services)"
    )
    config.addinivalue_line(
        "markers", "unit: Unit tests (no external dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: Slow tests (>1s execution time)"
    )


def pytest_collection_modifyitems(config, items):
    """
    Pytest collection hook.

    Automatically adds 'slow' marker to E2E tests.

    Args:
        config: Pytest config object
        items: List of collected test items
    """
    for item in items:
        # Add 'slow' marker to all E2E tests
        if "e2e" in item.keywords:
            item.add_marker(pytest.mark.slow)
