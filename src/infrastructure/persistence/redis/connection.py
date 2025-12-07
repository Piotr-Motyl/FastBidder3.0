"""
Redis Connection Pool Management.

Provides singleton connection pool for Redis with health checks and retry logic.
Used by RedisProgressTracker and other Redis-based services.

Responsibility:
    - Manage Redis connection pool (max 10 connections)
    - Health check with PING
    - Retry logic with exponential backoff
    - Thread-safe singleton pattern

Architecture Notes:
    - Infrastructure Layer (external dependency on Redis)
    - Singleton pattern for connection pool reuse
    - Thread-safe with threading.Lock
    - Environment-based configuration

Business Rules:
    - Max connections: 10 (configurable via REDIS_MAX_CONNECTIONS)
    - Connection timeout: 5s (configurable via REDIS_TIMEOUT)
    - Retry attempts: 3 (configurable via REDIS_RETRY_ATTEMPTS)
    - Exponential backoff: 1s, 2s, 4s (base=1s, multiplier=2)
    - Socket keepalive: enabled (default Redis behavior)
    - Decode responses: True (return strings not bytes)

Error Handling:
    - ConnectionError: Log and retry with exponential backoff
    - TimeoutError: Log and retry
    - RedisError: Log and raise after all retries exhausted
    - Health check failure: Return False (don't raise exception)

Examples:
    >>> # Get Redis client (creates pool on first call)
    >>> client = get_redis_client()
    >>> client.set("key", "value")
    >>>
    >>> # Health check
    >>> if health_check():
    ...     print("Redis is healthy")
    >>>
    >>> # Cleanup on shutdown
    >>> close_connections()
"""

import logging
import os
import threading
import time
from typing import Optional

from redis import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

# Configure logger for this module
logger = logging.getLogger(__name__)

# Singleton connection pool (thread-safe)
_redis_pool: Optional[ConnectionPool] = None
_pool_lock = threading.Lock()


def get_redis_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    db: int = 0,
    max_connections: Optional[int] = None,
    timeout: Optional[int] = None,
) -> Redis:
    """
    Get Redis client with connection pooling (singleton pattern).

    Creates connection pool on first call, reuses pool on subsequent calls.
    Thread-safe using lock. Implements retry logic with exponential backoff.

    Args:
        host: Redis hostname (default from env: REDIS_HOST or "localhost")
        port: Redis port (default from env: REDIS_PORT or 6379)
        db: Redis database number (default 0)
        max_connections: Max pool size (default from env: REDIS_MAX_CONNECTIONS or 10)
        timeout: Connection timeout in seconds (default from env: REDIS_TIMEOUT or 5)

    Returns:
        Redis client instance with connection pool

    Raises:
        RedisError: If connection fails after all retry attempts

    Examples:
        >>> # Get client with defaults (from environment)
        >>> client = get_redis_client()
        >>>
        >>> # Get client with custom settings
        >>> client = get_redis_client(host="redis.example.com", port=6380, db=1)
        >>>
        >>> # Use client
        >>> client.set("key", "value")
        >>> value = client.get("key")

    Implementation Details:
        - Singleton pattern: pool created once, reused across calls
        - Thread-safe: uses threading.Lock for pool creation
        - Retry logic: 3 attempts with exponential backoff (1s, 2s, 4s)
        - Decode responses: True (returns strings not bytes)
        - Socket keepalive: enabled for connection health
    """
    global _redis_pool

    # Get configuration from environment or use defaults
    redis_host = host or os.getenv("REDIS_HOST", "localhost")
    redis_port = port or int(os.getenv("REDIS_PORT", "6379"))
    redis_db = db
    max_conn = max_connections or int(os.getenv("REDIS_MAX_CONNECTIONS", "10"))
    conn_timeout = timeout or int(os.getenv("REDIS_TIMEOUT", "5"))

    # Create connection pool if not exists (thread-safe singleton)
    if _redis_pool is None:
        with _pool_lock:
            # Double-check locking pattern
            if _redis_pool is None:
                logger.info(
                    f"Creating Redis connection pool: "
                    f"host={redis_host}, port={redis_port}, db={redis_db}, "
                    f"max_connections={max_conn}, timeout={conn_timeout}s"
                )

                _redis_pool = ConnectionPool(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    max_connections=max_conn,
                    socket_timeout=conn_timeout,
                    socket_connect_timeout=conn_timeout,
                    socket_keepalive=True,
                    decode_responses=True,  # Return strings not bytes
                )

    # Create Redis client from pool
    client = Redis(connection_pool=_redis_pool)

    # Test connection with retry logic (exponential backoff)
    retry_attempts = int(os.getenv("REDIS_RETRY_ATTEMPTS", "3"))
    backoff_base = 1  # Base delay in seconds
    last_error: Optional[Exception] = None

    for attempt in range(retry_attempts):
        try:
            # Test connection with PING
            client.ping()
            logger.debug(f"Redis connection established (attempt {attempt + 1})")
            return client

        except (ConnectionError, TimeoutError) as e:
            last_error = e
            if attempt < retry_attempts - 1:  # Not last attempt
                # Calculate exponential backoff delay
                delay = backoff_base * (2**attempt)
                logger.warning(
                    f"Redis connection failed (attempt {attempt + 1}/{retry_attempts}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                # Last attempt failed
                logger.error(
                    f"Redis connection failed after {retry_attempts} attempts: {e}"
                )

    # All retries exhausted
    raise RedisError(
        f"Failed to connect to Redis after {retry_attempts} attempts. "
        f"Last error: {last_error}"
    )


def health_check() -> bool:
    """
    Check Redis health with PING test.

    Tests connection to Redis server by sending PING command.
    Returns True if Redis responds with PONG, False otherwise.

    Returns:
        True if Redis is healthy (PING successful), False otherwise

    Examples:
        >>> # Check if Redis is available
        >>> if health_check():
        ...     print("Redis is healthy")
        ... else:
        ...     print("Redis is down or unreachable")

    Implementation Details:
        - Uses get_redis_client() to get pooled connection
        - Sends PING command to Redis
        - Returns False on any error (doesn't raise exception)
        - Logs warning on health check failure
    """
    try:
        # Get Redis client from pool
        client = get_redis_client()

        # Send PING command
        response = client.ping()

        # PING returns True if successful
        if response:
            logger.debug("Redis health check: OK")
            return True
        else:
            logger.warning("Redis health check: PING returned False")
            return False

    except RedisError as e:
        logger.warning(f"Redis health check failed: {e}")
        return False

    except Exception as e:
        logger.error(f"Unexpected error in Redis health check: {e}")
        return False


def close_connections() -> None:
    """
    Close all Redis connections in the pool.

    Closes connection pool and resets singleton. Should be called on application
    shutdown to gracefully close all connections.

    Thread-safe using lock.

    Examples:
        >>> # On application shutdown
        >>> close_connections()

    Implementation Details:
        - Thread-safe: uses threading.Lock
        - Resets global _redis_pool to None
        - Calls disconnect() on connection pool
        - Logs closure operation
        - Safe to call multiple times (idempotent)
    """
    global _redis_pool

    with _pool_lock:
        if _redis_pool is not None:
            logger.info("Closing Redis connection pool")

            try:
                # Disconnect all connections in pool
                _redis_pool.disconnect()

            except Exception as e:
                logger.error(f"Error closing Redis connection pool: {e}")

            finally:
                # Reset singleton to None
                _redis_pool = None
                logger.info("Redis connection pool closed")

        else:
            logger.debug("Redis connection pool already closed or not initialized")
