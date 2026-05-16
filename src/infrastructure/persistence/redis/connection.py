"""
Redis connection pool management.

Singleton pool (max 10 connections), retry with exponential backoff (1s, 2s, 4s).
Config from env: REDIS_HOST, REDIS_PORT, REDIS_MAX_CONNECTIONS, REDIS_TIMEOUT, REDIS_RETRY_ATTEMPTS.
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
    Return Redis client with singleton connection pool.

    Creates pool on first call (thread-safe), reuses on subsequent calls.
    Raises RedisError after REDIS_RETRY_ATTEMPTS failed pings (exponential backoff).
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
    """Return True if Redis PING succeeds, False otherwise (never raises)."""
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
    """Disconnect all connections and reset singleton pool. Safe to call multiple times."""
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
