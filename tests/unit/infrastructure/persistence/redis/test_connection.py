"""
Tests for Redis Connection Pool Management.

Covers:
- Singleton connection pool
- Thread-safe pool creation
- Retry logic with exponential backoff
- Health check with PING
- Connection cleanup
- Configuration from environment
"""

import threading
import time
from unittest.mock import MagicMock, Mock, patch, call

import pytest
from redis import ConnectionPool, Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from src.infrastructure.persistence.redis.connection import (
    get_redis_client,
    health_check,
    close_connections,
    _redis_pool,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton pool before and after each test."""
    # Import module-level variables
    import src.infrastructure.persistence.redis.connection as conn_module

    # Reset before test
    conn_module._redis_pool = None

    yield

    # Reset after test
    conn_module._redis_pool = None


@pytest.fixture
def mock_connection_pool():
    """Create mock ConnectionPool."""
    pool_mock = MagicMock(spec=ConnectionPool)
    pool_mock.disconnect.return_value = None
    return pool_mock


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client."""
    client_mock = MagicMock(spec=Redis)
    client_mock.ping.return_value = True
    return client_mock


# ============================================================================
# HAPPY PATH TESTS - get_redis_client()
# ============================================================================


def test_get_redis_client_creates_connection_pool():
    """Test get_redis_client creates connection pool on first call."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            client = get_redis_client()

            # Verify ConnectionPool was created
            pool_class.assert_called_once()
            assert client is not None


def test_get_redis_client_reuses_pool_on_subsequent_calls():
    """Test get_redis_client reuses pool (singleton pattern)."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            # First call
            client1 = get_redis_client()
            # Second call
            client2 = get_redis_client()

            # ConnectionPool should be created only once
            assert pool_class.call_count == 1


def test_get_redis_client_uses_default_config():
    """Test get_redis_client uses default configuration."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            get_redis_client()

            # Verify ConnectionPool was called with defaults
            pool_class.assert_called_once()
            call_kwargs = pool_class.call_args[1]

            assert call_kwargs["host"] == "localhost"
            assert call_kwargs["port"] == 6379
            assert call_kwargs["db"] == 0
            assert call_kwargs["max_connections"] == 10
            assert call_kwargs["decode_responses"] is True


def test_get_redis_client_uses_custom_config():
    """Test get_redis_client accepts custom configuration."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            get_redis_client(
                host="redis.example.com",
                port=6380,
                db=1,
                max_connections=20,
                timeout=10,
            )

            call_kwargs = pool_class.call_args[1]
            assert call_kwargs["host"] == "redis.example.com"
            assert call_kwargs["port"] == 6380
            assert call_kwargs["db"] == 1
            assert call_kwargs["max_connections"] == 20


def test_get_redis_client_reads_config_from_env():
    """Test get_redis_client reads configuration from environment variables."""
    with patch.dict(
        "os.environ",
        {
            "REDIS_HOST": "redis-prod.example.com",
            "REDIS_PORT": "6380",
            "REDIS_MAX_CONNECTIONS": "15",
            "REDIS_TIMEOUT": "10",
        },
    ):
        with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
            with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
                mock_client = MagicMock()
                mock_client.ping.return_value = True
                redis_class.return_value = mock_client

                get_redis_client()

                call_kwargs = pool_class.call_args[1]
                assert call_kwargs["host"] == "redis-prod.example.com"
                assert call_kwargs["port"] == 6380
                assert call_kwargs["max_connections"] == 15
                assert call_kwargs["socket_timeout"] == 10


def test_get_redis_client_tests_connection_with_ping():
    """Test get_redis_client tests connection with PING."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            get_redis_client()

            # Verify PING was called
            mock_client.ping.assert_called()


def test_get_redis_client_enables_socket_keepalive():
    """Test get_redis_client enables socket keepalive."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            get_redis_client()

            call_kwargs = pool_class.call_args[1]
            assert call_kwargs["socket_keepalive"] is True


# ============================================================================
# RETRY LOGIC TESTS - get_redis_client()
# ============================================================================


def test_get_redis_client_retries_on_connection_error():
    """Test get_redis_client retries on ConnectionError."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            with patch("time.sleep"):  # Mock sleep to speed up test
                mock_client = MagicMock()
                # Fail twice, then succeed
                mock_client.ping.side_effect = [
                    ConnectionError("Connection refused"),
                    ConnectionError("Connection refused"),
                    True,  # Success on third attempt
                ]
                redis_class.return_value = mock_client

                client = get_redis_client()

                # Should succeed after retries
                assert client is not None
                assert mock_client.ping.call_count == 3


def test_get_redis_client_retries_on_timeout_error():
    """Test get_redis_client retries on TimeoutError."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            with patch("time.sleep"):
                mock_client = MagicMock()
                # Fail once with timeout, then succeed
                mock_client.ping.side_effect = [
                    TimeoutError("Connection timeout"),
                    True,
                ]
                redis_class.return_value = mock_client

                client = get_redis_client()

                assert client is not None
                assert mock_client.ping.call_count == 2


def test_get_redis_client_uses_exponential_backoff():
    """Test get_redis_client uses exponential backoff (1s, 2s, 4s)."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            with patch("time.sleep") as mock_sleep:
                mock_client = MagicMock()
                # Fail 3 times, then succeed
                mock_client.ping.side_effect = [
                    ConnectionError("Failed"),
                    ConnectionError("Failed"),
                    ConnectionError("Failed"),
                    True,
                ]
                redis_class.return_value = mock_client

                with patch.dict("os.environ", {"REDIS_RETRY_ATTEMPTS": "4"}):
                    client = get_redis_client()

                # Verify exponential backoff: 1s, 2s, 4s
                sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
                assert sleep_calls == [1, 2, 4]


def test_get_redis_client_raises_after_max_retries():
    """Test get_redis_client raises RedisError after all retries exhausted."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            with patch("time.sleep"):
                mock_client = MagicMock()
                # Always fail
                mock_client.ping.side_effect = ConnectionError("Connection refused")
                redis_class.return_value = mock_client

                with pytest.raises(RedisError, match="Failed to connect to Redis after 3 attempts"):
                    get_redis_client()


def test_get_redis_client_respects_retry_attempts_from_env():
    """Test get_redis_client respects REDIS_RETRY_ATTEMPTS from environment."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            with patch("time.sleep"):
                mock_client = MagicMock()
                mock_client.ping.side_effect = ConnectionError("Failed")
                redis_class.return_value = mock_client

                with patch.dict("os.environ", {"REDIS_RETRY_ATTEMPTS": "5"}):
                    with pytest.raises(RedisError, match="Failed to connect to Redis after 5 attempts"):
                        get_redis_client()

                    # Should have tried 5 times
                    assert mock_client.ping.call_count == 5


# ============================================================================
# THREAD SAFETY TESTS - get_redis_client()
# ============================================================================


def test_get_redis_client_is_thread_safe():
    """Test get_redis_client creates pool only once in multi-threaded environment."""
    pool_creation_count = 0

    def mock_pool_init(*args, **kwargs):
        nonlocal pool_creation_count
        pool_creation_count += 1
        return MagicMock()

    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool", side_effect=mock_pool_init):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            # Create multiple threads calling get_redis_client
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=get_redis_client)
                threads.append(thread)
                thread.start()

            # Wait for all threads to complete
            for thread in threads:
                thread.join()

            # Pool should be created only once
            assert pool_creation_count == 1


# ============================================================================
# HAPPY PATH TESTS - health_check()
# ============================================================================


def test_health_check_returns_true_when_redis_healthy():
    """Test health_check returns True when Redis responds to PING."""
    with patch("src.infrastructure.persistence.redis.connection.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_get_client.return_value = mock_client

        result = health_check()

        assert result is True
        mock_client.ping.assert_called_once()


def test_health_check_returns_false_when_ping_fails():
    """Test health_check returns False when PING fails."""
    with patch("src.infrastructure.persistence.redis.connection.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.ping.return_value = False
        mock_get_client.return_value = mock_client

        result = health_check()

        assert result is False


def test_health_check_returns_false_on_redis_error():
    """Test health_check returns False on RedisError (doesn't raise)."""
    with patch("src.infrastructure.persistence.redis.connection.get_redis_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.ping.side_effect = RedisError("Connection lost")
        mock_get_client.return_value = mock_client

        result = health_check()

        assert result is False


def test_health_check_returns_false_on_connection_error():
    """Test health_check returns False on ConnectionError."""
    with patch("src.infrastructure.persistence.redis.connection.get_redis_client") as mock_get_client:
        mock_get_client.side_effect = RedisError("Cannot connect")

        result = health_check()

        assert result is False


def test_health_check_handles_unexpected_exceptions():
    """Test health_check handles unexpected exceptions gracefully."""
    with patch("src.infrastructure.persistence.redis.connection.get_redis_client") as mock_get_client:
        mock_get_client.side_effect = Exception("Unexpected error")

        result = health_check()

        assert result is False


# ============================================================================
# HAPPY PATH TESTS - close_connections()
# ============================================================================


def test_close_connections_disconnects_pool():
    """Test close_connections disconnects connection pool."""
    import src.infrastructure.persistence.redis.connection as conn_module

    mock_pool = MagicMock()
    conn_module._redis_pool = mock_pool

    close_connections()

    # Verify disconnect was called
    mock_pool.disconnect.assert_called_once()


def test_close_connections_resets_singleton():
    """Test close_connections resets singleton to None."""
    import src.infrastructure.persistence.redis.connection as conn_module

    mock_pool = MagicMock()
    conn_module._redis_pool = mock_pool

    close_connections()

    # Verify pool is reset to None
    assert conn_module._redis_pool is None


def test_close_connections_handles_already_closed_pool():
    """Test close_connections handles already closed pool (idempotent)."""
    import src.infrastructure.persistence.redis.connection as conn_module

    conn_module._redis_pool = None

    # Should not raise
    close_connections()


def test_close_connections_handles_disconnect_error():
    """Test close_connections handles error during disconnect."""
    import src.infrastructure.persistence.redis.connection as conn_module

    mock_pool = MagicMock()
    mock_pool.disconnect.side_effect = Exception("Disconnect failed")
    conn_module._redis_pool = mock_pool

    # Should not raise - logs error
    close_connections()

    # Pool should still be reset
    assert conn_module._redis_pool is None


def test_close_connections_is_thread_safe():
    """Test close_connections is thread-safe."""
    import src.infrastructure.persistence.redis.connection as conn_module

    mock_pool = MagicMock()
    conn_module._redis_pool = mock_pool

    # Create multiple threads calling close_connections
    threads = []
    for _ in range(5):
        thread = threading.Thread(target=close_connections)
        threads.append(thread)
        thread.start()

    # Wait for all threads
    for thread in threads:
        thread.join()

    # Should not raise and pool should be None
    assert conn_module._redis_pool is None


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_lifecycle_create_health_close():
    """Integration test: Full lifecycle from creation to closing."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool"):
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            # 1. Get client (creates pool)
            client = get_redis_client()
            assert client is not None

            # 2. Health check
            is_healthy = health_check()
            assert is_healthy is True

            # 3. Close connections
            close_connections()

            # 4. Pool should be reset - next call creates new pool
            import src.infrastructure.persistence.redis.connection as conn_module

            assert conn_module._redis_pool is None


def test_connection_pool_configuration_complete():
    """Integration test: Verify complete ConnectionPool configuration."""
    with patch("src.infrastructure.persistence.redis.connection.ConnectionPool") as pool_class:
        with patch("src.infrastructure.persistence.redis.connection.Redis") as redis_class:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            redis_class.return_value = mock_client

            get_redis_client(
                host="redis.example.com",
                port=6380,
                db=2,
                max_connections=15,
                timeout=10,
            )

            # Verify all configuration parameters
            call_kwargs = pool_class.call_args[1]
            assert call_kwargs["host"] == "redis.example.com"
            assert call_kwargs["port"] == 6380
            assert call_kwargs["db"] == 2
            assert call_kwargs["max_connections"] == 15
            assert call_kwargs["socket_timeout"] == 10
            assert call_kwargs["socket_connect_timeout"] == 10
            assert call_kwargs["socket_keepalive"] is True
            assert call_kwargs["decode_responses"] is True
