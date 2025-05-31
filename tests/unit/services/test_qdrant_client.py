"""Unit tests for QdrantClient module."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.client import QdrantClient


@pytest.fixture
def config():
    """Create test configuration."""
    return UnifiedConfig(
        qdrant={
            "url": "http://localhost:6333",
            "api_key": "test-key",
            "timeout": 30,
            "prefer_grpc": False,
        }
    )


@pytest.fixture
def client(config):
    """Create QdrantClient instance."""
    return QdrantClient(config)


class TestQdrantClientLifecycle:
    """Test client lifecycle management."""

    async def test_client_creation(self, client):
        """Test client creation without initialization."""
        assert client._client is None
        assert not client._initialized

    async def test_initialize_success(self, client):
        """Test successful client initialization."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful connection validation
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()

            assert client._initialized
            assert client._client is mock_client

            # Verify client was created with correct parameters
            mock_client_class.assert_called_once_with(
                url="http://localhost:6333",
                api_key="test-key",
                timeout=30,
                prefer_grpc=False,
            )

    async def test_initialize_connection_failure(self, client):
        """Test initialization failure due to connection issues."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock connection failure
            mock_client.get_collections.side_effect = ResponseHandlingException(
                "Connection failed"
            )

            with pytest.raises(QdrantServiceError, match="Failed to initialize"):
                await client.initialize()

            assert not client._initialized
            assert client._client is None

    async def test_initialize_idempotent(self, client):
        """Test that multiple initialize calls are safe."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()
            await client.initialize()  # Second call should be no-op

            # Should only create client once
            mock_client_class.assert_called_once()

    async def test_cleanup(self, client):
        """Test client cleanup."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()
            await client.cleanup()

            mock_client.close.assert_called_once()
            assert not client._initialized
            assert client._client is None

    async def test_cleanup_not_initialized(self, client):
        """Test cleanup when not initialized."""
        # Should not raise error
        await client.cleanup()
        assert not client._initialized

    async def test_validate_initialized(self, client):
        """Test validation of initialization state."""
        with pytest.raises(QdrantServiceError, match="not initialized"):
            client._validate_initialized()

    async def test_get_client(self, client):
        """Test getting initialized client."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()

            result = await client.get_client()
            assert result is mock_client

    async def test_get_client_not_initialized(self, client):
        """Test getting client when not initialized."""
        with pytest.raises(QdrantServiceError, match="not initialized"):
            await client.get_client()


class TestQdrantClientHealthCheck:
    """Test health check functionality."""

    async def test_health_check_success(self, client):
        """Test successful health check."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock collections response
            mock_collections = MagicMock()
            mock_collections.collections = [MagicMock(name="test_collection")]
            mock_client.get_collections.return_value = mock_collections

            # Mock cluster status (optional)
            mock_client.cluster_status.return_value = {"status": "ready"}

            await client.initialize()

            health = await client.health_check()

            assert health["status"] == "healthy"
            assert health["collections_count"] == 1
            assert "response_time_ms" in health
            assert health["url"] == "http://localhost:6333"

    async def test_health_check_failure(self, client):
        """Test health check with connection failure."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()

            # Simulate failure during health check
            mock_client.get_collections.side_effect = Exception("Connection lost")

            with pytest.raises(QdrantServiceError, match="Health check failed"):
                await client.health_check()

    async def test_health_check_not_initialized(self, client):
        """Test health check when not initialized."""
        with pytest.raises(QdrantServiceError, match="not initialized"):
            await client.health_check()


class TestQdrantClientValidation:
    """Test configuration validation."""

    async def test_validate_configuration_success(self, client):
        """Test successful configuration validation."""
        result = await client.validate_configuration()

        assert result["valid"] is True
        assert result["connection_status"] == "not_initialized"
        assert result["configuration"]["url"] == "http://localhost:6333"
        assert result["configuration"]["api_key_configured"] is True

    async def test_validate_configuration_invalid_url(self, config):
        """Test validation with invalid URL."""
        config.qdrant.url = "invalid-url"
        client = QdrantClient(config)

        result = await client.validate_configuration()

        assert result["valid"] is False
        assert any("Invalid URL format" in issue for issue in result["config_issues"])

    async def test_validate_configuration_with_connection(self, client):
        """Test validation with active connection."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()

            result = await client.validate_configuration()

            assert result["connection_status"] == "connected"

    async def test_validate_connection_unauthorized(self, client):
        """Test connection validation with unauthorized error."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock unauthorized error
            mock_client.get_collections.side_effect = ResponseHandlingException(
                "Unauthorized"
            )

            with pytest.raises(QdrantServiceError, match="Unauthorized access"):
                await client.initialize()

    async def test_validate_connection_timeout(self, client):
        """Test connection validation with timeout error."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock timeout error
            mock_client.get_collections.side_effect = ResponseHandlingException(
                "Connection timeout"
            )

            with pytest.raises(QdrantServiceError, match="Connection failed"):
                await client.initialize()


class TestQdrantClientOperations:
    """Test client operations."""

    async def test_reconnect(self, client):
        """Test client reconnection."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            # Create different mock instances for each call
            first_client = AsyncMock()
            second_client = AsyncMock()
            mock_client_class.side_effect = [first_client, second_client]

            first_client.get_collections.return_value = MagicMock()
            second_client.get_collections.return_value = MagicMock()

            await client.initialize()
            old_client = client._client

            await client.reconnect()

            # Should have closed old client and created new one
            first_client.close.assert_called_once()
            assert client._initialized
            assert client._client is second_client
            assert client._client is not old_client

    async def test_test_operation_basic_connectivity(self, client):
        """Test basic connectivity test operation."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock collections response with proper name attributes
            collection1 = MagicMock()
            collection1.name = "test1"
            collection2 = MagicMock()
            collection2.name = "test2"

            mock_collections = MagicMock()
            mock_collections.collections = [collection1, collection2]
            mock_client.get_collections.return_value = mock_collections

            await client.initialize()

            result = await client.test_operation("basic_connectivity")

            assert result["success"] is True
            assert result["operation"] == "basic_connectivity"
            assert "execution_time_ms" in result
            assert result["result"]["collections_found"] == 2
            assert "test1" in result["result"]["collections"]

    async def test_test_operation_unknown(self, client):
        """Test unknown test operation."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()

            result = await client.test_operation("unknown_operation")

            assert result["success"] is False
            assert "Unknown test operation" in result["error"]

    async def test_test_operation_failure(self, client):
        """Test test operation with failure."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock()

            await client.initialize()

            # Simulate failure
            mock_client.get_collections.side_effect = Exception("Test failure")

            result = await client.test_operation("basic_connectivity")

            assert result["success"] is False
            assert "Test failure" in result["error"]
            assert "execution_time_ms" in result
