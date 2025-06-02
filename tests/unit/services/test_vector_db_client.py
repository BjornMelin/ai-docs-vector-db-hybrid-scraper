"""Tests for QdrantClient service."""

import logging
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.client import QdrantClient


class TestQdrantClient:
    """Test cases for QdrantClient service."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = MagicMock(spec=UnifiedConfig)
        config.qdrant = MagicMock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = "test-api-key"
        config.qdrant.timeout = 30
        config.qdrant.prefer_grpc = False
        return config

    @pytest.fixture
    def client(self, mock_config):
        """Create QdrantClient instance."""
        return QdrantClient(mock_config)

    @pytest.fixture
    def mock_async_client(self):
        """Create mock AsyncQdrantClient."""
        return AsyncMock(spec=AsyncQdrantClient)

    async def test_initialization_success(self, client, mock_config):
        """Test successful client initialization."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.return_value = MagicMock(collections=[])

            await client.initialize()

            assert client._initialized is True
            assert client._client is mock_client
            mock_client_class.assert_called_once_with(
                url=mock_config.qdrant.url,
                api_key=mock_config.qdrant.api_key,
                timeout=mock_config.qdrant.timeout,
                prefer_grpc=mock_config.qdrant.prefer_grpc,
            )
            mock_client.get_collections.assert_called_once()

    async def test_initialization_already_initialized(self, client):
        """Test initialization when already initialized."""
        client._initialized = True
        await client.initialize()
        # Should return early without creating new client

    async def test_initialization_connection_failure(self, client):
        """Test initialization failure during connection validation."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get_collections.side_effect = ResponseHandlingException(
                "Connection failed"
            )

            with pytest.raises(
                QdrantServiceError, match="Failed to initialize Qdrant client"
            ):
                await client.initialize()

            assert client._initialized is False
            assert client._client is None

    async def test_initialization_generic_error(self, client):
        """Test initialization failure with generic error."""
        with patch(
            "src.services.vector_db.client.AsyncQdrantClient"
        ) as mock_client_class:
            mock_client_class.side_effect = ValueError("Invalid configuration")

            with pytest.raises(
                QdrantServiceError, match="Failed to initialize Qdrant client"
            ):
                await client.initialize()

            assert client._initialized is False
            assert client._client is None

    async def test_cleanup_success(self, client):
        """Test successful cleanup."""
        mock_client = AsyncMock()
        client._client = mock_client
        client._initialized = True

        await client.cleanup()

        mock_client.close.assert_called_once()
        assert client._client is None
        assert client._initialized is False

    async def test_cleanup_with_error(self, client, caplog):
        """Test cleanup with error during close."""
        mock_client = AsyncMock()
        mock_client.close.side_effect = Exception("Close failed")
        client._client = mock_client
        client._initialized = True

        with caplog.at_level(logging.WARNING):
            await client.cleanup()

        assert "Error during client cleanup" in caplog.text
        assert client._client is None
        assert client._initialized is False

    async def test_cleanup_no_client(self, client):
        """Test cleanup when no client exists."""
        client._client = None
        client._initialized = False

        await client.cleanup()
        # Should complete without error

    async def test_get_client_success(self, client):
        """Test successful client retrieval."""
        mock_client = AsyncMock()
        client._client = mock_client
        client._initialized = True

        result = await client.get_client()

        assert result is mock_client

    async def test_get_client_not_initialized(self, client):
        """Test client retrieval when not initialized."""
        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            await client.get_client()

    async def test_get_client_no_client(self, client):
        """Test client retrieval when client is None."""
        client._initialized = True
        client._client = None

        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            await client.get_client()

    async def test_health_check_success(self, client):
        """Test successful health check."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(), MagicMock()]
        mock_client.get_collections.return_value = mock_collections
        mock_client.cluster_status.return_value = {"status": "healthy"}

        client._client = mock_client
        client._initialized = True

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["url"] == client.config.qdrant.url
        assert result["collections_count"] == 2
        assert "response_time_ms" in result
        assert "client_config" in result
        assert "cluster_info" in result

    async def test_health_check_cluster_not_supported(self, client):
        """Test health check when cluster status not supported."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        mock_client.cluster_status.side_effect = Exception("Not supported")

        client._client = mock_client
        client._initialized = True

        result = await client.health_check()

        assert result["status"] == "healthy"
        assert result["cluster_info"] == {"status": "single_node"}

    async def test_health_check_not_initialized(self, client):
        """Test health check when not initialized."""
        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            await client.health_check()

    async def test_health_check_failure(self, client):
        """Test health check failure."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = Exception("Health check failed")

        client._client = mock_client
        client._initialized = True

        with pytest.raises(QdrantServiceError, match="Health check failed"):
            await client.health_check()

    async def test_validate_configuration_success(self, client, mock_config):
        """Test successful configuration validation."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MagicMock()

        client._client = mock_client
        client._initialized = True

        result = await client.validate_configuration()

        assert result["valid"] is True
        assert result["connection_status"] == "connected"
        assert len(result["config_issues"]) == 0
        assert "configuration" in result

    async def test_validate_configuration_invalid_url(self, client, mock_config):
        """Test configuration validation with invalid URL."""
        mock_config.qdrant.url = "invalid-url"

        result = await client.validate_configuration()

        assert result["valid"] is False
        assert any("Invalid URL format" in issue for issue in result["config_issues"])

    async def test_validate_configuration_timeout_warnings(self, client, mock_config):
        """Test configuration validation with timeout warnings."""
        mock_config.qdrant.timeout = 5  # Too low

        result = await client.validate_configuration()

        assert any(
            "Consider increasing timeout" in rec for rec in result["recommendations"]
        )

        mock_config.qdrant.timeout = 400  # Too high

        result = await client.validate_configuration()

        assert any("Timeout is very high" in rec for rec in result["recommendations"])

    async def test_validate_configuration_grpc_http_warning(self, client, mock_config):
        """Test configuration validation with GRPC/HTTP mismatch."""
        mock_config.qdrant.prefer_grpc = True
        mock_config.qdrant.url = "http://localhost:6333"

        result = await client.validate_configuration()

        assert any(
            "GRPC preferred but HTTP URL" in rec for rec in result["recommendations"]
        )

    async def test_validate_configuration_connection_failed(self, client, mock_config):
        """Test configuration validation with connection failure."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = Exception("Connection failed")

        client._client = mock_client
        client._initialized = True

        result = await client.validate_configuration()

        assert "connection_failed" in result["connection_status"]

    async def test_validate_configuration_not_initialized(self, client):
        """Test configuration validation when not initialized."""
        result = await client.validate_configuration()

        assert result["connection_status"] == "not_initialized"

    async def test_validate_configuration_error(self, client):
        """Test configuration validation with error."""
        # Mock the config.qdrant.url property to raise an exception
        type(client.config.qdrant).url = PropertyMock(
            side_effect=Exception("Config error")
        )

        with pytest.raises(QdrantServiceError, match="Configuration validation failed"):
            await client.validate_configuration()

    async def test_validate_connection_success(self, client):
        """Test successful connection validation."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = MagicMock()
        client._client = mock_client

        await client._validate_connection()
        # Should not raise exception

    async def test_validate_connection_no_client(self, client):
        """Test connection validation with no client."""
        client._client = None

        with pytest.raises(QdrantServiceError, match="Client not initialized"):
            await client._validate_connection()

    async def test_validate_connection_unauthorized(self, client):
        """Test connection validation with unauthorized error."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = ResponseHandlingException(
            "Unauthorized access"
        )
        client._client = mock_client

        with pytest.raises(QdrantServiceError, match="Unauthorized access to Qdrant"):
            await client._validate_connection()

    async def test_validate_connection_timeout(self, client):
        """Test connection validation with timeout error."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = ResponseHandlingException(
            "Connection timeout"
        )
        client._client = mock_client

        with pytest.raises(QdrantServiceError, match="Connection failed to Qdrant"):
            await client._validate_connection()

    async def test_validate_connection_generic_response_error(self, client):
        """Test connection validation with generic response error."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = ResponseHandlingException(
            "Server error"
        )
        client._client = mock_client

        with pytest.raises(QdrantServiceError, match="Qdrant connection check failed"):
            await client._validate_connection()

    async def test_validate_connection_generic_error(self, client):
        """Test connection validation with generic error."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = ValueError("Generic error")
        client._client = mock_client

        with pytest.raises(QdrantServiceError, match="Qdrant connection check failed"):
            await client._validate_connection()

    async def test_validate_initialized_success(self, client):
        """Test successful initialization validation."""
        mock_client = AsyncMock()
        client._client = mock_client
        client._initialized = True

        client._validate_initialized()
        # Should not raise exception

    async def test_validate_initialized_not_initialized(self, client):
        """Test initialization validation when not initialized."""
        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            client._validate_initialized()

    async def test_validate_initialized_no_client(self, client):
        """Test initialization validation with no client."""
        client._initialized = True
        client._client = None

        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            client._validate_initialized()

    async def test_reconnect_success(self, client):
        """Test successful reconnection."""
        with (
            patch.object(client, "cleanup") as mock_cleanup,
            patch.object(client, "initialize") as mock_initialize,
        ):
            await client.reconnect()

            mock_cleanup.assert_called_once()
            mock_initialize.assert_called_once()

    async def test_reconnect_cleanup_error(self, client):
        """Test reconnection with cleanup error."""
        with patch.object(client, "cleanup", side_effect=Exception("Cleanup failed")):
            with pytest.raises(Exception, match="Cleanup failed"):
                await client.reconnect()

    async def test_reconnect_initialize_error(self, client):
        """Test reconnection with initialization error."""
        with (
            patch.object(client, "cleanup") as mock_cleanup,
            patch.object(
                client, "initialize", side_effect=QdrantServiceError("Init failed")
            ),
        ):
            with pytest.raises(QdrantServiceError, match="Init failed"):
                await client.reconnect()

            mock_cleanup.assert_called_once()

    async def test_test_operation_basic_connectivity(self, client):
        """Test basic connectivity test operation."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()

        # Create proper mock collections with name attribute
        collection1 = MagicMock()
        collection1.name = "collection1"
        collection2 = MagicMock()
        collection2.name = "collection2"

        mock_collections.collections = [collection1, collection2]
        mock_client.get_collections.return_value = mock_collections

        client._client = mock_client
        client._initialized = True

        result = await client.test_operation("basic_connectivity")

        assert result["success"] is True
        assert result["operation"] == "basic_connectivity"
        assert "execution_time_ms" in result
        assert result["result"]["collections_found"] == 2
        assert result["result"]["collections"] == ["collection1", "collection2"]

    async def test_test_operation_unknown_operation(self, client):
        """Test test operation with unknown operation name."""
        client._client = AsyncMock()
        client._initialized = True

        result = await client.test_operation("unknown_operation")

        assert result["success"] is False
        assert result["operation"] == "unknown_operation"
        assert "Unknown test operation" in result["error"]

    async def test_test_operation_not_initialized(self, client):
        """Test test operation when not initialized."""
        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            await client.test_operation()

    async def test_test_operation_error(self, client):
        """Test test operation with error."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = Exception("Test error")

        client._client = mock_client
        client._initialized = True

        result = await client.test_operation("basic_connectivity")

        assert result["success"] is False
        assert result["operation"] == "basic_connectivity"
        assert "Test error" in result["error"]
        assert result["error_type"] == "Exception"

    async def test_test_operation_default_operation(self, client):
        """Test test operation with default operation name."""
        mock_client = AsyncMock()
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections

        client._client = mock_client
        client._initialized = True

        result = await client.test_operation()

        assert result["operation"] == "basic_connectivity"
        assert result["success"] is True

    async def test_inheritance_from_base_service(self, client):
        """Test that QdrantClient inherits from BaseService."""
        from src.services.base import BaseService

        assert isinstance(client, BaseService)

    async def test_config_assignment(self, client, mock_config):
        """Test config is properly assigned."""
        assert client.config is mock_config

    async def test_context_manager_usage(self, mock_config):
        """Test QdrantClient can be used as context manager."""
        client = QdrantClient(mock_config)

        with (
            patch.object(client, "initialize") as mock_init,
            patch.object(client, "cleanup") as mock_cleanup,
        ):
            async with client:
                pass

            mock_init.assert_called_once()
            mock_cleanup.assert_called_once()
