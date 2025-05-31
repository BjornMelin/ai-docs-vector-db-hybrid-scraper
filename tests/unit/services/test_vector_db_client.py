"""Tests for Qdrant client management service."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException
from src.config import UnifiedConfig
from src.services.errors import QdrantServiceError
from src.services.vector_db.client import QdrantClient


class TestQdrantClient:
    """Test QdrantClient service."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return UnifiedConfig()
    
    @pytest.fixture
    def client_service(self, config):
        """Create QdrantClient instance."""
        return QdrantClient(config)
    
    async def test_initialization_success(self, client_service):
        """Test successful client initialization."""
        with patch("src.services.vector_db.client.AsyncQdrantClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client_class.return_value = mock_client
            
            await client_service.initialize()
            
            assert client_service._initialized is True
            assert client_service._client is not None
            mock_client_class.assert_called_once()
            mock_client.get_collections.assert_called_once()
    
    async def test_initialization_already_initialized(self, client_service):
        """Test initialization when already initialized."""
        client_service._initialized = True
        
        with patch("src.services.vector_db.client.AsyncQdrantClient") as mock_client_class:
            await client_service.initialize()
            
            # Should not create new client
            mock_client_class.assert_not_called()
    
    async def test_initialization_connection_failure(self, client_service):
        """Test initialization with connection failure."""
        with patch("src.services.vector_db.client.AsyncQdrantClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_collections.side_effect = ResponseHandlingException("Connection failed")
            mock_client_class.return_value = mock_client
            
            with pytest.raises(QdrantServiceError, match="Failed to initialize Qdrant client"):
                await client_service.initialize()
            
            assert client_service._initialized is False
    
    async def test_initialization_generic_error(self, client_service):
        """Test initialization with generic error."""
        with patch("src.services.vector_db.client.AsyncQdrantClient") as mock_client_class:
            mock_client_class.side_effect = Exception("Unexpected error")
            
            with pytest.raises(QdrantServiceError, match="Failed to initialize Qdrant client"):
                await client_service.initialize()
            
            assert client_service._initialized is False
    
    async def test_cleanup_success(self, client_service):
        """Test successful cleanup."""
        # Setup initialized client
        mock_client = AsyncMock()
        client_service._client = mock_client
        client_service._initialized = True
        
        await client_service.cleanup()
        
        assert client_service._initialized is False
        assert client_service._client is None
        mock_client.close.assert_called_once()
    
    async def test_cleanup_no_client(self, client_service):
        """Test cleanup when no client exists."""
        client_service._client = None
        client_service._initialized = False
        
        await client_service.cleanup()
        
        assert client_service._initialized is False
        assert client_service._client is None
    
    async def test_cleanup_client_close_error(self, client_service):
        """Test cleanup when client.close() raises error."""
        mock_client = AsyncMock()
        mock_client.close.side_effect = Exception("Close failed")
        client_service._client = mock_client
        client_service._initialized = True
        
        # Should not raise error, just log
        await client_service.cleanup()
        
        assert client_service._initialized is False
        assert client_service._client is None
    
    async def test_get_client_initialized(self, client_service):
        """Test get_client when initialized."""
        mock_client = AsyncMock()
        client_service._client = mock_client
        client_service._initialized = True
        
        result = await client_service.get_client()
        
        assert result is mock_client
    
    async def test_get_client_not_initialized(self, client_service):
        """Test get_client when not initialized."""
        client_service._client = None
        client_service._initialized = False
        
        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            await client_service.get_client()
    
    async def test_validate_configuration_success(self, client_service):
        """Test successful configuration validation."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = Mock(collections=[])
        client_service._client = mock_client
        client_service._initialized = True
        
        result = await client_service.validate_configuration()
        
        assert "connection_status" in result
        assert result["connection_status"] == "connected"
        assert "configuration" in result
        mock_client.get_collections.assert_called_once()
    
    async def test_validate_configuration_not_initialized(self, client_service):
        """Test configuration validation when not initialized."""
        client_service._initialized = False
        
        # validate_configuration doesn't check initialization - it just validates config
        result = await client_service.validate_configuration()
        
        assert "configuration" in result
    
    async def test_validate_configuration_failure(self, client_service):
        """Test configuration validation with connection failure."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = ResponseHandlingException("Connection failed")
        client_service._client = mock_client
        client_service._initialized = True
        
        result = await client_service.validate_configuration()
        
        assert "connection_status" in result
        # Should not raise error, but connection status should reflect failure
    
    async def test_health_check_success(self, client_service):
        """Test successful health check."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = Mock(collections=[])
        client_service._client = mock_client
        client_service._initialized = True
        
        result = await client_service.health_check()
        
        assert "status" in result
        assert result["status"] == "healthy"
        assert "response_time_ms" in result
        assert "collections_count" in result
        assert result["collections_count"] == 0
    
    async def test_health_check_not_initialized(self, client_service):
        """Test health check when not initialized."""
        client_service._initialized = False
        
        with pytest.raises(QdrantServiceError, match="Qdrant client not initialized"):
            await client_service.health_check()
    
    async def test_health_check_connection_error(self, client_service):
        """Test health check with connection error."""
        mock_client = AsyncMock()
        mock_client.get_collections.side_effect = Exception("Connection error")
        client_service._client = mock_client
        client_service._initialized = True
        
        with pytest.raises(QdrantServiceError):
            await client_service.health_check()
    
    async def test_reconnect_success(self, client_service):
        """Test successful reconnection."""
        with patch("src.services.vector_db.client.AsyncQdrantClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get_collections.return_value = Mock(collections=[])
            mock_client_class.return_value = mock_client
            
            await client_service.reconnect()
            
            assert client_service._initialized is True
            assert client_service._client is not None
    
    async def test_test_operation_success(self, client_service):
        """Test successful test operation."""
        mock_client = AsyncMock()
        mock_client.get_collections.return_value = Mock(collections=[])
        client_service._client = mock_client
        client_service._initialized = True
        
        result = await client_service.test_operation()
        
        assert result["success"] is True
        assert "execution_time_ms" in result
        assert "operation" in result
        mock_client.get_collections.assert_called_once()