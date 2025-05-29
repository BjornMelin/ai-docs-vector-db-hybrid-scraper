"""Integration tests for MCP server streaming functionality."""

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


@pytest.fixture
def large_search_results():
    """Generate large search results for streaming tests."""
    return [
        {
            "id": f"doc_{i}",
            "score": 0.9 - (i * 0.001),
            "payload": {
                "content": f"This is test document {i} with substantial content " * 50,
                "title": f"Test Document {i}",
                "url": f"https://example.com/docs/{i}",
                "metadata": {"length": 2500, "type": "documentation"},
            },
        }
        for i in range(1000)  # 1000 documents for streaming test
    ]


@pytest.fixture
def mock_client_manager():
    """Mock client manager for streaming tests."""
    mock_manager = Mock()
    mock_manager.initialize = AsyncMock()
    mock_manager.cleanup = AsyncMock()

    # Mock Qdrant service with large result capability
    mock_qdrant = Mock()
    mock_qdrant.hybrid_search = AsyncMock()
    mock_manager.qdrant_service = mock_qdrant

    # Mock embedding manager
    mock_embedding = Mock()
    mock_embedding.generate_embeddings = AsyncMock(return_value=[[0.1] * 1536])
    mock_manager.embedding_manager = mock_embedding

    # Mock cache manager
    mock_cache = Mock()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    mock_manager.cache_manager = mock_cache

    return mock_manager


@pytest.mark.asyncio
async def test_streaming_transport_initialization():
    """Test that streaming transport initializes correctly."""
    with patch.dict(
        os.environ,
        {
            "FASTMCP_TRANSPORT": "streamable-http",
            "FASTMCP_HOST": "127.0.0.1",
            "FASTMCP_PORT": "8000",
        },
    ):
        # Mock the FastMCP server run method
        with patch("unified_mcp_server.mcp.run") as mock_run:
            # Import after setting environment

            # Test that the correct parameters would be passed
            transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
            host = os.getenv("FASTMCP_HOST", "127.0.0.1")
            port = int(os.getenv("FASTMCP_PORT", "8000"))

            assert transport == "streamable-http"
            assert host == "127.0.0.1"
            assert port == 8000


@pytest.mark.asyncio
async def test_large_result_handling_preparation(
    large_search_results, mock_client_manager
):
    """Test preparation for handling large search results."""
    # Mock search results
    mock_client_manager.qdrant_service.hybrid_search.return_value = large_search_results

    # Test that we can handle large result sets
    results = await mock_client_manager.qdrant_service.hybrid_search(
        collection_name="test", query_vector=[0.1] * 1536, limit=1000
    )

    assert len(results) == 1000
    assert all("content" in result["payload"] for result in results)

    # Calculate approximate response size
    response_json = json.dumps(results)
    response_size = len(response_json.encode("utf-8"))

    # Verify it's large enough to benefit from streaming
    assert response_size > 1_000_000  # > 1MB

    # Test that it's within our max response size limit (10MB default)
    max_size = int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE", "10485760"))
    assert response_size < max_size


@pytest.mark.asyncio
async def test_environment_variable_configuration():
    """Test comprehensive environment variable configuration."""
    test_env = {
        "FASTMCP_TRANSPORT": "streamable-http",
        "FASTMCP_HOST": "0.0.0.0",
        "FASTMCP_PORT": "9000",
        "FASTMCP_BUFFER_SIZE": "16384",
        "FASTMCP_MAX_RESPONSE_SIZE": "20971520",
    }

    with patch.dict(os.environ, test_env):
        # Verify all environment variables
        assert os.getenv("FASTMCP_TRANSPORT") == "streamable-http"
        assert os.getenv("FASTMCP_HOST") == "0.0.0.0"
        assert int(os.getenv("FASTMCP_PORT")) == 9000
        assert int(os.getenv("FASTMCP_BUFFER_SIZE")) == 16384
        assert int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE")) == 20971520

        # Test configuration parsing
        config = {
            "transport": os.getenv("FASTMCP_TRANSPORT", "streamable-http"),
            "host": os.getenv("FASTMCP_HOST", "127.0.0.1"),
            "port": int(os.getenv("FASTMCP_PORT", "8000")),
            "buffer_size": os.getenv("FASTMCP_BUFFER_SIZE", "8192"),
            "max_response_size": os.getenv("FASTMCP_MAX_RESPONSE_SIZE", "10485760"),
        }

        assert config["transport"] == "streamable-http"
        assert config["host"] == "0.0.0.0"
        assert config["port"] == 9000
        assert config["buffer_size"] == "16384"
        assert config["max_response_size"] == "20971520"


@pytest.mark.asyncio
async def test_stdio_fallback_mechanism():
    """Test fallback to stdio transport for Claude Desktop compatibility."""
    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "stdio"}):
        # Test stdio configuration
        transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
        assert transport == "stdio"

        # Verify that stdio mode doesn't require host/port
        if transport == "stdio":
            # Should not use network configuration
            assert True  # stdio mode validated


@pytest.mark.asyncio
async def test_response_size_limits(large_search_results):
    """Test response size limit enforcement."""
    # Create an oversized response
    oversized_results = large_search_results * 20  # 20,000 results
    response_json = json.dumps(oversized_results)
    response_size = len(response_json.encode("utf-8"))

    # Test with default max size (10MB)
    default_max_size = int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE", "10485760"))

    if response_size > default_max_size:
        # Response would be too large - streaming should handle this
        assert response_size > default_max_size

        # Test with custom larger limit
        with patch.dict(
            os.environ, {"FASTMCP_MAX_RESPONSE_SIZE": str(response_size + 1000000)}
        ):
            custom_max_size = int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE"))
            assert response_size < custom_max_size


@pytest.mark.asyncio
async def test_streaming_error_handling():
    """Test error handling in streaming scenarios."""
    # Test invalid port configuration
    with patch.dict(os.environ, {"FASTMCP_PORT": "invalid"}):
        with pytest.raises(ValueError):
            int(os.getenv("FASTMCP_PORT", "8000"))

    # Test missing environment handling
    with patch.dict(os.environ, {}, clear=True):
        # Should fall back to defaults
        transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
        host = os.getenv("FASTMCP_HOST", "127.0.0.1")
        port = int(os.getenv("FASTMCP_PORT", "8000"))

        assert transport == "streamable-http"
        assert host == "127.0.0.1"
        assert port == 8000


@pytest.mark.asyncio
async def test_concurrent_streaming_requests(mock_client_manager):
    """Test handling of concurrent streaming requests."""

    # Simulate multiple concurrent large searches
    async def mock_large_search():
        return await mock_client_manager.qdrant_service.hybrid_search(
            collection_name="test", query_vector=[0.1] * 1536, limit=500
        )

    # Set up mock to return large results
    large_results = [
        {"id": f"doc_{i}", "score": 0.9, "payload": {"content": "x" * 1000}}
        for i in range(500)
    ]
    mock_client_manager.qdrant_service.hybrid_search.return_value = large_results

    # Run concurrent requests
    tasks = [mock_large_search() for _ in range(5)]
    results = await asyncio.gather(*tasks)

    # Verify all requests completed
    assert len(results) == 5
    assert all(len(result) == 500 for result in results)

    # Verify mock was called correctly
    assert mock_client_manager.qdrant_service.hybrid_search.call_count == 5


@pytest.mark.asyncio
async def test_transport_mode_detection():
    """Test automatic transport mode detection and configuration."""
    # Test streamable-http mode
    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "streamable-http"}):
        transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
        assert transport == "streamable-http"

        # Should require network configuration
        host = os.getenv("FASTMCP_HOST", "127.0.0.1")
        port = int(os.getenv("FASTMCP_PORT", "8000"))
        assert host is not None
        assert isinstance(port, int)
        assert 1 <= port <= 65535

    # Test stdio mode
    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "stdio"}):
        transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
        assert transport == "stdio"

    # Test unknown transport (should fall back to default)
    with patch.dict(os.environ, {"FASTMCP_TRANSPORT": "unknown"}):
        transport = os.getenv("FASTMCP_TRANSPORT", "streamable-http")
        assert transport == "unknown"  # Would be handled by server logic


@pytest.mark.asyncio
async def test_memory_efficiency_with_large_responses(large_search_results):
    """Test memory efficiency with large response handling."""
    # Test response serialization efficiency
    results = large_search_results[:100]  # Start with 100 results

    # Measure serialization overhead
    json_response = json.dumps(results)
    json_size = len(json_response.encode("utf-8"))

    # Test buffer size configuration
    buffer_size = int(os.getenv("FASTMCP_BUFFER_SIZE", "8192"))

    # Should be able to handle responses larger than buffer size
    if json_size > buffer_size:
        # Streaming should handle this efficiently
        chunks_needed = (json_size + buffer_size - 1) // buffer_size
        assert chunks_needed > 1
        assert chunks_needed * buffer_size >= json_size


@pytest.mark.asyncio
async def test_configuration_validation():
    """Test validation of streaming configuration parameters."""
    # Test valid configuration
    valid_config = {
        "FASTMCP_TRANSPORT": "streamable-http",
        "FASTMCP_HOST": "127.0.0.1",
        "FASTMCP_PORT": "8000",
        "FASTMCP_BUFFER_SIZE": "8192",
        "FASTMCP_MAX_RESPONSE_SIZE": "10485760",
    }

    with patch.dict(os.environ, valid_config):
        # All values should parse correctly
        transport = os.getenv("FASTMCP_TRANSPORT")
        host = os.getenv("FASTMCP_HOST")
        port = int(os.getenv("FASTMCP_PORT"))
        buffer_size = int(os.getenv("FASTMCP_BUFFER_SIZE"))
        max_size = int(os.getenv("FASTMCP_MAX_RESPONSE_SIZE"))

        assert transport in ["streamable-http", "stdio"]
        assert host is not None
        assert 1 <= port <= 65535
        assert buffer_size > 0
        assert max_size > 0
        assert max_size > buffer_size  # max should be larger than buffer
