"""Mock components and fixtures for streaming tests."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

# Add src to path for imports
src_path = str(Path(__file__).parent.parent.parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)


class MockStreamingTransport:
    """Mock streaming transport for testing."""

    def __init__(self, transport_type: str = "streamable-http"):
        self.transport_type = transport_type
        self.is_connected = False
        self.buffer_size = 8192
        self.max_response_size = 10485760
        self.sent_chunks = []
        self.total_bytes_sent = 0

    async def connect(self, host: str = "127.0.0.1", port: int = 8000):
        """Mock connection establishment."""
        self.is_connected = True
        self.host = host
        self.port = port
        return True

    async def disconnect(self):
        """Mock disconnection."""
        self.is_connected = False
        self.sent_chunks.clear()
        self.total_bytes_sent = 0

    async def send_chunk(self, data: bytes) -> bool:
        """Mock sending a data chunk."""
        if not self.is_connected:
            raise ConnectionError("Transport not connected")

        if len(data) > self.buffer_size:
            # Simulate chunking for large data
            chunks = [
                data[i : i + self.buffer_size]
                for i in range(0, len(data), self.buffer_size)
            ]
            for chunk in chunks:
                self.sent_chunks.append(chunk)
                self.total_bytes_sent += len(chunk)
                await asyncio.sleep(0.001)  # Simulate transmission delay
        else:
            self.sent_chunks.append(data)
            self.total_bytes_sent += len(data)
            await asyncio.sleep(0.001)

        return True

    async def send_response(self, response_data: Any) -> bool:
        """Mock sending a complete response."""
        if not self.is_connected:
            raise ConnectionError("Transport not connected")

        # Serialize response
        json_data = json.dumps(response_data)
        data_bytes = json_data.encode("utf-8")

        # Check size limits
        if len(data_bytes) > self.max_response_size:
            raise ValueError(
                f"Response size {len(data_bytes)} exceeds limit {self.max_response_size}"
            )

        # Send via chunking
        return await self.send_chunk(data_bytes)

    def get_stats(self) -> dict[str, Any]:
        """Get transmission statistics."""
        return {
            "transport_type": self.transport_type,
            "is_connected": self.is_connected,
            "chunks_sent": len(self.sent_chunks),
            "total_bytes_sent": self.total_bytes_sent,
            "buffer_size": self.buffer_size,
            "max_response_size": self.max_response_size,
        }


class MockFastMCPServer:
    """Mock FastMCP server for testing streaming functionality."""

    def __init__(self, name: str, instructions: str = ""):
        self.name = name
        self.instructions = instructions
        self.transport = None
        self.lifespan = None
        self.tools = {}
        self.is_running = False

    def tool(self):
        """Mock tool decorator."""

        def decorator(func):
            self.tools[func.__name__] = func
            return func

        return decorator

    async def run(
        self,
        transport: str = "stdio",
        host: str = "127.0.0.1",
        port: int = 8000,
        **kwargs,
    ):
        """Mock server run method."""
        self.transport = MockStreamingTransport(transport)
        self.is_running = True

        if transport == "streamable-http":
            await self.transport.connect(host, port)

        # Mock server lifecycle
        if self.lifespan:
            async with self.lifespan():
                await asyncio.sleep(0.1)  # Simulate running
        else:
            await asyncio.sleep(0.1)

        return True

    async def stop(self):
        """Mock server stop method."""
        if self.transport:
            await self.transport.disconnect()
        self.is_running = False

    def get_server_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        stats = {
            "name": self.name,
            "is_running": self.is_running,
            "tool_count": len(self.tools),
            "transport": None,
        }

        if self.transport:
            stats["transport"] = self.transport.get_stats()

        return stats


@pytest.fixture
def mock_fastmcp_server():
    """Fixture providing a mock FastMCP server."""
    return MockFastMCPServer("test-server", "Test streaming server")


@pytest.fixture
def mock_streaming_transport():
    """Fixture providing a mock streaming transport."""
    return MockStreamingTransport("streamable-http")


@pytest.fixture
def large_mock_response():
    """Fixture providing large response data for streaming tests."""
    return [
        {
            "id": f"large_doc_{i}",
            "score": 0.95 - (i * 0.0001),
            "payload": {
                "content": f"Large document content {i} " * 100,  # ~2KB per doc
                "title": f"Large Document {i}",
                "url": f"https://example.com/large/{i}",
                "metadata": {"size": "large", "index": i, "generated": True},
            },
        }
        for i in range(500)  # 500 docs ~= 1MB response
    ]


class TestMockComponents:
    """Test the mock components themselves."""

    @pytest.mark.asyncio
    async def test_mock_streaming_transport(self, mock_streaming_transport):
        """Test mock streaming transport functionality."""
        transport = mock_streaming_transport

        # Test initial state
        assert not transport.is_connected
        assert transport.transport_type == "streamable-http"
        assert len(transport.sent_chunks) == 0

        # Test connection
        await transport.connect("localhost", 9000)
        assert transport.is_connected
        assert transport.host == "localhost"
        assert transport.port == 9000

        # Test sending data
        test_data = b"test data chunk"
        result = await transport.send_chunk(test_data)
        assert result is True
        assert len(transport.sent_chunks) == 1
        assert transport.sent_chunks[0] == test_data
        assert transport.total_bytes_sent == len(test_data)

        # Test disconnection
        await transport.disconnect()
        assert not transport.is_connected
        assert len(transport.sent_chunks) == 0
        assert transport.total_bytes_sent == 0

    @pytest.mark.asyncio
    async def test_mock_fastmcp_server(self, mock_fastmcp_server):
        """Test mock FastMCP server functionality."""
        server = mock_fastmcp_server

        # Test initial state
        assert server.name == "test-server"
        assert not server.is_running
        assert len(server.tools) == 0

        # Test tool registration
        @server.tool()
        async def test_tool():
            return {"result": "test"}

        assert len(server.tools) == 1
        assert "test_tool" in server.tools

        # Test server lifecycle
        run_task = asyncio.create_task(server.run("streamable-http", "localhost", 8080))
        await asyncio.sleep(0.05)  # Let server start

        assert server.is_running
        assert server.transport is not None
        assert server.transport.transport_type == "streamable-http"

        # Wait for server to complete
        await run_task

        # Test stats
        stats = server.get_server_stats()
        assert stats["name"] == "test-server"
        assert stats["tool_count"] == 1
        assert "transport" in stats

    @pytest.mark.asyncio
    async def test_large_response_handling(
        self, mock_streaming_transport, large_mock_response
    ):
        """Test handling of large responses through mock transport."""
        transport = mock_streaming_transport
        await transport.connect()

        # Send large response
        result = await transport.send_response(large_mock_response)
        assert result is True

        # Verify chunking occurred
        assert len(transport.sent_chunks) > 1  # Should be chunked
        assert transport.total_bytes_sent > 100000  # Should be >100KB

        # Verify all data was sent
        all_data = b"".join(transport.sent_chunks)
        original_data = json.dumps(large_mock_response).encode("utf-8")
        assert all_data == original_data

        stats = transport.get_stats()
        assert stats["chunks_sent"] > 1
        assert stats["total_bytes_sent"] == len(original_data)

    @pytest.mark.asyncio
    async def test_response_size_limits(self, mock_streaming_transport):
        """Test response size limit enforcement."""
        transport = mock_streaming_transport
        transport.max_response_size = 1024  # 1KB limit
        await transport.connect()

        # Create response that exceeds limit
        large_response = {"data": "x" * 2000}  # >1KB

        with pytest.raises(ValueError, match="Response size .* exceeds limit"):
            await transport.send_response(large_response)

        # Verify no data was sent
        assert len(transport.sent_chunks) == 0
        assert transport.total_bytes_sent == 0

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_streaming_transport):
        """Test connection error handling."""
        transport = mock_streaming_transport

        # Try to send without connecting
        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send_chunk(b"test data")

        with pytest.raises(ConnectionError, match="Transport not connected"):
            await transport.send_response({"test": "data"})

        # Verify no data was sent
        assert len(transport.sent_chunks) == 0
        assert transport.total_bytes_sent == 0


class TestStreamingIntegrationMocks:
    """Test integration scenarios with mocks."""

    @pytest.mark.asyncio
    async def test_end_to_end_streaming_simulation(
        self, mock_fastmcp_server, large_mock_response
    ):
        """Test end-to-end streaming simulation."""
        server = mock_fastmcp_server

        # Register a tool that returns large response
        @server.tool()
        async def large_search():
            return large_mock_response

        # Start server with streaming
        start_task = asyncio.create_task(
            server.run("streamable-http", "localhost", 8080)
        )
        await asyncio.sleep(0.05)  # Let server start

        # Simulate tool execution
        if server.transport and server.transport.is_connected:
            result = await server.tools["large_search"]()
            await server.transport.send_response(result)

            # Verify streaming occurred
            stats = server.transport.get_stats()
            assert stats["chunks_sent"] > 1
            assert stats["total_bytes_sent"] > 100000

        # Complete server lifecycle
        await start_task

    @pytest.mark.asyncio
    async def test_fallback_to_stdio_simulation(self, mock_fastmcp_server):
        """Test fallback to stdio transport simulation."""
        server = mock_fastmcp_server

        # Test stdio mode (no network transport)
        start_task = asyncio.create_task(server.run("stdio"))
        await asyncio.sleep(0.05)

        # In stdio mode, transport should be different
        assert server.transport.transport_type == "stdio"
        assert not hasattr(server.transport, "host")  # No network config

        await start_task

    @pytest.mark.asyncio
    async def test_concurrent_requests_simulation(self, mock_fastmcp_server):
        """Test concurrent request handling simulation."""
        server = mock_fastmcp_server

        # Register tool for concurrent testing
        request_count = 0

        @server.tool()
        async def concurrent_tool():
            nonlocal request_count
            request_count += 1
            await asyncio.sleep(0.01)  # Simulate processing
            return {"request_id": request_count, "data": "x" * 1000}

        # Start server
        start_task = asyncio.create_task(server.run("streamable-http"))
        await asyncio.sleep(0.05)

        # Simulate concurrent requests
        if server.transport and server.transport.is_connected:
            tasks = []
            for _ in range(5):

                async def request():
                    result = await server.tools["concurrent_tool"]()
                    await server.transport.send_response(result)

                tasks.append(asyncio.create_task(request()))

            # Wait for all requests to complete
            await asyncio.gather(*tasks)

            # Verify all requests were processed
            assert request_count == 5

            # Verify transport handled multiple responses
            stats = server.transport.get_stats()
            assert stats["chunks_sent"] >= 5  # At least one chunk per request

        await start_task

    @pytest.mark.asyncio
    async def test_error_recovery_simulation(self, mock_fastmcp_server):
        """Test error recovery scenarios."""
        server = mock_fastmcp_server

        # Register tool that can simulate errors
        @server.tool()
        async def error_prone_tool(should_error: bool = False):
            if should_error:
                raise ValueError("Simulated error")
            return {"status": "success"}

        # Start server
        start_task = asyncio.create_task(server.run("streamable-http"))
        await asyncio.sleep(0.05)

        if server.transport and server.transport.is_connected:
            # Test successful request
            result = await server.tools["error_prone_tool"](False)
            await server.transport.send_response(result)

            # Test error handling
            try:
                await server.tools["error_prone_tool"](True)
            except ValueError:
                # Error should be caught and handled gracefully
                error_response = {"error": "Tool execution failed"}
                await server.transport.send_response(error_response)

            # Verify transport continued working after error
            stats = server.transport.get_stats()
            assert stats["chunks_sent"] >= 2  # Both responses sent

        await start_task


# Additional test fixtures for comprehensive testing
@pytest.fixture
def streaming_environment_config():
    """Fixture providing streaming environment configuration."""
    return {
        "FASTMCP_TRANSPORT": "streamable-http",
        "FASTMCP_HOST": "127.0.0.1",
        "FASTMCP_PORT": "8000",
        "FASTMCP_BUFFER_SIZE": "8192",
        "FASTMCP_MAX_RESPONSE_SIZE": "10485760",
    }


@pytest.fixture
def performance_test_data():
    """Fixture providing data for performance testing."""
    return {
        "small_response": [{"id": f"small_{i}", "data": "x" * 100} for i in range(10)],
        "medium_response": [
            {"id": f"medium_{i}", "data": "x" * 1000} for i in range(100)
        ],
        "large_response": [
            {"id": f"large_{i}", "data": "x" * 5000} for i in range(500)
        ],
    }


@pytest.fixture
def mock_search_results():
    """Fixture providing mock search results for various test scenarios."""

    def generate_results(count: int, content_size: int = 1000):
        return [
            {
                "id": f"search_result_{i}",
                "score": 0.9 - (i * 0.001),
                "payload": {
                    "content": "Search result content " * (content_size // 20),
                    "title": f"Search Result {i}",
                    "url": f"https://example.com/search/{i}",
                    "metadata": {"index": i, "generated": True},
                },
            }
            for i in range(count)
        ]

    return generate_results
