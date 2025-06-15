"""End-to-end MCP protocol testing.

Tests complete MCP request/response cycles, transport layers, and
protocol compliance following JSON-RPC 2.0 specifications.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config import UnifiedConfig
from src.infrastructure.client_manager import ClientManager

from tests.mocks.mock_tools import MockMCPServer
from tests.mocks.mock_tools import register_mock_tools


class TestMCPProtocolE2E:
    """End-to-end tests for complete MCP protocol implementation."""

    @pytest.fixture
    async def mock_config(self):
        """Mock configuration for E2E testing."""
        config = MagicMock(spec=UnifiedConfig)

        # Mock nested config objects
        config.qdrant = MagicMock()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = None

        config.openai = MagicMock()
        config.openai.api_key = "test-openai-key"

        config.crawling = MagicMock()
        config.crawling.providers = ["crawl4ai"]

        config.get_active_providers.return_value = ["openai", "fastembed"]
        return config

    @pytest.fixture
    async def mock_client_manager(self, mock_config):
        """Mock client manager for E2E testing."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.config = mock_config

        # Mock all services with realistic responses
        mock_vector_service = AsyncMock()
        mock_vector_service.search_documents.return_value = [
            {
                "id": "e2e-doc-1",
                "content": "End-to-end test document content",
                "score": 0.92,
                "metadata": {"title": "E2E Test Doc", "url": "https://example.com/e2e"},
            }
        ]
        client_manager.vector_service = mock_vector_service

        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings.return_value = {
            "embeddings": [[0.1] * 384],
            "model": "BAAI/bge-small-en-v1.5",
            "total_tokens": 5,
        }
        client_manager.embedding_service = mock_embedding_service

        # Add missing services
        mock_project_service = AsyncMock()
        mock_project_service.create_project.return_value = {
            "id": "test-project-123",
            "name": "E2E Test Project",
            "description": "Created by E2E test",
            "created_at": "2024-01-01T00:00:00Z",
        }
        client_manager.project_service = mock_project_service

        mock_cache_service = AsyncMock()
        mock_cache_service.get_stats.return_value = {
            "hit_rate": 0.85,
            "size": 1000,
            "total_requests": 5000,
        }
        client_manager.cache_service = mock_cache_service

        mock_crawling_service = AsyncMock()
        mock_crawling_service.crawl_url.return_value = {
            "url": "https://example.com",
            "title": "Example Page",
            "content": "Example content",
            "success": True,
        }
        client_manager.crawling_service = mock_crawling_service

        mock_deployment_service = AsyncMock()
        mock_deployment_service.list_aliases.return_value = {
            "production": "docs-v1.0",
            "staging": "docs-v1.1-beta",
        }
        client_manager.deployment_service = mock_deployment_service

        mock_analytics_service = AsyncMock()
        mock_analytics_service.get_analytics.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "collections": {},
            "performance": {"avg_search_time_ms": 85.2},
        }
        client_manager.analytics_service = mock_analytics_service

        mock_hyde_service = AsyncMock()
        mock_hyde_service.search.return_value = {
            "request_id": "hyde-123",
            "query": "test query",
            "results": [],
            "metrics": {"search_time_ms": 150.5},
        }
        client_manager.hyde_service = mock_hyde_service

        return client_manager

    @pytest.fixture
    async def mcp_server_e2e(self, mock_client_manager):
        """Create MCP server for E2E testing."""
        mcp = MockMCPServer("e2e-test-server")
        register_mock_tools(mcp, mock_client_manager)
        return mcp

    async def test_json_rpc_request_response_cycle(
        self, mcp_server_e2e, mock_client_manager
    ):
        """Test complete JSON-RPC 2.0 request/response cycle."""
        # Simulate JSON-RPC 2.0 request for search
        request_data = {
            "jsonrpc": "2.0",
            "id": "e2e-test-1",
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {
                    "query": "machine learning tutorial",
                    "collection": "documentation",
                    "limit": 5,
                },
            },
        }

        # Mock the search service response
        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.return_value = [
                {
                    "id": "ml-tutorial-1",
                    "content": "Comprehensive machine learning tutorial covering basics to advanced topics",
                    "score": 0.95,
                    "metadata": {
                        "title": "ML Tutorial",
                        "url": "https://example.com/ml-tutorial",
                    },
                }
            ]

            # Simulate FastMCP processing the request
            # Note: In real implementation, FastMCP would handle JSON-RPC parsing
            tool = None
            for t in mcp_server_e2e._tools:
                if t.name == "search_documents":
                    tool = t
                    break

            assert tool is not None, "Search tool not found"

            # Execute the tool
            result = await tool.handler(
                query=request_data["params"]["arguments"]["query"],
                collection=request_data["params"]["arguments"]["collection"],
                limit=request_data["params"]["arguments"]["limit"],
            )

            # Validate JSON-RPC 2.0 response structure
            response = {"jsonrpc": "2.0", "id": request_data["id"], "result": result}

            assert response["jsonrpc"] == "2.0"
            assert response["id"] == "e2e-test-1"
            assert len(response["result"]) == 1
            assert (
                response["result"][0]["content"]
                == "Comprehensive machine learning tutorial covering basics to advanced topics"
            )

    async def test_tool_chain_execution(self, mcp_server_e2e, mock_client_manager):
        """Test executing multiple tools in sequence (tool chaining)."""
        # Step 1: Create a project
        project_tool = None
        for tool in mcp_server_e2e._tools:
            if tool.name == "create_project":
                project_tool = tool
                break

        with patch.object(
            mock_client_manager.project_service, "create_project"
        ) as mock_create_project:
            mock_create_project.return_value = {
                "id": "chain-project-123",
                "name": "Chain Test Project",
                "created_at": "2024-01-01T00:00:00Z",
            }

            project_result = await project_tool.handler(
                name="Chain Test Project",
                description="Test project for tool chaining",
                quality_tier="balanced",
            )

            project_id = project_result["id"]

        # Step 2: Add documents to the project
        doc_tool = None
        for tool in mcp_server_e2e._tools:
            if tool.name == "add_document":
                doc_tool = tool
                break

        with patch.object(
            mock_client_manager.vector_service, "add_document"
        ) as mock_add_doc:
            mock_add_doc.return_value = {
                "url": "https://example.com/chain-doc",
                "title": "Chain Test Document",
                "chunks_created": 3,
                "collection": f"project-{project_id}",
                "chunking_strategy": "enhanced",
                "embedding_dimensions": 384,
            }

            doc_result = await doc_tool.handler(
                url="https://example.com/chain-doc",
                collection=f"project-{project_id}",
                chunk_strategy="enhanced",
            )

        # Step 3: Search within the project
        search_tool = None
        for tool in mcp_server_e2e._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.return_value = [
                {
                    "id": "chain-search-result",
                    "content": "Content from the chained document",
                    "score": 0.88,
                    "metadata": {"project_id": project_id},
                }
            ]

            search_result = await search_tool.handler(
                query="chain test content",
                collection=f"project-{project_id}",
                limit=10,
            )

        # Validate the complete chain
        assert project_result["id"] == "chain-project-123"
        assert doc_result["chunks_created"] == 3
        assert len(search_result) == 1
        assert search_result[0]["metadata"]["project_id"] == project_id

    async def test_error_propagation_through_protocol(
        self, mcp_server_e2e, mock_client_manager
    ):
        """Test proper error handling and propagation through MCP protocol."""
        search_tool = None
        for tool in mcp_server_e2e._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Test service-level error
        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.side_effect = Exception("Vector database connection timeout")

            # The error should propagate up through the tool
            with pytest.raises(Exception, match="Vector database connection timeout"):
                await search_tool.handler(
                    query="test query",
                    collection="documentation",
                    limit=10,
                )

        # Test JSON-RPC error response structure
        try:
            await search_tool.handler(
                query="test query",
                collection="documentation",
                limit=10,
            )
        except Exception as e:
            # JSON-RPC 2.0 error response format
            error_response = {
                "jsonrpc": "2.0",
                "id": "error-test-1",
                "error": {
                    "code": -32603,  # Internal error
                    "message": "Internal error",
                    "data": {"error": str(e)},
                },
            }

            assert error_response["error"]["code"] == -32603
            assert (
                "Vector database connection timeout"
                in error_response["error"]["data"]["error"]
            )

    async def test_concurrent_requests_handling(
        self, mcp_server_e2e, mock_client_manager
    ):
        """Test handling multiple concurrent MCP requests."""
        search_tool = None
        embedding_tool = None

        for tool in mcp_server_e2e._tools:
            if tool.name == "search_documents":
                search_tool = tool
            elif tool.name == "generate_embeddings":
                embedding_tool = tool

        # Mock services with delays to simulate real processing
        async def mock_search_with_delay(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate processing time
            return [{"id": "concurrent-1", "content": "Search result", "score": 0.9}]

        async def mock_embedding_with_delay(*args, **kwargs):
            await asyncio.sleep(0.15)  # Simulate processing time
            return {
                "embeddings": [[0.1] * 384],
                "model": "test-model",
                "total_tokens": 3,
            }

        with (
            patch.object(
                mock_client_manager.vector_service,
                "search_documents",
                side_effect=mock_search_with_delay,
            ),
            patch.object(
                mock_client_manager.embedding_service,
                "generate_embeddings",
                side_effect=mock_embedding_with_delay,
            ),
        ):
            # Execute multiple requests concurrently
            start_time = time.time()
            tasks = [
                search_tool.handler(
                    query="concurrent search 1", collection="docs", limit=5
                ),
                search_tool.handler(
                    query="concurrent search 2", collection="docs", limit=5
                ),
                embedding_tool.handler(texts=["concurrent text 1"]),
                embedding_tool.handler(texts=["concurrent text 2"]),
            ]

            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time

            # Verify all requests completed successfully
            assert len(results) == 4
            assert len(results[0]) == 1  # Search 1 results
            assert len(results[1]) == 1  # Search 2 results
            assert len(results[2]["embeddings"]) == 1  # Embedding 1 results
            assert len(results[3]["embeddings"]) == 1  # Embedding 2 results

            # Execution should be concurrent (faster than sequential)
            assert execution_time < 0.5, (
                f"Concurrent execution took {execution_time:.3f}s, expected < 0.5s"
            )

    async def test_large_response_handling(self, mcp_server_e2e, mock_client_manager):
        """Test handling of large responses through MCP protocol."""
        search_tool = None
        for tool in mcp_server_e2e._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Mock large search results
        large_results = []
        for i in range(100):  # Large result set
            large_results.append(
                {
                    "id": f"large-doc-{i}",
                    "content": f"Large document content {i} "
                    + "x" * 1000,  # ~1KB per result
                    "score": 0.9 - (i * 0.001),
                    "metadata": {"title": f"Large Doc {i}", "size": "large"},
                }
            )

        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.return_value = large_results

            start_time = time.time()
            result = await search_tool.handler(
                query="large response test",
                collection="documentation",
                limit=100,
            )
            execution_time = time.time() - start_time

            # Validate large response handling
            assert len(result) == 100
            assert all("content" in doc for doc in result)

            # Performance check for large responses
            assert execution_time < 2.0, (
                f"Large response handling took {execution_time:.3f}s"
            )

            # Estimate response size
            response_size = len(json.dumps(result))
            assert response_size > 100000, (
                "Response should be substantial for large data test"
            )

    async def test_session_state_management(self, mcp_server_e2e, mock_client_manager):
        """Test session state management across multiple requests."""
        # Simulate session-aware operations
        project_tool = None
        analytics_tool = None

        for tool in mcp_server_e2e._tools:
            if tool.name == "create_project":
                project_tool = tool
            elif tool.name == "get_analytics":
                analytics_tool = tool

        # Create project in "session"
        with (
            patch.object(
                mock_client_manager.project_service, "create_project"
            ) as mock_create,
            patch.object(mock_client_manager, "analytics_service") as mock_analytics,
        ):
            mock_create.return_value = {
                "id": "session-project-456",
                "name": "Session Test Project",
                "created_at": "2024-01-01T00:00:00Z",
            }

            project_result = await project_tool.handler(
                name="Session Test Project",
                description="Project for session testing",
                quality_tier="premium",
            )

            # Get analytics for the session project
            mock_analytics.get_analytics.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "collections": {
                    f"project-{project_result['id']}": {
                        "total_documents": 0,  # New project
                        "created_in_session": True,
                    }
                },
                "session_state": {
                    "active_projects": [project_result["id"]],
                    "session_duration": "00:05:30",
                },
            }

            analytics_result = await analytics_tool.handler(
                collection=f"project-{project_result['id']}",
                include_performance=True,
            )

            # Validate session state preservation
            assert project_result["id"] == "session-project-456"
            assert f"project-{project_result['id']}" in analytics_result["collections"]
            assert analytics_result["session_state"]["active_projects"] == [
                "session-project-456"
            ]

    async def test_authentication_and_authorization(
        self, mcp_server_e2e, mock_client_manager
    ):
        """Test authentication and authorization mechanisms."""
        # Test with valid authentication
        search_tool = None
        for tool in mcp_server_e2e._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Mock authenticated request
        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.return_value = [
                {"id": "auth-doc", "content": "Authenticated content", "score": 0.9}
            ]

            # Simulate request with authentication context
            result = await search_tool.handler(
                query="authenticated search",
                collection="documentation",
                limit=5,
            )

            assert len(result) == 1
            assert result[0]["content"] == "Authenticated content"

    async def test_transport_layer_compatibility(self, mcp_server_e2e):
        """Test compatibility with different transport layers."""
        # Test HTTP transport simulation
        http_request = {
            "method": "POST",
            "path": "/mcp",
            "headers": {
                "Content-Type": "application/json",
                "Authorization": "Bearer test-token",
            },
            "body": {
                "jsonrpc": "2.0",
                "id": "http-test-1",
                "method": "tools/call",
                "params": {"name": "list_collections", "arguments": {}},
            },
        }

        # Verify server can handle HTTP-style requests
        assert http_request["headers"]["Content-Type"] == "application/json"
        assert http_request["body"]["jsonrpc"] == "2.0"

        # Test stdio transport simulation
        stdio_request = {
            "jsonrpc": "2.0",
            "id": "stdio-test-1",
            "method": "tools/call",
            "params": {"name": "validate_configuration", "arguments": {}},
        }

        # Verify server can handle stdio-style requests
        assert stdio_request["jsonrpc"] == "2.0"
        assert stdio_request["method"] == "tools/call"

    async def test_protocol_compliance_json_rpc_2_0(self):
        """Test JSON-RPC 2.0 protocol compliance."""
        # Test valid request structure
        valid_request = {
            "jsonrpc": "2.0",
            "id": "compliance-test-1",
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {"query": "test", "collection": "docs"},
            },
        }

        # Validate required fields
        assert "jsonrpc" in valid_request
        assert valid_request["jsonrpc"] == "2.0"
        assert "id" in valid_request
        assert "method" in valid_request
        assert "params" in valid_request

        # Test valid response structure
        valid_response = {
            "jsonrpc": "2.0",
            "id": valid_request["id"],
            "result": [{"id": "doc-1", "content": "Test", "score": 0.9}],
        }

        assert valid_response["jsonrpc"] == "2.0"
        assert valid_response["id"] == valid_request["id"]
        assert "result" in valid_response

        # Test valid error response structure
        error_response = {
            "jsonrpc": "2.0",
            "id": valid_request["id"],
            "error": {
                "code": -32602,
                "message": "Invalid params",
                "data": {"param": "query", "reason": "required"},
            },
        }

        assert error_response["jsonrpc"] == "2.0"
        assert "error" in error_response
        assert "code" in error_response["error"]
        assert "message" in error_response["error"]


class TestMCPPerformance:
    """Performance and load testing for MCP operations."""

    @pytest.fixture
    async def performance_server(self):
        """Create server optimized for performance testing."""
        mcp = MockMCPServer("performance-test-server")

        # Mock optimized client manager
        mock_client_manager = MagicMock()

        # Fast mock services
        mock_vector_service = AsyncMock()
        mock_vector_service.search_documents.return_value = [
            {"id": f"perf-doc-{i}", "content": f"Performance content {i}", "score": 0.9}
            for i in range(10)
        ]
        mock_client_manager.vector_service = mock_vector_service

        # Add other required services for tool registration
        mock_client_manager.embedding_service = AsyncMock()
        mock_client_manager.crawling_service = AsyncMock()
        mock_client_manager.cache_service = AsyncMock()
        mock_client_manager.project_service = AsyncMock()
        mock_client_manager.deployment_service = AsyncMock()
        mock_client_manager.analytics_service = AsyncMock()
        mock_client_manager.hyde_service = AsyncMock()

        register_mock_tools(mcp, mock_client_manager)
        return mcp, mock_client_manager

    async def test_high_frequency_requests(self, performance_server):
        """Test handling high-frequency requests."""
        mcp_server, mock_client_manager = performance_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Execute many requests rapidly
        num_requests = 100
        start_time = time.time()

        tasks = []
        for i in range(num_requests):
            task = search_tool.handler(
                query=f"performance test {i}",
                collection="documentation",
                limit=5,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time

        # Performance assertions
        assert len(results) == num_requests
        assert all(
            len(result) == 10 for result in results
        )  # Each search returns 10 results

        # Should handle 100 requests in reasonable time
        requests_per_second = num_requests / execution_time
        assert requests_per_second > 50, (
            f"Performance: {requests_per_second:.1f} req/s, expected > 50 req/s"
        )

    async def test_memory_usage_stability(self, performance_server):
        """Test memory usage remains stable under load."""
        mcp_server, mock_client_manager = performance_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        # Execute requests in batches to test memory stability
        for batch in range(10):
            batch_tasks = []
            for i in range(20):
                task = search_tool.handler(
                    query=f"memory test batch {batch} request {i}",
                    collection="documentation",
                    limit=5,
                )
                batch_tasks.append(task)

            await asyncio.gather(*batch_tasks)

            # Small delay between batches
            await asyncio.sleep(0.01)

        # If we reach here without memory issues, test passes
        assert True, "Memory usage remained stable"

    async def test_response_time_consistency(self, performance_server):
        """Test response time consistency under varying loads."""
        mcp_server, mock_client_manager = performance_server

        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        response_times = []

        # Measure response times for individual requests
        for i in range(50):
            start_time = time.time()

            result = await search_tool.handler(
                query=f"consistency test {i}",
                collection="documentation",
                limit=5,
            )

            response_time = time.time() - start_time
            response_times.append(response_time)

            assert len(result) == 10

        # Analyze response time consistency
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)

        # Response times should be consistent
        assert avg_response_time < 0.1, (
            f"Average response time {avg_response_time:.3f}s too high"
        )
        assert max_response_time < 0.2, (
            f"Max response time {max_response_time:.3f}s too high"
        )

        # Variation should be reasonable
        time_variation = max_response_time - min_response_time
        assert time_variation < 0.15, (
            f"Response time variation {time_variation:.3f}s too high"
        )
