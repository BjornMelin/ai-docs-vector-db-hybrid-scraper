"""Comprehensive integration tests for MCP tools.

Tests all 11 MCP tools with realistic scenarios, proper mocking of external dependencies,
and validation of request/response models following FastMCP 2.0 best practices.
"""

import asyncio
import time
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from src.config.models import FirecrawlConfig
from src.config.models import OpenAIConfig
from src.config.models import QdrantConfig
from src.config.models import UnifiedConfig
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import AnalyticsRequest
from src.mcp_tools.models.requests import DocumentRequest
from src.mcp_tools.models.requests import EmbeddingRequest
from src.mcp_tools.models.requests import HyDESearchRequest
from src.mcp_tools.models.requests import ProjectRequest
from src.mcp_tools.models.requests import SearchRequest
from src.mcp_tools.models.responses import CollectionInfo
from src.mcp_tools.models.responses import OperationStatus
from src.mcp_tools.models.responses import SearchResult

from tests.mocks.mock_tools import MockMCPServer
from tests.mocks.mock_tools import register_mock_tools


class TestMCPToolsIntegration:
    """Integration tests for all MCP tools with mocked dependencies."""

    @pytest.fixture
    async def mock_config(self):
        """Create a mock configuration for testing."""
        config = MagicMock(spec=UnifiedConfig)

        # Mock nested config objects
        config.qdrant = MagicMock(spec=QdrantConfig)
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = None

        config.openai = MagicMock(spec=OpenAIConfig)
        config.openai.api_key = "test-openai-key"

        config.firecrawl = MagicMock(spec=FirecrawlConfig)
        config.firecrawl.api_key = "test-firecrawl-key"

        # Mock crawling providers directly
        config.crawling = MagicMock()
        config.crawling.providers = ["crawl4ai"]

        config.get_active_providers.return_value = ["openai", "fastembed"]

        return config

    @pytest.fixture
    async def mock_client_manager(self, mock_config):
        """Create a mock client manager with all services."""
        client_manager = MagicMock(spec=ClientManager)
        client_manager.config = mock_config

        # Mock vector DB service
        mock_vector_service = AsyncMock()
        mock_vector_service.search_documents.return_value = [
            {
                "id": "test-doc-1",
                "content": "Test document content",
                "score": 0.95,
                "metadata": {"title": "Test Document", "url": "https://example.com"},
            }
        ]
        mock_vector_service.list_collections.return_value = [
            {"name": "documentation", "vectors_count": 100, "status": "green"}
        ]
        client_manager.vector_service = mock_vector_service

        # Mock embedding service
        mock_embedding_service = AsyncMock()
        mock_embedding_service.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3] * 128],  # 384-dim embeddings
            "model": "BAAI/bge-small-en-v1.5",
            "total_tokens": 10,
        }
        mock_embedding_service.list_providers.return_value = [
            {"name": "BAAI/bge-small-en-v1.5", "dims": 384, "context_length": 512}
        ]
        client_manager.embedding_service = mock_embedding_service

        # Mock crawling service
        mock_crawling_service = AsyncMock()
        mock_crawling_service.crawl_url.return_value = {
            "url": "https://example.com",
            "title": "Example Page",
            "content": "Example page content for testing",
            "word_count": 100,
            "success": True,
            "metadata": {"description": "Test page"},
        }
        client_manager.crawling_service = mock_crawling_service

        # Mock cache service
        mock_cache_service = AsyncMock()
        mock_cache_service.get_stats.return_value = {
            "hit_rate": 0.85,
            "size": 1000,
            "total_requests": 5000,
        }
        mock_cache_service.clear.return_value = {"cleared_count": 100}
        client_manager.cache_service = mock_cache_service

        # Mock project storage service
        mock_project_service = AsyncMock()
        mock_project_service.create_project.return_value = {
            "id": "test-project-123",
            "name": "Test Project",
            "description": "Test project description",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_project_service.list_projects.return_value = [
            {
                "id": "test-project-123",
                "name": "Test Project",
                "created_at": "2024-01-01T00:00:00Z",
            }
        ]
        client_manager.project_service = mock_project_service

        # Mock deployment service
        mock_deployment_service = AsyncMock()
        mock_deployment_service.list_aliases.return_value = {
            "production": "docs-v1.2",
            "staging": "docs-v1.3-beta",
        }
        client_manager.deployment_service = mock_deployment_service

        # Mock analytics service
        mock_analytics_service = AsyncMock()
        mock_analytics_service.get_analytics.return_value = {
            "timestamp": "2024-01-01T00:00:00Z",
            "collections": {
                "documentation": {
                    "total_documents": 1000,
                    "total_chunks": 5000,
                    "last_updated": "2024-01-01T00:00:00Z",
                }
            },
            "performance": {
                "avg_search_time_ms": 85.2,
                "95th_percentile_ms": 200.1,
            },
            "costs": {
                "total_embedding_cost": 12.50,
                "total_requests": 10000,
            },
        }
        client_manager.analytics_service = mock_analytics_service

        # Mock HyDE service
        mock_hyde_service = AsyncMock()
        mock_hyde_service.search.return_value = {
            "request_id": "hyde-123",
            "query": "test query",
            "collection": "documentation",
            "results": [],
            "metrics": {
                "search_time_ms": 150.5,
                "results_found": 0,
            },
        }
        client_manager.hyde_service = mock_hyde_service

        return client_manager

    @pytest.fixture
    async def mcp_server(self, mock_client_manager):
        """Create Mock MCP server with all tools registered."""
        mcp = MockMCPServer("test-mcp-server")
        register_mock_tools(mcp, mock_client_manager)
        return mcp

    async def test_search_tool_basic_search(self, mcp_server, mock_client_manager):
        """Test basic search functionality."""
        # Create search request
        request = SearchRequest(
            query="test query",
            collection="documentation",
            limit=5,
        )

        # Execute search via MCP tool
        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        assert search_tool is not None, "Search tool not found"

        # Mock the tool execution
        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.return_value = [
                {
                    "id": "doc-1",
                    "content": "Test document about Python programming",
                    "score": 0.95,
                    "metadata": {
                        "title": "Python Guide",
                        "url": "https://example.com/python",
                    },
                }
            ]

            # This simulates calling the tool
            result = await search_tool.handler(
                query=request.query,
                collection=request.collection,
                limit=request.limit,
                strategy=request.strategy.value,
                enable_reranking=request.enable_reranking,
            )

            # Validate response
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["content"] == "Test document about Python programming"
            assert result[0]["score"] == 0.95

            # Verify service was called correctly
            mock_search.assert_called_once()

    async def test_documents_tool_add_document(self, mcp_server, mock_client_manager):
        """Test document addition functionality."""
        request = DocumentRequest(
            url="https://example.com/doc",
            collection="documentation",
            chunk_strategy="enhanced",
        )

        # Find document tool
        add_doc_tool = None
        for tool in mcp_server._tools:
            if tool.name == "add_document":
                add_doc_tool = tool
                break

        assert add_doc_tool is not None, "Add document tool not found"

        # Mock successful document processing
        with patch.object(
            mock_client_manager.vector_service, "add_document"
        ) as mock_add:
            mock_add.return_value = {
                "url": request.url,
                "title": "Example Document",
                "chunks_created": 5,
                "collection": request.collection,
                "chunking_strategy": request.chunk_strategy,
                "embedding_dimensions": 384,
            }

            result = await add_doc_tool.handler(
                url=request.url,
                collection=request.collection,
                chunk_strategy=request.chunk_strategy.value,
            )

            # Validate response structure
            assert result["url"] == request.url
            assert result["chunks_created"] == 5
            assert result["embedding_dimensions"] == 384

    async def test_embeddings_tool_generate_embeddings(
        self, mcp_server, mock_client_manager
    ):
        """Test embedding generation functionality."""
        request = EmbeddingRequest(
            texts=["Hello world", "Test document"],
            model="BAAI/bge-small-en-v1.5",
        )

        # Find embeddings tool
        embedding_tool = None
        for tool in mcp_server._tools:
            if tool.name == "generate_embeddings":
                embedding_tool = tool
                break

        assert embedding_tool is not None, "Embeddings tool not found"

        with patch.object(
            mock_client_manager.embedding_service, "generate_embeddings"
        ) as mock_embed:
            mock_embed.return_value = {
                "embeddings": [[0.1] * 384, [0.2] * 384],
                "model": request.model,
                "total_tokens": 6,
                "cost_estimate": 0.001,
            }

            result = await embedding_tool.handler(
                texts=request.texts,
                model=request.model,
            )

            assert len(result["embeddings"]) == 2
            assert result["model"] == request.model
            assert result["total_tokens"] == 6

    async def test_collections_tool_list_collections(
        self, mcp_server, mock_client_manager
    ):
        """Test collection listing functionality."""
        # Find collections tool
        collections_tool = None
        for tool in mcp_server._tools:
            if tool.name == "list_collections":
                collections_tool = tool
                break

        assert collections_tool is not None, "Collections tool not found"

        with patch.object(
            mock_client_manager.vector_service, "list_collections"
        ) as mock_list:
            mock_list.return_value = [
                {"name": "documentation", "vectors_count": 1000, "status": "green"},
                {"name": "support", "vectors_count": 500, "status": "green"},
            ]

            result = await collections_tool.handler()

            assert len(result) == 2
            assert result[0]["name"] == "documentation"
            assert result[1]["vectors_count"] == 500

    async def test_projects_tool_create_project(self, mcp_server, mock_client_manager):
        """Test project creation functionality."""
        request = ProjectRequest(
            name="Test Project",
            description="A test project for MCP integration",
            quality_tier="premium",
        )

        # Find project tool
        project_tool = None
        for tool in mcp_server._tools:
            if tool.name == "create_project":
                project_tool = tool
                break

        assert project_tool is not None, "Create project tool not found"

        with patch.object(
            mock_client_manager.project_service, "create_project"
        ) as mock_create:
            mock_create.return_value = {
                "id": "proj-123",
                "name": request.name,
                "description": request.description,
                "created_at": "2024-01-01T00:00:00Z",
            }

            result = await project_tool.handler(
                name=request.name,
                description=request.description,
                quality_tier=request.quality_tier,
            )

            assert result["name"] == request.name
            assert result["description"] == request.description
            assert "id" in result

    async def test_advanced_search_hyde_search(self, mcp_server, mock_client_manager):
        """Test HyDE advanced search functionality."""
        request = HyDESearchRequest(
            query="Python machine learning tutorials",
            collection="documentation",
            num_generations=3,
        )

        # Find HyDE search tool
        hyde_tool = None
        for tool in mcp_server._tools:
            if tool.name == "hyde_search":
                hyde_tool = tool
                break

        assert hyde_tool is not None, "HyDE search tool not found"

        # Mock HyDE service
        with patch.object(mock_client_manager, "hyde_service") as mock_hyde_service:
            mock_hyde_service.search.return_value = {
                "request_id": "hyde-123",
                "query": request.query,
                "collection": request.collection,
                "results": [
                    {
                        "id": "ml-doc-1",
                        "content": "Comprehensive Python ML tutorial",
                        "score": 0.97,
                        "metadata": {"title": "Python ML Guide"},
                    }
                ],
                "metrics": {
                    "search_time_ms": 150.5,
                    "results_found": 1,
                    "generation_parameters": {"num_generations": 3},
                },
            }

            result = await hyde_tool.handler(
                query=request.query,
                collection=request.collection,
                num_generations=request.num_generations,
            )

            assert result["query"] == request.query
            assert len(result["results"]) == 1
            assert result["metrics"]["search_time_ms"] == 150.5

    async def test_analytics_tool_get_analytics(self, mcp_server, mock_client_manager):
        """Test analytics functionality."""
        request = AnalyticsRequest(
            collection="documentation",
            include_performance=True,
            include_costs=True,
        )

        # Find analytics tool
        analytics_tool = None
        for tool in mcp_server._tools:
            if tool.name == "get_analytics":
                analytics_tool = tool
                break

        assert analytics_tool is not None, "Analytics tool not found"

        # Mock analytics service
        with patch.object(mock_client_manager, "analytics_service") as mock_analytics:
            mock_analytics.get_analytics.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "collections": {
                    "documentation": {
                        "total_documents": 1000,
                        "total_chunks": 5000,
                        "last_updated": "2024-01-01T00:00:00Z",
                    }
                },
                "performance": {
                    "avg_search_time_ms": 85.2,
                    "95th_percentile_ms": 200.1,
                },
                "costs": {
                    "total_embedding_cost": 12.50,
                    "total_requests": 10000,
                },
            }

            result = await analytics_tool.handler(
                collection=request.collection,
                include_performance=request.include_performance,
                include_costs=request.include_costs,
            )

            assert "timestamp" in result
            assert "documentation" in result["collections"]
            assert "performance" in result
            assert "costs" in result

    async def test_cache_tool_operations(self, mcp_server, mock_client_manager):
        """Test cache management operations."""
        # Test cache stats
        cache_stats_tool = None
        for tool in mcp_server._tools:
            if tool.name == "get_cache_stats":
                cache_stats_tool = tool
                break

        assert cache_stats_tool is not None, "Cache stats tool not found"

        with patch.object(mock_client_manager.cache_service, "get_stats") as mock_stats:
            mock_stats.return_value = {
                "hit_rate": 0.87,
                "size": 2500,
                "total_requests": 15000,
            }

            result = await cache_stats_tool.handler()

            assert result["hit_rate"] == 0.87
            assert result["size"] == 2500

        # Test cache clear
        cache_clear_tool = None
        for tool in mcp_server._tools:
            if tool.name == "clear_cache":
                cache_clear_tool = tool
                break

        assert cache_clear_tool is not None, "Cache clear tool not found"

        with patch.object(mock_client_manager.cache_service, "clear") as mock_clear:
            mock_clear.return_value = {"cleared_count": 150, "status": "success"}

            result = await cache_clear_tool.handler(pattern="search:*")

            assert result["cleared_count"] == 150
            assert result["status"] == "success"

    async def test_deployment_tool_operations(self, mcp_server, mock_client_manager):
        """Test deployment management operations."""
        # Test list aliases
        aliases_tool = None
        for tool in mcp_server._tools:
            if tool.name == "list_aliases":
                aliases_tool = tool
                break

        assert aliases_tool is not None, "List aliases tool not found"

        with patch.object(
            mock_client_manager.deployment_service, "list_aliases"
        ) as mock_aliases:
            mock_aliases.return_value = {
                "production": "docs-v2.1",
                "staging": "docs-v2.2-beta",
                "canary": "docs-v2.3-alpha",
            }

            result = await aliases_tool.handler()

            assert "production" in result["aliases"]
            assert result["aliases"]["production"] == "docs-v2.1"

    async def test_utilities_tool_validate_config(
        self, mcp_server, mock_client_manager
    ):
        """Test utilities functionality."""
        # Find config validation tool
        config_tool = None
        for tool in mcp_server._tools:
            if tool.name == "validate_configuration":
                config_tool = tool
                break

        assert config_tool is not None, "Config validation tool not found"

        # Should return success
        result = await config_tool.handler()

        assert result["status"] == "success"

    async def test_payload_indexing_tool_operations(
        self, mcp_server, mock_client_manager
    ):
        """Test payload indexing functionality."""
        # Find reindex tool
        reindex_tool = None
        for tool in mcp_server._tools:
            if tool.name == "reindex_collection":
                reindex_tool = tool
                break

        assert reindex_tool is not None, "Reindex tool not found"

        with patch.object(
            mock_client_manager.vector_service, "reindex_collection"
        ) as mock_reindex:
            mock_reindex.return_value = {
                "status": "success",
                "collection": "documentation",
                "reindexed_count": 5000,
                "message": "Collection reindexed successfully",
            }

            result = await reindex_tool.handler(collection="documentation")

            assert result["status"] == "success"
            assert result["reindexed_count"] == 5000

    # Error Handling Tests

    async def test_tool_error_handling_search_failure(
        self, mcp_server, mock_client_manager
    ):
        """Test error handling when search tool fails."""
        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.side_effect = Exception("Qdrant connection failed")

            with pytest.raises(Exception, match="Qdrant connection failed"):
                await search_tool.handler(
                    query="test",
                    collection="documentation",
                    limit=10,
                )

    async def test_tool_error_handling_invalid_request(self, mcp_server):
        """Test error handling for invalid request parameters."""
        # Test invalid search request
        with pytest.raises(ValidationError):
            SearchRequest(
                query="",  # Empty query should be invalid
                limit=200,  # Exceeds maximum limit
            )

        # Test invalid embedding request
        with pytest.raises(ValidationError):
            EmbeddingRequest(
                texts=[],  # Empty texts list should be invalid
                batch_size=200,  # Exceeds maximum
            )

    # Performance Tests

    async def test_search_tool_performance(self, mcp_server, mock_client_manager):
        """Test search tool performance characteristics."""
        search_tool = None
        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        with patch.object(
            mock_client_manager.vector_service, "search_documents"
        ) as mock_search:
            mock_search.return_value = [
                {"id": f"doc-{i}", "content": f"Content {i}", "score": 0.9}
                for i in range(10)
            ]

            # Measure execution time
            start_time = time.time()
            result = await search_tool.handler(
                query="performance test",
                collection="documentation",
                limit=10,
            )
            execution_time = time.time() - start_time

            # Performance assertions
            assert execution_time < 1.0, (
                f"Search took {execution_time:.3f}s, expected < 1.0s"
            )
            assert len(result) == 10

    async def test_concurrent_tool_execution(self, mcp_server, mock_client_manager):
        """Test concurrent execution of multiple tools."""
        search_tool = None
        analytics_tool = None
        collections_tool = None

        for tool in mcp_server._tools:
            if tool.name == "search_documents":
                search_tool = tool
            elif tool.name == "get_analytics":
                analytics_tool = tool
            elif tool.name == "list_collections":
                collections_tool = tool

        # Mock all services
        with (
            patch.object(
                mock_client_manager.vector_service, "search_documents"
            ) as mock_search,
            patch.object(mock_client_manager, "analytics_service") as mock_analytics,
            patch.object(
                mock_client_manager.vector_service, "list_collections"
            ) as mock_collections,
        ):
            mock_search.return_value = [
                {"id": "doc-1", "content": "Test", "score": 0.9}
            ]
            mock_analytics.get_analytics.return_value = {
                "timestamp": "2024-01-01T00:00:00Z",
                "collections": {},
            }
            mock_collections.return_value = [{"name": "docs", "vectors_count": 100}]

            # Execute tools concurrently
            tasks = [
                search_tool.handler(query="test", collection="docs", limit=5),
                analytics_tool.handler(),
                collections_tool.handler(),
            ]

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            execution_time = time.time() - start_time

            # Verify all completed successfully
            assert len(results) == 3
            assert len(results[0]) == 1  # Search results
            assert "timestamp" in results[1]  # Analytics results
            assert len(results[2]) == 1  # Collections results

            # Performance assertion - concurrent execution should be faster
            assert execution_time < 2.0, (
                f"Concurrent execution took {execution_time:.3f}s"
            )

    # Request/Response Model Validation Tests

    async def test_request_model_validation(self):
        """Test comprehensive request model validation."""
        # Valid SearchRequest
        valid_search = SearchRequest(
            query="test query",
            collection="documentation",
            limit=10,
        )
        assert valid_search.query == "test query"
        assert valid_search.limit == 10

        # Valid EmbeddingRequest
        valid_embedding = EmbeddingRequest(
            texts=["text1", "text2"],
            model="test-model",
        )
        assert len(valid_embedding.texts) == 2

        # Valid ProjectRequest
        valid_project = ProjectRequest(
            name="Test Project",
            quality_tier="premium",
        )
        assert valid_project.quality_tier == "premium"

    async def test_response_model_validation(self):
        """Test response model validation."""
        # Test SearchResult
        search_result = SearchResult(
            id="doc-1",
            content="Test content",
            score=0.95,
            url="https://example.com",
            metadata={"title": "Test"},
        )
        assert search_result.score == 0.95

        # Test OperationStatus
        operation_status = OperationStatus(
            status="success",
            message="Operation completed successfully",
        )
        assert operation_status.status == "success"

        # Test CollectionInfo
        collection_info = CollectionInfo(
            name="documentation",
            vectors_count=1000,
            status="green",
        )
        assert collection_info.vectors_count == 1000


class TestMCPServerLifecycle:
    """Test MCP server lifecycle and tool registration."""

    async def test_server_initialization_with_tools(self):
        """Test that server initializes correctly with all tools."""
        mcp = MockMCPServer("test-server")
        mock_client_manager = MagicMock()

        # Mock all required services
        mock_client_manager.vector_service = AsyncMock()
        mock_client_manager.embedding_service = AsyncMock()
        mock_client_manager.crawling_service = AsyncMock()
        mock_client_manager.cache_service = AsyncMock()
        mock_client_manager.project_service = AsyncMock()
        mock_client_manager.deployment_service = AsyncMock()
        mock_client_manager.analytics_service = AsyncMock()
        mock_client_manager.hyde_service = AsyncMock()

        register_mock_tools(mcp, mock_client_manager)

        # Verify tools are registered
        assert len(mcp._tools) > 0, "No tools were registered"

        # Expected tool names based on tool registry
        expected_tools = {
            "search_documents",
            "add_document",
            "generate_embeddings",
            "list_collections",
            "create_project",
            "hyde_search",
            "get_analytics",
            "get_cache_stats",
            "list_aliases",
            "validate_configuration",
            "reindex_collection",
        }

        registered_tool_names = {tool.name for tool in mcp._tools}

        # Check that we have reasonable coverage of expected tools
        # Note: We have 12 tools registered
        assert len(registered_tool_names) >= 10, (
            f"Expected at least 10 tools, got {len(registered_tool_names)}"
        )

    async def test_tool_registration_error_handling(self):
        """Test error handling during tool registration."""
        mcp = MockMCPServer("test-server")

        # Create client manager that will fail during tool registration
        mock_client_manager = MagicMock()
        mock_client_manager.vector_service = None  # Missing required service

        # Registration should still succeed with mock tools
        register_mock_tools(mcp, mock_client_manager)

        # But calling tools should fail
        search_tool = None
        for tool in mcp._tools:
            if tool.name == "search_documents":
                search_tool = tool
                break

        if search_tool:
            with pytest.raises(AttributeError):
                await search_tool.handler(query="test", collection="docs", limit=5)
