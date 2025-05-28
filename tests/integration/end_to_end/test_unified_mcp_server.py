"""
Tests for the unified MCP server
"""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch
from uuid import uuid4

import pytest
from qdrant_client.models import ScoredPoint


@pytest.fixture
def mock_context():
    """Mock MCP context for tool calls"""
    context = MagicMock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.error = AsyncMock()
    context.warning = AsyncMock()
    return context


@pytest.fixture
def mock_service_manager():
    """Mock the service manager and its dependencies"""
    with patch("src.unified_mcp_server.service_manager") as mock_sm:
        # Mock all service attributes
        mock_sm.config = MagicMock()
        mock_sm.config.openai_api_key = "test-key"
        mock_sm.config.firecrawl_api_key = "test-key"
        mock_sm.config.qdrant_url = "http://localhost:6333"
        mock_sm.config.redis_url = None
        mock_sm.config.cache_config = MagicMock(max_items=1000)

        mock_sm.embedding_manager = MagicMock()
        mock_sm.crawl_manager = MagicMock()
        mock_sm.qdrant_service = MagicMock()
        mock_sm.cache_manager = MagicMock()
        mock_sm.projects = {}
        mock_sm._initialized = True

        # Mock async methods
        mock_sm.initialize = AsyncMock()
        mock_sm.cleanup = AsyncMock()

        # Configure embedding manager
        mock_sm.embedding_manager.generate_embeddings = AsyncMock(
            return_value=[[0.1] * 1536]  # Return a list of embeddings directly
        )
        mock_sm.embedding_manager.get_current_provider_info = MagicMock(
            return_value={"name": "openai", "model": "text-embedding-3-small"}
        )
        mock_sm.embedding_manager.rerank_results = AsyncMock(
            return_value=[
                {
                    "original": MagicMock(
                        content="Test content",
                        score=0.95,
                        url="https://example.com",
                        title="Test",
                        metadata={},
                    ),
                    "score": 0.95,
                }
            ]
        )

        # Configure crawl manager
        mock_sm.crawl_manager.crawl_single = AsyncMock(
            return_value=MagicMock(
                markdown="# Test Document\n\nContent here.",
                metadata={"title": "Test"},
            )
        )

        # Configure qdrant service
        mock_sm.qdrant_service.search = AsyncMock(
            return_value=[
                ScoredPoint(
                    id=str(uuid4()),
                    score=0.95,
                    version=1,
                    payload={
                        "content": "Test content",
                        "url": "https://example.com",
                        "title": "Test",
                    },
                )
            ]
        )
        mock_sm.qdrant_service.search_by_vector = AsyncMock(
            return_value=[
                ScoredPoint(
                    id=str(uuid4()),
                    score=0.90,
                    version=1,
                    payload={
                        "content": "Similar content",
                        "url": "https://example.com/similar",
                        "title": "Similar",
                    },
                )
            ]
        )
        mock_sm.qdrant_service.hybrid_search = AsyncMock(
            return_value=[
                ScoredPoint(
                    id=str(uuid4()),
                    score=0.95,
                    version=1,
                    payload={
                        "content": "Test content",
                        "url": "https://example.com",
                        "title": "Test",
                    },
                )
            ]
        )
        mock_sm.qdrant_service.create_collection = AsyncMock()
        mock_sm.qdrant_service.upsert_points = AsyncMock()
        mock_sm.qdrant_service.list_collections = AsyncMock(
            return_value=["documentation", "test"]
        )
        mock_sm.qdrant_service.get_collection_info = AsyncMock(
            return_value=MagicMock(
                vectors_count=1000,
                indexed_vectors_count=1000,
                status="green",
                config={},
            )
        )
        mock_sm.qdrant_service.delete_collection = AsyncMock()
        mock_sm.qdrant_service.optimize_collection = AsyncMock()

        # Mock the _client attribute for direct calls
        mock_sm.qdrant_service._client = MagicMock()
        mock_sm.qdrant_service._client.get_collections = AsyncMock(
            return_value=MagicMock(
                collections=[MagicMock(name="documentation"), MagicMock(name="test")]
            )
        )
        mock_sm.qdrant_service._client.update_collection_aliases = AsyncMock()

        # Configure cache manager
        mock_sm.cache_manager.get = AsyncMock(return_value=None)
        mock_sm.cache_manager.set = AsyncMock()
        mock_sm.cache_manager.get_stats = AsyncMock(
            return_value={
                "hit_rate": 0.85,
                "total_requests": 1000,
                "hits": 850,
                "misses": 150,
            }
        )
        mock_sm.cache_manager.clear_all = AsyncMock(return_value=100)
        mock_sm.cache_manager.clear_pattern = AsyncMock(return_value=10)

        yield mock_sm


class TestSearchTools:
    """Test search and retrieval tools"""

    @pytest.mark.asyncio
    async def test_search_documents(self, mock_service_manager, mock_context):
        """Test document search with various strategies"""
        from src.unified_mcp_server import SearchRequest
        from src.unified_mcp_server import SearchStrategy
        from src.unified_mcp_server import search_documents

        request = SearchRequest(
            query="test query",
            collection="documentation",
            limit=5,
            strategy=SearchStrategy.HYBRID,
            enable_reranking=True,
        )

        result = await search_documents(request, mock_context)

        assert len(result) == 1
        assert result[0].content == "Test content"
        assert result[0].score == 0.95
        assert result[0].url == "https://example.com"

        # Verify service calls
        mock_service_manager.qdrant_service.hybrid_search.assert_called_once()
        mock_service_manager.cache_manager.get.assert_called_once()
        mock_service_manager.cache_manager.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_similar(self, mock_service_manager):
        """Test similarity search"""
        from src.unified_mcp_server import search_similar

        result = await search_similar(
            content="Find similar to this",
            collection="documentation",
            limit=5,
            threshold=0.8,
        )

        assert len(result) == 1
        assert (
            result[0].content == "Test content"
        )  # hybrid_search returns same mock data
        assert result[0].score == 0.95

        # Verify embedding generation
        mock_service_manager.embedding_manager.generate_embeddings.assert_called_once()
        # search_similar uses hybrid_search, not search_by_vector
        mock_service_manager.qdrant_service.hybrid_search.assert_called_once()


class TestEmbeddingTools:
    """Test embedding management tools"""

    @pytest.mark.asyncio
    async def test_generate_embeddings(self, mock_service_manager):
        """Test embedding generation"""
        from src.unified_mcp_server import EmbeddingRequest
        from src.unified_mcp_server import generate_embeddings

        request = EmbeddingRequest(
            texts=["Text 1", "Text 2"],
            batch_size=32,
        )

        result = await generate_embeddings(request)

        assert result["count"] == 1  # Mocked to return 1 embedding
        assert result["dimensions"] == 1536
        assert result["provider"] == "openai"
        assert "cost_estimate" in result

    @pytest.mark.asyncio
    async def test_list_embedding_providers(self, mock_service_manager):
        """Test listing embedding providers"""
        from src.unified_mcp_server import list_embedding_providers

        result = await list_embedding_providers()

        assert len(result) == 2
        assert any(p["name"] == "openai" for p in result)
        assert any(p["name"] == "fastembed" for p in result)


class TestDocumentTools:
    """Test document management tools"""

    @pytest.mark.asyncio
    async def test_add_document(self, mock_service_manager):
        """Test adding a single document"""
        from src.unified_mcp_server import ChunkingStrategy
        from src.unified_mcp_server import DocumentRequest
        from src.unified_mcp_server import add_document

        request = DocumentRequest(
            url="https://example.com/doc",
            collection="test",
            chunking_strategy=ChunkingStrategy.ENHANCED,
            chunk_size=1600,
            chunk_overlap=200,
        )

        result = await add_document(request)

        assert result["url"] == request.url
        assert result["title"] == "Test"
        assert result["chunks_created"] > 0
        assert result["collection"] == "test"

        # Verify service calls
        mock_service_manager.crawl_manager.crawl_single.assert_called_once()
        mock_service_manager.embedding_manager.generate_embeddings.assert_called_once()
        mock_service_manager.qdrant_service.create_collection.assert_called_once()
        mock_service_manager.qdrant_service.upsert_points.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_batch(self, mock_service_manager):
        """Test batch document processing"""
        from src.unified_mcp_server import BatchRequest
        from src.unified_mcp_server import add_documents_batch

        request = BatchRequest(
            urls=[
                "https://example.com/doc1",
                "https://example.com/doc2",
                "https://example.com/doc3",
            ],
            collection="test",
            parallel_limit=2,
        )

        result = await add_documents_batch(request)

        assert result["total"] == 3
        assert len(result["successful"]) == 3
        assert len(result["failed"]) == 0


class TestProjectTools:
    """Test project management tools"""

    @pytest.mark.asyncio
    async def test_create_project(self, mock_service_manager):
        """Test project creation"""
        from src.unified_mcp_server import ProjectRequest
        from src.unified_mcp_server import create_project

        request = ProjectRequest(
            name="Test Project",
            description="A test project",
            quality_tier="balanced",
            urls=["https://example.com/doc1"],
        )

        result = await create_project(request)

        assert result["name"] == request.name
        assert result["description"] == request.description
        assert result["quality_tier"] == request.quality_tier
        assert "id" in result
        assert "collection" in result
        assert result["document_count"] == 1

        # Verify collection creation
        mock_service_manager.qdrant_service.create_collection.assert_called()

    @pytest.mark.asyncio
    async def test_list_projects(self, mock_service_manager):
        """Test listing projects"""
        from src.unified_mcp_server import list_projects

        # Create a test project first
        mock_service_manager.projects = {
            "test-id": {
                "id": "test-id",
                "name": "Test Project",
                "collection": "project_test-id",
                "created_at": datetime.now(UTC).isoformat(),
            }
        }

        result = await list_projects()

        assert len(result) == 1
        assert result[0]["name"] == "Test Project"
        assert result[0]["vector_count"] == 1000  # From mock

    @pytest.mark.asyncio
    async def test_search_project(self, mock_service_manager):
        """Test searching within a project"""
        from src.unified_mcp_server import search_project

        # Create a test project
        project_id = "test-id"
        mock_service_manager.projects = {
            project_id: {
                "id": project_id,
                "collection": f"project_{project_id}",
                "quality_tier": "premium",
            }
        }

        result = await search_project(
            project_id=project_id,
            query="test query",
            limit=5,
        )

        assert len(result) == 1
        assert result[0].content == "Test content"


class TestCollectionTools:
    """Test collection management tools"""

    @pytest.mark.asyncio
    async def test_list_collections(self, mock_service_manager):
        """Test listing collections"""
        from src.unified_mcp_server import list_collections

        result = await list_collections()

        assert len(result) == 2
        assert all("name" in c for c in result)
        assert all("vector_count" in c for c in result)

    @pytest.mark.asyncio
    async def test_delete_collection(self, mock_service_manager):
        """Test deleting a collection"""
        from src.unified_mcp_server import delete_collection

        result = await delete_collection("test-collection")

        assert result["status"] == "success"
        mock_service_manager.qdrant_service.delete_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_collection(self, mock_service_manager):
        """Test optimizing a collection"""
        from src.unified_mcp_server import optimize_collection

        result = await optimize_collection("test-collection")

        assert result["status"] == "optimized"
        assert result["collection"] == "test-collection"
        # Verify that client methods were called
        mock_service_manager.qdrant_service._client.update_collection_aliases.assert_called_once()
        assert mock_service_manager.qdrant_service.get_collection_info.call_count == 2


class TestAnalyticsTools:
    """Test analytics and monitoring tools"""

    @pytest.mark.asyncio
    async def test_get_analytics(self, mock_service_manager):
        """Test getting analytics"""
        from src.unified_mcp_server import AnalyticsRequest
        from src.unified_mcp_server import get_analytics

        request = AnalyticsRequest(
            include_performance=True,
            include_costs=True,
        )

        result = await get_analytics(request)

        assert "timestamp" in result
        assert "collections" in result
        assert "cache_metrics" in result
        assert "costs" in result

        # Check cache metrics
        assert result["cache_metrics"]["hit_rate"] == 0.85
        assert result["cache_metrics"]["total_requests"] == 1000

    @pytest.mark.asyncio
    async def test_get_system_health(self, mock_service_manager):
        """Test system health check"""
        from src.unified_mcp_server import get_system_health

        result = await get_system_health()

        assert result["status"] == "healthy"
        assert "timestamp" in result
        assert "services" in result

        # Check service statuses
        assert result["services"]["qdrant"]["status"] == "healthy"
        assert result["services"]["embeddings"]["status"] == "healthy"
        assert result["services"]["cache"]["status"] == "healthy"


class TestCacheTools:
    """Test cache management tools"""

    @pytest.mark.asyncio
    async def test_clear_cache(self, mock_service_manager):
        """Test clearing cache"""
        from src.unified_mcp_server import clear_cache

        result = await clear_cache()

        assert result["status"] == "success"
        assert result["cleared_count"] == 100
        mock_service_manager.cache_manager.clear_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cache_pattern(self, mock_service_manager):
        """Test clearing cache with pattern"""
        from src.unified_mcp_server import clear_cache

        result = await clear_cache(pattern="search:*")

        assert result["status"] == "success"
        assert result["cleared_count"] == 10
        assert result["pattern"] == "search:*"
        mock_service_manager.cache_manager.clear_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache_stats(self, mock_service_manager):
        """Test getting cache statistics"""
        from src.unified_mcp_server import get_cache_stats

        result = await get_cache_stats()

        assert result["hit_rate"] == 0.85
        assert result["total_requests"] == 1000


class TestUtilityTools:
    """Test utility tools"""

    @pytest.mark.asyncio
    async def test_estimate_costs(self, mock_service_manager):
        """Test cost estimation"""
        from src.unified_mcp_server import estimate_costs

        result = await estimate_costs(
            text_count=1000,
            average_length=1500,
            include_storage=True,
        )

        assert result["text_count"] == 1000
        assert "embedding_cost" in result
        assert "storage_gb" in result
        assert "total_cost" in result

    @pytest.mark.asyncio
    async def test_validate_configuration(self, mock_service_manager):
        """Test configuration validation"""
        from src.unified_mcp_server import validate_configuration

        # Temporarily remove firecrawl key to test warning
        mock_service_manager.config.firecrawl_api_key = None

        result = await validate_configuration()

        assert result["valid"] is True
        assert len(result["warnings"]) == 1  # Firecrawl not configured
        assert result["config"]["openai"] == "configured"
        assert result["config"]["cache"]["l2_enabled"] is False


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_search_with_cache_hit(self, mock_service_manager):
        """Test search when cache has results"""
        from src.unified_mcp_server import SearchRequest
        from src.unified_mcp_server import search_documents

        # Mock cache hit
        cached_results = [
            {
                "id": "cached-1",
                "content": "Cached content",
                "score": 0.99,
                "url": "https://cached.com",
            }
        ]
        mock_service_manager.cache_manager.get = AsyncMock(return_value=cached_results)

        request = SearchRequest(query="cached query")
        result = await search_documents(request)

        assert len(result) == 1
        assert result[0].content == "Cached content"

        # Verify no search was performed
        mock_service_manager.qdrant_service.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_document_crawl_failure(self, mock_service_manager):
        """Test document addition when crawl fails"""
        from src.unified_mcp_server import DocumentRequest
        from src.unified_mcp_server import add_document

        mock_service_manager.crawl_manager.crawl_single = AsyncMock(return_value=None)

        request = DocumentRequest(url="https://fail.com")

        with pytest.raises(ValueError) as exc_info:
            await add_document(request)

        assert "Failed to crawl" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_project_not_found(self, mock_service_manager):
        """Test searching non-existent project"""
        from src.unified_mcp_server import search_project

        with pytest.raises(ValueError) as exc_info:
            await search_project(
                project_id="non-existent",
                query="test",
            )

        assert "Project non-existent not found" in str(exc_info.value)
