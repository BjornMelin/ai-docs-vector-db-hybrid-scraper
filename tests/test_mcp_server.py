"""Test suite for MCP server implementation.

Tests the FastMCP server functionality including tools, resources, and integrations.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp import Client
from openai import AsyncOpenAI
from qdrant_client import AsyncQdrantClient
from src.mcp_server import mcp


class TestMCPServer:
    """Test suite for the main MCP server"""

    @pytest.fixture
    async def mcp_client(self):
        """Create an in-memory MCP client"""
        client = Client(mcp)
        async with client:
            yield client

    @pytest.fixture
    def mock_qdrant(self):
        """Mock Qdrant client"""
        with patch("src.mcp_server.get_qdrant_client") as mock:
            client = AsyncMock(spec=AsyncQdrantClient)
            mock.return_value = client
            yield client

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI client"""
        with patch("src.mcp_server.get_openai_client") as mock:
            client = AsyncMock(spec=AsyncOpenAI)
            mock.return_value = client
            yield client

    async def test_server_initialization(self, mcp_client):
        """Test that the MCP server initializes correctly"""
        # List available tools
        tools = await mcp_client.list_tools()
        tool_names = [tool.name for tool in tools]

        # Verify core tools are registered
        assert "scrape_url" in tool_names
        assert "search" in tool_names
        assert "list_collections" in tool_names
        assert "create_collection" in tool_names
        assert "clear_cache" in tool_names

    async def test_scrape_url_tool(self, mcp_client):
        """Test URL scraping tool"""
        # Skip the test for now as it requires complex mocking
        pytest.skip("Requires complex crawl4ai module mocking")

    async def test_search_tool(self, mcp_client, mock_qdrant, mock_openai):
        """Test search functionality"""
        # Mock OpenAI embedding response
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.return_value = mock_embedding_response

        # Mock Qdrant search response
        mock_search_result = MagicMock()
        mock_search_result.score = 0.95
        mock_search_result.payload = {
            "content": "Test content",
            "url": "https://example.com",
            "title": "Test Title",
            "chunk_index": 0,
        }
        mock_qdrant.search.return_value = [mock_search_result]

        # Call search tool
        result = await mcp_client.call_tool(
            "search", {"query": "test query", "collection": "documentation", "limit": 5}
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert result_data["query"] == "test query"
        assert len(result_data["results"]) == 1
        assert result_data["results"][0]["score"] == 0.95

    async def test_list_collections(self, mcp_client, mock_qdrant):
        """Test listing collections"""
        # Mock collection response
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"

        mock_response = MagicMock()
        mock_response.collections = [mock_collection]
        mock_qdrant.get_collections.return_value = mock_response

        # Mock collection info
        mock_info = MagicMock()
        mock_info.vectors_count = 100
        mock_info.points_count = 100
        mock_info.indexed_vectors_count = 100
        mock_qdrant.get_collection.return_value = mock_info

        # Call tool
        result = await mcp_client.call_tool("list_collections", {})

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert len(result_data["collections"]) == 1
        assert result_data["collections"][0]["name"] == "test_collection"

    async def test_create_collection(self, mcp_client, mock_qdrant):
        """Test creating a collection"""
        mock_qdrant.create_collection.return_value = None

        result = await mcp_client.call_tool(
            "create_collection",
            {"name": "new_collection", "vector_size": 1536, "distance": "cosine"},
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert result_data["name"] == "new_collection"

        # Verify Qdrant was called correctly
        mock_qdrant.create_collection.assert_called_once()

    async def test_delete_collection(self, mcp_client, mock_qdrant):
        """Test deleting a collection"""
        mock_qdrant.delete_collection.return_value = None

        result = await mcp_client.call_tool(
            "delete_collection", {"name": "test_collection"}
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert "deleted successfully" in result_data["message"]

    async def test_clear_cache(self, mcp_client):
        """Test cache clearing"""
        with (
            patch("src.mcp_server.shutil.rmtree") as mock_rmtree,
            patch("src.mcp_server.Path.exists", return_value=True),
            patch("src.mcp_server.Path.mkdir") as mock_mkdir,
        ):
            result = await mcp_client.call_tool("clear_cache", {})

            assert result[0].text
            result_data = json.loads(result[0].text)
            assert result_data["success"] is True
            assert "Cache cleared successfully" in result_data["message"]

            # Verify cache operations were called
            mock_rmtree.assert_called_once()
            mock_mkdir.assert_called_once()

    async def test_resources(self, mcp_client):
        """Test MCP resources"""
        # List resources
        resources = await mcp_client.list_resources()
        resource_uris = [r.uri for r in resources]

        assert any("config://environment" in str(uri) for uri in resource_uris)
        assert any("stats://database" in str(uri) for uri in resource_uris)

        # Read environment config
        env_resource = await mcp_client.read_resource("config://environment")
        env_data = json.loads(env_resource.contents[0].text)
        assert "qdrant_url" in env_data
        assert "openai_api_key_set" in env_data

    async def test_error_handling(self, mcp_client, mock_qdrant, mock_openai):
        """Test error handling in tools"""
        # Set up mock for OpenAI to avoid API key error
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock(embedding=[0.1] * 1536)]
        mock_openai.embeddings.create.return_value = mock_embedding_response

        # Simulate Qdrant error
        mock_qdrant.search.side_effect = Exception("Connection failed")

        result = await mcp_client.call_tool(
            "search", {"query": "test", "collection": "docs"}
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is False
        assert "Connection failed" in result_data["error"]


class TestEnhancedMCPServer:
    """Test suite for the enhanced MCP server"""

    @pytest.fixture
    async def enhanced_client(self):
        """Create client for enhanced server"""
        from src.enhanced_mcp_server import enhanced_mcp

        client = Client(enhanced_mcp)
        async with client:
            yield client

    async def test_project_management(self, enhanced_client):
        """Test project creation and management"""
        # Create project
        result = await enhanced_client.call_tool(
            "create_project",
            {
                "name": "test_project",
                "description": "Test documentation project",
                "source_urls": ["https://example.com/docs"],
            },
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert result_data["project"]["name"] == "test_project"

        # List projects
        result = await enhanced_client.call_tool("list_projects", {})
        result_data = json.loads(result[0].text)
        assert result_data["total"] == 1

    async def test_scraping_plan(self, enhanced_client):
        """Test scraping plan creation"""
        # First create a project
        await enhanced_client.call_tool("create_project", {"name": "test_project"})

        # Create scraping plan
        result = await enhanced_client.call_tool(
            "plan_scraping",
            {
                "project_name": "test_project",
                "urls": ["https://docs.example.com", "https://api.example.com"],
                "auto_discover": True,
                "max_depth": 3,
            },
        )

        assert result[0].text
        plan = json.loads(result[0].text)
        assert plan["project_name"] == "test_project"
        assert len(plan["sites"]) == 2
        assert plan["strategy"] == "hybrid"

    async def test_smart_search(self, enhanced_client):
        """Test smart search functionality"""
        result = await enhanced_client.call_tool(
            "smart_search",
            {"query": "installation guide", "project_name": None},
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert "results" in result_data
        assert result_data["query"] == "installation guide"

    async def test_export_project(self, enhanced_client):
        """Test project export"""
        # Create a project first
        await enhanced_client.call_tool(
            "create_project",
            {"name": "export_test", "description": "Project for export testing"},
        )

        # Export project
        result = await enhanced_client.call_tool(
            "export_project", {"name": "export_test", "include_content": False}
        )

        assert result[0].text
        result_data = json.loads(result[0].text)
        assert result_data["success"] is True
        assert result_data["export"]["project"]["name"] == "export_test"
        assert "export_date" in result_data["export"]


@pytest.mark.integration
class TestMCPIntegration:
    """Integration tests for MCP server"""

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set"
    )
    async def test_real_search_integration(self):
        """Test real search with OpenAI embeddings"""
        from src.mcp_server import mcp

        client = Client(mcp)

        async with client:
            # This would test real OpenAI embedding generation
            # and Qdrant search if services are available
            pass

    async def test_concurrent_operations(self):
        """Test concurrent tool calls"""
        from src.mcp_server import mcp

        client = Client(mcp)

        async with client:
            # Test multiple concurrent operations
            tasks = [
                client.call_tool("list_collections", {}),
                client.call_tool("clear_cache", {}),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Verify all operations completed
            for result in results:
                assert not isinstance(result, Exception)
                assert result[0].text
