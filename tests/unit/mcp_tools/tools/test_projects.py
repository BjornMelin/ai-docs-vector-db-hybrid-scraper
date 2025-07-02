"""Tests for MCP project management tools."""

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml
from pydantic import ValidationError

# Import the actual enums before any mocking happens
from src.config import SearchStrategy
from src.mcp_tools.models.requests import ProjectRequest
from src.mcp_tools.models.responses import ProjectInfo
from src.mcp_tools.tools.projects import register_tools


if TYPE_CHECKING:
    from fastmcp import Context
else:
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


@pytest.fixture
def mock_client_manager():
    """Create a mock client manager."""
    manager = Mock()

    # Mock project storage service
    mock_project_storage = Mock()
    mock_project_storage.create_project = AsyncMock()
    mock_project_storage.delete_project = AsyncMock()
    mock_project_storage.update_project = AsyncMock()
    mock_project_storage.list_projects = AsyncMock()
    mock_project_storage.get_project = AsyncMock()
    mock_project_storage.save_project = AsyncMock()
    manager.get_project_storage = AsyncMock(return_value=mock_project_storage)

    # Mock other services
    mock_qdrant = Mock()
    mock_qdrant.create_collection = AsyncMock()
    mock_qdrant.create_collection_with_quality = AsyncMock()
    mock_qdrant.delete_collection = AsyncMock()
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)
    manager.qdrant_service = mock_qdrant  # Direct access

    mock_crawling = Mock()
    mock_crawling.crawl_url = AsyncMock()
    manager.get_crawling_service = Mock(return_value=mock_crawling)

    # Also set up crawl manager for projects.py
    mock_crawl_manager = Mock()
    mock_crawl_manager.scrape_url = AsyncMock()
    manager.get_crawl_manager = AsyncMock(return_value=mock_crawl_manager)

    mock_document_service = Mock()
    mock_document_service.add_document = AsyncMock()
    manager.get_document_service = Mock(return_value=mock_document_service)

    return manager


@pytest.mark.asyncio
async def test_project_tools_registration(mock_client_manager):
    """Test that project tools are properly registered."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    expected_tools = [
        "create_project",
        "delete_project",
        "update_project",
        "list_projects",
        "search_project",
    ]

    for tool_name in expected_tools:
        assert tool_name in registered_tools, f"Tool {tool_name} not registered"


@pytest.mark.asyncio
async def test_create_project_success(mock_client_manager, mock_context):
    """Test successful project creation."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.save_project = AsyncMock()
    project_storage.create_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "A test project",
        "quality_tier": "balanced",
        "created_at": "2024-01-01T00:00:00Z",
        "status": "active",
    }

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection_with_quality.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test project creation
    request = ProjectRequest(
        name="Test Project", description="A test project", quality_tier="balanced"
    )

    result = await registered_tools["create_project"](request, mock_context)

    assert isinstance(result, ProjectInfo)
    assert result.name == "Test Project"
    assert result.description == "A test project"
    assert result.quality_tier == "balanced"

    # Verify calls
    project_storage.save_project.assert_called_once()
    mock_client_manager.qdrant_service.create_collection.assert_called_once()
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_create_project_with_initial_urls(mock_client_manager, mock_context):
    """Test project creation with initial URLs."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.create_project.return_value = {
        "id": "proj_456",
        "name": "Docs Project",
        "quality_tier": "premium",
        "created_at": "2024-01-01T00:00:00Z",
        "status": "active",
    }

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection_with_quality.return_value = True

    # Setup crawl manager for URL processing
    crawl_manager = await mock_client_manager.get_crawl_manager()

    # Test with initial URLs
    urls = ["https://example.com/docs", "https://example.com/api"]

    # Mock crawl results for both URLs to return successful results
    crawl_manager.scrape_url.side_effect = [
        {
            "url": url,
            "success": True,
            "content": f"Content for {url}",
            "title": f"Title for {url}",
            "metadata": {"type": "documentation"},
        }
        for url in urls
    ]

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Docs Project",
        description="Documentation project",
        quality_tier="premium",
        urls=urls,
    )

    result = await registered_tools["create_project"](request, mock_context)

    assert result.name == "Docs Project"
    assert result.description == "Documentation project"
    assert result.quality_tier == "premium"
    assert result.document_count == len(urls)

    # Verify URL processing - the function should call scrape_url for each URL
    crawl_manager = await mock_client_manager.get_crawl_manager()
    assert crawl_manager.scrape_url.call_count == len(urls)


@pytest.mark.asyncio
async def test_create_project_invalid_quality_tier(mock_client_manager, _mock_context):
    """Test project creation with invalid quality tier."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool
    register_tools(mock_mcp, mock_client_manager)

    # Test with invalid quality tier
    with pytest.raises(ValidationError):
        ProjectRequest(
            name="Test Project", description="Test", quality_tier="invalid_tier"
        )


@pytest.mark.asyncio
async def test_delete_project_success(mock_client_manager, mock_context):
    """Test successful project deletion."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection": "project_proj_123",
    }
    project_storage.delete_project.return_value = True

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.delete_collection.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test deletion
    result = await registered_tools["delete_project"]("proj_123", True, mock_context)

    assert result.status == "deleted"
    assert result.details["project_id"] == "proj_123"

    # Verify calls
    project_storage.get_project.assert_called_once_with("proj_123")
    project_storage.delete_project.assert_called_once_with("proj_123")
    qdrant.delete_collection.assert_called_once_with("project_proj_123")


@pytest.mark.asyncio
async def test_delete_project_not_found(mock_client_manager, mock_context):
    """Test deletion of non-existent project."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test deletion of non-existent project
    with pytest.raises(ValueError, match="Project .* not found"):
        await registered_tools["delete_project"]("missing_proj", True, mock_context)


@pytest.mark.asyncio
async def test_update_project_description_success(mock_client_manager, mock_context):
    """Test successful project description update."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "Original description",
        "updated_at": "2024-01-01T00:00:00Z",
    }
    project_storage.update_project.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test update
    result = await registered_tools["update_project"](
        "proj_123", None, "Updated description", mock_context
    )

    assert result.name == "Test Project"
    assert result.description == "Updated description"

    # Verify calls
    project_storage.update_project.assert_called_once_with(
        "proj_123", {"description": "Updated description"}
    )


@pytest.mark.asyncio
async def test_list_projects_success(mock_client_manager, mock_context):
    """Test successful project listing."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.list_projects.return_value = [
        {
            "id": "proj_1",
            "name": "Project 1",
            "quality_tier": "economy",
            "created_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "proj_2",
            "name": "Project 2",
            "quality_tier": "premium",
            "created_at": "2024-01-02T00:00:00Z",
        },
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test listing
    result = await registered_tools["list_projects"](mock_context)

    # Should return all projects as a list of ProjectInfo objects
    assert len(result) == 2
    assert result[0].id == "proj_1"
    assert result[0].quality_tier == "economy"
    assert result[1].id == "proj_2"
    assert result[1].quality_tier == "premium"


@pytest.mark.asyncio
async def test_list_projects_all(mock_client_manager, mock_context):
    """Test listing all projects."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.list_projects.return_value = [
        {"id": "proj_1", "name": "Project 1", "quality_tier": "economy"},
        {"id": "proj_2", "name": "Project 2", "quality_tier": "balanced"},
        {"id": "proj_3", "name": "Project 3", "quality_tier": "premium"},
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test listing all
    result = await registered_tools["list_projects"](mock_context)

    assert len(result) == 3
    assert result[0].id == "proj_1"
    assert result[1].id == "proj_2"
    assert result[2].id == "proj_3"


@pytest.mark.asyncio
async def test_get_project_details_success(mock_client_manager, mock_context):
    """Test getting project details."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "A test project",
        "quality_tier": "balanced",
        "collection": "project_proj_123",
        "created_at": "2024-01-01T00:00:00Z",
        "urls": ["https://example.com"],
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test get details
    result = await registered_tools["get_project_details"](
        project_id="proj_123",
        ctx=mock_context,
    )

    assert result["project"]["id"] == "proj_123"
    assert result["project"]["name"] == "Test Project"
    assert len(result["project"]["urls"]) == 1


@pytest.mark.asyncio
async def test_get_project_details_not_found(mock_client_manager, mock_context):
    """Test getting details of non-existent project."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test with non-existent project
    with pytest.raises(ValueError, match="Project not found"):
        await registered_tools["get_project_details"](
            project_id="missing_proj",
            ctx=mock_context,
        )


@pytest.mark.asyncio
async def test_add_project_urls_success(mock_client_manager, mock_context):
    """Test adding URLs to project."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection": "project_proj_123",
        "urls": ["https://example.com/existing"],
    }

    crawling = mock_client_manager.get_crawling_service()
    document_service = mock_client_manager.get_document_service()

    # Mock successful crawl and document addition
    new_urls = ["https://example.com/new1", "https://example.com/new2"]

    crawling.crawl_url.side_effect = [
        {
            "url": url,
            "content": f"Content for {url}",
            "title": f"Title for {url}",
        }
        for url in new_urls
    ]

    document_service.add_document.return_value = {
        "document_id": "doc_new",
        "chunks": 3,
        "tokens": 300,
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test adding URLs
    result = await registered_tools["add_project_urls"](
        project_id="proj_123",
        urls=new_urls,
        ctx=mock_context,
    )

    assert result["status"] == "urls_added"
    assert result["urls_added"] == len(new_urls)
    assert result["_total_urls"] == 3  # 1 existing + 2 new

    # Verify processing
    assert crawling.crawl_url.call_count == len(new_urls)
    assert document_service.add_document.call_count == len(new_urls)


@pytest.mark.asyncio
async def test_add_project_urls_duplicate_handling(mock_client_manager, mock_context):
    """Test adding duplicate URLs to project."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection": "project_proj_123",
        "urls": ["https://example.com/existing"],
    }
    project_storage.update_project = AsyncMock()

    # Setup crawling and document services (even though only new URL will be processed)
    crawling = mock_client_manager.get_crawling_service()
    crawling.crawl_url = AsyncMock(
        return_value={
            "url": "https://example.com/new",
            "content": "New content",
            "title": "New Title",
        }
    )

    document_service = mock_client_manager.get_document_service()
    document_service.add_document = AsyncMock()

    register_tools(mock_mcp, mock_client_manager)

    # Test adding duplicate URL
    result = await registered_tools["add_project_urls"](
        project_id="proj_123",
        urls=["https://example.com/existing", "https://example.com/new"],
        ctx=mock_context,
    )

    # Should only add the new URL
    assert result["urls_added"] == 1
    assert result["_total_urls"] == 2


@pytest.mark.asyncio
async def test_export_project_success(mock_client_manager, mock_context):
    """Test successful project export."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "A test project",
        "quality_tier": "balanced",
        "collection": "project_proj_123",
        "created_at": "2024-01-01T00:00:00Z",
        "urls": ["https://example.com"],
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test export
    with patch.object(json, "dumps") as mock_json_dumps:
        mock_json_dumps.return_value = '{"project_data": "exported"}'

        result = await registered_tools["export_project"](
            project_id="proj_123",
            format="json",
            ctx=mock_context,
        )

        assert result["status"] == "exported"
        assert result["format"] == "json"
        assert "data" in result

        # Verify JSON export was called
        mock_json_dumps.assert_called_once()


@pytest.mark.asyncio
async def test_export_project_yaml_format(mock_client_manager, mock_context):
    """Test project export in YAML format."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "quality_tier": "premium",
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test YAML export
    with patch.object(yaml, "dump") as mock_yaml_dump:
        mock_yaml_dump.return_value = "project_data: exported"

        result = await registered_tools["export_project"](
            project_id="proj_123",
            format="yaml",
            ctx=mock_context,
        )

        assert result["format"] == "yaml"
        mock_yaml_dump.assert_called_once()


@pytest.mark.asyncio
async def test_export_project_invalid_format(mock_client_manager, mock_context):
    """Test project export with invalid format."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {"id": "proj_123"}

    register_tools(mock_mcp, mock_client_manager)

    # Test with invalid format
    with pytest.raises(ValueError, match="Unsupported format"):
        await registered_tools["export_project"](
            project_id="proj_123",
            format="xml",  # Invalid format
            ctx=mock_context,
        )


@pytest.mark.asyncio
async def test_error_handling_collection_creation_failure(
    mock_client_manager, mock_context
):
    """Test error handling when collection creation fails."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()

    # Mock both qdrant_service direct access and get_qdrant_service()
    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection.side_effect = Exception("Collection creation failed")
    # Ensure direct attribute access also points to the same mock
    mock_client_manager.qdrant_service = qdrant

    # Project should be deleted on collection creation failure
    project_storage.delete_project.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Mock uuid4 to return predictable UUID
    with patch("src.mcp_tools.tools.projects.uuid4") as mock_uuid:
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="proj_123")

        # Test that exception is propagated
        request = ProjectRequest(name="Test Project", quality_tier="balanced")
        with pytest.raises(Exception, match="Collection creation failed"):
            await registered_tools["create_project"](request, mock_context)

        # Verify cleanup was attempted
        project_storage.delete_project.assert_called_once_with("proj_123")


@pytest.mark.asyncio
async def test_without_context_parameter(mock_client_manager):
    """Test tools work without context parameter."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.list_projects.return_value = [
        {"id": "proj_1", "name": "Project 1", "quality_tier": "economy"},
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test without context
    result = await registered_tools["list_projects"](ctx=None)

    assert len(result) == 1
    assert result[0].id == "proj_1"


@pytest.mark.asyncio
async def test_quality_tier_enum_handling(mock_client_manager, mock_context):
    """Test handling of QualityTier enum values."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.create_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "quality_tier": "premium",
    }

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection_with_quality.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test with enum value
    request = ProjectRequest(name="Test Project", quality_tier="premium")

    result = await registered_tools["create_project"](request, mock_context)

    assert result.quality_tier == "premium"


# Additional comprehensive tests for error handling and edge cases


@pytest.mark.asyncio
async def test_create_project_url_processing_failures(
    mock_client_manager, mock_context
):
    """Test project creation with URL processing failures."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.save_project = AsyncMock()
    project_storage.update_project = AsyncMock()

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection = AsyncMock()

    # Setup crawl manager to fail for some URLs
    crawl_manager = await mock_client_manager.get_crawl_manager()
    crawl_manager.scrape_url.side_effect = [
        {"url": "https://example.com/doc1", "success": True, "content": "content1"},
        Exception("Network error"),  # Second URL fails
        {
            "url": "https://example.com/doc3",
            "success": False,
            "error": "Timeout",
        },  # Third URL fails with success=False
    ]

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Test Project",
        quality_tier="balanced",
        urls=[
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3",
        ],
    )

    # Mock uuid4 for predictable ID
    with patch("src.mcp_tools.tools.projects.uuid4") as mock_uuid:
        mock_uuid.return_value = Mock()
        mock_uuid.return_value.__str__ = Mock(return_value="proj_123")

        result = await registered_tools["create_project"](request, mock_context)

        # Should succeed with only 1 successful URL
        assert result.document_count == 1  # Only first URL succeeded
        assert result.urls == request.urls  # All URLs should be stored

        # Verify warning logs for failed URLs
        mock_context.warning.assert_called()
        warning_calls = [call.args[0] for call in mock_context.warning.call_args_list]
        assert any("Failed to process URL" in call for call in warning_calls)


@pytest.mark.asyncio
async def test_list_projects_with_collection_stats_failure(
    mock_client_manager, mock_context
):
    """Test list_projects when collection stats retrieval fails."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.list_projects.return_value = [
        {
            "id": "proj_1",
            "name": "Project 1",
            "collection": "project_proj_1",
            "quality_tier": "economy",
        }
    ]

    # Make collection info retrieval fail
    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.get_collection_info.side_effect = Exception("Qdrant connection error")

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["list_projects"](mock_context)

    assert len(result) == 1
    assert result[0].vector_count == 0  # Should default to 0 on error
    assert result[0].indexed_count == 0  # Should default to 0 on error

    # Verify warning was logged
    mock_context.warning.assert_called()
    warning_call = mock_context.warning.call_args[0][0]
    assert "Failed to get stats for project" in warning_call


@pytest.mark.asyncio
async def test_search_project_not_found(mock_client_manager, mock_context):
    """Test search_project with non-existent project."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test search with non-existent project
    with pytest.raises(ValueError, match="Project .* not found"):
        await registered_tools["search_project"](
            "missing_proj", "test query", 10, None, mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_search_project_success(mock_client_manager, mock_context):
    """Test successful project search."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection": "project_proj_123",
    }

    # Mock embedding manager
    embedding_manager = Mock()
    embedding_manager.generate_embeddings = AsyncMock(
        return_value={
            "embeddings": [[0.1, 0.2, 0.3]],
            "sparse_embeddings": [[0.0, 0.1, 0.0, 0.2]],
        }
    )
    mock_client_manager.embedding_manager = embedding_manager

    # Mock qdrant service
    qdrant = await mock_client_manager.get_qdrant_service()
    mock_point = Mock()
    mock_point.id = "doc_1"
    mock_point.score = 0.95
    mock_point.payload = {
        "content": "Test document content",
        "url": "https://example.com/doc1",
        "title": "Test Document",
    }
    qdrant.hybrid_search = AsyncMock(return_value=[mock_point])

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["search_project"](
        "proj_123", "test query", 10, SearchStrategy.HYBRID, mock_context
    )

    assert len(result) == 1
    assert result[0].id == "doc_1"
    assert result[0].content == "Test document content"
    assert result[0].score == 0.95

    # Verify service calls
    embedding_manager.generate_embeddings.assert_called_once()
    qdrant.hybrid_search.assert_called_once()


@pytest.mark.asyncio
async def test_update_project_not_found(mock_client_manager, mock_context):
    """Test update_project with non-existent project."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test update with non-existent project
    with pytest.raises(ValueError, match="Project .* not found"):
        await registered_tools["update_project"](
            "missing_proj", "New Name", "New Description", mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_update_project_name_only(mock_client_manager, mock_context):
    """Test updating only project name."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Old Name",
        "description": "Original description",
    }
    project_storage.update_project = AsyncMock()

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["update_project"](
        "proj_123", "New Name", None, mock_context
    )

    assert result.name == "New Name"
    assert result.description == "Original description"

    # Verify update called with only name
    project_storage.update_project.assert_called_once_with(
        "proj_123", {"name": "New Name"}
    )


@pytest.mark.asyncio
async def test_update_project_no_changes(mock_client_manager, mock_context):
    """Test update_project with no actual changes."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "Test description",
    }
    project_storage.update_project = AsyncMock()

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["update_project"](
        "proj_123", None, None, mock_context
    )

    assert result.name == "Test Project"
    assert result.description == "Test description"

    # Verify no update call was made
    project_storage.update_project.assert_not_called()


@pytest.mark.asyncio
async def test_delete_project_collection_deletion_failure(
    mock_client_manager, mock_context
):
    """Test delete_project when collection deletion fails."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection": "project_proj_123",
    }
    project_storage.delete_project = AsyncMock()

    # Make collection deletion fail
    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.delete_collection.side_effect = Exception("Collection deletion failed")

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["delete_project"]("proj_123", True, mock_context)

    # Should still succeed and delete project even if collection deletion fails
    assert result.status == "deleted"
    assert result.details["project_id"] == "proj_123"

    # Verify project was still deleted
    project_storage.delete_project.assert_called_once_with("proj_123")

    # Verify warning was logged for collection deletion failure
    mock_context.warning.assert_called()
    warning_call = mock_context.warning.call_args[0][0]
    assert "Failed to delete collection" in warning_call


@pytest.mark.asyncio
async def test_delete_project_without_collection_deletion(
    mock_client_manager, mock_context
):
    """Test delete_project without deleting collection."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection": "project_proj_123",
    }
    project_storage.delete_project = AsyncMock()

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.delete_collection = AsyncMock()

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["delete_project"]("proj_123", False, mock_context)

    assert result.status == "deleted"
    assert result.details["collection_deleted"] == "False"

    # Verify collection deletion was NOT called
    qdrant.delete_collection.assert_not_called()

    # Verify project was deleted
    project_storage.delete_project.assert_called_once_with("proj_123")


@pytest.mark.asyncio
async def test_search_project_embedding_failure(mock_client_manager, mock_context):
    """Test search_project when embedding generation fails."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "collection": "project_proj_123",
    }

    # Make embedding generation fail
    embedding_manager = Mock()
    embedding_manager.generate_embeddings = AsyncMock(
        side_effect=Exception("Embedding error")
    )
    mock_client_manager.embedding_manager = embedding_manager

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(Exception, match="Embedding error"):
        await registered_tools["search_project"](
            "proj_123", "test query", 10, None, mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_search_project_qdrant_failure(mock_client_manager, mock_context):
    """Test search_project when Qdrant search fails."""
    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = await mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "collection": "project_proj_123",
    }

    # Mock successful embedding generation
    embedding_manager = Mock()
    embedding_manager.generate_embeddings = AsyncMock(
        return_value={"embeddings": [[0.1, 0.2, 0.3]], "sparse_embeddings": None}
    )
    mock_client_manager.embedding_manager = embedding_manager

    # Make Qdrant search fail
    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.hybrid_search.side_effect = Exception("Qdrant search error")

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(Exception, match="Qdrant search error"):
        await registered_tools["search_project"](
            "proj_123", "test query", 10, None, mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()
