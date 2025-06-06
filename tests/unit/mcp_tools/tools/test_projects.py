"""Tests for MCP project management tools."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.enums import QualityTier

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
    manager.get_project_storage = Mock(return_value=mock_project_storage)

    # Mock other services
    mock_qdrant = Mock()
    mock_qdrant.create_collection_with_quality = AsyncMock()
    mock_qdrant.delete_collection = AsyncMock()
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

    mock_crawling = Mock()
    mock_crawling.crawl_url = AsyncMock()
    manager.get_crawling_service = Mock(return_value=mock_crawling)

    mock_document_service = Mock()
    mock_document_service.add_document = AsyncMock()
    manager.get_document_service = Mock(return_value=mock_document_service)

    return manager


@pytest.mark.asyncio
async def test_project_tools_registration(mock_client_manager):
    """Test that project tools are properly registered."""
    from src.mcp_tools.tools.projects import register_tools

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
        "update_project_description",
        "list_projects",
        "get_project_details",
        "add_project_urls",
        "export_project",
    ]

    for tool_name in expected_tools:
        assert tool_name in registered_tools, f"Tool {tool_name} not registered"


@pytest.mark.asyncio
async def test_create_project_success(mock_client_manager, mock_context):
    """Test successful project creation."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
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
    result = await registered_tools["create_project"](
        name="Test Project",
        description="A test project",
        quality_tier="balanced",
        urls=["https://example.com/docs"],
        ctx=mock_context,
    )

    assert result["id"] == "proj_123"
    assert result["name"] == "Test Project"
    assert result["status"] == "created"
    assert "collection_name" in result

    # Verify calls
    project_storage.create_project.assert_called_once()
    qdrant.create_collection_with_quality.assert_called_once()
    mock_context.info.assert_called()


@pytest.mark.asyncio
async def test_create_project_with_initial_urls(mock_client_manager, mock_context):
    """Test project creation with initial URLs."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.create_project.return_value = {
        "id": "proj_456",
        "name": "Docs Project",
        "quality_tier": "premium",
        "created_at": "2024-01-01T00:00:00Z",
        "status": "active",
    }

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection_with_quality.return_value = True

    crawling = mock_client_manager.get_crawling_service()
    crawling.crawl_url.return_value = {
        "url": "https://example.com/docs",
        "content": "Documentation content",
        "title": "Example Docs",
        "metadata": {"type": "documentation"},
    }

    document_service = mock_client_manager.get_document_service()
    document_service.add_document.return_value = {
        "document_id": "doc_789",
        "chunks": 5,
        "tokens": 500,
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test with initial URLs
    urls = ["https://example.com/docs", "https://example.com/api"]

    # Mock crawl results for both URLs
    crawling.crawl_url.side_effect = [
        {
            "url": url,
            "content": f"Content for {url}",
            "title": f"Title for {url}",
            "metadata": {"type": "documentation"},
        }
        for url in urls
    ]

    result = await registered_tools["create_project"](
        name="Docs Project",
        description="Documentation project",
        quality_tier="premium",
        urls=urls,
        ctx=mock_context,
    )

    assert result["status"] == "created"
    assert result["documents_added"] == len(urls)
    assert result["quality_tier"] == "premium"

    # Verify URL processing
    assert crawling.crawl_url.call_count == len(urls)
    assert document_service.add_document.call_count == len(urls)


@pytest.mark.asyncio
async def test_create_project_invalid_quality_tier(mock_client_manager, mock_context):
    """Test project creation with invalid quality tier."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool
    register_tools(mock_mcp, mock_client_manager)

    # Test with invalid quality tier
    with pytest.raises(ValueError, match="Invalid quality tier"):
        await registered_tools["create_project"](
            name="Test Project",
            description="Test",
            quality_tier="invalid_tier",
            ctx=mock_context,
        )


@pytest.mark.asyncio
async def test_delete_project_success(mock_client_manager, mock_context):
    """Test successful project deletion."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection_name": "project_proj_123",
    }
    project_storage.delete_project.return_value = True

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.delete_collection.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test deletion
    result = await registered_tools["delete_project"](
        project_id="proj_123",
        ctx=mock_context,
    )

    assert result["status"] == "deleted"
    assert result["project_id"] == "proj_123"

    # Verify calls
    project_storage.get_project.assert_called_once_with("proj_123")
    project_storage.delete_project.assert_called_once_with("proj_123")
    qdrant.delete_collection.assert_called_once_with("project_proj_123")


@pytest.mark.asyncio
async def test_delete_project_not_found(mock_client_manager, mock_context):
    """Test deletion of non-existent project."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = None

    register_tools(mock_mcp, mock_client_manager)

    # Test deletion of non-existent project
    with pytest.raises(ValueError, match="Project not found"):
        await registered_tools["delete_project"](
            project_id="missing_proj",
            ctx=mock_context,
        )


@pytest.mark.asyncio
async def test_update_project_description_success(mock_client_manager, mock_context):
    """Test successful project description update."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.update_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "Updated description",
        "updated_at": "2024-01-02T00:00:00Z",
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test update
    result = await registered_tools["update_project_description"](
        project_id="proj_123",
        description="Updated description",
        ctx=mock_context,
    )

    assert result["status"] == "updated"
    assert result["project"]["description"] == "Updated description"

    # Verify calls
    project_storage.update_project.assert_called_once_with(
        project_id="proj_123",
        description="Updated description",
    )


@pytest.mark.asyncio
async def test_list_projects_success(mock_client_manager, mock_context):
    """Test successful project listing."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
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
    result = await registered_tools["list_projects"](
        quality_tier="premium",
        ctx=mock_context,
    )

    assert len(result["projects"]) == 1  # Only premium project
    assert result["projects"][0]["id"] == "proj_2"
    assert result["total"] == 1


@pytest.mark.asyncio
async def test_list_projects_all(mock_client_manager, mock_context):
    """Test listing all projects."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.list_projects.return_value = [
        {"id": "proj_1", "name": "Project 1", "quality_tier": "economy"},
        {"id": "proj_2", "name": "Project 2", "quality_tier": "balanced"},
        {"id": "proj_3", "name": "Project 3", "quality_tier": "premium"},
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test listing all
    result = await registered_tools["list_projects"](ctx=mock_context)

    assert len(result["projects"]) == 3
    assert result["total"] == 3


@pytest.mark.asyncio
async def test_get_project_details_success(mock_client_manager, mock_context):
    """Test getting project details."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "A test project",
        "quality_tier": "balanced",
        "collection_name": "project_proj_123",
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
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
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
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection_name": "project_proj_123",
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
    assert result["total_urls"] == 3  # 1 existing + 2 new

    # Verify processing
    assert crawling.crawl_url.call_count == len(new_urls)
    assert document_service.add_document.call_count == len(new_urls)


@pytest.mark.asyncio
async def test_add_project_urls_duplicate_handling(mock_client_manager, mock_context):
    """Test adding duplicate URLs to project."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "collection_name": "project_proj_123",
        "urls": ["https://example.com/existing"],
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test adding duplicate URL
    result = await registered_tools["add_project_urls"](
        project_id="proj_123",
        urls=["https://example.com/existing", "https://example.com/new"],
        ctx=mock_context,
    )

    # Should only add the new URL
    assert result["urls_added"] == 1
    assert result["total_urls"] == 2


@pytest.mark.asyncio
async def test_export_project_success(mock_client_manager, mock_context):
    """Test successful project export."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "description": "A test project",
        "quality_tier": "balanced",
        "collection_name": "project_proj_123",
        "created_at": "2024-01-01T00:00:00Z",
        "urls": ["https://example.com"],
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test export
    with patch("src.mcp_tools.tools.projects.json.dumps") as mock_json_dumps:
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
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.get_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "quality_tier": "premium",
    }

    register_tools(mock_mcp, mock_client_manager)

    # Test YAML export
    with patch("src.mcp_tools.tools.projects.yaml") as mock_yaml:
        mock_yaml.dump.return_value = "project_data: exported"

        result = await registered_tools["export_project"](
            project_id="proj_123",
            format="yaml",
            ctx=mock_context,
        )

        assert result["format"] == "yaml"
        mock_yaml.dump.assert_called_once()


@pytest.mark.asyncio
async def test_export_project_invalid_format(mock_client_manager, mock_context):
    """Test project export with invalid format."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    project_storage = mock_client_manager.get_project_storage()
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
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.create_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
    }

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection_with_quality.side_effect = Exception(
        "Collection creation failed"
    )

    # Project should be deleted on collection creation failure
    project_storage.delete_project.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is propagated
    with pytest.raises(Exception, match="Collection creation failed"):
        await registered_tools["create_project"](
            name="Test Project",
            quality_tier="balanced",
            ctx=mock_context,
        )

    # Verify cleanup was attempted
    project_storage.delete_project.assert_called_once_with("proj_123")


@pytest.mark.asyncio
async def test_without_context_parameter(mock_client_manager):
    """Test tools work without context parameter."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.list_projects.return_value = [
        {"id": "proj_1", "name": "Project 1", "quality_tier": "economy"},
    ]

    register_tools(mock_mcp, mock_client_manager)

    # Test without context
    result = await registered_tools["list_projects"](ctx=None)

    assert len(result["projects"]) == 1
    assert result["projects"][0]["id"] == "proj_1"


@pytest.mark.asyncio
async def test_quality_tier_enum_handling(mock_client_manager, mock_context):
    """Test handling of QualityTier enum values."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    project_storage = mock_client_manager.get_project_storage()
    project_storage.create_project.return_value = {
        "id": "proj_123",
        "name": "Test Project",
        "quality_tier": QualityTier.PREMIUM.value,
    }

    qdrant = await mock_client_manager.get_qdrant_service()
    qdrant.create_collection_with_quality.return_value = True

    register_tools(mock_mcp, mock_client_manager)

    # Test with enum value
    result = await registered_tools["create_project"](
        name="Test Project",
        quality_tier="premium",
        ctx=mock_context,
    )

    assert result["quality_tier"] == "premium"
