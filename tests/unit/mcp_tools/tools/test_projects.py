"""Tests for MCP project management tools using ProjectStorage."""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest
from src.config.enums import SearchStrategy
from src.mcp_tools.models.requests import ProjectRequest
from src.mcp_tools.models.responses import OperationStatus
from src.mcp_tools.models.responses import ProjectInfo
from src.mcp_tools.models.responses import SearchResult


@pytest.fixture
def mock_context():
    """Create a mock context for testing."""
    context = Mock()
    context.info = AsyncMock()
    context.debug = AsyncMock()
    context.warning = AsyncMock()
    context.error = AsyncMock()
    return context


@pytest.fixture
def mock_project_storage():
    """Create a mock project storage."""
    storage = Mock()
    storage.save_project = AsyncMock()
    storage.update_project = AsyncMock()
    storage.delete_project = AsyncMock()
    storage.get_project = AsyncMock()
    storage.list_projects = AsyncMock()
    return storage


@pytest.fixture
def mock_client_manager(mock_project_storage):
    """Create a mock client manager with all required services."""
    manager = Mock()

    # Mock get_project_storage to return our mock
    manager.get_project_storage = AsyncMock(return_value=mock_project_storage)

    # Mock Qdrant service
    mock_qdrant = Mock()
    mock_qdrant.create_collection = AsyncMock()
    mock_qdrant.get_collection_info = AsyncMock()
    mock_qdrant.delete_collection = AsyncMock()
    mock_qdrant.hybrid_search = AsyncMock()
    manager.qdrant_service = mock_qdrant

    # Mock crawl manager
    mock_crawl = Mock()
    mock_crawl.crawl_single = AsyncMock()
    manager.crawl_manager = mock_crawl

    # Mock embedding manager
    mock_embedding = Mock()
    mock_embedding.generate_embeddings = AsyncMock()
    manager.embedding_manager = mock_embedding

    return manager


def register_tool(mock_mcp):
    """Helper to capture registered tools."""
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool
    return registered_tools


class TestCreateProject:
    """Test create_project functionality."""

    @pytest.mark.asyncio
    async def test_create_basic_project(self, mock_client_manager, mock_context):
        """Test creating a basic project."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        request = ProjectRequest(
            name="Test Project",
            description="A test project",
            quality_tier="balanced",
            urls=[],
        )

        result = await tools["create_project"](request, mock_context)

        # Verify result
        assert isinstance(result, ProjectInfo)
        assert result.name == "Test Project"
        assert result.description == "A test project"
        assert result.quality_tier == "balanced"
        assert result.document_count == 0
        assert result.collection.startswith("project_")

        # Verify storage was called
        mock_client_manager.get_project_storage.assert_called_once()
        storage = mock_client_manager.get_project_storage.return_value
        storage.save_project.assert_called_once()

        # Get saved project data
        project_id, project_data = storage.save_project.call_args[0]
        assert project_data["name"] == "Test Project"
        assert project_data["description"] == "A test project"
        assert project_data["quality_tier"] == "balanced"

        # Verify collection creation
        mock_client_manager.qdrant_service.create_collection.assert_called_once()
        create_args = mock_client_manager.qdrant_service.create_collection.call_args[1]
        assert create_args["vector_size"] == 384  # balanced tier
        assert create_args["sparse_vector_name"] == "sparse"  # hybrid enabled
        assert create_args["enable_quantization"] is True

    @pytest.mark.asyncio
    async def test_create_premium_project(self, mock_client_manager, mock_context):
        """Test creating a premium quality project."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        request = ProjectRequest(
            name="Premium Project",
            description="Premium tier project",
            quality_tier="premium",
            urls=[],
        )

        result = await tools["create_project"](request, mock_context)

        # Verify premium settings
        assert result.quality_tier == "premium"

        create_args = mock_client_manager.qdrant_service.create_collection.call_args[1]
        assert create_args["vector_size"] == 1536  # premium tier
        assert (
            create_args["enable_quantization"] is False
        )  # no quantization for premium

    @pytest.mark.asyncio
    async def test_create_project_with_urls(self, mock_client_manager, mock_context):
        """Test creating a project with initial URLs."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        # Mock successful crawl
        mock_result = Mock()
        mock_result.markdown = "Document content"
        mock_client_manager.crawl_manager.crawl_single.return_value = mock_result

        request = ProjectRequest(
            name="Project with URLs",
            description="Has URLs",
            quality_tier="balanced",
            urls=["https://example.com/doc1", "https://example.com/doc2"],
        )

        result = await tools["create_project"](request, mock_context)

        # Verify URLs were processed
        assert result.urls == ["https://example.com/doc1", "https://example.com/doc2"]
        assert result.document_count == 2

        # Verify crawl was called
        assert mock_client_manager.crawl_manager.crawl_single.call_count == 2

        # Verify storage update
        storage = mock_client_manager.get_project_storage.return_value
        assert storage.update_project.called


class TestListProjects:
    """Test list_projects functionality."""

    @pytest.mark.asyncio
    async def test_list_projects_success(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test listing projects."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        # Setup test projects
        projects = [
            {
                "id": "proj1",
                "name": "Project 1",
                "description": "First project",
                "collection": "project_proj1",
                "quality_tier": "balanced",
                "document_count": 5,
                "urls": ["https://example.com/1"],
                "created_at": datetime.now(UTC).isoformat(),
            },
            {
                "id": "proj2",
                "name": "Project 2",
                "description": "Second project",
                "collection": "project_proj2",
                "quality_tier": "premium",
                "document_count": 3,
                "urls": ["https://example.com/2"],
                "created_at": datetime.now(UTC).isoformat(),
            },
        ]
        mock_project_storage.list_projects.return_value = projects

        # Mock collection info
        mock_info = Mock()
        mock_info.vectors_count = 10
        mock_info.indexed_vectors_count = 8
        mock_client_manager.qdrant_service.get_collection_info.return_value = mock_info

        result = await tools["list_projects"](mock_context)

        # Verify results
        assert len(result) == 2
        assert all(isinstance(p, ProjectInfo) for p in result)
        assert result[0].name == "Project 1"
        assert result[1].name == "Project 2"
        assert result[0].vector_count == 10
        assert result[0].indexed_count == 8

    @pytest.mark.asyncio
    async def test_list_empty_projects(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test listing when no projects exist."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        mock_project_storage.list_projects.return_value = []

        result = await tools["list_projects"](mock_context)

        assert len(result) == 0
        assert isinstance(result, list)


class TestSearchProject:
    """Test search_project functionality."""

    @pytest.mark.asyncio
    async def test_search_project_success(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test searching within a project."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        # Setup project
        project = {
            "id": "proj1",
            "name": "Test Project",
            "collection": "project_proj1",
            "quality_tier": "balanced",
        }
        mock_project_storage.get_project.return_value = project

        # Mock embeddings
        mock_client_manager.embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "sparse_embeddings": [[0.5, 0.0, 0.7]],
        }

        # Mock search results
        mock_point = Mock()
        mock_point.id = "doc1"
        mock_point.score = 0.95
        mock_point.payload = {
            "content": "Test content",
            "url": "https://example.com/doc",
            "title": "Test Document",
        }
        mock_client_manager.qdrant_service.hybrid_search.return_value = [mock_point]

        result = await tools["search_project"](
            project_id="proj1",
            query="test query",
            limit=10,
            strategy=SearchStrategy.HYBRID,
            ctx=mock_context,
        )

        # Verify results
        assert len(result) == 1
        assert isinstance(result[0], SearchResult)
        assert result[0].id == "doc1"
        assert result[0].score == 0.95
        assert result[0].content == "Test content"

    @pytest.mark.asyncio
    async def test_search_project_not_found(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test searching when project doesn't exist."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        mock_project_storage.get_project.return_value = None

        with pytest.raises(ValueError, match="Project proj1 not found"):
            await tools["search_project"](
                project_id="proj1", query="test", ctx=mock_context
            )


class TestUpdateProject:
    """Test update_project functionality."""

    @pytest.mark.asyncio
    async def test_update_project_success(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test updating project metadata."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        # Setup project
        project = {
            "id": "proj1",
            "name": "Original Name",
            "description": "Original Description",
            "quality_tier": "balanced",
            "collection": "project_proj1",
        }
        mock_project_storage.get_project.return_value = project.copy()

        result = await tools["update_project"](
            project_id="proj1",
            name="Updated Name",
            description="Updated Description",
            ctx=mock_context,
        )

        # Verify result
        assert result.name == "Updated Name"
        assert result.description == "Updated Description"

        # Verify storage update
        mock_project_storage.update_project.assert_called_once()
        update_args = mock_project_storage.update_project.call_args[0]
        assert update_args[0] == "proj1"
        assert update_args[1]["name"] == "Updated Name"
        assert update_args[1]["description"] == "Updated Description"

    @pytest.mark.asyncio
    async def test_update_project_not_found(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test updating non-existent project."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        mock_project_storage.get_project.return_value = None

        with pytest.raises(ValueError, match="Project proj1 not found"):
            await tools["update_project"](
                project_id="proj1", name="New Name", ctx=mock_context
            )


class TestDeleteProject:
    """Test delete_project functionality."""

    @pytest.mark.asyncio
    async def test_delete_project_with_collection(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test deleting project and collection."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        # Setup project
        project = {"id": "proj1", "name": "Test Project", "collection": "project_proj1"}
        mock_project_storage.get_project.return_value = project

        result = await tools["delete_project"](
            project_id="proj1", delete_collection=True, ctx=mock_context
        )

        # Verify result
        assert isinstance(result, OperationStatus)
        assert result.status == "deleted"
        assert result.details["project_id"] == "proj1"
        assert result.details["collection_deleted"] == "True"

        # Verify deletion calls
        mock_client_manager.qdrant_service.delete_collection.assert_called_once_with(
            "project_proj1"
        )
        mock_project_storage.delete_project.assert_called_once_with("proj1")

    @pytest.mark.asyncio
    async def test_delete_project_without_collection(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test deleting project without collection."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        project = {"id": "proj1", "name": "Test Project", "collection": "project_proj1"}
        mock_project_storage.get_project.return_value = project

        await tools["delete_project"](
            project_id="proj1", delete_collection=False, ctx=mock_context
        )

        # Verify collection was not deleted
        mock_client_manager.qdrant_service.delete_collection.assert_not_called()

        # Verify project was deleted
        mock_project_storage.delete_project.assert_called_once_with("proj1")

    @pytest.mark.asyncio
    async def test_delete_project_collection_error_continues(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test that project deletion continues even if collection deletion fails."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        project = {"id": "proj1", "name": "Test Project", "collection": "project_proj1"}
        mock_project_storage.get_project.return_value = project

        # Make collection deletion fail
        mock_client_manager.qdrant_service.delete_collection.side_effect = Exception(
            "Collection error"
        )

        result = await tools["delete_project"](
            project_id="proj1", delete_collection=True, ctx=mock_context
        )

        # Verify project was still deleted
        assert result.status == "deleted"
        mock_project_storage.delete_project.assert_called_once_with("proj1")

        # Verify warning was logged
        mock_context.warning.assert_called()


class TestToolsWithoutContext:
    """Test that all tools work without context parameter."""

    @pytest.mark.asyncio
    async def test_all_tools_without_context(
        self, mock_client_manager, mock_project_storage
    ):
        """Test all tools work with ctx=None."""
        from src.mcp_tools.tools.projects import register_tools

        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)

        # Setup basic project
        project = {
            "id": "proj1",
            "name": "Test Project",
            "description": "Test",
            "collection": "project_proj1",
            "quality_tier": "balanced",
        }
        mock_project_storage.get_project.return_value = project
        mock_project_storage.list_projects.return_value = [project]

        # Mock other services
        mock_info = Mock()
        mock_info.vectors_count = 5
        mock_info.indexed_vectors_count = 5
        mock_client_manager.qdrant_service.get_collection_info.return_value = mock_info

        mock_client_manager.embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]]
        }
        mock_client_manager.qdrant_service.hybrid_search.return_value = []

        # Test all tools with ctx=None

        # Create project
        request = ProjectRequest(
            name="New Project", description="Test", quality_tier="balanced", urls=[]
        )
        result = await tools["create_project"](request, ctx=None)
        assert isinstance(result, ProjectInfo)

        # List projects
        result = await tools["list_projects"](ctx=None)
        assert isinstance(result, list)

        # Search project
        result = await tools["search_project"](
            project_id="proj1", query="test", ctx=None
        )
        assert isinstance(result, list)

        # Update project
        result = await tools["update_project"](
            project_id="proj1", name="Updated", ctx=None
        )
        assert isinstance(result, ProjectInfo)

        # Delete project
        result = await tools["delete_project"](
            project_id="proj1", delete_collection=False, ctx=None
        )
        assert isinstance(result, OperationStatus)
