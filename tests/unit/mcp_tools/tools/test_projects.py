"""Tests for MCP project management tools using ProjectStorage."""

from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

<<<<<<< HEAD
=======
from datetime import UTC
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock

>>>>>>> dev
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

<<<<<<< HEAD
    # Mock get_project_storage to return our mock
    manager.get_project_storage = AsyncMock(return_value=mock_project_storage)
=======
    # Mock projects dictionary
    manager.projects = {}

    # Mock project storage
    mock_project_storage = Mock()
    mock_project_storage.save_project = AsyncMock()
    mock_project_storage.update_project = AsyncMock()
    mock_project_storage.delete_project = AsyncMock()
    manager.project_storage = mock_project_storage
>>>>>>> dev

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


<<<<<<< HEAD
def register_tool(mock_mcp):
    """Helper to capture registered tools."""
=======
@pytest.mark.asyncio
async def test_project_tools_registration(mock_client_manager, mock_context):
    """Test that project tools are properly registered."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
>>>>>>> dev
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool
<<<<<<< HEAD
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
=======

    register_tools(mock_mcp, mock_client_manager)

    expected_tools = [
        "create_project", "list_projects", "search_project",
        "update_project", "delete_project"
    ]

    for tool in expected_tools:
        assert tool in registered_tools


@pytest.mark.asyncio
async def test_create_project_basic_success(mock_client_manager, mock_context):
    """Test successful basic project creation."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    # Test create_project function
    request = ProjectRequest(
        name="Test Project",
        description="A test project for unit testing",
        quality_tier="balanced",
        urls=[]
    )

    result = await registered_tools["create_project"](request, mock_context)

    # Verify result
    assert isinstance(result, ProjectInfo)
    assert result.name == "Test Project"
    assert result.description == "A test project for unit testing"
    assert result.quality_tier == "balanced"
    assert result.document_count == 0
    assert result.urls == []
    assert result.collection.startswith("project_")

    # Verify project was stored in memory
    assert len(mock_client_manager.projects) == 1
    project_id = list(mock_client_manager.projects.keys())[0]
    stored_project = mock_client_manager.projects[project_id]
    assert stored_project["name"] == "Test Project"
    assert stored_project["description"] == "A test project for unit testing"

    # Verify persistent storage was called
    mock_client_manager.project_storage.save_project.assert_called_once()

    # Verify collection creation with balanced tier settings
    mock_client_manager.qdrant_service.create_collection.assert_called_once()
    call_args = mock_client_manager.qdrant_service.create_collection.call_args[1]
    assert call_args["collection_name"] == stored_project["collection"]
    assert call_args["vector_size"] == 384  # balanced tier
    assert call_args["distance"] == "Cosine"
    assert call_args["sparse_vector_name"] == "sparse"  # hybrid enabled
    assert call_args["enable_quantization"] is True  # not premium

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_create_project_premium_tier(mock_client_manager, mock_context):
    """Test project creation with premium quality tier."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Premium Project",
        description="Premium quality project",
        quality_tier="premium",
        urls=[]
    )

    result = await registered_tools["create_project"](request, mock_context)

    # Verify premium tier settings
    assert result.quality_tier == "premium"

    # Verify collection creation with premium settings
    call_args = mock_client_manager.qdrant_service.create_collection.call_args[1]
    assert call_args["vector_size"] == 1536  # premium tier
    assert call_args["sparse_vector_name"] == "sparse"  # hybrid enabled
    assert call_args["enable_quantization"] is False  # premium = no quantization
>>>>>>> dev

    @pytest.mark.asyncio
    async def test_list_projects_success(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test listing projects."""
        from src.mcp_tools.tools.projects import register_tools

<<<<<<< HEAD
        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)
=======
@pytest.mark.asyncio
async def test_create_project_basic_tier(mock_client_manager, mock_context):
    """Test project creation with basic quality tier."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Basic Project",
        description="Basic quality project",
        quality_tier="economy",
        urls=[]
    )

    result = await registered_tools["create_project"](request, mock_context)

    # Verify economy tier settings
    assert result.quality_tier == "economy"

    # Verify collection creation with economy settings
    call_args = mock_client_manager.qdrant_service.create_collection.call_args[1]
    assert call_args["vector_size"] == 384  # basic tier
    assert call_args["sparse_vector_name"] is None  # no hybrid
    assert call_args["enable_quantization"] is True
>>>>>>> dev

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

<<<<<<< HEAD
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
=======
@pytest.mark.asyncio
async def test_create_project_with_urls_success(mock_client_manager, mock_context):
    """Test project creation with initial URLs that process successfully."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup successful crawl results
    mock_crawl_result = Mock()
    mock_crawl_result.markdown = "# Test Document\n\nThis is test content."
    mock_client_manager.crawl_manager.crawl_single.return_value = mock_crawl_result

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Project with URLs",
        description="Project with initial URLs",
        quality_tier="balanced",
        urls=["https://example.com/doc1", "https://example.com/doc2"]
    )

    result = await registered_tools["create_project"](request, mock_context)

    # Verify URLs were processed
    assert result.urls == ["https://example.com/doc1", "https://example.com/doc2"]
    assert result.document_count == 2  # Both URLs processed successfully

    # Verify crawl was called for each URL
    assert mock_client_manager.crawl_manager.crawl_single.call_count == 2
    mock_client_manager.crawl_manager.crawl_single.assert_any_call("https://example.com/doc1")
    mock_client_manager.crawl_manager.crawl_single.assert_any_call("https://example.com/doc2")

    # Verify project was updated with URL info
    mock_client_manager.project_storage.update_project.assert_called_once()
    update_args = mock_client_manager.project_storage.update_project.call_args[0]
    update_data = update_args[1]  # Second positional argument
    assert update_data["urls"] == ["https://example.com/doc1", "https://example.com/doc2"]
    assert update_data["document_count"] == 2


@pytest.mark.asyncio
async def test_create_project_with_urls_partial_failure(mock_client_manager, mock_context):
    """Test project creation with URLs where some fail to process."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup crawl results - first succeeds, second fails
    def crawl_side_effect(url):
        if url == "https://example.com/doc1":
            mock_result = Mock()
            mock_result.markdown = "Success content"
            return mock_result
        elif url == "https://example.com/doc2":
            raise Exception("Crawl failed")
        else:
            return None

    mock_client_manager.crawl_manager.crawl_single.side_effect = crawl_side_effect

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Project with Mixed URLs",
        description="Project with some failing URLs",
        quality_tier="balanced",
        urls=["https://example.com/doc1", "https://example.com/doc2"]
    )

    result = await registered_tools["create_project"](request, mock_context)

    # Verify only successful URL was counted
    assert result.urls == ["https://example.com/doc1", "https://example.com/doc2"]
    assert result.document_count == 1  # Only one successful

    # Verify warning was logged for failed URL
    mock_context.warning.assert_called()


@pytest.mark.asyncio
async def test_create_project_without_context(mock_client_manager):
    """Test project creation without context parameter."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="No Context Project",
        description="Project created without context",
        quality_tier="economy",
        urls=[]
    )

    # Test without ctx parameter (None)
    result = await registered_tools["create_project"](request, ctx=None)

    # Should still work without context
    assert isinstance(result, ProjectInfo)
    assert result.name == "No Context Project"


@pytest.mark.asyncio
async def test_create_project_error_handling(mock_client_manager, mock_context):
    """Test error handling in project creation."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Make collection creation fail
    mock_client_manager.qdrant_service.create_collection.side_effect = Exception("Collection creation failed")

    register_tools(mock_mcp, mock_client_manager)

    request = ProjectRequest(
        name="Failed Project",
        description="Project that will fail",
        quality_tier="balanced",
        urls=[]
    )

    with pytest.raises(Exception, match="Collection creation failed"):
        await registered_tools["create_project"](request, mock_context)

    # Verify error logging
    mock_context.error.assert_called()


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

    # Setup mock projects in memory
    project1 = {
        "id": "proj1",
        "name": "Project 1",
        "description": "First project",
        "collection": "project_proj1",
        "quality_tier": "balanced",
        "document_count": 5,
        "urls": ["https://example.com/1"],
        "created_at": datetime.now(UTC).isoformat()
    }

    project2 = {
        "id": "proj2",
        "name": "Project 2",
        "description": "Second project",
        "collection": "project_proj2",
        "quality_tier": "premium",
        "document_count": 3,
        "urls": ["https://example.com/2"],
        "created_at": datetime.now(UTC).isoformat()
    }

    mock_client_manager.projects = {"proj1": project1, "proj2": project2}

    # Setup collection info mocks
    def get_collection_info_side_effect(collection_name):
        mock_info = Mock()
        if collection_name == "project_proj1":
            mock_info.vectors_count = 10
            mock_info.indexed_vectors_count = 8
        else:  # project_proj2
            mock_info.vectors_count = 15
            mock_info.indexed_vectors_count = 12
        return mock_info

    mock_client_manager.qdrant_service.get_collection_info.side_effect = get_collection_info_side_effect

    register_tools(mock_mcp, mock_client_manager)

    # Test list_projects function
    result = await registered_tools["list_projects"](mock_context)

    # Verify results
    assert len(result) == 2
    assert all(isinstance(p, ProjectInfo) for p in result)

    # Find projects by name for verification
    proj1_result = next(p for p in result if p.name == "Project 1")
    proj2_result = next(p for p in result if p.name == "Project 2")

    # Verify project 1
    assert proj1_result.id == "proj1"
    assert proj1_result.description == "First project"
    assert proj1_result.quality_tier == "balanced"
    assert proj1_result.vector_count == 10
    assert proj1_result.indexed_count == 8

    # Verify project 2
    assert proj2_result.id == "proj2"
    assert proj2_result.description == "Second project"
    assert proj2_result.quality_tier == "premium"
    assert proj2_result.vector_count == 15
    assert proj2_result.indexed_count == 12

    # Verify service calls
    assert mock_client_manager.qdrant_service.get_collection_info.call_count == 2

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_list_projects_with_collection_errors(mock_client_manager, mock_context):
    """Test project listing when collection info retrieval fails."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Test Project",
        "description": "Test",
        "collection": "project_proj1",
        "quality_tier": "balanced",
        "document_count": 1,
        "urls": [],
        "created_at": datetime.now(UTC).isoformat()
    }
    mock_client_manager.projects = {"proj1": project}

    # Make collection info fail
    mock_client_manager.qdrant_service.get_collection_info.side_effect = Exception("Collection not found")

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["list_projects"](mock_context)

    # Should still return project with default vector counts
    assert len(result) == 1
    assert result[0].vector_count == 0
    assert result[0].indexed_count == 0

    # Verify warning was logged
    mock_context.warning.assert_called()
>>>>>>> dev

    @pytest.mark.asyncio
    async def test_search_project_success(
        self, mock_client_manager, mock_project_storage, mock_context
    ):
        """Test searching within a project."""
        from src.mcp_tools.tools.projects import register_tools

<<<<<<< HEAD
        mock_mcp = MagicMock()
        tools = register_tool(mock_mcp)
        register_tools(mock_mcp, mock_client_manager)
=======
@pytest.mark.asyncio
async def test_list_projects_empty(mock_client_manager, mock_context):
    """Test listing projects when none exist."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Empty projects dictionary
    mock_client_manager.projects = {}

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["list_projects"](mock_context)

    # Should return empty list
    assert len(result) == 0
    assert isinstance(result, list)
>>>>>>> dev

        # Setup project
        project = {
            "id": "proj1",
            "name": "Test Project",
            "collection": "project_proj1",
            "quality_tier": "balanced",
        }
        mock_project_storage.get_project.return_value = project

<<<<<<< HEAD
        # Mock embeddings
        mock_client_manager.embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "sparse_embeddings": [[0.5, 0.0, 0.7]],
        }
=======
@pytest.mark.asyncio
async def test_search_project_success(mock_client_manager, mock_context):
    """Test successful project search."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Test Project",
        "collection": "project_proj1",
        "quality_tier": "balanced"
    }
    mock_client_manager.projects = {"proj1": project}

    # Setup embedding generation
    mock_embedding_result = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]],
        "sparse_embeddings": [[0.8, 0.0, 0.6, 0.0]]
    }
    mock_client_manager.embedding_manager.generate_embeddings.return_value = mock_embedding_result

    # Setup search results
    mock_search_point = Mock()
    mock_search_point.id = "doc1"
    mock_search_point.score = 0.95
    mock_search_point.payload = {
        "content": "Test document content",
        "url": "https://example.com/doc1",
        "title": "Test Document",
        "type": "documentation"
    }

    mock_client_manager.qdrant_service.hybrid_search.return_value = [mock_search_point]

    register_tools(mock_mcp, mock_client_manager)

    # Test search_project function
    result = await registered_tools["search_project"](
        project_id="proj1",
        query="machine learning algorithms",
        limit=5,
        strategy=SearchStrategy.HYBRID,
        ctx=mock_context
    )

    # Verify results
    assert len(result) == 1
    assert isinstance(result[0], SearchResult)
    assert result[0].id == "doc1"
    assert result[0].content == "Test document content"
    assert result[0].score == 0.95
    assert result[0].url == "https://example.com/doc1"
    assert result[0].title == "Test Document"
    assert result[0].metadata["type"] == "documentation"

    # Verify embedding generation
    mock_client_manager.embedding_manager.generate_embeddings.assert_called_once_with(
        ["machine learning algorithms"], generate_sparse=True
    )

    # Verify search call
    mock_client_manager.qdrant_service.hybrid_search.assert_called_once()
    call_args = mock_client_manager.qdrant_service.hybrid_search.call_args[1]
    assert call_args["collection_name"] == "project_proj1"
    assert call_args["query_vector"] == [0.1, 0.2, 0.3, 0.4]
    assert call_args["sparse_vector"] == [0.8, 0.0, 0.6, 0.0]
    assert call_args["limit"] == 5
    assert call_args["fusion_type"] == "rrf"

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()
>>>>>>> dev

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

<<<<<<< HEAD
        result = await tools["search_project"](
=======
@pytest.mark.asyncio
async def test_search_project_vector_only_strategy(mock_client_manager, mock_context):
    """Test project search with VECTOR_ONLY strategy."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {"id": "proj1", "collection": "project_proj1"}
    mock_client_manager.projects = {"proj1": project}

    # Setup embedding generation (no sparse embeddings for VECTOR_ONLY)
    mock_embedding_result = {
        "embeddings": [[0.1, 0.2, 0.3, 0.4]]
    }
    mock_client_manager.embedding_manager.generate_embeddings.return_value = mock_embedding_result

    mock_client_manager.qdrant_service.hybrid_search.return_value = []

    register_tools(mock_mcp, mock_client_manager)

    # Test with DENSE strategy
    result = await registered_tools["search_project"](
        project_id="proj1",
        query="test query",
        strategy=SearchStrategy.DENSE,
        ctx=mock_context
    )

    # Verify embedding generation without sparse
    mock_client_manager.embedding_manager.generate_embeddings.assert_called_once_with(
        ["test query"], generate_sparse=False
    )

    # Verify search call without sparse vector
    call_args = mock_client_manager.qdrant_service.hybrid_search.call_args[1]
    assert call_args["sparse_vector"] is None
    assert call_args["fusion_type"] is None


@pytest.mark.asyncio
async def test_search_project_not_found(mock_client_manager, mock_context):
    """Test project search when project doesn't exist."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Empty projects
    mock_client_manager.projects = {}

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(ValueError, match="Project nonexistent not found"):
        await registered_tools["search_project"](
            project_id="nonexistent",
            query="test query",
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_update_project_success(mock_client_manager, mock_context):
    """Test successful project update."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Original Name",
        "description": "Original Description",
        "quality_tier": "balanced",
        "collection": "project_proj1",
        "document_count": 5,
        "urls": [],
        "created_at": datetime.now(UTC).isoformat()
    }
    mock_client_manager.projects = {"proj1": project}

    register_tools(mock_mcp, mock_client_manager)

    # Test update_project function
    result = await registered_tools["update_project"](
        project_id="proj1",
        name="Updated Name",
        description="Updated Description",
        ctx=mock_context
    )

    # Verify result
    assert isinstance(result, ProjectInfo)
    assert result.name == "Updated Name"
    assert result.description == "Updated Description"
    assert result.id == "proj1"

    # Verify project was updated in memory
    assert mock_client_manager.projects["proj1"]["name"] == "Updated Name"
    assert mock_client_manager.projects["proj1"]["description"] == "Updated Description"

    # Verify persistent storage update
    mock_client_manager.project_storage.update_project.assert_called_once()
    update_args = mock_client_manager.project_storage.update_project.call_args[0]
    update_data = update_args[1]  # Second positional argument
    assert update_args[0] == "proj1"
    assert update_data["name"] == "Updated Name"
    assert update_data["description"] == "Updated Description"

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_update_project_partial_update(mock_client_manager, mock_context):
    """Test partial project update (only name or description)."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Original Name",
        "description": "Original Description",
        "quality_tier": "balanced"
    }
    mock_client_manager.projects = {"proj1": project}

    register_tools(mock_mcp, mock_client_manager)

    # Test updating only name
    result = await registered_tools["update_project"](
        project_id="proj1",
        name="New Name Only",
        description=None,  # Don't update description
        ctx=mock_context
    )

    # Verify only name was updated
    assert result.name == "New Name Only"
    assert result.description == "Original Description"  # Unchanged

    # Verify storage update
    update_args = mock_client_manager.project_storage.update_project.call_args[0]
    update_data = update_args[1]  # Second positional argument
    assert "name" in update_data
    assert "description" not in update_data


@pytest.mark.asyncio
async def test_update_project_no_changes(mock_client_manager, mock_context):
    """Test project update with no actual changes."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Test Name",
        "description": "Test Description"
    }
    mock_client_manager.projects = {"proj1": project}

    register_tools(mock_mcp, mock_client_manager)

    # Test with no updates (both None)
    result = await registered_tools["update_project"](
        project_id="proj1",
        name=None,
        description=None,
        ctx=mock_context
    )

    # Verify no changes
    assert result.name == "Test Name"
    assert result.description == "Test Description"

    # Verify storage update was not called
    mock_client_manager.project_storage.update_project.assert_not_called()


@pytest.mark.asyncio
async def test_update_project_not_found(mock_client_manager, mock_context):
    """Test project update when project doesn't exist."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Empty projects
    mock_client_manager.projects = {}

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(ValueError, match="Project nonexistent not found"):
        await registered_tools["update_project"](
            project_id="nonexistent",
            name="New Name",
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_delete_project_success_with_collection(mock_client_manager, mock_context):
    """Test successful project deletion including collection."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Test Project",
        "collection": "project_proj1"
    }
    mock_client_manager.projects = {"proj1": project}

    register_tools(mock_mcp, mock_client_manager)

    # Test delete_project function
    result = await registered_tools["delete_project"](
        project_id="proj1",
        delete_collection=True,
        ctx=mock_context
    )

    # Verify result
    assert isinstance(result, OperationStatus)
    assert result.status == "deleted"
    assert result.details["project_id"] == "proj1"
    assert result.details["collection_deleted"] == "True"

    # Verify collection deletion
    mock_client_manager.qdrant_service.delete_collection.assert_called_once_with("project_proj1")

    # Verify project removed from memory
    assert "proj1" not in mock_client_manager.projects

    # Verify persistent storage deletion
    mock_client_manager.project_storage.delete_project.assert_called_once_with("proj1")

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_delete_project_without_collection(mock_client_manager, mock_context):
    """Test project deletion without deleting collection."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Test Project",
        "collection": "project_proj1"
    }
    mock_client_manager.projects = {"proj1": project}

    register_tools(mock_mcp, mock_client_manager)

    # Test delete without collection
    result = await registered_tools["delete_project"](
        project_id="proj1",
        delete_collection=False,
        ctx=mock_context
    )

    # Verify result
    assert result.status == "deleted"
    assert result.details["collection_deleted"] == "False"

    # Verify collection deletion was NOT called
    mock_client_manager.qdrant_service.delete_collection.assert_not_called()

    # Verify project still removed from memory and storage
    assert "proj1" not in mock_client_manager.projects
    mock_client_manager.project_storage.delete_project.assert_called_once_with("proj1")


@pytest.mark.asyncio
async def test_delete_project_collection_deletion_fails(mock_client_manager, mock_context):
    """Test project deletion when collection deletion fails."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {
        "id": "proj1",
        "name": "Test Project",
        "collection": "project_proj1"
    }
    mock_client_manager.projects = {"proj1": project}

    # Make collection deletion fail
    mock_client_manager.qdrant_service.delete_collection.side_effect = Exception("Collection deletion failed")

    register_tools(mock_mcp, mock_client_manager)

    # Test delete_project - should complete despite collection deletion failure
    result = await registered_tools["delete_project"](
        project_id="proj1",
        delete_collection=True,
        ctx=mock_context
    )

    # Verify project deletion still succeeded
    assert result.status == "deleted"
    assert "proj1" not in mock_client_manager.projects

    # Verify warning was logged for collection deletion failure
    mock_context.warning.assert_called()

    # Verify project storage deletion still occurred
    mock_client_manager.project_storage.delete_project.assert_called_once()


@pytest.mark.asyncio
async def test_delete_project_not_found(mock_client_manager, mock_context):
    """Test project deletion when project doesn't exist."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Empty projects
    mock_client_manager.projects = {}

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(ValueError, match="Project nonexistent not found"):
        await registered_tools["delete_project"](
            project_id="nonexistent",
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_list_projects_error_handling(mock_client_manager, mock_context):
    """Test error handling in list_projects."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Make projects access fail on len() call
    mock_client_manager.projects = Mock()
    type(mock_client_manager.projects).__len__ = Mock(side_effect=TypeError("object of type 'Mock' has no len()"))

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(TypeError):
        await registered_tools["list_projects"](mock_context)

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_search_project_error_handling(mock_client_manager, mock_context):
    """Test error handling in search_project."""
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mock project
    project = {"id": "proj1", "collection": "project_proj1"}
    mock_client_manager.projects = {"proj1": project}

    # Make embedding generation fail
    mock_client_manager.embedding_manager.generate_embeddings.side_effect = Exception("Embedding failed")

    register_tools(mock_mcp, mock_client_manager)

    with pytest.raises(Exception, match="Embedding failed"):
        await registered_tools["search_project"](
>>>>>>> dev
            project_id="proj1",
            query="test query",
            limit=10,
            strategy=SearchStrategy.HYBRID,
            ctx=mock_context,
        )

<<<<<<< HEAD
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
=======
    # Verify error logging
    mock_context.error.assert_called()
>>>>>>> dev


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
<<<<<<< HEAD

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
=======
    from src.mcp_tools.tools.projects import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup basic mocks
    project = {
        "id": "proj1",
        "name": "Test Project",
        "description": "Test",
        "collection": "project_proj1",
        "quality_tier": "basic",
        "document_count": 0,
        "urls": [],
        "created_at": datetime.now(UTC).isoformat()
    }
    mock_client_manager.projects = {"proj1": project}

    # Mock collection info
    mock_info = Mock()
    mock_info.vectors_count = 5
    mock_info.indexed_vectors_count = 5
    mock_client_manager.qdrant_service.get_collection_info.return_value = mock_info

    # Mock embedding and search
    mock_client_manager.embedding_manager.generate_embeddings.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]]
    }
    mock_client_manager.qdrant_service.hybrid_search.return_value = []

    register_tools(mock_mcp, mock_client_manager)

    # Test all tools work without context (ctx=None)

    # Test list_projects
    result = await registered_tools["list_projects"](ctx=None)
    assert len(result) == 1

    # Test search_project
    result = await registered_tools["search_project"](
        project_id="proj1",
        query="test",
        ctx=None
    )
    assert isinstance(result, list)

    # Test update_project
    result = await registered_tools["update_project"](
        project_id="proj1",
        name="Updated",
        ctx=None
    )
    assert isinstance(result, ProjectInfo)

    # Test delete_project
    result = await registered_tools["delete_project"](
        project_id="proj1",
        delete_collection=False,
        ctx=None
    )
    assert isinstance(result, OperationStatus)
>>>>>>> dev
