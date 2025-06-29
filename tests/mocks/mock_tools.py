"""Mock MCP tools for testing without FastMCP Context issues."""

from typing import Any, Protocol
from unittest.mock import AsyncMock


class MockContext(Protocol):
    """Mock Context protocol for testing."""

    async def info(self, msg: str) -> None: ...
    async def debug(self, msg: str) -> None: ...
    async def warning(self, msg: str) -> None: ...
    async def error(self, msg: str) -> None: ...


class MockTool:
    """Mock MCP tool for testing."""

    def __init__(self, name: str, handler: Any):
        self.name = name
        self.handler = handler


class MockMCPServer:
    """Mock MCP server for testing."""

    def __init__(self, name: str):
        self.name = name
        self._tools: list[MockTool] = []
        self._resources = []
        self._prompts = []

    def tool(self, *_args, **__kwargs):
        """Mock tool decorator."""

        def decorator(func):
            # Extract tool name from function
            tool_name = func.__name__
            # Create mock tool
            tool = MockTool(name=tool_name, handler=func)
            self._tools.append(tool)
            return func

        return decorator

    def resource(self, *_args, **__kwargs):
        """Mock resource decorator."""

        def decorator(func):
            self._resources.append(func)
            return func

        return decorator

    def prompt(self, *_args, **__kwargs):
        """Mock prompt decorator."""

        def decorator(func):
            self._prompts.append(func)
            return func

        return decorator


def create_mock_tools(client_manager) -> dict[str, AsyncMock]:
    """Create mock tool functions for testing."""

    # Mock search tool
    async def mock_search_documents(
        query: str,
        collection: str = "documentation",
        limit: int = 10,
        strategy: str = "hybrid",
        enable_reranking: bool = True,
        **__kwargs,
    ) -> list[dict[str, Any]]:
        """Mock search documents tool."""
        result = await client_manager.vector_service.search_documents(
            query=query,
            collection=collection,
            limit=limit,
            strategy=strategy,
            enable_reranking=enable_reranking,
        )

        # Validate response structure
        if result is None:
            msg = "Invalid response: received None from search service"
            raise ValueError(msg)
        if not isinstance(result, list):
            msg = f"Invalid response: expected list, got {type(result).__name__}"
            raise ValueError(msg)

        return result

    # Mock embeddings tool
    async def mock_generate_embeddings(
        texts: list[str], model: str | None = None, **_kwargs
    ) -> dict[str, Any]:
        """Mock generate embeddings tool."""
        return await client_manager.embedding_service.generate_embeddings(
            texts=texts,
            model=model,
        )

    # Mock document tool
    async def mock_add_document(
        url: str,
        collection: str = "documentation",
        chunk_strategy: str = "enhanced",
        **__kwargs,
    ) -> dict[str, Any]:
        """Mock add document tool."""
        # First crawl the URL
        await client_manager.crawling_service.crawl_url(url)

        # Then add to vector service
        return await client_manager.vector_service.add_document(
            url=url,
            collection=collection,
            chunk_strategy=chunk_strategy,
        )

    # Mock collections tool
    async def mock_list_collections(**__kwargs) -> list[dict[str, Any]]:
        """Mock list collections tool."""
        return await client_manager.vector_service.list_collections()

    # Mock project tool
    async def mock_create_project(
        name: str,
        description: str | None = None,
        quality_tier: str = "balanced",
        **__kwargs,
    ) -> dict[str, Any]:
        """Mock create project tool."""
        return await client_manager.project_service.create_project(
            name=name,
            description=description,
            quality_tier=quality_tier,
        )

    # Mock analytics tool
    async def mock_get_analytics(
        collection: str | None = None,
        include_performance: bool = True,
        include_costs: bool = True,
        **__kwargs,
    ) -> dict[str, Any]:
        """Mock get analytics tool."""
        return await client_manager.analytics_service.get_analytics(
            collection=collection,
            include_performance=include_performance,
            include_costs=include_costs,
        )

    # Mock cache tools
    async def mock_get_cache_stats(**__kwargs) -> dict[str, Any]:
        """Mock get cache stats tool."""
        return await client_manager.cache_service.get_stats()

    async def mock_clear_cache(
        pattern: str | None = None, **__kwargs
    ) -> dict[str, Any]:
        """Mock clear cache tool."""
        return await client_manager.cache_service.clear(pattern=pattern)

    # Mock deployment tool
    async def mock_list_aliases(**__kwargs) -> dict[str, Any]:
        """Mock list aliases tool."""
        result = await client_manager.deployment_service.list_aliases()
        return {"aliases": result}

    # Mock utilities tool
    async def mock_validate_configuration(**__kwargs) -> dict[str, Any]:
        """Mock validate configuration tool."""
        # This would normally validate the actual config
        return {"status": "success", "message": "Configuration is valid"}

    # Mock payload indexing tool
    async def mock_reindex_collection(collection: str, **__kwargs) -> dict[str, Any]:
        """Mock reindex collection tool."""
        return await client_manager.vector_service.reindex_collection(
            collection=collection
        )

    # Mock HyDE search tool
    async def mock_hyde_search(
        query: str,
        collection: str = "documentation",
        num_generations: int = 5,
        **__kwargs,
    ) -> dict[str, Any]:
        """Mock HyDE search tool."""
        return await client_manager.hyde_service.search(
            query=query,
            collection=collection,
            num_generations=num_generations,
        )

    # Return all mock tools
    return {
        "search_documents": AsyncMock(side_effect=mock_search_documents),
        "generate_embeddings": AsyncMock(side_effect=mock_generate_embeddings),
        "add_document": AsyncMock(side_effect=mock_add_document),
        "list_collections": AsyncMock(side_effect=mock_list_collections),
        "create_project": AsyncMock(side_effect=mock_create_project),
        "get_analytics": AsyncMock(side_effect=mock_get_analytics),
        "get_cache_stats": AsyncMock(side_effect=mock_get_cache_stats),
        "clear_cache": AsyncMock(side_effect=mock_clear_cache),
        "list_aliases": AsyncMock(side_effect=mock_list_aliases),
        "validate_configuration": AsyncMock(side_effect=mock_validate_configuration),
        "reindex_collection": AsyncMock(side_effect=mock_reindex_collection),
        "hyde_search": AsyncMock(side_effect=mock_hyde_search),
    }


def register_mock_tools(mcp_server: MockMCPServer, client_manager) -> None:
    """Register mock tools with the mock MCP server."""
    mock_tools = create_mock_tools(client_manager)

    for tool_name, tool_func in mock_tools.items():
        # Create a mock tool and add it to the server
        tool = MockTool(name=tool_name, handler=tool_func)
        mcp_server._tools.append(tool)
