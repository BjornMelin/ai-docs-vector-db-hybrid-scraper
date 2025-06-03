"""Tests for MCP search tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.mcp_tools.models.requests import SearchRequest
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
def mock_client_manager():
    """Create a mock client manager."""
    manager = Mock()

    # Mock QdrantService
    mock_qdrant = Mock()
    mock_qdrant._client = Mock()
    mock_qdrant._client.retrieve = AsyncMock()
    mock_qdrant.hybrid_search = AsyncMock()
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

    return manager


@pytest.mark.asyncio
async def test_search_documents_tool_registration(mock_client_manager, mock_context):
    """Test that search_documents tool is properly registered."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    assert "search_documents" in registered_tools
    assert "search_similar" in registered_tools


@pytest.mark.asyncio
async def test_search_documents_basic(mock_client_manager, mock_context):
    """Test basic search_documents functionality."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock search_documents_core
    mock_search_results = [
        SearchResult(
            id="doc1",
            content="Test content 1",
            score=0.95,
            url="https://example.com/1",
            title="Test Document 1",
            metadata={"type": "documentation"}
        ),
        SearchResult(
            id="doc2",
            content="Test content 2",
            score=0.85,
            url="https://example.com/2",
            title="Test Document 2",
            metadata={"type": "guide"}
        )
    ]

    with patch('src.mcp_tools.tools._search_utils.search_documents_core', new_callable=AsyncMock) as mock_core:
        mock_core.return_value = mock_search_results

        register_tools(mock_mcp, mock_client_manager)

        # Test search_documents function
        request = SearchRequest(
            query="test query",
            collection="documentation",
            limit=10
        )

        result = await registered_tools["search_documents"](request, mock_context)

        assert len(result) == 2
        assert result[0].id == "doc1"
        assert result[0].content == "Test content 1"
        assert result[0].score == 0.95
        assert result[1].id == "doc2"

        # Verify search_documents_core was called correctly
        mock_core.assert_called_once_with(request, mock_client_manager, mock_context)


@pytest.mark.asyncio
async def test_search_similar_success(mock_client_manager, mock_context):
    """Test successful search_similar functionality."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock retrieved document with vector
    mock_retrieved_doc = Mock()
    mock_retrieved_doc.vector = Mock()
    mock_retrieved_doc.vector.dense = [0.1, 0.2, 0.3, 0.4]

    # Mock hybrid search results
    mock_search_results = [
        {
            "id": "source_doc",
            "score": 1.0,
            "payload": {"content": "Source content", "url": "https://example.com/source"}
        },
        {
            "id": "similar_doc1",
            "score": 0.85,
            "payload": {
                "content": "Similar content 1",
                "url": "https://example.com/similar1",
                "title": "Similar Document 1"
            }
        },
        {
            "id": "similar_doc2",
            "score": 0.75,
            "payload": {
                "content": "Similar content 2",
                "url": "https://example.com/similar2",
                "title": "Similar Document 2"
            }
        }
    ]

    # Setup mocks
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = [mock_retrieved_doc]
    mock_qdrant.hybrid_search.return_value = mock_search_results

    register_tools(mock_mcp, mock_client_manager)

    # Test search_similar function
    result = await registered_tools["search_similar"](
        query_id="source_doc",
        collection="documentation",
        limit=2,
        score_threshold=0.7,
        ctx=mock_context
    )

    # Should exclude source document and return 2 similar documents
    assert len(result) == 2
    assert result[0].id == "similar_doc1"
    assert result[0].content == "Similar content 1"
    assert result[0].score == 0.85
    assert result[1].id == "similar_doc2"

    # Verify correct API calls
    mock_qdrant._client.retrieve.assert_called_once_with(
        collection_name="documentation",
        ids=["source_doc"],
        with_vectors=True,
        with_payload=True
    )

    mock_qdrant.hybrid_search.assert_called_once_with(
        collection_name="documentation",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=None,
        limit=3,  # limit + 1 to exclude self
        score_threshold=0.7,
        fusion_type="rrf",
        search_accuracy="balanced"
    )

    # Verify context logging
    mock_context.info.assert_called()
    mock_context.debug.assert_called()


@pytest.mark.asyncio
async def test_search_similar_document_not_found(mock_client_manager, mock_context):
    """Test search_similar when source document is not found."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock empty retrieval result
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = []

    register_tools(mock_mcp, mock_client_manager)

    # Test search_similar function with non-existent document
    with pytest.raises(ValueError, match="Document missing_doc not found"):
        await registered_tools["search_similar"](
            query_id="missing_doc",
            collection="documentation",
            limit=10,
            score_threshold=0.7,
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_search_similar_vector_formats(mock_client_manager, mock_context):
    """Test search_similar with different vector formats."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Test case 1: vector as list
    mock_retrieved_doc1 = Mock()
    mock_retrieved_doc1.vector = [0.1, 0.2, 0.3, 0.4]

    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = [mock_retrieved_doc1]
    mock_qdrant.hybrid_search.return_value = [
        {
            "id": "similar_doc",
            "score": 0.8,
            "payload": {"content": "Similar content", "url": "https://example.com"}
        }
    ]

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["search_similar"](
        query_id="doc1",
        collection="documentation",
        limit=10,
        ctx=mock_context
    )

    assert len(result) == 1
    mock_qdrant.hybrid_search.assert_called_with(
        collection_name="documentation",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=None,
        limit=11,
        score_threshold=0.7,
        fusion_type="rrf",
        search_accuracy="balanced"
    )


@pytest.mark.asyncio
async def test_search_similar_dict_vector_format(mock_client_manager, mock_context):
    """Test search_similar with dictionary vector format."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Test case: vector as dict with "dense" key
    mock_retrieved_doc = Mock()
    mock_retrieved_doc.vector = {"dense": [0.1, 0.2, 0.3, 0.4], "sparse": []}

    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = [mock_retrieved_doc]
    mock_qdrant.hybrid_search.return_value = []

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["search_similar"](
        query_id="doc1",
        collection="documentation",
        limit=5,
        ctx=mock_context
    )

    assert len(result) == 0
    mock_qdrant.hybrid_search.assert_called_with(
        collection_name="documentation",
        query_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=None,
        limit=6,
        score_threshold=0.7,
        fusion_type="rrf",
        search_accuracy="balanced"
    )


@pytest.mark.asyncio
async def test_search_similar_exclude_self(mock_client_manager, mock_context):
    """Test that search_similar excludes the source document from results."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock retrieved document
    mock_retrieved_doc = Mock()
    mock_retrieved_doc.vector = Mock()
    mock_retrieved_doc.vector.dense = [0.1, 0.2, 0.3, 0.4]

    # Mock search results including the source document
    mock_search_results = [
        {
            "id": "source_doc",  # This should be excluded
            "score": 1.0,
            "payload": {"content": "Source content"}
        },
        {
            "id": "similar_doc1",
            "score": 0.9,
            "payload": {"content": "Similar content 1", "title": "Similar 1"}
        },
        {
            "id": "similar_doc2",
            "score": 0.8,
            "payload": {"content": "Similar content 2", "title": "Similar 2"}
        }
    ]

    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = [mock_retrieved_doc]
    mock_qdrant.hybrid_search.return_value = mock_search_results

    register_tools(mock_mcp, mock_client_manager)

    result = await registered_tools["search_similar"](
        query_id="source_doc",
        collection="documentation",
        limit=5,
        ctx=mock_context
    )

    # Should only return similar documents, not the source
    assert len(result) == 2
    assert result[0].id == "similar_doc1"
    assert result[1].id == "similar_doc2"

    # Verify no source document in results
    assert all(r.id != "source_doc" for r in result)


@pytest.mark.asyncio
async def test_search_similar_error_handling(mock_client_manager, mock_context):
    """Test error handling in search_similar."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock QdrantService to raise an exception
    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.side_effect = Exception("Qdrant connection error")

    register_tools(mock_mcp, mock_client_manager)

    # Test that exception is properly handled and re-raised
    with pytest.raises(Exception, match="Qdrant connection error"):
        await registered_tools["search_similar"](
            query_id="test_doc",
            collection="documentation",
            limit=10,
            ctx=mock_context
        )

    # Verify error logging
    mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_search_similar_without_context(mock_client_manager):
    """Test search_similar functionality without context parameter."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock retrieved document
    mock_retrieved_doc = Mock()
    mock_retrieved_doc.vector = Mock()
    mock_retrieved_doc.vector.dense = [0.1, 0.2, 0.3, 0.4]

    # Mock search results
    mock_search_results = [
        {
            "id": "similar_doc1",
            "score": 0.9,
            "payload": {"content": "Similar content", "url": "https://example.com"}
        }
    ]

    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = [mock_retrieved_doc]
    mock_qdrant.hybrid_search.return_value = mock_search_results

    register_tools(mock_mcp, mock_client_manager)

    # Test without ctx parameter (None)
    result = await registered_tools["search_similar"](
        query_id="test_doc",
        collection="documentation",
        limit=5,
        score_threshold=0.8,
        ctx=None
    )

    assert len(result) == 1
    assert result[0].id == "similar_doc1"
    assert result[0].content == "Similar content"


@pytest.mark.asyncio
async def test_search_similar_default_parameters(mock_client_manager, mock_context):
    """Test search_similar with default parameters."""
    from src.mcp_tools.tools.search import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock retrieved document
    mock_retrieved_doc = Mock()
    mock_retrieved_doc.vector = [0.1, 0.2, 0.3, 0.4]

    mock_qdrant = await mock_client_manager.get_qdrant_service()
    mock_qdrant._client.retrieve.return_value = [mock_retrieved_doc]
    mock_qdrant.hybrid_search.return_value = []

    register_tools(mock_mcp, mock_client_manager)

    # Test with only required parameter
    result = await registered_tools["search_similar"](
        query_id="test_doc",
        ctx=mock_context
    )

    assert len(result) == 0

    # Verify default parameters were used
    mock_qdrant.hybrid_search.assert_called_with(
        collection_name="documentation",  # default
        query_vector=[0.1, 0.2, 0.3, 0.4],
        sparse_vector=None,
        limit=11,  # default 10 + 1
        score_threshold=0.7,  # default
        fusion_type="rrf",
        search_accuracy="balanced"
    )
