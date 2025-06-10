"""Tests for MCP document management tools."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.enums import ChunkingStrategy
from src.mcp_tools.models.requests import BatchRequest
from src.mcp_tools.models.requests import DocumentRequest
from src.mcp_tools.models.responses import AddDocumentResponse
from src.mcp_tools.models.responses import DocumentBatchResponse


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

    # Mock services
    mock_cache = Mock()
    mock_cache.get = AsyncMock(return_value=None)  # No cache hit by default
    mock_cache.set = AsyncMock()
    manager.get_cache_manager = AsyncMock(return_value=mock_cache)

    mock_crawl = Mock()
    mock_crawl.scrape_url = AsyncMock()
    manager.get_crawl_manager = AsyncMock(return_value=mock_crawl)

    mock_embedding = Mock()
    mock_embedding.generate_embeddings = AsyncMock()
    manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)

    mock_qdrant = Mock()
    mock_qdrant.create_collection = AsyncMock()
    mock_qdrant.upsert_points = AsyncMock()
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

    return manager


@pytest.fixture
def mock_crawl_result():
    """Create a mock crawl result."""
    return {
        "success": True,
        "content": "# Test Document\n\nThis is test content for the document.",
        "title": "Test Document",
        "url": "https://example.com/test",
        "metadata": {
            "title": "Test Document",
            "url": "https://example.com/test",
            "description": "A test document",
        },
    }


@pytest.fixture
def mock_chunks():
    """Create mock chunks."""
    return [
        {
            "content": "# Test Document",
            "metadata": {"chunk_type": "header", "section": "title"},
        },
        {
            "content": "This is test content for the document.",
            "metadata": {"chunk_type": "paragraph", "section": "body"},
        },
    ]


@pytest.mark.asyncio
async def test_add_document_tool_registration(mock_client_manager, mock_context):
    """Test that document tools are properly registered."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    register_tools(mock_mcp, mock_client_manager)

    assert "add_document" in registered_tools
    assert "add_documents_batch" in registered_tools


@pytest.mark.asyncio
async def test_add_document_success(
    mock_client_manager, mock_context, mock_crawl_result, mock_chunks
):
    """Test successful document addition."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    crawl_manager = await mock_client_manager.get_crawl_manager()
    crawl_manager.scrape_url.return_value = mock_crawl_result

    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [
        [0.1, 0.2, 0.3, 0.4],  # embedding for first chunk
        [0.5, 0.6, 0.7, 0.8],  # embedding for second chunk
    ]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    # Mock security validator
    with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
        mock_validator = Mock()
        mock_validator.validate_url.return_value = "https://example.com/test"
        mock_security.from_unified_config.return_value = mock_validator

        # Mock enhanced chunker
        with patch(
            "src.mcp_tools.tools.documents.EnhancedChunker"
        ) as mock_chunker_class:
            mock_chunker = Mock()
            mock_chunker.chunk_content.return_value = mock_chunks
            mock_chunker_class.return_value = mock_chunker

            register_tools(mock_mcp, mock_client_manager)

            # Test add_document function
            request = DocumentRequest(
                url="https://example.com/test",
                collection="test_collection",
                chunk_strategy=ChunkingStrategy.ENHANCED,
                chunk_size=512,
                chunk_overlap=50,
            )

            result = await registered_tools["add_document"](request, mock_context)

            # Verify result
            assert isinstance(result, AddDocumentResponse)
            assert result.url == "https://example.com/test"
            assert result.title == "Test Document"
            assert result.chunks_created == 2
            assert result.collection == "test_collection"
            assert result.chunking_strategy == "enhanced"
            assert result.embedding_dimensions == 4

            # Verify services were called correctly
            crawl_manager.scrape_url.assert_called_once_with("https://example.com/test")
            embedding_manager.generate_embeddings.assert_called_once()

            # Verify chunks were processed correctly
            args, kwargs = embedding_manager.generate_embeddings.call_args
            assert len(args[0]) == 2  # Two chunks
            assert args[0][0] == "# Test Document"
            assert args[0][1] == "This is test content for the document."

            # Verify collection creation
            qdrant_service = await mock_client_manager.get_qdrant_service()
            qdrant_service.create_collection.assert_called_once_with(
                collection_name="test_collection",
                vector_size=4,
                distance="Cosine",
                sparse_vector_name="sparse",
                enable_quantization=True,
            )

            # Verify points insertion
            qdrant_service.upsert_points.assert_called_once()
            points = qdrant_service.upsert_points.call_args[1]["points"]
            assert len(points) == 2
            assert points[0]["vector"] == [0.1, 0.2, 0.3, 0.4]
            assert points[1]["vector"] == [0.5, 0.6, 0.7, 0.8]


@pytest.mark.asyncio
async def test_add_document_cache_hit(mock_client_manager, mock_context):
    """Test document addition with cache hit."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup cache hit
    cached_response = {
        "url": "https://example.com/cached",
        "title": "Cached Document",
        "chunks_created": 3,
        "collection": "cached_collection",
        "chunking_strategy": "basic",
        "embedding_dimensions": 768,
    }

    cache_manager = await mock_client_manager.get_cache_manager()
    cache_manager.get.return_value = cached_response

    # Mock security validator
    with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
        mock_validator = Mock()
        mock_validator.validate_url.return_value = "https://example.com/cached"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        request = DocumentRequest(
            url="https://example.com/cached", collection="cached_collection"
        )

        result = await registered_tools["add_document"](request, mock_context)

        # Verify cached result is returned
        assert isinstance(result, AddDocumentResponse)
        assert result.url == "https://example.com/cached"
        assert result.title == "Cached Document"
        assert result.chunks_created == 3

        # Verify cache was checked
        cache_manager.get.assert_called_once_with("doc:https://example.com/cached")

        # Verify no crawling occurred
        crawl_manager = await mock_client_manager.get_crawl_manager()
        crawl_manager.scrape_url.assert_not_called()


@pytest.mark.asyncio
async def test_add_document_crawl_failure(mock_client_manager, mock_context):
    """Test document addition when crawling fails."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup crawl failure
    crawl_manager = await mock_client_manager.get_crawl_manager()
    crawl_manager.scrape_url.return_value = None

    # Mock security validator
    with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
        mock_validator = Mock()
        mock_validator.validate_url.return_value = "https://example.com/fail"
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        request = DocumentRequest(
            url="https://example.com/fail", collection="test_collection"
        )

        with pytest.raises(ValueError, match="Failed to scrape"):
            await registered_tools["add_document"](request, mock_context)

        # Verify error logging
        mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_add_document_basic_chunking_no_sparse_vector(
    mock_client_manager, mock_context, mock_crawl_result, mock_chunks
):
    """Test document addition with basic chunking strategy (no sparse vector)."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    crawl_manager = await mock_client_manager.get_crawl_manager()
    crawl_manager.scrape_url.return_value = mock_crawl_result

    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    # Mock security validator and chunker
    with (
        patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
        patch("src.mcp_tools.tools.documents.EnhancedChunker") as mock_chunker_class,
    ):
        mock_validator = Mock()
        mock_validator.validate_url.return_value = "https://example.com/test"
        mock_security.from_unified_config.return_value = mock_validator

        mock_chunker = Mock()
        mock_chunker.chunk_content.return_value = [mock_chunks[0]]  # Only one chunk
        mock_chunker_class.return_value = mock_chunker

        register_tools(mock_mcp, mock_client_manager)

        request = DocumentRequest(
            url="https://example.com/test",
            collection="test_collection",
            chunk_strategy=ChunkingStrategy.BASIC,  # Use BASIC strategy
        )

        await registered_tools["add_document"](request, mock_context)

        # Verify collection creation without sparse vector
        qdrant_service = await mock_client_manager.get_qdrant_service()
        qdrant_service.create_collection.assert_called_once_with(
            collection_name="test_collection",
            vector_size=4,
            distance="Cosine",
            sparse_vector_name=None,  # No sparse vector for BASIC strategy
            enable_quantization=True,
        )


@pytest.mark.asyncio
async def test_add_documents_batch_success(mock_client_manager, mock_context):
    """Test successful batch document addition."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup all mocks for successful processing
    crawl_manager = await mock_client_manager.get_crawl_manager()
    mock_crawl_result = {
        "success": True,
        "content": "Test content",
        "title": "Test",
        "url": "https://example.com/test",
        "metadata": {"title": "Test", "url": "https://example.com/test"},
    }
    crawl_manager.scrape_url.return_value = mock_crawl_result

    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    # Mock security validator and chunker
    with (
        patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
        patch("src.mcp_tools.tools.documents.EnhancedChunker") as mock_chunker_class,
    ):
        mock_validator = Mock()
        mock_validator.validate_url.side_effect = lambda x: x  # Return input unchanged
        mock_security.from_unified_config.return_value = mock_validator

        mock_chunker = Mock()
        mock_chunker.chunk_content.return_value = [
            {"content": "Test chunk", "metadata": {}}
        ]
        mock_chunker_class.return_value = mock_chunker

        register_tools(mock_mcp, mock_client_manager)

        request = BatchRequest(
            urls=["https://example.com/doc1", "https://example.com/doc2"],
            collection="test_collection",
            max_concurrent=2,
        )

        result = await registered_tools["add_documents_batch"](request, mock_context)

        # Verify result
        assert isinstance(result, DocumentBatchResponse)
        assert len(result.successful) == 2
        assert len(result.failed) == 0
        assert result.total == 2


@pytest.mark.asyncio
async def test_add_documents_batch_with_failures(mock_client_manager, mock_context):
    """Test batch document addition with some failures."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock security validator
    with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
        mock_validator = Mock()

        def validate_url_side_effect(url):
            if "invalid" in url:
                raise ValueError("Invalid URL")
            return url

        mock_validator.validate_url.side_effect = validate_url_side_effect
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        request = BatchRequest(
            urls=[
                "https://example.com/valid1",
                "https://example.com/invalid-url",
                "https://example.com/valid2",
            ],
            collection="test_collection",
            max_concurrent=3,
        )

        register_tools(mock_mcp, mock_client_manager)

        result = await registered_tools["add_documents_batch"](request, mock_context)

        # Should have 0 successful and 3 failed (all URLs fail validation)
        assert len(result.successful) == 0
        assert len(result.failed) == 3
        assert result.total == 3
        assert any("invalid" in url for url in result.failed)


@pytest.mark.asyncio
async def test_add_document_error_handling(mock_client_manager, mock_context):
    """Test error handling in add_document."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Mock security validator to raise exception
    with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
        mock_validator = Mock()
        mock_validator.validate_url.side_effect = Exception(
            "Security validation failed"
        )
        mock_security.from_unified_config.return_value = mock_validator

        register_tools(mock_mcp, mock_client_manager)

        request = DocumentRequest(
            url="https://example.com/error", collection="test_collection"
        )

        with pytest.raises(Exception, match="Security validation failed"):
            await registered_tools["add_document"](request, mock_context)

        # Verify error logging
        mock_context.error.assert_called()


@pytest.mark.asyncio
async def test_add_document_chunking_config(
    mock_client_manager, mock_context, mock_crawl_result, mock_chunks
):
    """Test that chunking configuration is properly passed."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    crawl_manager = await mock_client_manager.get_crawl_manager()
    crawl_manager.scrape_url.return_value = mock_crawl_result

    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    # Mock security validator and chunker
    with (
        patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
        patch("src.mcp_tools.tools.documents.EnhancedChunker") as mock_chunker_class,
        patch("src.mcp_tools.tools.documents.ChunkingConfig") as mock_config_class,
    ):
        mock_validator = Mock()
        mock_validator.validate_url.return_value = "https://example.com/test"
        mock_security.from_unified_config.return_value = mock_validator

        mock_chunker = Mock()
        mock_chunker.chunk_content.return_value = [mock_chunks[0]]
        mock_chunker_class.return_value = mock_chunker

        mock_config = Mock()
        mock_config_class.return_value = mock_config

        register_tools(mock_mcp, mock_client_manager)

        request = DocumentRequest(
            url="https://example.com/test",
            collection="test_collection",
            chunk_strategy=ChunkingStrategy.AST,
            chunk_size=1024,
            chunk_overlap=100,
        )

        await registered_tools["add_document"](request, mock_context)

        # Verify ChunkingConfig was created with correct parameters
        mock_config_class.assert_called_once_with(
            strategy=ChunkingStrategy.AST, chunk_size=1024, chunk_overlap=100
        )

        # Verify EnhancedChunker was initialized with config
        mock_chunker_class.assert_called_once_with(mock_config)


@pytest.mark.asyncio
async def test_add_document_cache_set(
    mock_client_manager, mock_context, mock_crawl_result, mock_chunks
):
    """Test that successful results are cached."""
    from src.mcp_tools.tools.documents import register_tools

    mock_mcp = MagicMock()
    registered_tools = {}

    def capture_tool(func):
        registered_tools[func.__name__] = func
        return func

    mock_mcp.tool.return_value = capture_tool

    # Setup mocks
    crawl_manager = await mock_client_manager.get_crawl_manager()
    crawl_manager.scrape_url.return_value = mock_crawl_result

    embedding_manager = await mock_client_manager.get_embedding_manager()
    mock_embedding_result = Mock()
    mock_embedding_result.embeddings = [[0.1, 0.2, 0.3, 0.4]]
    embedding_manager.generate_embeddings.return_value = mock_embedding_result

    cache_manager = await mock_client_manager.get_cache_manager()

    # Mock security validator and chunker
    with (
        patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
        patch("src.mcp_tools.tools.documents.EnhancedChunker") as mock_chunker_class,
    ):
        mock_validator = Mock()
        mock_validator.validate_url.return_value = "https://example.com/test"
        mock_security.from_unified_config.return_value = mock_validator

        mock_chunker = Mock()
        mock_chunker.chunk_content.return_value = [mock_chunks[0]]
        mock_chunker_class.return_value = mock_chunker

        register_tools(mock_mcp, mock_client_manager)

        request = DocumentRequest(
            url="https://example.com/test", collection="test_collection"
        )

        result = await registered_tools["add_document"](request, mock_context)

        # Verify result was cached
        cache_manager.set.assert_called_once()
        args, kwargs = cache_manager.set.call_args
        cache_key = args[0]
        cached_data = args[1]
        ttl = kwargs.get("ttl", args[2] if len(args) > 2 else None)

        assert cache_key == "doc:https://example.com/test"
        assert ttl == 86400  # 24 hours

        # Verify cached data matches result
        assert cached_data["url"] == result.url
        assert cached_data["title"] == result.title
        assert cached_data["chunks_created"] == result.chunks_created
