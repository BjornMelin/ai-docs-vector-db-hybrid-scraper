"""Comprehensive tests for Document MCP tools.

- Real-world functionality focus
- Complete coverage of document management tools
- Zero flaky tests
- Modern pytest patterns
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.config.enums import ChunkingStrategy
from src.infrastructure.client_manager import ClientManager
from src.mcp_tools.models.requests import BatchRequest, DocumentRequest
from src.mcp_tools.models.responses import AddDocumentResponse, DocumentBatchResponse
from src.mcp_tools.tools.documents import register_tools


class MockContext:
    """Mock context for testing."""

    def __init__(self):
        self.logs = {"info": [], "debug": [], "warning": [], "error": []}

    async def info(self, msg: str):
        self.logs["info"].append(msg)

    async def debug(self, msg: str):
        self.logs["debug"].append(msg)

    async def warning(self, msg: str):
        self.logs["warning"].append(msg)

    async def error(self, msg: str):
        self.logs["error"].append(msg)


@pytest.fixture
def mock_client_manager():
    """Create mock client manager with all required services."""
    manager = Mock(spec=ClientManager)

    # Mock cache manager
    mock_cache = AsyncMock()
    mock_cache.get = AsyncMock(return_value=None)  # No cached results
    manager.get_cache_manager = AsyncMock(return_value=mock_cache)

    # Mock crawl manager (browser manager)
    mock_crawl = AsyncMock()
    mock_crawl.scrape_url = AsyncMock(
        return_value={
            "content": "Sample document content for testing. This is a comprehensive document with multiple paragraphs.",
            "title": "Test Document",
            "url": "https://example.com/test-doc",
            "metadata": {
                "title": "Test Document",
                "url": "https://example.com/test-doc",
            },
            "success": True,
        }
    )
    manager.get_crawl_manager = AsyncMock(return_value=mock_crawl)

    # Mock embedding manager
    mock_embedding = AsyncMock()
    mock_embedding.embed_batch = AsyncMock(
        return_value=[[0.1, 0.2, 0.3] for _ in range(3)]
    )
    manager.get_embedding_manager = AsyncMock(return_value=mock_embedding)

    # Mock qdrant service
    mock_qdrant = AsyncMock()
    mock_qdrant.add_documents = AsyncMock(
        return_value={"success": True, "document_ids": ["doc1", "doc2", "doc3"]}
    )
    manager.get_qdrant_service = AsyncMock(return_value=mock_qdrant)

    # Mock content intelligence service
    mock_content_intel = AsyncMock()
    mock_content_intel.analyze_content = AsyncMock(
        return_value=None
    )  # Return None to skip CI analysis
    manager.get_content_intelligence_service = AsyncMock(
        return_value=mock_content_intel
    )

    return manager


@pytest.fixture
def mock_mcp():
    """Create mock MCP server."""
    mcp = Mock()
    mcp.tools = {}

    def tool_decorator():
        def decorator(func):
            mcp.tools[func.__name__] = func
            return func

        return decorator

    mcp.tool = tool_decorator
    return mcp


@pytest.fixture
def mock_context():
    """Create mock context."""
    return MockContext()


@pytest.fixture(autouse=True)
def setup_tools(mock_mcp, mock_client_manager):
    """Register tools for testing."""
    register_tools(mock_mcp, mock_client_manager)


@pytest.fixture
def sample_document_request():
    """Create sample document request."""
    return DocumentRequest(
        url="https://example.com/test-doc",
        collection="test_collection",
        chunk_strategy=ChunkingStrategy.ENHANCED,
        chunk_size=500,
        chunk_overlap=50,
    )


@pytest.fixture
def sample_batch_request():
    """Create sample batch request."""
    return BatchRequest(
        urls=[
            "https://example.com/doc1",
            "https://example.com/doc2",
            "https://example.com/doc3",
        ],
        collection="batch_collection",
        chunk_strategy=ChunkingStrategy.BASIC,
        max_concurrent=2,
    )


class TestDocumentToolRegistration:
    """Test tool registration functionality."""

    def test_all_tools_registered(self, mock_mcp):
        """Test that all expected tools are registered."""
        expected_tools = ["add_document", "add_documents_batch"]

        for tool_name in expected_tools:
            assert tool_name in mock_mcp.tools
            assert callable(mock_mcp.tools[tool_name])


class TestAddDocument:
    """Test single document addition tool."""

    async def test_successful_document_addition(
        self, mock_mcp, mock_context, mock_client_manager, sample_document_request
    ):
        """Test successful document processing and addition."""
        tool_func = mock_mcp.tools["add_document"]

        with (
            patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
            patch("src.mcp_tools.tools.documents.DocumentChunker") as mock_chunker,
            patch("src.mcp_tools.tools.documents.uuid4") as mock_uuid,
        ):
            # Setup mocks
            mock_uuid.return_value = Mock()
            mock_uuid.return_value.__str__ = Mock(return_value="test-doc-id")

            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.return_value = (
                "https://example.com/test-doc"
            )

            mock_chunker_instance = mock_chunker.return_value
            mock_chunker_instance.chunk_content.return_value = [
                {"content": "First chunk of the document content.", "metadata": {}},
                {"content": "Second chunk of the document content.", "metadata": {}},
                {"content": "Third chunk with more information.", "metadata": {}},
            ]

            # Mock embedding manager
            mock_embedding = mock_client_manager.get_embedding_manager.return_value
            mock_embedding.generate_embeddings.return_value = Mock(
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
            )

            # Execute tool
            result = await tool_func(sample_document_request, mock_context)

            # Verify successful response
            assert isinstance(result, AddDocumentResponse)
            assert result.url == "https://example.com/test-doc"
            assert result.chunks_created == 3
            assert result.collection == "test_collection"
            assert result.chunking_strategy == "enhanced"
            assert result.embedding_dimensions == 3

            # Verify service calls
            mock_client_manager.get_crawl_manager.assert_called_once()
            mock_client_manager.get_embedding_manager.assert_called_once()
            mock_client_manager.get_qdrant_service.assert_called_once()

            # Verify logging
            assert any(
                "Processing document" in msg for msg in mock_context.logs["info"]
            )
            assert any(
                "processed successfully" in msg for msg in mock_context.logs["info"]
            )

    async def test_invalid_url_rejected(
        self, mock_mcp, mock_context, sample_document_request
    ):
        """Test that invalid URLs are rejected by security validator."""
        tool_func = mock_mcp.tools["add_document"]

        with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.side_effect = ValueError(
                "URL not allowed by security policy"
            )

            try:
                await tool_func(sample_document_request, mock_context)
                raise AssertionError("Expected ValueError to be raised")
            except Exception:
                # Verify error is propagated correctly
                assert "URL not allowed" in str(e)

            # Verify error logging
            assert any(
                "Failed to process document" in msg
                for msg in mock_context.logs["error"]
            )

    async def test_crawl_manager_unavailable(
        self, mock_mcp, mock_context, mock_client_manager, sample_document_request
    ):
        """Test handling when crawl manager is unavailable."""
        tool_func = mock_mcp.tools["add_document"]
        mock_client_manager.get_crawl_manager.return_value = None

        with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.return_value = (
                "https://example.com/test-doc"
            )

            try:
                await tool_func(sample_document_request, mock_context)
                raise AssertionError("Expected AttributeError to be raised")
            except AttributeError as e:
                # Verify error is related to None crawl manager
                assert "'NoneType' object has no attribute 'scrape_url'" in str(e)

            # Verify error logging
            assert any(
                "Failed to process document" in msg
                for msg in mock_context.logs["error"]
            )

    async def test_scraping_failure(
        self, mock_mcp, mock_context, mock_client_manager, sample_document_request
    ):
        """Test handling when URL scraping fails."""
        tool_func = mock_mcp.tools["add_document"]

        # Make scraping fail
        mock_crawl = mock_client_manager.get_crawl_manager.return_value
        mock_crawl.scrape_url.return_value = {
            "content": "",
            "title": "",
            "url": "",
            "metadata": {},
            "success": False,
            "error": "Failed to scrape URL",
        }

        with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.return_value = (
                "https://example.com/test-doc"
            )

            try:
                await tool_func(sample_document_request, mock_context)
                raise AssertionError("Expected ValueError to be raised")
            except ValueError as e:
                # Verify error relates to scraping failure
                assert "Failed to scrape" in str(e)

            # Verify error logging
            assert any("Failed to scrape" in msg for msg in mock_context.logs["error"])

    async def test_embedding_failure(
        self, mock_mcp, mock_context, mock_client_manager, sample_document_request
    ):
        """Test handling when embedding generation fails."""
        tool_func = mock_mcp.tools["add_document"]

        # Make embedding fail
        mock_embedding = mock_client_manager.get_embedding_manager.return_value
        mock_embedding.generate_embeddings.side_effect = Exception(
            "Embedding service error"
        )

        with (
            patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
            patch("src.mcp_tools.tools.documents.DocumentChunker") as mock_chunker,
        ):
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.return_value = (
                "https://example.com/test-doc"
            )

            mock_chunker_instance = mock_chunker.return_value
            mock_chunker_instance.chunk_content.return_value = [
                {"content": "Test chunk", "metadata": {}}
            ]

            try:
                await tool_func(sample_document_request, mock_context)
                raise AssertionError("Expected Exception to be raised")
            except Exception:
                # Verify error relates to embedding failure
                assert "Embedding service error" in str(e)

            # Verify error logging
            assert any(
                "Failed to process document" in msg
                for msg in mock_context.logs["error"]
            )

    async def test_vector_db_storage_failure(
        self, mock_mcp, mock_context, mock_client_manager, sample_document_request
    ):
        """Test handling when vector database storage fails."""
        tool_func = mock_mcp.tools["add_document"]

        # Make vector DB storage fail
        mock_qdrant = mock_client_manager.get_qdrant_service.return_value
        mock_qdrant.upsert_points.side_effect = Exception("Vector DB storage error")

        with (
            patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
            patch("src.mcp_tools.tools.documents.DocumentChunker") as mock_chunker,
        ):
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.return_value = (
                "https://example.com/test-doc"
            )

            mock_chunker_instance = mock_chunker.return_value
            mock_chunker_instance.chunk_content.return_value = [
                {"content": "Test chunk", "metadata": {}}
            ]

            # Mock embedding manager
            mock_embedding = mock_client_manager.get_embedding_manager.return_value
            mock_embedding.generate_embeddings.return_value = Mock(
                embeddings=[[0.1, 0.2, 0.3]]
            )

            try:
                await tool_func(sample_document_request, mock_context)
                raise AssertionError("Expected Exception to be raised")
            except Exception:
                # Verify error relates to vector DB failure
                assert "Vector DB storage error" in str(e)

            # Verify error logging
            assert any(
                "Failed to process document" in msg
                for msg in mock_context.logs["error"]
            )


class TestAddDocumentBatch:
    """Test batch document processing tool."""

    async def test_successful_batch_processing(
        self, mock_mcp, mock_context, mock_client_manager, sample_batch_request
    ):
        """Test successful batch document processing."""
        tool_func = mock_mcp.tools["add_documents_batch"]

        with (
            patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
            patch("src.mcp_tools.tools.documents.DocumentChunker") as mock_chunker,
        ):
            # Setup mocks for successful processing
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.side_effect = lambda url: url

            mock_chunker_instance = mock_chunker.return_value
            mock_chunker_instance.chunk_content.return_value = [
                {"content": "Test chunk", "metadata": {}}
            ]

            # Mock embedding manager
            mock_embedding = mock_client_manager.get_embedding_manager.return_value
            mock_embedding.generate_embeddings.return_value = Mock(
                embeddings=[[0.1, 0.2, 0.3]]
            )

            result = await tool_func(sample_batch_request, mock_context)

            # Verify successful batch response
            assert isinstance(result, DocumentBatchResponse)
            assert result.total == 3
            assert len(result.successful) == 3
            assert len(result.failed) == 0

            # Verify all responses are AddDocumentResponse objects
            for response in result.successful:
                assert isinstance(response, AddDocumentResponse)
                assert response.collection == "batch_collection"

    async def test_partial_batch_failure(
        self, mock_mcp, mock_context, _mock_client_manager, sample_batch_request
    ):
        """Test batch processing with some failures."""
        tool_func = mock_mcp.tools["add_documents_batch"]

        with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
            # Make URL validation fail for second URL only
            call_count = 0

            def validate_url_side_effect(url):
                nonlocal call_count
                call_count += 1
                if call_count == 2:  # Second URL fails
                    raise ValueError("URL validation failed")
                return url

            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.side_effect = validate_url_side_effect

            result = await tool_func(sample_batch_request, mock_context)

            # Verify partial success response
            assert result.total == 3
            assert len(result.successful) == 2
            assert len(result.failed) == 1

    async def test_complete_batch_failure(
        self, mock_mcp, mock_context, sample_batch_request
    ):
        """Test batch processing when all documents fail."""
        tool_func = mock_mcp.tools["add_documents_batch"]

        with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
            # Make all URL validations fail
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.side_effect = ValueError(
                "All URLs rejected"
            )

            result = await tool_func(sample_batch_request, mock_context)

            # Verify complete failure response
            assert result.total == 3
            assert len(result.successful) == 0
            assert len(result.failed) == 3

    async def test_batch_with_concurrency_control(
        self, mock_mcp, mock_context, mock_client_manager
    ):
        """Test batch processing respects concurrency limits."""
        # Create batch request with low concurrency
        batch_request = BatchRequest(
            urls=[
                "https://example.com/doc1",
                "https://example.com/doc2",
            ],  # Reduce for faster test
            collection="test_collection",
            max_concurrent=1,  # Force sequential processing
        )

        tool_func = mock_mcp.tools["add_documents_batch"]

        with (
            patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
            patch("src.mcp_tools.tools.documents.DocumentChunker") as mock_chunker,
        ):
            # Setup mocks for successful processing
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.side_effect = lambda url: url

            mock_chunker_instance = mock_chunker.return_value
            mock_chunker_instance.chunk_content.return_value = [
                {"content": "Test chunk", "metadata": {}}
            ]

            # Mock embedding manager
            mock_embedding = mock_client_manager.get_embedding_manager.return_value
            mock_embedding.generate_embeddings.return_value = Mock(
                embeddings=[[0.1, 0.2, 0.3]]
            )

            result = await tool_func(batch_request, mock_context)

            # Verify successful processing
            assert result.total == 2
            assert len(result.successful) == 2


class TestDocumentIntegration:
    """Test integration scenarios and real-world usage patterns."""

    async def test_minimal_document_request(self, mock_mcp, mock_context):
        """Test document tool handles minimal request data correctly."""
        minimal_request = DocumentRequest(
            url="https://example.com/minimal", collection="minimal_collection"
        )

        tool_func = mock_mcp.tools["add_document"]

        with patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security:
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.side_effect = ValueError(
                "URL not allowed"
            )

            try:
                await tool_func(minimal_request, mock_context)
                raise AssertionError("Expected ValueError to be raised")
            except ValueError as e:
                # Should handle minimal data gracefully even when URL validation fails
                assert "URL not allowed" in str(e)

    async def test_comprehensive_document_workflow(
        self, mock_mcp, mock_context, mock_client_manager
    ):
        """Test a complete document processing workflow."""
        # Test single document addition
        doc_request = DocumentRequest(
            url="https://example.com/comprehensive-doc",
            collection="workflow_test",
            chunk_strategy=ChunkingStrategy.AST,
            chunk_size=800,
            chunk_overlap=80,
        )

        add_tool = mock_mcp.tools["add_document"]

        with (
            patch("src.mcp_tools.tools.documents.SecurityValidator") as mock_security,
            patch("src.mcp_tools.tools.documents.DocumentChunker") as mock_chunker,
        ):
            mock_security_instance = mock_security.from_unified_config.return_value
            mock_security_instance.validate_url.return_value = (
                "https://example.com/comprehensive-doc"
            )

            mock_chunker_instance = mock_chunker.return_value
            mock_chunker_instance.chunk_content.return_value = [
                {"content": "Comprehensive document chunk 1", "metadata": {}},
                {"content": "Comprehensive document chunk 2", "metadata": {}},
            ]

            # Mock embedding manager
            mock_embedding = mock_client_manager.get_embedding_manager.return_value
            mock_embedding.generate_embeddings.return_value = Mock(
                embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            )

            single_result = await add_tool(doc_request, mock_context)

            # Then test batch processing with related documents
            batch_request = BatchRequest(
                urls=[
                    "https://example.com/related-doc1",
                    "https://example.com/related-doc2",
                ],
                collection="workflow_test",
                chunk_strategy=ChunkingStrategy.AST,
                max_concurrent=2,
            )

            batch_tool = mock_mcp.tools["add_documents_batch"]

            batch_result = await batch_tool(batch_request, mock_context)

            # Verify complete workflow success
            assert single_result.url == "https://example.com/comprehensive-doc"
            assert single_result.chunks_created == 2
            assert batch_result.total == 2
            assert len(batch_result.successful) == 2

            # Verify consistent collection usage
            assert single_result.collection == "workflow_test"
            for response in batch_result.successful:
                assert response.collection == "workflow_test"

            # Verify comprehensive logging throughout workflow
            assert len(mock_context.logs["info"]) >= 4  # At least 2 per operation
