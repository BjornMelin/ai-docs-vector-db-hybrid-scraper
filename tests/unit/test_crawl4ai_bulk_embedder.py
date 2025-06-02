"""Comprehensive tests for crawl4ai_bulk_embedder module.

Complete rewrite focused on the new ClientManager-based architecture
with 80-90% test coverage and modern testing practices.
"""

import hashlib
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.enums import EmbeddingModel
from src.config.enums import EmbeddingProvider
from src.config.enums import SearchStrategy
from src.config.models import DocumentationSite
from src.crawl4ai_bulk_embedder import ESSENTIAL_SITES
from src.crawl4ai_bulk_embedder import ModernDocumentationScraper
from src.crawl4ai_bulk_embedder import ScrapingStats
from src.crawl4ai_bulk_embedder import VectorMetrics
from src.crawl4ai_bulk_embedder import create_advanced_config
from src.crawl4ai_bulk_embedder import main
from src.mcp.models.responses import CrawlResult


class TestVectorMetrics:
    """Test VectorMetrics Pydantic model."""

    def test_default_initialization(self):
        """Test default field values."""
        metrics = VectorMetrics()
        assert metrics.total_documents == 0
        assert metrics.total_chunks == 0
        assert metrics.successful_embeddings == 0
        assert metrics.failed_embeddings == 0
        assert metrics.processing_time == 0.0

    def test_custom_initialization(self):
        """Test initialization with custom values."""
        metrics = VectorMetrics(
            total_documents=25,
            total_chunks=150,
            successful_embeddings=140,
            failed_embeddings=10,
            processing_time=45.7,
        )
        assert metrics.total_documents == 25
        assert metrics.total_chunks == 150
        assert metrics.successful_embeddings == 140
        assert metrics.failed_embeddings == 10
        assert metrics.processing_time == 45.7

    def test_field_types(self):
        """Test field type validation."""
        metrics = VectorMetrics(
            total_documents=10,
            total_chunks=50,
            successful_embeddings=45,
            failed_embeddings=5,
            processing_time=120.5,
        )
        assert isinstance(metrics.total_documents, int)
        assert isinstance(metrics.total_chunks, int)
        assert isinstance(metrics.successful_embeddings, int)
        assert isinstance(metrics.failed_embeddings, int)
        assert isinstance(metrics.processing_time, float)


class TestScrapingStats:
    """Test ScrapingStats Pydantic model."""

    def test_default_initialization(self):
        """Test default field values."""
        stats = ScrapingStats()
        assert stats.total_processed == 0
        assert stats.successful_embeddings == 0
        assert stats.failed_crawls == 0
        assert stats.total_chunks == 0
        assert stats.unique_urls == 0
        assert stats.start_time is None
        assert stats.end_time is None

    def test_with_timestamps(self):
        """Test initialization with timestamps."""
        start = datetime.now()
        end = datetime.now()
        stats = ScrapingStats(
            total_processed=20,
            successful_embeddings=18,
            failed_crawls=2,
            total_chunks=80,
            unique_urls=20,
            start_time=start,
            end_time=end,
        )
        assert stats.total_processed == 20
        assert stats.successful_embeddings == 18
        assert stats.failed_crawls == 2
        assert stats.total_chunks == 80
        assert stats.unique_urls == 20
        assert stats.start_time == start
        assert stats.end_time == end

    def test_stats_modification(self):
        """Test modifying stats after creation."""
        stats = ScrapingStats()
        stats.total_processed = 5
        stats.unique_urls = 3
        assert stats.total_processed == 5
        assert stats.unique_urls == 3


class TestModernDocumentationScraper:
    """Comprehensive tests for ModernDocumentationScraper class."""

    @pytest.fixture
    def mock_config(self):
        """Create comprehensive mock configuration."""
        config = Mock()
        # Firecrawl configuration
        config.firecrawl.api_key = None
        # Chunking configuration
        config.chunking = Mock()
        # Embedding configuration
        config.embedding.enable_reranking = False
        config.embedding.reranker_model = "BAAI/bge-reranker-v2-m3"
        config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        config.embedding.provider = EmbeddingProvider.OPENAI
        config.embedding.search_strategy = SearchStrategy.DENSE
        config.embedding.enable_quantization = False
        # Qdrant configuration
        config.qdrant.collection_name = "test_collection"
        config.qdrant.quantization_enabled = False
        config.qdrant.sparse_vector_name = "sparse"
        # Crawl4AI configuration
        config.crawl4ai.max_concurrent_crawls = 3
        return config

    @pytest.fixture
    def mock_client_manager(self):
        """Create comprehensive mock ClientManager."""
        client_manager = Mock()
        client_manager.initialize = AsyncMock()
        client_manager.cleanup = AsyncMock()

        # Mock service getters
        client_manager.get_embedding_manager = AsyncMock()
        client_manager.get_qdrant_service = AsyncMock()
        client_manager.get_crawl_manager = AsyncMock()

        return client_manager

    @pytest.fixture
    def scraper(self, mock_config, mock_client_manager):
        """Create scraper instance with all mocked dependencies."""
        with (
            patch("src.crawl4ai_bulk_embedder.EnhancedChunker") as mock_chunker,
            patch("logging.getLogger") as mock_logger,
        ):
            scraper = ModernDocumentationScraper(mock_config, mock_client_manager)
            scraper.chunker = mock_chunker.return_value
            scraper.logger = mock_logger.return_value

            # Pre-populate service instances for testing
            scraper.embedding_manager = Mock()
            scraper.qdrant_service = Mock()
            scraper.crawl_manager = Mock()

            return scraper

    def test_initialization_basic(self, mock_config, mock_client_manager):
        """Test basic scraper initialization."""
        with patch("src.crawl4ai_bulk_embedder.EnhancedChunker"):
            scraper = ModernDocumentationScraper(mock_config, mock_client_manager)

            assert scraper.config == mock_config
            assert scraper.client_manager == mock_client_manager
            assert isinstance(scraper.stats, ScrapingStats)
            assert isinstance(scraper.processed_urls, set)
            assert len(scraper.processed_urls) == 0
            assert scraper.firecrawl_client is None
            assert scraper.reranker is None

    def test_initialization_with_firecrawl(self, mock_config, mock_client_manager):
        """Test initialization with Firecrawl API key."""
        mock_config.firecrawl.api_key = "fc-test-key"

        with (
            patch("src.crawl4ai_bulk_embedder.EnhancedChunker"),
            patch("src.crawl4ai_bulk_embedder.FirecrawlApp") as mock_firecrawl,
        ):
            scraper = ModernDocumentationScraper(mock_config, mock_client_manager)

            mock_firecrawl.assert_called_once_with(api_key="fc-test-key")
            assert scraper.firecrawl_client is not None

    def test_initialization_without_firecrawl_import(self, mock_config, mock_client_manager):
        """Test initialization when FirecrawlApp is not available."""
        mock_config.firecrawl.api_key = "fc-test-key"

        with (
            patch("src.crawl4ai_bulk_embedder.EnhancedChunker"),
            patch("src.crawl4ai_bulk_embedder.FirecrawlApp", None),
        ):
            scraper = ModernDocumentationScraper(mock_config, mock_client_manager)
            assert scraper.firecrawl_client is None

    @pytest.mark.asyncio
    async def test_initialize_services(self, scraper):
        """Test service initialization through ClientManager."""
        # Mock service instances
        mock_embedding_manager = Mock()
        mock_qdrant_service = Mock()
        mock_crawl_manager = Mock()

        scraper.client_manager.get_embedding_manager.return_value = mock_embedding_manager
        scraper.client_manager.get_qdrant_service.return_value = mock_qdrant_service
        scraper.client_manager.get_crawl_manager.return_value = mock_crawl_manager

        await scraper.initialize()

        # Verify ClientManager initialization
        scraper.client_manager.initialize.assert_called_once()

        # Verify service getter calls
        scraper.client_manager.get_embedding_manager.assert_called_once()
        scraper.client_manager.get_qdrant_service.assert_called_once()
        scraper.client_manager.get_crawl_manager.assert_called_once()

        # Verify services are assigned
        assert scraper.embedding_manager == mock_embedding_manager
        assert scraper.qdrant_service == mock_qdrant_service
        assert scraper.crawl_manager == mock_crawl_manager

    @pytest.mark.asyncio
    async def test_cleanup_services(self, scraper):
        """Test service cleanup through ClientManager."""
        await scraper.cleanup()
        scraper.client_manager.cleanup.assert_called_once()

    def test_initialize_reranker_disabled(self, scraper):
        """Test reranker initialization when disabled."""
        scraper.config.embedding.enable_reranking = False
        scraper._initialize_reranker()
        assert scraper.reranker is None

    def test_initialize_reranker_missing_dependency(self, scraper):
        """Test reranker initialization with missing FlagReranker."""
        scraper.config.embedding.enable_reranking = True

        with patch("src.crawl4ai_bulk_embedder.FlagReranker", None):
            scraper._initialize_reranker()
            assert scraper.reranker is None
            assert scraper.config.embedding.enable_reranking is False

    def test_initialize_reranker_success(self, scraper):
        """Test successful reranker initialization."""
        scraper.config.embedding.enable_reranking = True
        mock_reranker = Mock()

        with patch("src.crawl4ai_bulk_embedder.FlagReranker", return_value=mock_reranker):
            scraper._initialize_reranker()
            assert scraper.reranker == mock_reranker

    def test_initialize_reranker_exception(self, scraper):
        """Test reranker initialization with exception."""
        scraper.config.embedding.enable_reranking = True

        with patch(
            "src.crawl4ai_bulk_embedder.FlagReranker",
            side_effect=Exception("Reranker init failed"),
        ):
            scraper._initialize_reranker()
            assert scraper.reranker is None
            assert scraper.config.embedding.enable_reranking is False

    @pytest.mark.asyncio
    async def test_setup_collection_new(self, scraper):
        """Test collection setup when collection doesn't exist."""
        scraper.qdrant_service.list_collections = AsyncMock(return_value=[])
        scraper.qdrant_service.create_collection = AsyncMock()
        scraper.config.qdrant.collection_name = "new_collection"
        scraper.config.embedding.search_strategy = SearchStrategy.DENSE

        await scraper.setup_collection()

        scraper.qdrant_service.list_collections.assert_called_once()
        scraper.qdrant_service.create_collection.assert_called_once()

    @pytest.mark.asyncio
    async def test_setup_collection_exists(self, scraper):
        """Test collection setup when collection already exists."""
        scraper.qdrant_service.list_collections = AsyncMock(
            return_value=["test_collection"]
        )
        scraper.qdrant_service.create_collection = AsyncMock()
        scraper.config.qdrant.collection_name = "test_collection"

        await scraper.setup_collection()

        scraper.qdrant_service.list_collections.assert_called_once()
        scraper.qdrant_service.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_collection_with_hybrid_search(self, scraper):
        """Test collection setup with hybrid search strategy."""
        scraper.qdrant_service.list_collections = AsyncMock(return_value=[])
        scraper.qdrant_service.create_collection = AsyncMock()
        scraper.config.qdrant.collection_name = "hybrid_collection"
        scraper.config.embedding.search_strategy = SearchStrategy.HYBRID

        await scraper.setup_collection()

        scraper.qdrant_service.create_collection.assert_called_once()
        call_args = scraper.qdrant_service.create_collection.call_args
        assert call_args.kwargs["enable_sparse"] is True

    def test_get_vector_size_openai_models(self, scraper):
        """Test vector size calculation for OpenAI models."""
        # Test TEXT_EMBEDDING_3_SMALL
        scraper.config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert scraper._get_vector_size() == 1536

        # Test TEXT_EMBEDDING_3_LARGE
        scraper.config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_LARGE
        assert scraper._get_vector_size() == 1536

    def test_get_vector_size_bge_models(self, scraper):
        """Test vector size calculation for BGE models."""
        # Test BGE small model
        scraper.config.embedding.dense_model = EmbeddingModel.BGE_SMALL_EN_V15
        assert scraper._get_vector_size() == 384

        # Test BGE large model
        scraper.config.embedding.dense_model = EmbeddingModel.BGE_LARGE_EN_V15
        assert scraper._get_vector_size() == 1024

    def test_get_vector_size_nvidia_model(self, scraper):
        """Test vector size calculation for NVIDIA model."""
        scraper.config.embedding.dense_model = EmbeddingModel.NV_EMBED_V2
        assert scraper._get_vector_size() == 4096

    def test_get_vector_size_unknown_model(self, scraper):
        """Test vector size calculation for unknown model (fallback)."""
        scraper.config.embedding.dense_model = Mock()
        scraper.config.embedding.dense_model.value = "unknown-model"
        assert scraper._get_vector_size() == 1536

    @pytest.mark.asyncio
    async def test_create_embedding_dense_only(self, scraper):
        """Test embedding creation for dense-only strategy."""
        scraper.config.embedding.search_strategy = SearchStrategy.DENSE
        mock_result = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "cost": 0.001,
        }
        scraper.embedding_manager.generate_embeddings = AsyncMock(return_value=mock_result)

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        assert sparse_data is None
        scraper.embedding_manager.generate_embeddings.assert_called_once_with(
            ["test text"], generate_sparse=False
        )

    @pytest.mark.asyncio
    async def test_create_embedding_hybrid_with_sparse(self, scraper):
        """Test embedding creation for hybrid strategy with sparse embeddings."""
        scraper.config.embedding.search_strategy = SearchStrategy.HYBRID
        mock_result = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "sparse_embeddings": [{"indices": [1, 2], "values": [0.5, 0.8]}],
            "provider": "fastembed",
            "model": "bge-small-en-v1.5",
            "cost": 0.0,
        }
        scraper.embedding_manager.generate_embeddings = AsyncMock(return_value=mock_result)

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        assert sparse_data == {"sparse": {"indices": [1, 2], "values": [0.5, 0.8]}}
        scraper.embedding_manager.generate_embeddings.assert_called_once_with(
            ["test text"], generate_sparse=True
        )

    @pytest.mark.asyncio
    async def test_create_embedding_empty_result(self, scraper):
        """Test embedding creation with empty result."""
        scraper.config.embedding.search_strategy = SearchStrategy.DENSE
        mock_result = {"embeddings": [], "provider": "openai", "cost": 0.0}
        scraper.embedding_manager.generate_embeddings = AsyncMock(return_value=mock_result)

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == []
        assert sparse_data is None

    @pytest.mark.asyncio
    async def test_create_embedding_text_truncation(self, scraper):
        """Test text truncation for long inputs."""
        scraper.config.embedding.search_strategy = SearchStrategy.DENSE
        long_text = "x" * 10000  # Text longer than 8000 chars
        mock_result = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "openai",
            "cost": 0.001,
        }
        scraper.embedding_manager.generate_embeddings = AsyncMock(return_value=mock_result)

        await scraper.create_embedding(long_text)

        # Verify text was truncated to 8000 characters
        call_args = scraper.embedding_manager.generate_embeddings.call_args[0][0]
        assert len(call_args[0]) == 8000

    @pytest.mark.asyncio
    async def test_create_embedding_exception_handling(self, scraper):
        """Test embedding creation exception handling."""
        scraper.config.embedding.search_strategy = SearchStrategy.DENSE
        scraper.embedding_manager.generate_embeddings = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == []
        assert sparse_data is None

    def test_generate_point_id_consistency(self, scraper):
        """Test point ID generation consistency and collision resistance."""
        url = "https://example.com/docs/page1"
        chunk_index = 5

        # Generate ID
        point_id = scraper._generate_point_id(url, chunk_index)

        # Verify format: MD5 hash + chunk index
        expected_hash = hashlib.md5(url.encode()).hexdigest()
        expected_id = f"{expected_hash}_{chunk_index}"
        assert point_id == expected_id

        # Test consistency
        point_id2 = scraper._generate_point_id(url, chunk_index)
        assert point_id == point_id2

        # Test different inputs produce different IDs
        point_id3 = scraper._generate_point_id(url, chunk_index + 1)
        assert point_id != point_id3

        point_id4 = scraper._generate_point_id("https://different.com", chunk_index)
        assert point_id != point_id4

    def test_create_qdrant_point_dense_only(self, scraper):
        """Test Qdrant point creation for dense-only strategy."""
        from qdrant_client.models import PointStruct

        scraper.config.embedding.search_strategy = SearchStrategy.DENSE
        point_id = "test_point_123"
        dense_embedding = [0.1, 0.2, 0.3]
        payload = {"content": "test content", "url": "https://example.com"}

        point = scraper._create_qdrant_point(point_id, dense_embedding, None, payload)

        assert isinstance(point, PointStruct)
        assert point.id == point_id
        assert point.vector == dense_embedding
        assert point.payload == payload

    def test_create_qdrant_point_hybrid_with_sparse(self, scraper):
        """Test Qdrant point creation for hybrid strategy with sparse vectors."""
        from qdrant_client.models import PointStruct
        from qdrant_client.models import SparseVector

        scraper.config.embedding.search_strategy = SearchStrategy.HYBRID
        scraper.config.qdrant.sparse_vector_name = "sparse"

        point_id = "test_point_456"
        dense_embedding = [0.1, 0.2, 0.3]
        sparse_data = {"sparse": {"indices": [1, 2], "values": [0.5, 0.8]}}
        payload = {"content": "test content", "url": "https://example.com"}

        point = scraper._create_qdrant_point(point_id, dense_embedding, sparse_data, payload)

        assert isinstance(point, PointStruct)
        assert point.id == point_id
        assert isinstance(point.vector, dict)
        assert "dense" in point.vector
        assert "sparse" in point.vector
        assert point.vector["dense"] == dense_embedding
        assert isinstance(point.vector["sparse"], SparseVector)
        assert point.payload == payload

    def test_create_qdrant_point_hybrid_without_sparse_data(self, scraper):
        """Test Qdrant point creation for hybrid strategy without sparse data."""
        from qdrant_client.models import PointStruct

        scraper.config.embedding.search_strategy = SearchStrategy.HYBRID
        point_id = "test_point_789"
        dense_embedding = [0.1, 0.2, 0.3]
        payload = {"content": "test content", "url": "https://example.com"}

        # No sparse data provided
        point = scraper._create_qdrant_point(point_id, dense_embedding, None, payload)

        assert isinstance(point, PointStruct)
        assert point.id == point_id
        assert point.vector == dense_embedding  # Falls back to dense-only
        assert point.payload == payload

    def test_rerank_results_disabled(self, scraper):
        """Test reranking when disabled."""
        scraper.config.embedding.enable_reranking = False
        scraper.reranker = None
        passages = [{"content": "passage1"}, {"content": "passage2"}]

        result = scraper.rerank_results("test query", passages)

        assert result == passages

    def test_rerank_results_empty_input(self, scraper):
        """Test reranking with empty passages."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()

        result = scraper.rerank_results("test query", [])

        assert result == []

    def test_rerank_results_single_passage(self, scraper):
        """Test reranking with single passage."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()
        passages = [{"content": "single passage"}]

        result = scraper.rerank_results("test query", passages)

        assert result == passages

    def test_rerank_results_multiple_passages(self, scraper):
        """Test reranking with multiple passages."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()
        scraper.reranker.compute_score.return_value = [0.9, 0.7, 0.8]

        passages = [
            {"content": "passage1"},
            {"content": "passage2"},
            {"content": "passage3"},
        ]

        result = scraper.rerank_results("test query", passages)

        # Should be reordered by score (highest first)
        assert len(result) == 3
        assert result[0]["content"] == "passage1"  # Score 0.9
        assert result[1]["content"] == "passage3"  # Score 0.8
        assert result[2]["content"] == "passage2"  # Score 0.7

    def test_rerank_results_exception_handling(self, scraper):
        """Test reranking exception handling."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()
        scraper.reranker.compute_score.side_effect = Exception("Reranking failed")

        passages = [{"content": "passage1"}, {"content": "passage2"}]
        result = scraper.rerank_results("test query", passages)

        # Should return original order on exception
        assert result == passages

    def test_chunk_content_delegation(self, scraper):
        """Test content chunking delegation to EnhancedChunker."""
        expected_chunks = [
            {"content": "chunk1", "chunk_index": 0},
            {"content": "chunk2", "chunk_index": 1},
        ]
        scraper.chunker.chunk_content.return_value = expected_chunks

        result = scraper.chunk_content("test content", "test title", "test url")

        assert result == expected_chunks
        scraper.chunker.chunk_content.assert_called_once_with(
            "test content", "test title", "test url"
        )

    def test_create_filter_chain(self, scraper):
        """Test filter chain creation for documentation sites."""
        site = DocumentationSite(
            name="test_site",
            url="https://example.com",
            url_patterns=["*docs*", "*guides*"],
        )

        with (
            patch("src.crawl4ai_bulk_embedder.URLPatternFilter") as mock_url_filter,
            patch("src.crawl4ai_bulk_embedder.ContentTypeFilter") as mock_content_filter,
            patch("src.crawl4ai_bulk_embedder.FilterChain") as mock_filter_chain,
        ):
            scraper.create_filter_chain(site)

            mock_url_filter.assert_called_once_with(patterns=["*docs*", "*guides*"])
            mock_content_filter.assert_called_once_with(allowed_types=["text/html"])
            mock_filter_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_documentation_site_success(self, scraper):
        """Test successful documentation site crawling."""
        site = DocumentationSite(
            name="test_site", url="https://example.com", max_pages=10
        )

        mock_crawl_result = {
            "success": True,
            "total": 2,
            "provider": "crawl4ai",
            "pages": [
                {
                    "url": "https://example.com/page1",
                    "title": "Page 1",
                    "content": "Content 1",
                    "metadata": {"depth": 1},
                },
                {
                    "url": "https://example.com/page2",
                    "title": "Page 2",
                    "content": "Content 2",
                    "metadata": {"depth": 2},
                },
            ],
        }

        scraper.crawl_manager.crawl_site = AsyncMock(return_value=mock_crawl_result)

        results = await scraper.crawl_documentation_site(site)

        assert len(results) == 2
        assert all(isinstance(result, CrawlResult) for result in results)
        assert results[0].url == "https://example.com/page1"
        assert results[1].url == "https://example.com/page2"
        assert "https://example.com/page1" in scraper.processed_urls
        assert "https://example.com/page2" in scraper.processed_urls
        assert scraper.stats.unique_urls == 2

    @pytest.mark.asyncio
    async def test_crawl_documentation_site_failure(self, scraper):
        """Test documentation site crawling failure."""
        site = DocumentationSite(
            name="test_site", url="https://example.com", max_pages=10
        )

        mock_crawl_result = {"success": False, "error": "Network error"}
        scraper.crawl_manager.crawl_site = AsyncMock(return_value=mock_crawl_result)

        results = await scraper.crawl_documentation_site(site)

        assert len(results) == 0
        assert scraper.stats.failed_crawls == 1

    @pytest.mark.asyncio
    async def test_crawl_documentation_site_exception(self, scraper):
        """Test documentation site crawling with exception."""
        site = DocumentationSite(
            name="test_site", url="https://example.com", max_pages=10
        )

        scraper.crawl_manager.crawl_site = AsyncMock(
            side_effect=Exception("Crawling error")
        )

        results = await scraper.crawl_documentation_site(site)

        assert len(results) == 0
        assert scraper.stats.failed_crawls == 1

    def test_display_comprehensive_stats(self, scraper):
        """Test comprehensive stats display."""
        # Set up test data
        scraper.stats.total_processed = 15
        scraper.stats.unique_urls = 12
        scraper.stats.successful_embeddings = 75
        scraper.stats.total_chunks = 80
        scraper.stats.failed_crawls = 3
        scraper.stats.start_time = datetime(2024, 1, 1, 10, 0, 0)
        scraper.stats.end_time = datetime(2024, 1, 1, 10, 10, 0)

        with patch("src.crawl4ai_bulk_embedder.console") as mock_console:
            scraper.display_comprehensive_stats()

            # Verify console output was called multiple times
            assert mock_console.print.call_count >= 3

    @pytest.mark.asyncio
    async def test_demo_reranking_search_disabled(self, scraper):
        """Test demo reranking search when disabled."""
        scraper.config.embedding.enable_reranking = False
        scraper.reranker = None

        with patch("src.crawl4ai_bulk_embedder.console") as mock_console:
            result = await scraper.demo_reranking_search("test query")

            assert result == []
            mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_demo_reranking_search_enabled(self, scraper):
        """Test demo reranking search when enabled."""
        scraper.config.embedding.enable_reranking = True
        scraper.config.embedding.rerank_top_k = 10
        scraper.reranker = Mock()
        scraper.reranker.compute_score.return_value = [0.9, 0.8, 0.7]

        with patch("src.crawl4ai_bulk_embedder.console") as mock_console:
            result = await scraper.demo_reranking_search("test query", limit=3)

            assert len(result) <= 3
            mock_console.print.assert_called()


class TestCreateAdvancedConfig:
    """Test create_advanced_config function with new architecture."""

    @patch("src.crawl4ai_bulk_embedder.get_config")
    @patch("src.crawl4ai_bulk_embedder.sys.exit")
    def test_missing_openai_api_key(self, mock_exit, mock_get_config):
        """Test config creation with missing OpenAI API key."""
        mock_config = Mock()
        mock_config.openai.api_key = None
        mock_get_config.return_value = mock_config

        with patch("src.crawl4ai_bulk_embedder.console") as mock_console:
            create_advanced_config()

            mock_console.print.assert_called()
            mock_exit.assert_called_once_with(1)

    @patch("src.crawl4ai_bulk_embedder.get_config")
    def test_fastembed_provider_configuration(self, mock_get_config):
        """Test configuration for FastEmbed provider."""
        mock_config = Mock()
        mock_config.openai.api_key = "test-key"
        mock_config.embedding_provider = EmbeddingProvider.FASTEMBED
        mock_get_config.return_value = mock_config

        with (
            patch("importlib.util.find_spec", return_value=None),
            patch("src.crawl4ai_bulk_embedder.console"),
        ):
            result = create_advanced_config()

            # Should return the same config without modification
            assert result == mock_config

    @patch("src.crawl4ai_bulk_embedder.get_config")
    def test_fastembed_with_sparse_available(self, mock_get_config):
        """Test FastEmbed configuration with sparse models available."""
        mock_config = Mock()
        mock_config.openai.api_key = "test-key"
        mock_config.embedding_provider = EmbeddingProvider.FASTEMBED
        mock_get_config.return_value = mock_config

        mock_spec = Mock()
        with (
            patch("importlib.util.find_spec", return_value=mock_spec),
            patch("src.crawl4ai_bulk_embedder.console"),
        ):
            result = create_advanced_config()

            assert result == mock_config

    @patch("src.crawl4ai_bulk_embedder.get_config")
    def test_openai_provider_configuration(self, mock_get_config):
        """Test configuration for OpenAI provider."""
        mock_config = Mock()
        mock_config.openai.api_key = "test-key"
        mock_config.embedding_provider = EmbeddingProvider.OPENAI
        mock_get_config.return_value = mock_config

        with patch("src.crawl4ai_bulk_embedder.console"):
            result = create_advanced_config()

            assert result == mock_config


class TestMainFunction:
    """Test main function execution with new architecture."""

    @pytest.mark.asyncio
    @patch("src.crawl4ai_bulk_embedder.create_advanced_config")
    @patch("src.infrastructure.client_manager.ClientManager")
    @patch("src.crawl4ai_bulk_embedder.ModernDocumentationScraper")
    async def test_main_execution_success(
        self, mock_scraper_class, mock_client_manager_class, mock_create_config
    ):
        """Test successful main function execution."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embedding.provider.value = "openai"
        mock_config.embedding.dense_model.value = "text-embedding-3-small"
        mock_config.embedding.search_strategy.value = "dense"
        mock_config.chunking.chunk_size = 1600
        mock_config.embedding.enable_quantization = False
        mock_config.embedding.enable_reranking = False
        mock_config.firecrawl.api_key = None
        mock_config.documentation_sites = None
        mock_create_config.return_value = mock_config

        mock_client_manager = Mock()
        mock_client_manager_class.return_value = mock_client_manager

        mock_scraper = AsyncMock()
        mock_scraper_class.return_value = mock_scraper

        with (
            patch("src.crawl4ai_bulk_embedder.console"),
            patch("src.crawl4ai_bulk_embedder.ESSENTIAL_SITES", []),
        ):
            await main()

            # Verify proper initialization sequence
            mock_client_manager_class.assert_called_once_with(mock_config)
            mock_scraper_class.assert_called_once_with(mock_config, mock_client_manager)
            mock_scraper.initialize.assert_called_once()
            mock_scraper.scrape_multiple_sites.assert_called_once()
            mock_scraper.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.crawl4ai_bulk_embedder.create_advanced_config")
    @patch("src.infrastructure.client_manager.ClientManager")
    @patch("src.crawl4ai_bulk_embedder.ModernDocumentationScraper")
    @patch("src.crawl4ai_bulk_embedder.sys.exit")
    async def test_main_execution_failure(
        self, mock_exit, mock_scraper_class, mock_client_manager_class, mock_create_config
    ):
        """Test main function execution with failure."""
        # Setup mocks
        mock_config = Mock()
        mock_config.embedding.provider.value = "openai"
        mock_config.embedding.dense_model.value = "text-embedding-3-small"
        mock_config.embedding.search_strategy.value = "dense"
        mock_config.chunking.chunk_size = 1600
        mock_config.embedding.enable_quantization = False
        mock_config.embedding.enable_reranking = False
        mock_config.firecrawl.api_key = None
        mock_config.documentation_sites = None
        mock_create_config.return_value = mock_config

        mock_client_manager = Mock()
        mock_client_manager_class.return_value = mock_client_manager

        mock_scraper = AsyncMock()
        mock_scraper.initialize.side_effect = Exception("Initialization failed")
        mock_scraper_class.return_value = mock_scraper

        with patch("src.crawl4ai_bulk_embedder.console"):
            await main()

            # Verify error handling
            mock_scraper.display_comprehensive_stats.assert_called_once()
            mock_scraper.cleanup.assert_called_once()
            mock_exit.assert_called_once_with(1)

    @pytest.mark.asyncio
    @patch("src.crawl4ai_bulk_embedder.create_advanced_config")
    @patch("src.infrastructure.client_manager.ClientManager")
    @patch("src.crawl4ai_bulk_embedder.ModernDocumentationScraper")
    async def test_main_with_custom_sites(
        self, mock_scraper_class, mock_client_manager_class, mock_create_config
    ):
        """Test main function with custom documentation sites."""
        # Setup config with custom sites
        mock_config = Mock()
        mock_config.embedding.provider.value = "fastembed"
        mock_config.embedding.dense_model.value = "bge-small-en-v1.5"
        mock_config.embedding.search_strategy.value = "hybrid"
        mock_config.chunking.chunk_size = 1600
        mock_config.embedding.enable_quantization = True
        mock_config.embedding.enable_reranking = True
        mock_config.firecrawl.api_key = "fc-test-key"

        custom_sites = [
            DocumentationSite(name="Custom Site", url="https://custom.com")
        ]
        mock_config.documentation_sites = custom_sites
        mock_create_config.return_value = mock_config

        mock_client_manager = Mock()
        mock_client_manager_class.return_value = mock_client_manager

        mock_scraper = AsyncMock()
        mock_scraper_class.return_value = mock_scraper

        with patch("src.crawl4ai_bulk_embedder.console"):
            await main()

            # Verify custom sites are used
            mock_scraper.scrape_multiple_sites.assert_called_once_with(custom_sites)


class TestEssentialSites:
    """Test ESSENTIAL_SITES configuration."""

    def test_essential_sites_structure(self):
        """Test that ESSENTIAL_SITES has proper structure."""
        assert isinstance(ESSENTIAL_SITES, list)
        assert len(ESSENTIAL_SITES) > 0

        for site in ESSENTIAL_SITES:
            assert isinstance(site, DocumentationSite)
            assert site.name
            assert site.url
            assert isinstance(site.max_pages, int)
            assert site.max_pages > 0

    def test_essential_sites_content(self):
        """Test specific essential sites content."""
        site_names = {site.name for site in ESSENTIAL_SITES}

        # Check for key documentation sites
        expected_sites = {
            "Qdrant Documentation",
            "FastEmbed Documentation",
            "Crawl4AI Documentation",
            "Pydantic v2 Documentation",
            "OpenAI Embeddings Documentation",
        }

        for expected_site in expected_sites:
            assert expected_site in site_names, f"Missing essential site: {expected_site}"
