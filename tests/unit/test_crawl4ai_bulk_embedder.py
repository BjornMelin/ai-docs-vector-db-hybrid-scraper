"""Unit tests for crawl4ai_bulk_embedder module."""

from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.config.enums import EmbeddingModel
from src.config.enums import EmbeddingProvider
from src.config.enums import SearchStrategy
from src.config.models import DocumentationSite
from src.crawl4ai_bulk_embedder import ModernDocumentationScraper
from src.crawl4ai_bulk_embedder import ScrapingStats
from src.crawl4ai_bulk_embedder import VectorMetrics
from src.crawl4ai_bulk_embedder import create_advanced_config
from src.mcp.models.responses import CrawlResult


class TestVectorMetrics:
    """Test VectorMetrics model."""

    def test_default_values(self):
        """Test default field values."""
        metrics = VectorMetrics()
        assert metrics.total_documents == 0
        assert metrics.total_chunks == 0
        assert metrics.successful_embeddings == 0
        assert metrics.failed_embeddings == 0
        assert metrics.processing_time == 0.0

    def test_field_validation(self):
        """Test field validation."""
        metrics = VectorMetrics(
            total_documents=10,
            total_chunks=50,
            successful_embeddings=45,
            failed_embeddings=5,
            processing_time=120.5,
        )
        assert metrics.total_documents == 10
        assert metrics.total_chunks == 50
        assert metrics.successful_embeddings == 45
        assert metrics.failed_embeddings == 5
        assert metrics.processing_time == 120.5

    def test_negative_values_allowed(self):
        """Test that negative values are allowed (for edge cases)."""
        # Pydantic v2 allows negative values by default unless constrained
        metrics = VectorMetrics(total_documents=-1)
        assert metrics.total_documents == -1


class TestScrapingStats:
    """Test ScrapingStats model."""

    def test_default_values(self):
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
        """Test with start and end timestamps."""
        start = datetime.now()
        end = datetime.now()
        stats = ScrapingStats(
            total_processed=10,
            successful_embeddings=8,
            failed_crawls=2,
            total_chunks=40,
            unique_urls=10,
            start_time=start,
            end_time=end,
        )
        assert stats.start_time == start
        assert stats.end_time == end
        assert stats.total_processed == 10


class TestModernDocumentationScraper:
    """Test ModernDocumentationScraper class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.firecrawl.api_key = None
        config.chunking = Mock()
        config.embedding.enable_reranking = False
        config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        config.embedding.provider = EmbeddingProvider.OPENAI
        config.embedding.search_strategy = SearchStrategy.DENSE
        config.embedding.enable_quantization = False
        config.qdrant.collection_name = "test_collection"
        config.qdrant.quantization_enabled = False
        config.crawl4ai.max_concurrent_crawls = 3
        return config

    @pytest.fixture
    def scraper(self, mock_config):
        """Create scraper instance with mocked dependencies."""
        with (
            patch("src.crawl4ai_bulk_embedder.RateLimitManager") as mock_rate_limiter,
            patch(
                "src.crawl4ai_bulk_embedder.EmbeddingManager"
            ) as mock_embedding_manager,
            patch("src.crawl4ai_bulk_embedder.QdrantService") as mock_qdrant_service,
            patch("src.crawl4ai_bulk_embedder.CrawlManager") as mock_crawl_manager,
            patch("src.crawl4ai_bulk_embedder.EnhancedChunker") as mock_chunker,
        ):
            scraper = ModernDocumentationScraper(mock_config)
            scraper.rate_limiter = mock_rate_limiter.return_value
            scraper.embedding_manager = mock_embedding_manager.return_value
            scraper.qdrant_service = mock_qdrant_service.return_value
            scraper.crawl_manager = mock_crawl_manager.return_value
            scraper.chunker = mock_chunker.return_value
            return scraper

    def test_init_without_firecrawl(self, mock_config):
        """Test initialization without Firecrawl API key."""
        mock_config.firecrawl.api_key = None
        with (
            patch("src.crawl4ai_bulk_embedder.RateLimitManager"),
            patch("src.crawl4ai_bulk_embedder.EmbeddingManager"),
            patch("src.crawl4ai_bulk_embedder.QdrantService"),
            patch("src.crawl4ai_bulk_embedder.CrawlManager"),
            patch("src.crawl4ai_bulk_embedder.EnhancedChunker"),
        ):
            scraper = ModernDocumentationScraper(mock_config)
            assert scraper.firecrawl_client is None
            assert scraper.config == mock_config
            assert isinstance(scraper.stats, ScrapingStats)
            assert isinstance(scraper.processed_urls, set)

    def test_init_with_firecrawl(self, mock_config):
        """Test initialization with Firecrawl API key."""
        mock_config.firecrawl.api_key = "fc-test-key"
        with (
            patch("src.crawl4ai_bulk_embedder.RateLimitManager"),
            patch("src.crawl4ai_bulk_embedder.EmbeddingManager"),
            patch("src.crawl4ai_bulk_embedder.QdrantService"),
            patch("src.crawl4ai_bulk_embedder.CrawlManager"),
            patch("src.crawl4ai_bulk_embedder.EnhancedChunker"),
            patch("src.crawl4ai_bulk_embedder.FirecrawlApp") as mock_firecrawl,
        ):
            scraper = ModernDocumentationScraper(mock_config)
            mock_firecrawl.assert_called_once_with(api_key="fc-test-key")
            assert scraper.firecrawl_client is not None

    @pytest.mark.asyncio
    async def test_initialize(self, scraper):
        """Test service initialization."""
        scraper.rate_limiter.initialize = AsyncMock()
        scraper.embedding_manager.initialize = AsyncMock()
        scraper.qdrant_service.initialize = AsyncMock()
        scraper.crawl_manager.initialize = AsyncMock()

        await scraper.initialize()

        scraper.rate_limiter.initialize.assert_called_once()
        scraper.embedding_manager.initialize.assert_called_once()
        scraper.qdrant_service.initialize.assert_called_once()
        scraper.crawl_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self, scraper):
        """Test service cleanup."""
        scraper.embedding_manager.cleanup = AsyncMock()
        scraper.qdrant_service.cleanup = AsyncMock()
        scraper.crawl_manager.cleanup = AsyncMock()

        await scraper.cleanup()

        scraper.embedding_manager.cleanup.assert_called_once()
        scraper.qdrant_service.cleanup.assert_called_once()
        scraper.crawl_manager.cleanup.assert_called_once()

    def test_initialize_reranker_disabled(self, scraper):
        """Test reranker initialization when disabled."""
        scraper.config.embedding.enable_reranking = False
        scraper._initialize_reranker()
        assert scraper.reranker is None

    def test_initialize_reranker_missing_dependency(self, scraper):
        """Test reranker initialization with missing dependency."""
        scraper.config.embedding.enable_reranking = True
        with patch("src.crawl4ai_bulk_embedder.FlagReranker", None):
            scraper._initialize_reranker()
            assert scraper.reranker is None
            assert scraper.config.embedding.enable_reranking is False

    def test_initialize_reranker_success(self, scraper):
        """Test successful reranker initialization."""
        scraper.config.embedding.enable_reranking = True
        scraper.config.embedding.reranker_model = "test-model"

        mock_reranker = Mock()
        with patch(
            "src.crawl4ai_bulk_embedder.FlagReranker", return_value=mock_reranker
        ):
            scraper._initialize_reranker()
            assert scraper.reranker == mock_reranker

    def test_initialize_reranker_exception(self, scraper):
        """Test reranker initialization with exception."""
        scraper.config.embedding.enable_reranking = True
        scraper.config.embedding.reranker_model = "test-model"

        with patch(
            "src.crawl4ai_bulk_embedder.FlagReranker",
            side_effect=Exception("Test error"),
        ):
            scraper._initialize_reranker()
            assert scraper.reranker is None
            assert scraper.config.embedding.enable_reranking is False

    @pytest.mark.asyncio
    async def test_setup_collection_create_new(self, scraper):
        """Test collection setup when collection doesn't exist."""
        scraper.qdrant_service.list_collections = AsyncMock(return_value=[])
        scraper.qdrant_service.create_collection = AsyncMock()
        scraper.config.qdrant.collection_name = "test_collection"
        scraper.config.embedding.search_strategy = SearchStrategy.DENSE

        await scraper.setup_collection()

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

        scraper.qdrant_service.create_collection.assert_not_called()

    def test_get_vector_size_openai_small(self, scraper):
        """Test vector size calculation for OpenAI small model."""
        scraper.config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert scraper._get_vector_size() == 1536

    def test_get_vector_size_openai_large(self, scraper):
        """Test vector size calculation for OpenAI large model."""
        scraper.config.embedding.dense_model = EmbeddingModel.TEXT_EMBEDDING_3_LARGE
        assert scraper._get_vector_size() == 1536

    def test_get_vector_size_bge_small(self, scraper):
        """Test vector size calculation for BGE small model."""
        scraper.config.embedding.dense_model = EmbeddingModel.BGE_SMALL_EN_V15
        assert scraper._get_vector_size() == 384

    def test_get_vector_size_bge_large(self, scraper):
        """Test vector size calculation for BGE large model."""
        scraper.config.embedding.dense_model = EmbeddingModel.BGE_LARGE_EN_V15
        assert scraper._get_vector_size() == 1024

    def test_get_vector_size_nv_embed(self, scraper):
        """Test vector size calculation for NVIDIA model."""
        scraper.config.embedding.dense_model = EmbeddingModel.NV_EMBED_V2
        assert scraper._get_vector_size() == 4096

    def test_get_vector_size_default(self, scraper):
        """Test vector size calculation for unknown model."""
        scraper.config.embedding.dense_model = Mock()
        scraper.config.embedding.dense_model.value = "unknown-model"
        assert scraper._get_vector_size() == 1536

    @pytest.mark.asyncio
    async def test_create_embedding_success(self, scraper):
        """Test successful embedding creation."""
        scraper.embedding_manager.generate_embeddings = AsyncMock(
            return_value=[[0.1, 0.2, 0.3]]
        )
        # Ensure no sparse embeddings attribute
        delattr(scraper.embedding_manager, "_last_sparse_embeddings") if hasattr(
            scraper.embedding_manager, "_last_sparse_embeddings"
        ) else None

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        assert sparse_data is None
        scraper.embedding_manager.generate_embeddings.assert_called_once_with(
            ["test text"]
        )

    @pytest.mark.asyncio
    async def test_create_embedding_with_sparse(self, scraper):
        """Test embedding creation with sparse data."""
        scraper.embedding_manager.generate_embeddings = AsyncMock(
            return_value=[[0.1, 0.2, 0.3]]
        )
        scraper.embedding_manager._last_sparse_embeddings = [
            {"indices": [1, 2], "values": [0.5, 0.8]}
        ]

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == [0.1, 0.2, 0.3]
        assert sparse_data == {"sparse": {"indices": [1, 2], "values": [0.5, 0.8]}}

    @pytest.mark.asyncio
    async def test_create_embedding_failure(self, scraper):
        """Test embedding creation failure."""
        scraper.embedding_manager.generate_embeddings = AsyncMock(
            side_effect=Exception("Embedding failed")
        )

        embedding, sparse_data = await scraper.create_embedding("test text")

        assert embedding == []
        assert sparse_data is None

    def test_rerank_results_disabled(self, scraper):
        """Test reranking when disabled."""
        scraper.config.embedding.enable_reranking = False
        scraper.reranker = None

        passages = [{"content": "test1"}, {"content": "test2"}]
        result = scraper.rerank_results("query", passages)

        assert result == passages

    def test_rerank_results_empty_passages(self, scraper):
        """Test reranking with empty passages."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()

        result = scraper.rerank_results("query", [])

        assert result == []

    def test_rerank_results_single_passage(self, scraper):
        """Test reranking with single passage."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()

        passages = [{"content": "test1"}]
        result = scraper.rerank_results("query", passages)

        assert result == passages

    def test_rerank_results_success(self, scraper):
        """Test successful reranking."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()
        scraper.reranker.compute_score.return_value = [0.9, 0.7, 0.8]

        passages = [
            {"content": "passage1"},
            {"content": "passage2"},
            {"content": "passage3"},
        ]

        result = scraper.rerank_results("query", passages)

        # Should be reordered by score (highest first)
        assert len(result) == 3
        assert result[0]["content"] == "passage1"  # Score 0.9
        assert result[1]["content"] == "passage3"  # Score 0.8
        assert result[2]["content"] == "passage2"  # Score 0.7

    def test_rerank_results_exception(self, scraper):
        """Test reranking with exception."""
        scraper.config.embedding.enable_reranking = True
        scraper.reranker = Mock()
        scraper.reranker.compute_score.side_effect = Exception("Reranking failed")

        passages = [{"content": "test1"}, {"content": "test2"}]
        result = scraper.rerank_results("query", passages)

        # Should return original order on exception
        assert result == passages

    def test_chunk_content(self, scraper):
        """Test content chunking."""
        scraper.chunker.chunk_content.return_value = [
            {"content": "chunk1", "chunk_index": 0},
            {"content": "chunk2", "chunk_index": 1},
        ]

        result = scraper.chunk_content("content", "title", "url")

        assert len(result) == 2
        scraper.chunker.chunk_content.assert_called_once_with("content", "title", "url")

    def test_create_filter_chain(self, scraper):
        """Test filter chain creation."""
        site = DocumentationSite(
            name="test",
            url="https://example.com",
            url_patterns=["*docs*"],
        )

        with (
            patch("src.crawl4ai_bulk_embedder.URLPatternFilter") as mock_url_filter,
            patch(
                "src.crawl4ai_bulk_embedder.ContentTypeFilter"
            ) as mock_content_filter,
            patch("src.crawl4ai_bulk_embedder.FilterChain") as mock_filter_chain,
        ):
            scraper.create_filter_chain(site)

            mock_url_filter.assert_called_once_with(patterns=["*docs*"])
            mock_content_filter.assert_called_once_with(allowed_types=["text/html"])
            mock_filter_chain.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawl_documentation_site_success(self, scraper):
        """Test successful site crawling."""
        site = DocumentationSite(name="test", url="https://example.com", max_pages=10)

        mock_result = {
            "success": True,
            "total": 5,
            "provider": "crawl4ai",
            "pages": [
                {
                    "url": "https://example.com/page1",
                    "title": "Page 1",
                    "content": "Content 1",
                    "metadata": {"depth": 1},
                }
            ],
        }

        scraper.crawl_manager.crawl_site = AsyncMock(return_value=mock_result)

        results = await scraper.crawl_documentation_site(site)

        assert len(results) == 1
        assert isinstance(results[0], CrawlResult)
        assert results[0].url == "https://example.com/page1"
        assert results[0].title == "Page 1"
        assert results[0].success is True
        assert "https://example.com/page1" in scraper.processed_urls

    @pytest.mark.asyncio
    async def test_crawl_documentation_site_failure(self, scraper):
        """Test site crawling failure."""
        site = DocumentationSite(name="test", url="https://example.com", max_pages=10)

        mock_result = {"success": False, "error": "Crawling failed"}
        scraper.crawl_manager.crawl_site = AsyncMock(return_value=mock_result)

        results = await scraper.crawl_documentation_site(site)

        assert len(results) == 0
        assert scraper.stats.failed_crawls == 1

    def test_display_comprehensive_stats(self, scraper):
        """Test stats display."""
        scraper.stats.total_processed = 10
        scraper.stats.unique_urls = 8
        scraper.stats.successful_embeddings = 50
        scraper.stats.total_chunks = 60
        scraper.stats.failed_crawls = 2
        scraper.stats.start_time = datetime(2024, 1, 1, 10, 0, 0)
        scraper.stats.end_time = datetime(2024, 1, 1, 10, 5, 0)

        with patch("src.crawl4ai_bulk_embedder.console") as mock_console:
            scraper.display_comprehensive_stats()
            # Verify console.print was called multiple times
            assert mock_console.print.call_count >= 5

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
            result = await scraper.demo_reranking_search("test query", limit=5)

            assert len(result) <= 5
            mock_console.print.assert_called()


class TestCreateAdvancedConfig:
    """Test create_advanced_config function."""

    @patch("src.crawl4ai_bulk_embedder.get_config")
    @patch("src.crawl4ai_bulk_embedder.sys.exit")
    def test_missing_openai_api_key(self, mock_exit, mock_get_config):
        """Test configuration creation with missing OpenAI API key."""
        mock_config = Mock()
        mock_config.openai.api_key = None
        mock_get_config.return_value = mock_config

        with patch("src.crawl4ai_bulk_embedder.console") as mock_console:
            create_advanced_config()

            mock_console.print.assert_called()
            mock_exit.assert_called_once_with(1)

    @patch("src.crawl4ai_bulk_embedder.get_config")
    def test_fastembed_provider_config(self, mock_get_config):
        """Test configuration for FastEmbed provider."""
        mock_config = Mock()
        mock_config.openai.api_key = "test-key"
        mock_config.embedding_provider = EmbeddingProvider.FASTEMBED
        mock_config.qdrant.quantization_enabled = True
        mock_get_config.return_value = mock_config

        with (
            patch("src.crawl4ai_bulk_embedder.SparseTextEmbedding", None),
            patch("src.crawl4ai_bulk_embedder.console"),
        ):
            result = create_advanced_config()

            assert result.embedding.provider == EmbeddingProvider.FASTEMBED
            assert result.embedding.dense_model == EmbeddingModel.BGE_SMALL_EN_V15
            assert result.embedding.search_strategy == SearchStrategy.DENSE

    @patch("src.crawl4ai_bulk_embedder.get_config")
    def test_fastembed_with_sparse_models(self, mock_get_config):
        """Test FastEmbed configuration with sparse models available."""
        mock_config = Mock()
        mock_config.openai.api_key = "test-key"
        mock_config.embedding_provider = EmbeddingProvider.FASTEMBED
        mock_config.qdrant.quantization_enabled = True
        mock_get_config.return_value = mock_config

        mock_sparse = Mock()
        with (
            patch("src.crawl4ai_bulk_embedder.SparseTextEmbedding", mock_sparse),
            patch("src.crawl4ai_bulk_embedder.console"),
        ):
            result = create_advanced_config()

            assert result.embedding.search_strategy == SearchStrategy.HYBRID
            assert result.embedding.sparse_model == EmbeddingModel.SPLADE_PP_EN_V1

    @patch("src.crawl4ai_bulk_embedder.get_config")
    def test_openai_provider_config(self, mock_get_config):
        """Test configuration for OpenAI provider."""
        mock_config = Mock()
        mock_config.openai.api_key = "test-key"
        mock_config.embedding_provider = EmbeddingProvider.OPENAI
        mock_get_config.return_value = mock_config

        with patch("src.crawl4ai_bulk_embedder.console"):
            result = create_advanced_config()

            assert result.embedding.provider == EmbeddingProvider.OPENAI
            assert result.embedding.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
            assert result.embedding.search_strategy == SearchStrategy.DENSE


class TestMainFunction:
    """Test main function."""

    @pytest.mark.asyncio
    @patch("src.crawl4ai_bulk_embedder.create_advanced_config")
    @patch("src.crawl4ai_bulk_embedder.ModernDocumentationScraper")
    async def test_main_success(self, mock_scraper_class, mock_create_config):
        """Test successful main execution."""
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

        mock_scraper = AsyncMock()
        mock_scraper_class.return_value = mock_scraper

        with (
            patch("src.crawl4ai_bulk_embedder.console"),
            patch("src.crawl4ai_bulk_embedder.ESSENTIAL_SITES", []),
        ):
            from src.crawl4ai_bulk_embedder import main

            await main()

            mock_scraper.initialize.assert_called_once()
            mock_scraper.scrape_multiple_sites.assert_called_once()
            mock_scraper.cleanup.assert_called_once()

    @pytest.mark.asyncio
    @patch("src.crawl4ai_bulk_embedder.create_advanced_config")
    @patch("src.crawl4ai_bulk_embedder.ModernDocumentationScraper")
    @patch("src.crawl4ai_bulk_embedder.sys.exit")
    async def test_main_failure(
        self, mock_exit, mock_scraper_class, mock_create_config
    ):
        """Test main execution with failure."""
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

        mock_scraper = AsyncMock()
        mock_scraper.initialize.side_effect = Exception("Init failed")
        mock_scraper_class.return_value = mock_scraper

        with patch("src.crawl4ai_bulk_embedder.console"):
            from src.crawl4ai_bulk_embedder import main

            await main()

            mock_scraper.display_comprehensive_stats.assert_called_once()
            mock_scraper.cleanup.assert_called_once()
            mock_exit.assert_called_once_with(1)
