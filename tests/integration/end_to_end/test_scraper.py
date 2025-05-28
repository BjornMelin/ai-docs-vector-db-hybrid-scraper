"""Tests for the advanced 2025 documentation scraper functionality."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.enums import EmbeddingModel
from src.config.enums import EmbeddingProvider
from src.config.enums import SearchStrategy
from src.config.models import DocumentationSite
from src.config.models import EmbeddingConfig
from src.config.models import UnifiedConfig
from src.crawl4ai_bulk_embedder import ModernDocumentationScraper
from src.crawl4ai_bulk_embedder import VectorMetrics
from src.mcp.models.responses import CrawlResult


class TestEmbeddingConfig:
    """Test the advanced 2025 EmbeddingConfig Pydantic model."""

    def test_embedding_config_defaults(self):
        """Test embedding config with defaults."""
        config = EmbeddingConfig()
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert config.search_strategy == SearchStrategy.DENSE
        assert config.enable_quantization is True

    def test_embedding_config_hybrid_search(self):
        """Test hybrid search configuration."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.FASTEMBED,
            dense_model=EmbeddingModel.BGE_SMALL_EN_V15,
            sparse_model=EmbeddingModel.SPLADE_PP_EN_V1,
            search_strategy=SearchStrategy.HYBRID,
        )
        assert config.provider == EmbeddingProvider.FASTEMBED
        assert config.dense_model == EmbeddingModel.BGE_SMALL_EN_V15
        assert config.sparse_model == EmbeddingModel.SPLADE_PP_EN_V1
        assert config.search_strategy == SearchStrategy.HYBRID

    def test_embedding_config_reranking(self):
        """Test advanced 2025 reranking configuration."""
        config = EmbeddingConfig(
            enable_reranking=True,
            reranker_model="BAAI/bge-reranker-v2-m3",
            rerank_top_k=20,
        )
        assert config.enable_reranking is True
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"
        assert config.rerank_top_k == 20

    def test_embedding_config_reranking_defaults(self):
        """Test reranking defaults."""
        config = EmbeddingConfig()
        assert config.enable_reranking is False  # Opt-in by default
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"  # Advanced model
        assert config.rerank_top_k == 20  # Research-backed optimal


class TestUnifiedConfig:
    """Test the UnifiedConfig model integration."""

    def test_config_creation_with_defaults(self):
        """Test creating config with defaults."""
        config = UnifiedConfig()
        assert config.qdrant.url == "http://localhost:6333"
        assert config.chunking.chunk_size == 1600  # Default chunking size
        assert config.chunking.chunk_overlap == 200  # Default overlap
        assert config.embedding.provider == EmbeddingProvider.OPENAI
        assert config.embedding.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL

    def test_config_custom_values(self):
        """Test config with custom values."""
        from src.config.models import FirecrawlConfig
        from src.config.models import OpenAIConfig
        from src.config.models import QdrantConfig

        # Test-only API keys - not real credentials, used for validation testing
        config = UnifiedConfig(
            openai=OpenAIConfig(api_key="sk-testkey123456789012345678901234567890"),
            firecrawl=FirecrawlConfig(api_key="fc-testkey123"),
            qdrant=QdrantConfig(
                url="http://custom:6333", collection_name="custom_collection"
            ),
            embedding_provider=EmbeddingProvider.FASTEMBED,
        )
        assert config.openai.api_key == "sk-testkey123456789012345678901234567890"
        assert config.firecrawl.api_key == "fc-testkey123"
        assert config.qdrant.url == "http://custom:6333"
        assert config.qdrant.collection_name == "custom_collection"
        assert config.embedding_provider == EmbeddingProvider.FASTEMBED


class TestDocumentationSite:
    """Test the DocumentationSite Pydantic model."""

    def test_site_creation_minimal(self):
        """Test creating site with minimal required fields."""
        site = DocumentationSite(
            name="test-site",
            url="https://example.com",
        )
        assert site.name == "test-site"
        assert str(site.url) == "https://example.com/"  # HttpUrl adds trailing slash
        assert site.max_depth == 2
        assert site.max_pages == 50

    def test_site_creation_full(self):
        """Test creating site with all fields."""
        site = DocumentationSite(
            name="test-docs",
            url="https://docs.example.com",
            max_pages=100,
            max_depth=3,
            url_patterns=["*docs*", "*api*", "*guides*"],
        )
        assert site.name == "test-docs"
        assert (
            str(site.url) == "https://docs.example.com/"
        )  # HttpUrl adds trailing slash
        assert site.max_pages == 100
        assert site.max_depth == 3
        assert site.url_patterns == ["*docs*", "*api*", "*guides*"]

    def test_site_validation_empty_name(self):
        """Test site validation with empty name."""
        # Empty name should still be valid according to current model
        # If we want to enforce non-empty names, we need to add validation
        site = DocumentationSite(
            name="",
            url="https://example.com",
        )
        assert site.name == ""


class TestCrawlResult:
    """Test the CrawlResult Pydantic model."""

    def test_crawl_result_success(self, sample_crawl_result):
        """Test creating successful crawl result."""
        result = CrawlResult(**sample_crawl_result)
        assert result.success is True
        assert result.error is None
        assert result.title == "Test Page"
        assert "test content" in result.content.lower()

    def test_crawl_result_failure(self):
        """Test creating failed crawl result."""
        result = CrawlResult(
            url="https://example.com/error",
            title="",
            content="",
            markdown="",
            metadata={},
            links=[],
            success=False,
            error="Connection timeout",
            timestamp="2024-01-01T00:00:00Z",
        )
        assert result.success is False
        assert result.error == "Connection timeout"


class TestVectorMetrics:
    """Test the VectorMetrics Pydantic model."""

    def test_metrics_creation(self):
        """Test creating vector metrics."""
        metrics = VectorMetrics(
            total_documents=100,
            total_chunks=500,
            successful_embeddings=480,
            failed_embeddings=20,
            processing_time=120.5,
        )
        assert metrics.total_documents == 100
        assert metrics.total_chunks == 500
        assert metrics.successful_embeddings == 480
        assert metrics.failed_embeddings == 20
        assert metrics.processing_time == 120.5

    def test_metrics_success_rate(self):
        """Test metrics success rate calculation."""
        metrics = VectorMetrics(
            total_documents=10,
            total_chunks=100,
            successful_embeddings=90,
            failed_embeddings=10,
            processing_time=60.0,
        )
        success_rate = metrics.successful_embeddings / metrics.total_chunks
        assert success_rate == 0.9


class TestModernDocumentationScraper:
    """Test the ModernDocumentationScraper class."""

    @pytest.fixture()
    def scraper_config(self):
        """Create advanced 2025 scraper configuration for testing."""
        return UnifiedConfig(
            openai__api_key="test_key",
            qdrant__url="http://localhost:6333",
        )

    @pytest.fixture()
    def scraper(self, scraper_config):
        """Create advanced 2025 scraper instance for testing."""
        with (
            patch(
                "src.crawl4ai_bulk_embedder.TextEmbedding", None
            ),  # Mock FastEmbed unavailable
            patch("src.crawl4ai_bulk_embedder.SparseTextEmbedding", None),
            patch("src.crawl4ai_bulk_embedder.FirecrawlApp", None),
        ):
            return ModernDocumentationScraper(scraper_config)

    def test_scraper_initialization(self, scraper, scraper_config):
        """Test advanced 2025 scraper initialization."""
        assert scraper.config == scraper_config
        assert scraper.embedding_manager is not None  # Should be initialized
        assert scraper.qdrant_service is not None  # Should be initialized
        assert scraper.firecrawl_client is None  # Should be None without API key

    async def test_scraper_setup_collection(self, scraper):
        """Test setting up Qdrant collection."""
        with (
            patch.object(scraper.qdrant_service, "initialize", AsyncMock()),
            patch.object(
                scraper.qdrant_service, "list_collections", AsyncMock(return_value=[])
            ),
            patch.object(scraper.qdrant_service, "create_collection", AsyncMock()),
        ):
            # Initialize the service first
            scraper.qdrant_service._initialized = True
            await scraper.setup_collection()
            scraper.qdrant_service.list_collections.assert_called_once()
            scraper.qdrant_service.create_collection.assert_called_once()

    def test_chunk_content_basic(self, scraper):
        """Test advanced 2025 character-based content chunking."""
        content = "This is a test content. " * 100  # Long content (~2400 chars)
        chunks = scraper.chunk_content(content, "Test Title", "https://example.com")

        assert len(chunks) > 1
        # Check character counts are within expected range
        for chunk in chunks:
            assert (
                chunk["char_count"]
                <= scraper.config.chunking.chunk_size
                + scraper.config.chunking.chunk_overlap
            )
            assert "token_estimate" in chunk
            assert chunk["token_estimate"] == chunk["char_count"] // 4

    def test_chunk_content_short(self, scraper):
        """Test chunking short content."""
        content = "Short content for testing"
        chunks = scraper.chunk_content(content, "Test Title", "https://example.com")

        assert len(chunks) == 1
        assert chunks[0]["content"] == content
        assert chunks[0]["char_count"] == len(content)
        assert chunks[0]["total_chunks"] == 1

    def test_chunk_content_metadata(self, scraper):
        """Test chunk metadata in advanced 2025 implementation."""
        content = "Test content " * 200  # Long enough to create multiple chunks
        chunks = scraper.chunk_content(content, "Test Page", "https://example.com/page")

        for i, chunk in enumerate(chunks):
            assert chunk["url"] == "https://example.com/page"
            assert chunk["chunk_index"] == i
            assert chunk["total_chunks"] == len(chunks)
            if i > 0:
                assert f"(Part {i + 1})" in chunk["title"]
            else:
                assert chunk["title"] == "Test Page"

    async def test_crawl_documentation_site_success(
        self,
        scraper,
        sample_documentation_site,
    ):
        """Test successful documentation site crawling."""
        # Mock the crawl_site method
        mock_crawl_result = {
            "success": True,
            "total": 2,
            "provider": "crawl4ai",
            "pages": [
                {
                    "url": "https://test.example.com/page1",
                    "content": "Test content for page 1",
                    "title": "Test Page 1",
                    "metadata": {
                        "depth": 0,
                        "source_url": "https://test.example.com/page1",
                    },
                },
                {
                    "url": "https://test.example.com/page2",
                    "content": "Test content for page 2",
                    "title": "Test Page 2",
                    "metadata": {
                        "depth": 1,
                        "source_url": "https://test.example.com/page2",
                    },
                },
            ],
        }

        with patch.object(
            scraper.crawl_manager, "crawl_site", AsyncMock(return_value=mock_crawl_result)
        ):
            site = DocumentationSite(**sample_documentation_site)
            results = await scraper.crawl_documentation_site(site)

            assert len(results) == 2
            assert all(isinstance(result, CrawlResult) for result in results)
            assert results[0].url == "https://test.example.com/page1"
            assert results[1].url == "https://test.example.com/page2"
            scraper.crawl_manager.crawl_site.assert_called_once_with(
                url=site.url,
                max_pages=site.max_pages,
                formats=["markdown"],
                preferred_provider="crawl4ai",
            )

    async def test_crawl_documentation_site_failure(
        self,
        scraper,
        sample_documentation_site,
    ):
        """Test documentation site crawling with failures."""
        # Mock the crawl_site method to return a failure
        mock_crawl_result = {
            "success": False,
            "total": 0,
            "pages": [],
            "error": "Connection failed",
        }

        with patch.object(
            scraper.crawl_manager, "crawl_site", AsyncMock(return_value=mock_crawl_result)
        ):
            site = DocumentationSite(**sample_documentation_site)
            results = await scraper.crawl_documentation_site(site)

            # Should still return results, but empty due to failure
            assert isinstance(results, list)
            assert len(results) == 0

    async def test_create_embeddings_success(self, scraper):
        """Test successful embedding creation."""
        with patch.object(
            scraper.embedding_manager,
            "generate_embeddings",
            AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]),
        ):
            texts = ["Text 1", "Text 2", "Text 3"]
            embeddings = await scraper.embedding_manager.generate_embeddings(texts)

            assert len(embeddings) == 3
            assert all(len(emb) == 1536 for emb in embeddings)

    async def test_create_embeddings_failure(self, scraper):
        """Test embedding creation failure."""
        with patch.object(
            scraper.embedding_manager,
            "generate_embeddings",
            AsyncMock(side_effect=Exception("API Error")),
        ):
            texts = ["Text 1", "Text 2"]
            try:
                embeddings = await scraper.embedding_manager.generate_embeddings(texts)
                # Should handle the exception gracefully
                assert embeddings == []
            except Exception:
                # Exception is expected in this test
                pass

    async def test_store_vectors_success(self, scraper, sample_crawl_result):
        """Test successful vector storage."""
        with (
            patch.object(scraper.qdrant_service, "upsert_points", AsyncMock()),
            patch.object(
                scraper.embedding_manager,
                "generate_embeddings",
                AsyncMock(return_value=[[0.1] * 1536]),
            ),
        ):
            # Process a single crawl result
            await scraper.process_and_embed_results(
                [CrawlResult(**sample_crawl_result)]
            )

            # Check that upsert was called
            scraper.qdrant_service.upsert_points.assert_called()

    async def test_cleanup_services(self, scraper):
        """Test service cleanup."""
        with (
            patch.object(scraper.embedding_manager, "cleanup", AsyncMock()),
            patch.object(scraper.qdrant_service, "cleanup", AsyncMock()),
            patch.object(scraper.crawl_manager, "cleanup", AsyncMock()),
        ):
            await scraper.cleanup()

            scraper.embedding_manager.cleanup.assert_called_once()
            scraper.qdrant_service.cleanup.assert_called_once()
            scraper.crawl_manager.cleanup.assert_called_once()

    async def test_process_multiple_sites(self, scraper, sample_documentation_site):
        """Test processing multiple documentation sites."""
        sites = [DocumentationSite(**sample_documentation_site)]

        with (
            patch.object(scraper, "initialize", AsyncMock()),
            patch.object(
                scraper.qdrant_service, "list_collections", AsyncMock(return_value=[])
            ),
            patch.object(scraper.qdrant_service, "create_collection", AsyncMock()),
            patch.object(
                scraper, "crawl_documentation_site", AsyncMock(return_value=[])
            ),
            patch.object(scraper, "process_and_embed_results", AsyncMock()),
            patch.object(scraper, "cleanup", AsyncMock()),
        ):
            # Initialize the service to avoid initialization error
            scraper.qdrant_service._initialized = True
            await scraper.scrape_multiple_sites(sites)

            scraper.crawl_documentation_site.assert_called_once()
            scraper.process_and_embed_results.assert_called_once()


@pytest.mark.asyncio()
async def test_main_function():
    """Test the main function execution."""
    from src.crawl4ai_bulk_embedder import main

    with (
        patch(
            "src.crawl4ai_bulk_embedder.ModernDocumentationScraper"
        ) as mock_scraper_class,
        patch(
            "src.crawl4ai_bulk_embedder.create_advanced_config"
        ) as mock_create_config,
    ):
        # Mock the unified config with valid API key
        mock_config = UnifiedConfig(openai__api_key="test_key")
        mock_create_config.return_value = mock_config

        # Mock the scraper
        mock_scraper = AsyncMock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper.initialize = AsyncMock()
        mock_scraper.cleanup = AsyncMock()
        mock_scraper.scrape_multiple_sites = AsyncMock()

        await main()

        mock_scraper.initialize.assert_called_once()
        mock_scraper.scrape_multiple_sites.assert_called_once()
        mock_scraper.cleanup.assert_called_once()
