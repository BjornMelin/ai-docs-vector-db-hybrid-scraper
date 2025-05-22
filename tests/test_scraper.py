"""Tests for the SOTA 2025 documentation scraper functionality."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from src.crawl4ai_bulk_embedder import CrawlResult
from src.crawl4ai_bulk_embedder import DocumentationSite
from src.crawl4ai_bulk_embedder import EmbeddingConfig
from src.crawl4ai_bulk_embedder import EmbeddingModel
from src.crawl4ai_bulk_embedder import EmbeddingProvider
from src.crawl4ai_bulk_embedder import ModernDocumentationScraper
from src.crawl4ai_bulk_embedder import ScrapingConfig
from src.crawl4ai_bulk_embedder import VectorMetrics
from src.crawl4ai_bulk_embedder import VectorSearchStrategy


class TestEmbeddingConfig:
    """Test the SOTA 2025 EmbeddingConfig Pydantic model."""

    def test_embedding_config_defaults(self):
        """Test embedding config with defaults."""
        config = EmbeddingConfig()
        assert config.provider == EmbeddingProvider.OPENAI
        assert config.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL
        assert config.search_strategy == VectorSearchStrategy.DENSE_ONLY
        assert config.enable_quantization is True

    def test_embedding_config_hybrid(self):
        """Test hybrid embedding configuration."""
        config = EmbeddingConfig(
            provider=EmbeddingProvider.HYBRID,
            dense_model=EmbeddingModel.BGE_SMALL_EN_V15,
            sparse_model=EmbeddingModel.SPLADE_PP_EN_V1,
            search_strategy=VectorSearchStrategy.HYBRID_RRF,
        )
        assert config.provider == EmbeddingProvider.HYBRID
        assert config.dense_model == EmbeddingModel.BGE_SMALL_EN_V15
        assert config.sparse_model == EmbeddingModel.SPLADE_PP_EN_V1
        assert config.search_strategy == VectorSearchStrategy.HYBRID_RRF

    def test_embedding_config_reranking(self):
        """Test SOTA 2025 reranking configuration."""
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
        assert config.reranker_model == "BAAI/bge-reranker-v2-m3"  # SOTA model
        assert config.rerank_top_k == 20  # Research-backed optimal


class TestScrapingConfig:
    """Test the SOTA 2025 ScrapingConfig Pydantic model."""

    def test_config_creation_with_required_fields(self):
        """Test creating config with required fields (SOTA 2025 defaults)."""
        config = ScrapingConfig(
            openai_api_key="test_key",
            qdrant_url="http://localhost:6333",
        )
        assert config.openai_api_key == "test_key"
        assert config.qdrant_url == "http://localhost:6333"
        assert config.chunk_size == 1600  # Updated SOTA default
        assert config.chunk_overlap == 320  # 20% overlap
        assert config.embedding.provider == EmbeddingProvider.OPENAI
        assert config.embedding.dense_model == EmbeddingModel.TEXT_EMBEDDING_3_SMALL

    def test_config_validation_missing_api_key(self):
        """Test config validation fails without API key."""
        with pytest.raises(ValidationError) as exc_info:
            ScrapingConfig()
        assert "openai_api_key" in str(exc_info.value)

    def test_config_custom_values(self):
        """Test config with custom SOTA 2025 values."""
        embedding_config = EmbeddingConfig(
            provider=EmbeddingProvider.FASTEMBED,
            dense_model=EmbeddingModel.BGE_LARGE_EN_V15,
            search_strategy=VectorSearchStrategy.HYBRID_RRF,
            sparse_model=EmbeddingModel.SPLADE_PP_EN_V1,
        )

        config = ScrapingConfig(
            openai_api_key="test_key",
            firecrawl_api_key="firecrawl_key",
            qdrant_url="http://custom:6333",
            collection_name="custom_collection",
            embedding=embedding_config,
            chunk_size=800,
            chunk_overlap=160,
            concurrent_limit=5,
            max_retries=2,
            enable_hybrid_search=True,
            enable_firecrawl_premium=True,
        )
        assert config.qdrant_url == "http://custom:6333"
        assert config.collection_name == "custom_collection"
        assert config.chunk_size == 800
        assert config.chunk_overlap == 160
        assert config.concurrent_limit == 5
        assert config.max_retries == 2
        assert config.enable_hybrid_search is True
        assert config.enable_firecrawl_premium is True
        assert config.embedding.provider == EmbeddingProvider.FASTEMBED


class TestDocumentationSite:
    """Test the DocumentationSite Pydantic model."""

    def test_site_creation_minimal(self):
        """Test creating site with minimal required fields."""
        site = DocumentationSite(
            name="test-site",
            url="https://example.com",
        )
        assert site.name == "test-site"
        assert site.url == "https://example.com"
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
        assert site.url == "https://docs.example.com"
        assert site.max_pages == 100
        assert site.max_depth == 3
        assert site.url_patterns == ["*docs*", "*api*", "*guides*"]

    def test_site_validation_invalid_url(self):
        """Test site validation with invalid URL."""
        with pytest.raises(ValidationError):
            DocumentationSite(
                name="test",
                url="not-a-url",
            )


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
        """Create SOTA 2025 scraper configuration for testing."""
        return ScrapingConfig(
            openai_api_key="test_key",
            qdrant_url="http://localhost:6333",
        )

    @pytest.fixture()
    def scraper(self, scraper_config):
        """Create SOTA 2025 scraper instance for testing."""
        with (
            patch(
                "src.crawl4ai_bulk_embedder.TextEmbedding", None
            ),  # Mock FastEmbed unavailable
            patch("src.crawl4ai_bulk_embedder.SparseTextEmbedding", None),
            patch("src.crawl4ai_bulk_embedder.FirecrawlApp", None),
        ):
            return ModernDocumentationScraper(scraper_config)

    def test_scraper_initialization(self, scraper, scraper_config):
        """Test SOTA 2025 scraper initialization."""
        assert scraper.config == scraper_config
        assert scraper.openai_client is not None  # Should be initialized
        assert scraper.qdrant_client is not None  # Should be initialized
        assert scraper.firecrawl_client is None  # Should be None without API key

    @patch("src.crawl4ai_bulk_embedder.AsyncQdrantClient")
    async def test_scraper_setup_collection(self, mock_qdrant, scraper):
        """Test setting up Qdrant collection."""
        mock_qdrant_instance = AsyncMock()
        mock_qdrant_instance.get_collections = AsyncMock(
            return_value=MagicMock(collections=[])
        )
        mock_qdrant_instance.create_collection = AsyncMock()
        scraper.qdrant_client = mock_qdrant_instance

        await scraper.setup_collection()

        mock_qdrant_instance.get_collections.assert_called_once()
        mock_qdrant_instance.create_collection.assert_called_once()

    def test_chunk_content_basic(self, scraper):
        """Test SOTA 2025 character-based content chunking."""
        content = "This is a test content. " * 100  # Long content (~2400 chars)
        chunks = scraper.chunk_content(content, "Test Title", "https://example.com")

        assert len(chunks) > 1
        # Check character counts are within expected range
        for chunk in chunks:
            assert (
                chunk["char_count"]
                <= scraper.config.chunk_size + scraper.config.chunk_overlap
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
        """Test chunk metadata in SOTA 2025 implementation."""
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

    @patch("src.crawl4ai_bulk_embedder.AsyncWebCrawler")
    async def test_crawl_documentation_site_success(
        self,
        mock_crawler_class,
        scraper,
        sample_documentation_site,
        mock_crawl4ai,
    ):
        """Test successful documentation site crawling."""
        mock_crawler_class.return_value = mock_crawl4ai
        site = DocumentationSite(**sample_documentation_site)

        with patch.object(scraper, "_setup_clients", AsyncMock()):
            results = await scraper.crawl_documentation_site(site)

        assert len(results) > 0
        assert all(isinstance(result, CrawlResult) for result in results)
        mock_crawl4ai.arun.assert_called()

    @patch("src.crawl4ai_bulk_embedder.AsyncWebCrawler")
    async def test_crawl_documentation_site_failure(
        self,
        mock_crawler_class,
        scraper,
        sample_documentation_site,
    ):
        """Test documentation site crawling with failures."""
        mock_crawler = MagicMock()
        mock_crawler.__aenter__ = AsyncMock(return_value=mock_crawler)
        mock_crawler.__aexit__ = AsyncMock(return_value=None)
        mock_crawler.arun = AsyncMock(
            return_value=MagicMock(success=False, error_message="Connection failed"),
        )
        mock_crawler_class.return_value = mock_crawler

        site = DocumentationSite(**sample_documentation_site)

        with patch.object(scraper, "_setup_clients", AsyncMock()):
            results = await scraper.crawl_documentation_site(site)

        # Should still return results, but with failures
        assert isinstance(results, list)

    async def test_create_embeddings_success(self, scraper, mock_openai_client):
        """Test successful embedding creation."""
        scraper.openai_client = mock_openai_client

        texts = ["Text 1", "Text 2", "Text 3"]
        embeddings = await scraper._create_embeddings(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 1536 for emb in embeddings)
        mock_openai_client.embeddings.create.assert_called()

    async def test_create_embeddings_failure(self, scraper):
        """Test embedding creation failure."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        scraper.openai_client = mock_client

        texts = ["Text 1", "Text 2"]
        embeddings = await scraper._create_embeddings(texts)

        # Should return empty list on failure
        assert embeddings == []

    async def test_store_vectors_success(
        self, scraper, mock_qdrant_client, sample_crawl_result
    ):
        """Test successful vector storage."""
        scraper.qdrant_client = mock_qdrant_client

        with patch.object(
            scraper, "_create_embeddings", AsyncMock(return_value=[[0.1] * 1536])
        ):
            metrics = await scraper._store_vectors(
                [CrawlResult(**sample_crawl_result)], "test_collection"
            )

        assert metrics.total_documents == 1
        assert metrics.successful_embeddings >= 0
        mock_qdrant_client.upsert.assert_called()

    async def test_cleanup_clients(
        self, scraper, mock_qdrant_client, mock_openai_client
    ):
        """Test client cleanup."""
        scraper.qdrant_client = mock_qdrant_client
        scraper.openai_client = mock_openai_client

        await scraper._cleanup_clients()

        mock_qdrant_client.close.assert_called_once()
        assert scraper.qdrant_client is None
        assert scraper.openai_client is None

    @patch("src.crawl4ai_bulk_embedder.load_documentation_sites")
    async def test_process_all_sites(
        self, mock_load_sites, scraper, sample_documentation_site
    ):
        """Test processing all documentation sites."""
        mock_load_sites.return_value = [DocumentationSite(**sample_documentation_site)]

        with (
            patch.object(
                scraper, "crawl_documentation_site", AsyncMock(return_value=[])
            ),
            patch.object(
                scraper,
                "_store_vectors",
                AsyncMock(
                    return_value=VectorMetrics(
                        total_documents=0,
                        total_chunks=0,
                        successful_embeddings=0,
                        failed_embeddings=0,
                        processing_time=0.0,
                    )
                ),
            ),
        ):
            await scraper.process_all_sites()

        mock_load_sites.assert_called_once()


@patch("src.crawl4ai_bulk_embedder.load_dotenv")
@patch("builtins.open")
def test_load_documentation_sites(
    mock_open, mock_load_dotenv, sample_documentation_site
):
    """Test loading documentation sites from JSON."""
    import json

    from src.crawl4ai_bulk_embedder import load_documentation_sites

    mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
        "sites": [sample_documentation_site],
    })

    sites = load_documentation_sites()

    assert len(sites) == 1
    assert isinstance(sites[0], DocumentationSite)
    assert sites[0].name == "test-docs"


@pytest.mark.asyncio()
async def test_main_function():
    """Test the main function execution."""
    from src.crawl4ai_bulk_embedder import main

    with patch(
        "src.crawl4ai_bulk_embedder.ModernDocumentationScraper"
    ) as mock_scraper_class:
        mock_scraper = AsyncMock()
        mock_scraper_class.return_value = mock_scraper
        mock_scraper.process_all_sites = AsyncMock()

        with patch(
            "os.getenv",
            side_effect=lambda key, default=None: {
                "OPENAI_API_KEY": "test_key",
                "QDRANT_URL": "http://localhost:6333",
            }.get(key, default),
        ):
            await main()

        mock_scraper.process_all_sites.assert_called_once()
