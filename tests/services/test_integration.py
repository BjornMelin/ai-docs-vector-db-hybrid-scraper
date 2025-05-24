"""Integration tests for service layer with mocked API responses."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from src.services.config import APIConfig
from src.services.crawling.manager import CrawlManager
from src.services.embeddings.manager import EmbeddingManager
from src.services.embeddings.manager import QualityTier
from src.services.qdrant_service import QdrantService


@pytest.fixture
def integration_config():
    """Integration test configuration."""
    return APIConfig(
        openai_api_key="sk-test-integration-key",
        firecrawl_api_key="fc-test-integration-key",
        qdrant_url="http://localhost:6333",
        enable_local_embeddings=True,
    )


class TestServiceIntegration:
    """Integration tests for complete service workflows."""

    @pytest.mark.asyncio
    async def test_document_processing_workflow(self, integration_config):
        """Test complete document processing workflow."""
        # Mock responses
        crawl_response = {
            "success": True,
            "content": "# Test Document\n\nThis is test content for integration.",
            "metadata": {"title": "Test Document", "url": "https://example.com"},
        }

        embedding_response = [[0.1, 0.2, 0.3] * 512]  # 1536 dimensions

        # Test workflow
        crawl_manager = CrawlManager(integration_config)
        await crawl_manager.initialize()
        try:
            with patch.object(
                crawl_manager.providers.get("crawl4ai", Mock()),
                "scrape_url",
                return_value=crawl_response,
            ):
                # Step 1: Crawl document
                crawl_result = await crawl_manager.scrape_url("https://example.com")
                assert crawl_result["success"]
                assert crawl_result["content"] == crawl_response["content"]
        finally:
            await crawl_manager.cleanup()

        embedding_manager = EmbeddingManager(integration_config)
        await embedding_manager.initialize()
        try:
            with patch(
                "src.services.embeddings.openai_provider.AsyncOpenAI"
            ) as mock_openai:
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client

                # Mock embedding response
                mock_response = MagicMock()
                mock_response.data = [MagicMock(embedding=embedding_response[0])]
                mock_client.embeddings.create.return_value = mock_response

                # Reinitialize to use mocked client
                await embedding_manager.cleanup()
                await embedding_manager.initialize()

                # Step 2: Generate embeddings
                embeddings = await embedding_manager.generate_embeddings(
                    [crawl_result["content"]], quality_tier=QualityTier.BALANCED
                )
                assert len(embeddings) == 1
                # FastEmbed default model has 384 dimensions
                assert len(embeddings[0]) == 384
        finally:
            await embedding_manager.cleanup()

        qdrant = QdrantService(integration_config)

        # Mock the AsyncQdrantClient to avoid actual connection
        with patch(
            "src.services.qdrant_service.AsyncQdrantClient"
        ) as mock_client_class:
            mock_qdrant = AsyncMock()
            mock_client_class.return_value = mock_qdrant

            # Mock the connection check
            mock_qdrant.get_collections = AsyncMock(
                return_value=MagicMock(collections=[])
            )
            mock_qdrant.close = AsyncMock()

            await qdrant.initialize()

        try:
            # Mock collection creation
            mock_qdrant.get_collections.return_value = MagicMock(collections=[])
            mock_qdrant.create_collection.return_value = True

            # Step 3: Create collection
            await qdrant.create_collection(
                "test_docs",
                vector_size=384,  # FastEmbed default size
                enable_quantization=True,
            )

            # Step 4: Upsert document
            point = {
                "id": "doc1",
                "vector": embeddings[0],
                "payload": {
                    "text": crawl_result["content"],
                    "title": crawl_result["metadata"]["title"],
                    "url": crawl_result["metadata"]["url"],
                },
            }

            mock_qdrant.upsert.return_value = True
            success = await qdrant.upsert_points("test_docs", [point])
            assert success
        finally:
            await qdrant.cleanup()

    @pytest.mark.asyncio
    async def test_hybrid_search_workflow(self, integration_config):
        """Test hybrid search with reranking workflow."""
        query_text = "test query for integration"

        embedding_manager = EmbeddingManager(integration_config)
        await embedding_manager.initialize()
        try:
            with patch(
                "src.services.embeddings.fastembed_provider.TextEmbedding"
            ) as mock_fastembed:
                # Use FastEmbed for this test
                mock_model = MagicMock()
                mock_fastembed.return_value = mock_model

                # Mock embedding response
                query_embedding = [0.2, 0.3, 0.4] * 128  # 384 dimensions
                mock_model.embed.return_value = [query_embedding]

                # Generate query embedding
                await embedding_manager.cleanup()
                await embedding_manager.initialize()

                embeddings = await embedding_manager.generate_embeddings(
                    [query_text], provider_name="fastembed"
                )
                assert len(embeddings) == 1
        finally:
            await embedding_manager.cleanup()

        qdrant = QdrantService(integration_config)

        # Mock the AsyncQdrantClient to avoid actual connection
        with patch(
            "src.services.qdrant_service.AsyncQdrantClient"
        ) as mock_client_class:
            mock_qdrant = AsyncMock()
            mock_client_class.return_value = mock_qdrant

            # Mock the connection check
            mock_qdrant.get_collections = AsyncMock(
                return_value=MagicMock(collections=[])
            )
            mock_qdrant.close = AsyncMock()

            await qdrant.initialize()

        try:
            # Mock search response
            mock_points = [
                MagicMock(
                    id="doc1",
                    score=0.95,
                    payload={
                        "text": "Relevant document content",
                        "title": "Test Doc 1",
                    },
                ),
                MagicMock(
                    id="doc2",
                    score=0.85,
                    payload={
                        "text": "Another relevant document",
                        "title": "Test Doc 2",
                    },
                ),
            ]

            mock_qdrant.query_points.return_value = MagicMock(points=mock_points)

            # Perform hybrid search
            results = await qdrant.hybrid_search(
                "test_docs", query_vector=embeddings[0], limit=5, fusion_type="rrf"
            )

            assert len(results) == 2
            assert results[0]["score"] == 0.95
            assert results[0]["payload"]["title"] == "Test Doc 1"
        finally:
            await qdrant.cleanup()

    @pytest.mark.asyncio
    async def test_error_handling_workflow(self, integration_config):
        """Test error handling across services."""
        from src.services.errors import EmbeddingServiceError

        # Test crawl error handling
        crawl_manager = CrawlManager(integration_config)
        await crawl_manager.initialize()
        try:
            with patch.object(
                crawl_manager.providers.get("firecrawl", Mock()),
                "scrape_url",
                side_effect=Exception("API rate limit exceeded"),
            ):
                # Should fallback to crawl4ai
                with patch.object(
                    crawl_manager.providers.get("crawl4ai", Mock()),
                    "scrape_url",
                    return_value={"success": True, "content": "Fallback content"},
                ):
                    result = await crawl_manager.scrape_url(
                        "https://example.com", preferred_provider="firecrawl"
                    )
                    assert result["success"]
                    assert result["provider"] == "crawl4ai"
        finally:
            await crawl_manager.cleanup()

        # Test embedding error handling
        embedding_manager = EmbeddingManager(integration_config)
        await embedding_manager.initialize()
        try:
            with patch(
                "src.services.embeddings.openai_provider.AsyncOpenAI"
            ) as mock_openai:
                mock_client = AsyncMock()
                mock_openai.return_value = mock_client

                # Simulate rate limit error
                mock_client.embeddings.create.side_effect = Exception(
                    "Rate limit exceeded"
                )

                await embedding_manager.cleanup()
                await embedding_manager.initialize()

                with pytest.raises(EmbeddingServiceError) as exc_info:
                    await embedding_manager.generate_embeddings(
                        ["test text"], provider_name="openai"
                    )
                assert "rate limit" in str(exc_info.value).lower()
        finally:
            await embedding_manager.cleanup()

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, integration_config):
        """Test rate limiting across multiple requests."""
        from src.services.rate_limiter import RateLimiter

        # Create a test rate limiter with no burst capacity
        test_limiter = RateLimiter(
            max_calls=2,
            time_window=1.0,  # 2 calls per second
            burst_multiplier=1.0,  # No burst capacity
        )

        # First two calls should succeed immediately
        await test_limiter.acquire()
        await test_limiter.acquire()

        # Third call should wait because we've hit the limit
        import time

        start_time = time.time()
        await test_limiter.acquire()
        elapsed = time.time() - start_time

        # Should have waited for tokens to refill
        # With 2 calls/s and no burst, need to wait for 1 token
        # Refill rate is 2 tokens/s, so wait time is 0.5s
        assert elapsed > 0.4  # At least 400ms wait

    @pytest.mark.asyncio
    async def test_cost_optimization_workflow(self, integration_config):
        """Test cost optimization features."""
        texts = ["Short text"] * 10 + ["Very long text " * 100] * 5

        manager = EmbeddingManager(integration_config)
        await manager.initialize()
        try:
            # Test cost estimation
            costs = manager.estimate_cost(texts)

            # Should have costs for available providers
            assert "openai" in costs or "fastembed" in costs

            if "openai" in costs:
                assert costs["openai"]["estimated_tokens"] > 0
                assert costs["openai"]["total_cost"] > 0

            if "fastembed" in costs:
                assert costs["fastembed"]["total_cost"] == 0  # Local model

            # Test optimal provider selection
            mock_fastembed = Mock()
            mock_fastembed.cost_per_token = 0.0  # Free local model
            mock_fastembed.model = "fastembed-default"

            mock_openai = Mock()
            mock_openai.cost_per_token = 0.00002  # OpenAI pricing
            mock_openai.model = "text-embedding-3-small"

            with patch.object(
                manager,
                "providers",
                {"fastembed": mock_fastembed, "openai": mock_openai},
            ):
                # For small text with budget, should prefer local
                provider = await manager.get_optimal_provider(
                    text_length=100, quality_required=False, budget_limit=0.001
                )
                assert provider == "fastembed"

                # For quality requirement, should prefer OpenAI
                provider = await manager.get_optimal_provider(
                    text_length=10000, quality_required=True
                )
                assert provider == "openai"
        finally:
            await manager.cleanup()


class TestEndToEndScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_documentation_site_indexing(self, integration_config):
        """Test indexing a documentation site end-to-end."""
        site_pages = [
            {
                "url": "https://docs.example.com/intro",
                "content": "# Introduction\n\nWelcome to our documentation.",
                "title": "Introduction",
            },
            {
                "url": "https://docs.example.com/api",
                "content": "# API Reference\n\nOur API provides...",
                "title": "API Reference",
            },
        ]

        crawl_manager = CrawlManager(integration_config)
        await crawl_manager.initialize()
        try:
            # Mock site crawling
            with patch.object(
                crawl_manager.providers.get("firecrawl", Mock()),
                "crawl_site",
                return_value={
                    "success": True,
                    "pages": site_pages,
                    "total": len(site_pages),
                },
            ):
                crawl_result = await crawl_manager.crawl_site(
                    "https://docs.example.com", max_pages=10
                )
                assert crawl_result["success"]
                assert len(crawl_result["pages"]) == 2
        finally:
            await crawl_manager.cleanup()

    @pytest.mark.asyncio
    async def test_incremental_updates(self, integration_config):
        """Test incremental documentation updates."""
        # This would test:
        # 1. Checking existing content hashes
        # 2. Only processing changed pages
        # 3. Updating vector database efficiently
        # Implementation depends on specific incremental update strategy
        pass

    @pytest.mark.asyncio
    async def test_multi_language_support(self, integration_config):
        """Test multi-language document processing."""
        multilang_docs = [
            {"text": "Hello world", "lang": "en"},
            {"text": "Bonjour le monde", "lang": "fr"},
            {"text": "Hola mundo", "lang": "es"},
            {"text": "你好世界", "lang": "zh"},
        ]

        manager = EmbeddingManager(integration_config)
        await manager.initialize()
        try:
            # FastEmbed models support multiple languages
            with patch.object(
                manager.providers.get("fastembed", Mock()),
                "generate_embeddings",
                return_value=[[0.1] * 384 for _ in multilang_docs],
            ):
                embeddings = await manager.generate_embeddings(
                    [doc["text"] for doc in multilang_docs], provider_name="fastembed"
                )
                assert len(embeddings) == len(multilang_docs)
        finally:
            await manager.cleanup()
