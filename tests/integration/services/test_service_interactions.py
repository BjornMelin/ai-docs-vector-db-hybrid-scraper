class TestError(Exception):
    """Custom exception for this module."""

    pass


"""Integration tests for service interactions and composition.

Tests the integration between different service layers including:
- Embedding services with cache interaction
- Vector database with embedding pipeline integration
- Browser automation with content intelligence
- Query processing orchestration
- Circuit breaker coordination across services
"""

import asyncio  # noqa: PLC0415
import contextlib
import time  # noqa: PLC0415
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Config
from src.services.functional.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    create_circuit_breaker,
)


class TestEmbeddingCacheIntegration:
    """Test embedding service integration with cache service."""

    @pytest.fixture
    async def integrated_services(self):
        """Setup integrated embedding and cache services."""
        config = MagicMock(spec=Config)
        config.cache.enable_caching = True
        config.cache.embedding_cache_ttl = 3600
        config.embedding_provider = MagicMock()

        # Mock services
        cache_manager = AsyncMock()
        embedding_manager = AsyncMock()
        client_manager = AsyncMock()

        return {
            "config": config,
            "cache_manager": cache_manager,
            "embedding_manager": embedding_manager,
            "client_manager": client_manager,
        }

    @pytest.mark.asyncio
    async def test_embedding_cache_hit_scenario(self, integrated_services):
        """Test embedding generation with cache hit."""
        cache_manager = integrated_services["cache_manager"]
        embedding_manager = integrated_services["embedding_manager"]

        # Setup cache hit
        cached_embedding = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "cached": True,
            "cache_timestamp": time.time(),
        }
        cache_manager.get.return_value = cached_embedding

        # Simulate service interaction
        text = "test document for embedding"
        cache_key = f"embedding:{hash(text)}"

        # Check cache first
        result = await cache_manager.get(cache_key)

        if result is not None:
            # Cache hit - return cached result
            result["cache_hit"] = True
        else:
            # Cache miss - generate new embedding
            result = await embedding_manager.generate_embeddings([text])
            await cache_manager.set(cache_key, result, ttl=3600)

        # Verify cache hit behavior
        assert result["cache_hit"] is True
        assert result["embeddings"] == [[0.1, 0.2, 0.3]]
        cache_manager.get.assert_called_once_with(cache_key)
        embedding_manager.generate_embeddings.assert_not_called()

    @pytest.mark.asyncio
    async def test_embedding_cache_miss_scenario(self, integrated_services):
        """Test embedding generation with cache miss."""
        cache_manager = integrated_services["cache_manager"]
        embedding_manager = integrated_services["embedding_manager"]

        # Setup cache miss
        cache_manager.get.return_value = None

        # Setup embedding generation
        generated_embedding = {
            "embeddings": [[0.4, 0.5, 0.6]],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "cost": 0.001,
            "latency_ms": 250.0,
        }
        embedding_manager.generate_embeddings.return_value = generated_embedding

        # Simulate service interaction
        text = "new document for embedding"
        cache_key = f"embedding:{hash(text)}"

        # Check cache first
        result = await cache_manager.get(cache_key)

        if result is None:
            # Cache miss - generate new embedding
            result = await embedding_manager.generate_embeddings([text])
            result["cache_hit"] = False
            await cache_manager.set(cache_key, result, ttl=3600)

        # Verify cache miss behavior
        assert result["cache_hit"] is False
        assert result["embeddings"] == [[0.4, 0.5, 0.6]]
        cache_manager.get.assert_called_once_with(cache_key)
        embedding_manager.generate_embeddings.assert_called_once_with([text])
        cache_manager.set.assert_called_once_with(cache_key, result, ttl=3600)

    @pytest.mark.asyncio
    async def test_embedding_cache_invalidation(self, integrated_services):
        """Test cache invalidation scenarios."""
        cache_manager = integrated_services["cache_manager"]
        embedding_manager = integrated_services["embedding_manager"]

        # Setup expired cache entry
        expired_embedding = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "openai",
            "model": "text-embedding-ada-002",  # Old model
            "cached": True,
            "cache_timestamp": time.time() - 7200,  # 2 hours ago
        }
        cache_manager.get.return_value = expired_embedding

        # Setup new embedding generation
        new_embedding = {
            "embeddings": [[0.7, 0.8, 0.9]],
            "provider": "openai",
            "model": "text-embedding-3-small",  # New model
            "cost": 0.001,
            "latency_ms": 200.0,
        }
        embedding_manager.generate_embeddings.return_value = new_embedding

        # Simulate cache invalidation logic
        text = "document for re-embedding"
        cache_key = f"embedding:{hash(text)}"

        cached_result = await cache_manager.get(cache_key)

        # Check if cache is stale (different model or too old)
        is_stale = cached_result and (
            cached_result.get("model") != "text-embedding-3-small"
            or time.time() - cached_result.get("cache_timestamp", 0) > 3600
        )

        if is_stale or cached_result is None:
            # Regenerate and update cache
            result = await embedding_manager.generate_embeddings([text])
            await cache_manager.set(cache_key, result, ttl=3600)
        else:
            result = cached_result

        # Verify cache invalidation and regeneration
        assert result["model"] == "text-embedding-3-small"
        assert result["embeddings"] == [[0.7, 0.8, 0.9]]
        embedding_manager.generate_embeddings.assert_called_once()
        cache_manager.set.assert_called_once()


class TestVectorDatabaseEmbeddingPipeline:
    """Test vector database integration with embedding pipeline."""

    @pytest.fixture
    async def vector_embedding_pipeline(self):
        """Setup vector database and embedding pipeline."""
        config = MagicMock(spec=Config)

        # Mock services
        vector_db_manager = AsyncMock()
        embedding_manager = AsyncMock()

        return {
            "config": config,
            "vector_db": vector_db_manager,
            "embedding": embedding_manager,
        }

    @pytest.mark.asyncio
    async def test_document_ingestion_pipeline(self, vector_embedding_pipeline):
        """Test complete document ingestion pipeline."""
        vector_db = vector_embedding_pipeline["vector_db"]
        embedding_manager = vector_embedding_pipeline["embedding"]

        # Setup embedding generation
        embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "model": "text-embedding-3-small",
            "provider": "openai",
            "cost": 0.002,
        }

        # Setup vector database storage
        vector_db.upsert.return_value = {
            "status": "success",
            "ids": ["doc_1", "doc_2"],
            "operation_id": "op_123",
        }

        # Test document ingestion
        documents = [
            {"id": "doc_1", "text": "First document", "metadata": {"source": "web"}},
            {"id": "doc_2", "text": "Second document", "metadata": {"source": "pdf"}},
        ]

        # Pipeline execution
        texts = [doc["text"] for doc in documents]
        embedding_result = await embedding_manager.generate_embeddings(texts)

        # Prepare vector points
        points = []
        for i, doc in enumerate(documents):
            points.append(
                {
                    "id": doc["id"],
                    "vector": embedding_result["embeddings"][i],
                    "payload": {
                        "text": doc["text"],
                        "metadata": doc["metadata"],
                        "embedding_model": embedding_result["model"],
                    },
                }
            )

        # Store in vector database
        storage_result = await vector_db.upsert(collection="documents", points=points)

        # Verify pipeline execution
        embedding_manager.generate_embeddings.assert_called_once_with(texts)
        vector_db.upsert.assert_called_once()
        assert storage_result["status"] == "success"
        assert len(storage_result["ids"]) == 2

    @pytest.mark.asyncio
    async def test_search_pipeline_with_reranking(self, vector_embedding_pipeline):
        """Test search pipeline with embedding and reranking."""
        vector_db = vector_embedding_pipeline["vector_db"]
        embedding_manager = vector_embedding_pipeline["embedding"]

        # Setup query embedding
        query_embedding_result = {
            "embeddings": [[0.1, 0.15, 0.25]],
            "model": "text-embedding-3-small",
            "latency_ms": 150.0,
        }
        embedding_manager.generate_embeddings.return_value = query_embedding_result

        # Setup vector search results
        search_results = {
            "matches": [
                {
                    "id": "doc_1",
                    "score": 0.95,
                    "payload": {
                        "text": "Relevant document",
                        "metadata": {"source": "web"},
                    },
                },
                {
                    "id": "doc_2",
                    "score": 0.87,
                    "payload": {
                        "text": "Somewhat relevant",
                        "metadata": {"source": "pdf"},
                    },
                },
                {
                    "id": "doc_3",
                    "score": 0.82,
                    "payload": {"text": "Less relevant", "metadata": {"source": "api"}},
                },
            ]
        }
        vector_db.search.return_value = search_results

        # Test search pipeline
        query = "find relevant documents"

        # Generate query embedding
        query_embedding = await embedding_manager.generate_embeddings([query])

        # Perform vector search
        search_result = await vector_db.search(
            collection="documents",
            query_vector=query_embedding["embeddings"][0],
            limit=10,
        )

        # Apply reranking (mock cross-encoder scoring)
        reranked_results = []
        for match in search_result["matches"]:
            # Mock cross-encoder score
            cross_encoder_score = match["score"] * 0.9  # Slightly adjust
            reranked_results.append(
                {
                    "id": match["id"],
                    "score": cross_encoder_score,
                    "vector_score": match["score"],
                    "text": match["payload"]["text"],
                    "metadata": match["payload"]["metadata"],
                }
            )

        # Sort by reranked scores
        reranked_results.sort(key=lambda x: x["score"], reverse=True)

        # Verify search pipeline
        embedding_manager.generate_embeddings.assert_called_once_with([query])
        vector_db.search.assert_called_once()
        assert len(reranked_results) == 3
        assert reranked_results[0]["score"] >= reranked_results[1]["score"]

    @pytest.mark.asyncio
    async def test_hybrid_search_integration(self, vector_embedding_pipeline):
        """Test hybrid search combining vector and keyword search."""
        vector_db = vector_embedding_pipeline["vector_db"]
        embedding_manager = vector_embedding_pipeline["embedding"]

        # Setup embedding for vector search
        embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.2, 0.3, 0.4]],
            "model": "text-embedding-3-small",
        }

        # Setup vector search results
        vector_results = {
            "matches": [
                {"id": "doc_1", "score": 0.9, "payload": {"text": "Vector match"}},
                {
                    "id": "doc_2",
                    "score": 0.8,
                    "payload": {"text": "Another vector match"},
                },
            ]
        }
        vector_db.search.return_value = vector_results

        # Setup keyword search results (mock)
        keyword_results = {
            "matches": [
                {
                    "id": "doc_2",
                    "score": 0.95,
                    "payload": {"text": "Keyword exact match"},
                },
                {
                    "id": "doc_3",
                    "score": 0.7,
                    "payload": {"text": "Partial keyword match"},
                },
            ]
        }

        # Test hybrid search
        query = "search query text"

        # Perform both searches
        query_embedding = await embedding_manager.generate_embeddings([query])
        vector_search = await vector_db.search(
            collection="documents",
            query_vector=query_embedding["embeddings"][0],
            limit=5,
        )

        # Mock keyword search
        keyword_search = keyword_results

        # Combine results using RRF (Reciprocal Rank Fusion)
        combined_scores = {}

        # Add vector search scores
        for i, match in enumerate(vector_search["matches"]):
            doc_id = match["id"]
            rrf_score = 1 / (i + 1 + 60)  # RRF with k=60
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

        # Add keyword search scores
        for i, match in enumerate(keyword_search["matches"]):
            doc_id = match["id"]
            rrf_score = 1 / (i + 1 + 60)  # RRF with k=60
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score

        # Sort by combined scores
        final_results = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Verify hybrid search
        assert len(final_results) >= 2
        assert final_results[0][1] >= final_results[1][1]  # Properly sorted
        # doc_2 should rank highest (appears in both searches)
        assert final_results[0][0] == "doc_2"


class TestBrowserContentIntelligenceIntegration:
    """Test browser automation integration with content intelligence."""

    @pytest.fixture
    async def browser_content_services(self):
        """Setup browser automation and content intelligence services."""
        config = MagicMock(spec=Config)

        # Mock services
        browser_manager = AsyncMock()
        content_intelligence = AsyncMock()

        return {
            "config": config,
            "browser": browser_manager,
            "content_ai": content_intelligence,
        }

    @pytest.mark.asyncio
    async def test_intelligent_content_extraction(self, browser_content_services):
        """Test intelligent content extraction pipeline."""
        browser_manager = browser_content_services["browser"]
        content_ai = browser_content_services["content_ai"]

        # Setup browser scraping
        scraped_content = {
            "success": True,
            "content": "<html><body><h1>Title</h1><p>Content paragraph</p></body></html>",
            "url": "https://example.com",
            "metadata": {
                "tier_used": "playwright",
                "scraping_time_ms": 1200,
                "page_load_time_ms": 800,
            },
        }
        browser_manager.scrape_url.return_value = scraped_content

        # Setup content intelligence analysis
        content_analysis = {
            "content_type": "article",
            "quality_score": 0.92,
            "extracted_text": "Title\nContent paragraph",
            "key_topics": ["technology", "web scraping"],
            "sentiment": "neutral",
            "readability_score": 0.85,
            "metadata": {"word_count": 50, "paragraph_count": 1, "heading_count": 1},
        }
        content_ai.analyze_content.return_value = content_analysis

        # Test integrated pipeline
        url = "https://example.com"

        # Scrape content
        scraping_result = await browser_manager.scrape_url(url)

        if scraping_result["success"]:
            # Analyze scraped content
            analysis_result = await content_ai.analyze_content(
                content=scraping_result["content"], url=scraping_result["url"]
            )

            # Combine results
            final_result = {
                "url": url,
                "scraping": scraping_result,
                "analysis": analysis_result,
                "quality_score": analysis_result["quality_score"],
                "processing_time_ms": scraping_result["metadata"]["scraping_time_ms"],
            }

        # Verify integrated processing
        browser_manager.scrape_url.assert_called_once_with(url)
        content_ai.analyze_content.assert_called_once()
        assert final_result["quality_score"] == 0.92
        assert final_result["analysis"]["content_type"] == "article"

    @pytest.mark.asyncio
    async def test_adaptive_scraping_based_on_content_type(
        self, browser_content_services
    ):
        """Test adaptive scraping strategy based on content type detection."""
        browser_manager = browser_content_services["browser"]
        content_ai = browser_content_services["content_ai"]

        # Mock content type detection
        content_ai.detect_content_type.return_value = {
            "content_type": "single_page_application",
            "confidence": 0.95,
            "characteristics": ["dynamic_content", "javascript_heavy"],
            "recommended_tier": "crawl4ai",
        }

        # Mock tier-specific scraping
        browser_manager.scrape_with_tier.return_value = {
            "success": True,
            "content": "SPA content extracted",
            "tier_used": "crawl4ai",
            "adaptation_reason": "spa_detection",
        }

        # Test adaptive scraping
        url = "https://spa-example.com"

        # Pre-analyze URL for content type
        content_type_info = await content_ai.detect_content_type(url)

        # Select appropriate tier based on content type
        recommended_tier = content_type_info["recommended_tier"]

        # Scrape with recommended tier
        scraping_result = await browser_manager.scrape_with_tier(
            url=url,
            tier=recommended_tier,
            reason=f"content_type:{content_type_info['content_type']}",
        )

        # Verify adaptive behavior
        content_ai.detect_content_type.assert_called_once_with(url)
        browser_manager.scrape_with_tier.assert_called_once()
        assert scraping_result["tier_used"] == "crawl4ai"
        assert scraping_result["adaptation_reason"] == "spa_detection"

    @pytest.mark.asyncio
    async def test_content_quality_feedback_loop(self, browser_content_services):
        """Test content quality feedback loop for tier optimization."""
        browser_manager = browser_content_services["browser"]
        content_ai = browser_content_services["content_ai"]

        # Mock multiple scraping attempts with different tiers
        scraping_attempts = [
            {
                "tier": "lightweight",
                "content": "Basic content",
                "quality_score": 0.3,
                "success": True,
            },
            {
                "tier": "playwright",
                "content": "Better content with JS",
                "quality_score": 0.7,
                "success": True,
            },
            {
                "tier": "crawl4ai",
                "content": "High quality extracted content",
                "quality_score": 0.95,
                "success": True,
            },
        ]

        # Test quality-based tier selection
        url = "https://complex-site.com"
        quality_threshold = 0.8

        for attempt in scraping_attempts:
            # Mock scraping with specific tier
            browser_manager.scrape_with_tier.return_value = {
                "success": attempt["success"],
                "content": attempt["content"],
                "tier_used": attempt["tier"],
            }

            # Mock quality analysis
            content_ai.assess_quality.return_value = {
                "quality_score": attempt["quality_score"],
                "issues": []
                if attempt["quality_score"] > quality_threshold
                else ["low_content_extraction"],
            }

            # Perform scraping and quality check
            scraping_result = await browser_manager.scrape_with_tier(
                url, attempt["tier"]
            )
            quality_result = await content_ai.assess_quality(scraping_result["content"])

            # Check if quality meets threshold
            if quality_result["quality_score"] >= quality_threshold:
                final_result = {
                    "selected_tier": attempt["tier"],
                    "quality_score": quality_result["quality_score"],
                    "content": scraping_result["content"],
                }
                break

        # Verify quality-driven tier selection
        assert final_result["selected_tier"] == "crawl4ai"
        assert final_result["quality_score"] == 0.95


class TestQueryProcessingOrchestration:
    """Test query processing orchestration across services."""

    @pytest.fixture
    async def query_processing_services(self):
        """Setup query processing orchestration services."""
        config = MagicMock(spec=Config)

        # Mock services
        query_processor = AsyncMock()
        vector_db = AsyncMock()
        embedding_manager = AsyncMock()
        cache_manager = AsyncMock()

        return {
            "config": config,
            "query_processor": query_processor,
            "vector_db": vector_db,
            "embedding": embedding_manager,
            "cache": cache_manager,
        }

    @pytest.mark.asyncio
    async def test_query_processing_orchestration(self, query_processing_services):
        """Test complete query processing orchestration."""
        services = query_processing_services

        # Setup query processing pipeline
        query = "find documents about machine learning"

        # Mock query classification
        services["query_processor"].classify_intent.return_value = {
            "intent": "information_retrieval",
            "complexity": "medium",
            "suggested_strategy": "hybrid_search",
            "confidence": 0.92,
        }

        # Mock query expansion
        services["query_processor"].expand_query.return_value = {
            "original_query": query,
            "expanded_terms": [
                "machine learning",
                "ML",
                "artificial intelligence",
                "neural networks",
            ],
            "synonyms": ["deep learning", "AI"],
            "related_concepts": ["data science", "algorithms"],
        }

        # Mock embedding generation
        services["embedding"].generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "model": "text-embedding-3-small",
        }

        # Mock vector search
        services["vector_db"].search.return_value = {
            "matches": [
                {"id": "doc_1", "score": 0.92, "payload": {"text": "ML tutorial"}},
                {
                    "id": "doc_2",
                    "score": 0.87,
                    "payload": {"text": "Neural networks guide"},
                },
            ]
        }

        # Mock result ranking
        services["query_processor"].rank_results.return_value = {
            "ranked_results": [
                {"id": "doc_1", "score": 0.94, "relevance": "high"},
                {"id": "doc_2", "score": 0.89, "relevance": "medium"},
            ],
            "ranking_strategy": "ml_based_reranking",
        }

        # Execute orchestrated query processing
        # Step 1: Classify query intent
        intent_result = await services["query_processor"].classify_intent(query)

        # Step 2: Expand query based on intent
        expansion_result = await services["query_processor"].expand_query(
            query, strategy=intent_result["suggested_strategy"]
        )

        # Step 3: Generate embeddings for expanded query
        embedding_result = await services["embedding"].generate_embeddings(
            [expansion_result["original_query"]]
        )

        # Step 4: Perform vector search
        search_result = await services["vector_db"].search(
            query_vector=embedding_result["embeddings"][0], limit=10
        )

        # Step 5: Rank and refine results
        final_result = await services["query_processor"].rank_results(
            search_result["matches"],
            original_query=query,
            intent=intent_result["intent"],
        )

        # Verify orchestration
        assert intent_result["intent"] == "information_retrieval"
        assert len(expansion_result["expanded_terms"]) == 4
        assert len(final_result["ranked_results"]) == 2
        assert (
            final_result["ranked_results"][0]["score"]
            >= final_result["ranked_results"][1]["score"]
        )

    @pytest.mark.asyncio
    async def test_query_caching_integration(self, query_processing_services):
        """Test query processing with caching integration."""
        services = query_processing_services

        query = "cached query example"
        query_hash = f"query:{hash(query)}"

        # Setup cached result
        cached_result = {
            "query": query,
            "results": [
                {"id": "doc_1", "score": 0.95, "text": "Cached result"},
                {"id": "doc_2", "score": 0.88, "text": "Another cached result"},
            ],
            "metadata": {
                "cached_at": time.time() - 300,  # 5 minutes ago
                "cache_hit": True,
                "processing_time_ms": 50.0,
            },
        }
        services["cache"].get.return_value = cached_result

        # Test cached query processing
        # Check cache first
        cached_query_result = await services["cache"].get(query_hash)

        if (
            cached_query_result
            and time.time() - cached_query_result["metadata"]["cached_at"] < 3600
        ):
            # Cache hit and not expired
            result = cached_query_result
            result["metadata"]["cache_hit"] = True
        else:
            # Cache miss or expired - process normally
            # ... (normal processing pipeline would go here)
            result = None

        # Verify caching behavior
        assert result is not None
        assert result["metadata"]["cache_hit"] is True
        assert len(result["results"]) == 2
        services["cache"].get.assert_called_once_with(query_hash)


class TestCircuitBreakerCoordination:
    """Test circuit breaker coordination across services."""

    @pytest.mark.asyncio
    async def test_service_circuit_breaker_coordination(self):
        """Test circuit breaker coordination across multiple services."""
        # Create circuit breakers for different services
        embedding_circuit = create_circuit_breaker("enterprise", failure_threshold=3)
        vector_db_circuit = create_circuit_breaker("enterprise", failure_threshold=2)
        cache_circuit = create_circuit_breaker("simple", failure_threshold=5)

        # Mock service operations
        async def embedding_operation():
            raise ConnectionError("Embedding service down")

        async def vector_db_operation():
            return {"status": "success", "data": "vector_result"}

        async def cache_operation():
            return {"status": "success", "data": "cached_data"}

        # Test coordinated service calls with circuit breakers
        service_results = {}

        # Try embedding service (will fail and open circuit)
        try:
            embedding_result = await embedding_circuit.call(embedding_operation)
            service_results["embedding"] = embedding_result
        except Exception as e:
            service_results["embedding"] = {"error": str(e), "circuit_open": False}

        # Continue with other services even if embedding fails
        try:
            vector_result = await vector_db_circuit.call(vector_db_operation)
            service_results["vector_db"] = vector_result
        except Exception as e:
            service_results["vector_db"] = {"error": str(e)}

        try:
            cache_result = await cache_circuit.call(cache_operation)
            service_results["cache"] = cache_result
        except Exception as e:
            service_results["cache"] = {"error": str(e)}

        # Test that embedding failures trigger circuit opening
        for _i in range(2):  # Need 2 more failures to open circuit (threshold=3)
            with contextlib.suppress(Exception):
                await embedding_circuit.call(embedding_operation)

        # Now embedding circuit should be open
        from src.services.functional.circuit_breaker import CircuitBreakerError  # noqa: PLC0415

        try:
            await embedding_circuit.call(embedding_operation)
            service_results["embedding"]["circuit_open"] = False
        except CircuitBreakerError:
            service_results["embedding"]["circuit_open"] = True

        # Verify circuit breaker coordination
        assert service_results["embedding"]["circuit_open"] is True
        assert service_results["vector_db"]["status"] == "success"
        assert service_results["cache"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self):
        """Test prevention of cascading failures across services."""
        # Setup circuit breakers with different thresholds
        primary_service_circuit = create_circuit_breaker(
            "enterprise", failure_threshold=2
        )
        fallback_service_circuit = create_circuit_breaker(
            "enterprise", failure_threshold=3
        )

        # Mock services
        async def primary_service():
            raise TestError("Primary service overloaded")

        async def fallback_service():
            return {"status": "fallback_success", "data": "fallback_data"}

        # Test cascading failure prevention
        async def protected_service_call():
            try:
                # Try primary service
                return await primary_service_circuit.call(primary_service)
            except Exception:
                # Primary failed, try fallback
                try:
                    return await fallback_service_circuit.call(fallback_service)
                except Exception:
                    return {"status": "all_services_failed"}

        # Execute multiple calls to trigger circuit opening
        results = []
        for _i in range(5):
            try:
                result = await protected_service_call()
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})

        # Verify fallback success and circuit behavior
        successful_fallbacks = [
            r for r in results if r.get("status") == "fallback_success"
        ]
        assert len(successful_fallbacks) > 0

        # Primary circuit should be open after failures
        assert primary_service_circuit.state.value == "open"
        # Fallback circuit should remain closed (successful calls)
        assert fallback_service_circuit.state.value == "closed"

    @pytest.mark.asyncio
    async def test_service_health_monitoring_integration(self):
        """Test circuit breaker integration with service health monitoring."""

        # Mock service health monitor
        class ServiceHealthMonitor:
            def __init__(self):
                self.service_health = {
                    "embedding_service": {"status": "healthy", "response_time": 200},
                    "vector_db_service": {"status": "degraded", "response_time": 2000},
                    "cache_service": {"status": "healthy", "response_time": 50},
                }
                self.circuit_breakers = {}

            def get_circuit_breaker(self, service_name):
                if service_name not in self.circuit_breakers:
                    # Configure circuit breaker based on service health
                    health = self.service_health[service_name]
                    if health["status"] == "degraded":
                        config = CircuitBreakerConfig.enterprise_mode()
                        config.failure_threshold = (
                            1  # More sensitive for degraded services
                        )
                    else:
                        config = CircuitBreakerConfig.simple_mode()

                    self.circuit_breakers[service_name] = CircuitBreaker(config)

                return self.circuit_breakers[service_name]

            async def call_service_with_protection(self, service_name, operation):
                circuit_breaker = self.get_circuit_breaker(service_name)
                return await circuit_breaker.call(operation)

        # Test health-aware circuit breaker configuration
        monitor = ServiceHealthMonitor()

        # Mock operations
        async def healthy_operation():
            await asyncio.sleep(0.01)
            return {"status": "success"}

        async def degraded_operation():
            await asyncio.sleep(0.05)  # Slow
            raise TestError("Service degraded")

        # Test calls to different services
        healthy_result = await monitor.call_service_with_protection(
            "embedding_service", healthy_operation
        )

        # Degraded service should have more sensitive circuit breaker
        try:
            await monitor.call_service_with_protection(
                "vector_db_service", degraded_operation
            )
        except Exception as e:
            pass  # Expected failure

        # Verify different circuit breaker configurations
        embedding_cb = monitor.circuit_breakers["embedding_service"]
        vector_db_cb = monitor.circuit_breakers["vector_db_service"]

        assert (
            embedding_cb.config.failure_threshold
            > vector_db_cb.config.failure_threshold
        )
        assert healthy_result["status"] == "success"
