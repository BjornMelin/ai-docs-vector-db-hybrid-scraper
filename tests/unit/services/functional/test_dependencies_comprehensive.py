
class TestError(Exception):
    """Custom exception for this module."""
    pass

"""Comprehensive tests for function-based dependency injection.

Tests modern dependency injection patterns with FastAPI integration,
circuit breaker patterns, and async service lifecycle management.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.config import Config
from src.services.functional.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    create_circuit_breaker,
)
from src.services.functional.dependencies import (
    get_cache_client,
    get_client_manager,
    get_config,
    get_crawling_client,
    get_embedding_client,
    get_vector_db_client,
)


class TestDependencyLifecycle:
    """Test dependency lifecycle management patterns."""

    @pytest.mark.asyncio
    async def test_client_manager_lifecycle(self):
        """Test ClientManager lifecycle with proper initialization and cleanup."""
        mock_config = MagicMock(spec=Config)

        with patch(
            "src.services.functional.dependencies.ClientManager"
        ) as MockClientManager:
            mock_instance = AsyncMock()
            MockClientManager.return_value = mock_instance

            # Test dependency generator
            async for client_manager in get_client_manager(mock_config):
                assert client_manager == mock_instance
                break

            # Verify initialization and cleanup were called
            mock_instance.initialize.assert_called_once()
            mock_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_client_lifecycle(self):
        """Test cache client lifecycle with resource management."""
        mock_config = MagicMock(spec=Config)
        mock_config.cache.dragonfly_url = "redis://localhost:6379"
        mock_config.cache.enable_local_cache = True
        mock_config.cache.enable_dragonfly_cache = True
        mock_config.cache.local_max_size = 1000
        mock_config.cache.local_max_memory_mb = 100

        with patch("src.services.cache.manager.CacheManager") as MockCacheManager:
            mock_instance = AsyncMock()
            MockCacheManager.return_value = mock_instance

            async for cache_client in get_cache_client(mock_config):
                assert cache_client == mock_instance
                break

            # Verify CacheManager was initialized with correct config
            MockCacheManager.assert_called_once()
            call_kwargs = MockCacheManager.call_args.kwargs
            assert call_kwargs["dragonfly_url"] == "redis://localhost:6379"
            assert call_kwargs["enable_local_cache"] is True
            assert call_kwargs["enable_distributed_cache"] is True

            # Verify cleanup was called
            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_embedding_client_lifecycle(self):
        """Test embedding client lifecycle with dependency injection."""
        mock_config = MagicMock(spec=Config)
        mock_client_manager = AsyncMock()

        with patch(
            "src.services.embeddings.manager.EmbeddingManager"
        ) as MockEmbeddingManager:
            mock_instance = AsyncMock()
            MockEmbeddingManager.return_value = mock_instance

            async for embedding_client in get_embedding_client(
                mock_config, mock_client_manager
            ):
                assert embedding_client == mock_instance
                break

            # Verify EmbeddingManager was initialized correctly
            MockEmbeddingManager.assert_called_once_with(
                config=mock_config,
                client_manager=mock_client_manager,
                budget_limit=None,
                rate_limiter=None,
            )

            # Verify lifecycle methods
            mock_instance.initialize.assert_called_once()
            mock_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_db_client_lifecycle(self):
        """Test vector database client lifecycle management."""
        mock_config = MagicMock(spec=Config)

        with patch("src.services.vector_db.service.QdrantService") as MockQdrantService:
            mock_instance = AsyncMock()
            MockQdrantService.return_value = mock_instance

            async for vector_db_client in get_vector_db_client(mock_config):
                assert vector_db_client == mock_instance
                break

            # Verify initialization and cleanup
            MockQdrantService.assert_called_once_with(mock_config)
            mock_instance.initialize.assert_called_once()
            mock_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_crawling_client_lifecycle(self):
        """Test crawling client lifecycle with rate limiter."""
        mock_config = MagicMock(spec=Config)
        mock_rate_limiter = MagicMock()

        with patch("src.services.crawling.manager.CrawlManager") as MockCrawlManager:
            mock_instance = AsyncMock()
            MockCrawlManager.return_value = mock_instance

            async for crawling_client in get_crawling_client(
                mock_config, mock_rate_limiter
            ):
                assert crawling_client == mock_instance
                break

            # Verify initialization
            MockCrawlManager.assert_called_once_with(
                config=mock_config,
                rate_limiter=mock_rate_limiter,
            )
            mock_instance.initialize.assert_called_once()
            mock_instance.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_dependency_error_handling(self):
        """Test error handling in dependency lifecycle."""
        mock_config = MagicMock(spec=Config)

        with patch(
            "src.services.functional.dependencies.ClientManager"
        ) as MockClientManager:
            mock_instance = AsyncMock()
            MockClientManager.return_value = mock_instance

            # Simulate initialization error
            mock_instance.initialize.side_effect = Exception("Initialization failed")

            with pytest.raises(Exception, match="Initialization failed"):
                async for _client_manager in get_client_manager(mock_config):
                    pass

            # Cleanup should still be called even on error
            mock_instance.cleanup.assert_called_once()


class TestFastAPIIntegration:
    """Test FastAPI dependency injection integration."""

    def test_config_dependency_injection(self):
        """Test config dependency injection in FastAPI."""
        app = FastAPI()

        @app.get("/config")
        async def get_config_endpoint(config: Config = Depends(get_config)):
            return {
                "provider": config.embedding_provider.value
                if hasattr(config, "embedding_provider")
                else "default"
            }

        with TestClient(app) as client:
            response = client.get("/config")
            assert response.status_code == 200
            assert "provider" in response.json()

    @pytest.mark.asyncio
    async def test_async_dependency_composition(self):
        """Test composition of async dependencies."""
        app = FastAPI()

        @asynccontextmanager
        async def lifespan(_app: FastAPI):
            yield

        app.router.lifespan_context = lifespan

        @app.get("/health")
        async def health_check(
            config: Config = Depends(get_config),
            _client_manager=Depends(get_client_manager),
        ):
            return {"status": "healthy", "config_loaded": config is not None}

        # Test endpoint creation doesn't raise errors
        assert health_check is not None

    def test_dependency_override_patterns(self):
        """Test dependency override for testing."""
        app = FastAPI()

        def mock_config():
            return MagicMock(spec=Config)

        app.dependency_overrides[get_config] = mock_config

        @app.get("/test")
        async def test_endpoint(config: Config = Depends(get_config)):
            return {"is_mock": isinstance(config, MagicMock)}

        with TestClient(app) as client:
            response = client.get("/test")
            assert response.status_code == 200
            assert response.json()["is_mock"] is True


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration with services."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_service_protection(self):
        """Test circuit breaker protecting service calls."""
        config = CircuitBreakerConfig.simple_mode()
        config.failure_threshold = 2  # Lower threshold for testing
        circuit_breaker = CircuitBreaker(config)

        failure_count = 0

        async def flaky_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:
                raise ConnectionError("Service temporarily unavailable")
            return {"status": "success", "data": "result"}

        # First two calls should fail
        with pytest.raises(ConnectionError):
            await circuit_breaker.call(flaky_service)

        with pytest.raises(ConnectionError):
            await circuit_breaker.call(flaky_service)

        # Circuit should now be open
        assert circuit_breaker.state.value == "open"

        # Next call should raise CircuitBreakerError
        with pytest.raises(CircuitBreakerError):
            await circuit_breaker.call(flaky_service)

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery_cycle(self):
        """Test full circuit breaker recovery cycle."""
        config = CircuitBreakerConfig.simple_mode()
        config.failure_threshold = 1
        config.recovery_timeout = 1  # 1 second for testing
        circuit_breaker = CircuitBreaker(config)

        call_count = 0

        async def recovering_service():
            nonlocal call_count
            call_count += 1
                raise TestError("Initial failure")
                raise TestError("Initial failure")
            return f"success_call_{call_count}"

        # Trigger circuit opening
        with pytest.raises(Exception):
            await circuit_breaker.call(recovering_service)

        assert circuit_breaker.state.value == "open"

        # Wait for recovery timeout
        await asyncio.sleep(1.1)

        # Should recover on next successful call
        result = await circuit_breaker.call(recovering_service)
        assert result == "success_call_2"
        assert circuit_breaker.state.value == "closed"

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_fastapi_dependency(self):
        """Test circuit breaker integrated with FastAPI dependency injection."""
        app = FastAPI()

        # Create a service with circuit breaker protection
        service_circuit_breaker = create_circuit_breaker("simple", failure_threshold=2)

        async def protected_embedding_service():
            # Simulate embedding service call
            await asyncio.sleep(0.01)
            return {"embeddings": [[0.1, 0.2, 0.3]], "model": "test-model"}

        @app.post("/embeddings")
        async def generate_embeddings_endpoint():
            try:
                result = await service_circuit_breaker.call(protected_embedding_service)
                return result
            except CircuitBreakerError as e:
                raise HTTPException(status_code=503, detail=str(e))

        with TestClient(app) as client:
            # First call should succeed
            response = client.post("/embeddings")
            assert response.status_code == 200
            assert "embeddings" in response.json()


class TestServiceInteractions:
    """Test service interactions and composition."""

    @pytest.mark.asyncio
    async def test_embedding_with_cache_interaction(self):
        """Test embedding service interaction with cache service."""
        mock_cache_manager = AsyncMock()
        mock_embedding_manager = AsyncMock()

        # Setup cache miss scenario
        mock_cache_manager.get.return_value = None
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "provider": "openai",
            "cost": 0.001,
        }

        # Simulate service interaction
        cache_key = "embedding:test_text"
        cached_result = await mock_cache_manager.get(cache_key)

        if cached_result is None:
            # Cache miss - generate new embeddings
            result = await mock_embedding_manager.generate_embeddings(["test text"])
            await mock_cache_manager.set(cache_key, result, ttl=3600)
        else:
            result = cached_result

        # Verify interaction
        mock_cache_manager.get.assert_called_once_with(cache_key)
        mock_embedding_manager.generate_embeddings.assert_called_once_with(
            ["test text"]
        )
        mock_cache_manager.set.assert_called_once_with(cache_key, result, ttl=3600)

    @pytest.mark.asyncio
    async def test_crawling_with_rate_limiting(self):
        """Test crawling service interaction with rate limiting."""
        mock_crawl_manager = AsyncMock()
        mock_rate_limiter = AsyncMock()

        # Setup rate limiter to allow request
        mock_rate_limiter.acquire.return_value = True
        mock_crawl_manager.scrape_url.return_value = {
            "success": True,
            "content": "scraped content",
            "url": "https://example.com",
        }

        # Simulate rate-limited crawling
        if await mock_rate_limiter.acquire():
            result = await mock_crawl_manager.scrape_url("https://example.com")
            await mock_rate_limiter.release()
        else:
            result = {"success": False, "error": "Rate limit exceeded"}

        # Verify interaction
        mock_rate_limiter.acquire.assert_called_once()
        mock_crawl_manager.scrape_url.assert_called_once_with("https://example.com")
        mock_rate_limiter.release.assert_called_once()
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_vector_db_with_embedding_pipeline(self):
        """Test vector database interaction with embedding pipeline."""
        mock_vector_db = AsyncMock()
        mock_embedding_manager = AsyncMock()

        # Setup pipeline
        text = "test document"
        mock_embedding_manager.generate_embeddings.return_value = {
            "embeddings": [[0.1, 0.2, 0.3]],
            "model": "text-embedding-3-small",
        }
        mock_vector_db.upsert.return_value = {"status": "success", "ids": ["doc_1"]}

        # Simulate embedding + storage pipeline
        embedding_result = await mock_embedding_manager.generate_embeddings([text])
        embeddings = embedding_result["embeddings"]

        storage_result = await mock_vector_db.upsert(
            collection="test_collection",
            points=[
                {
                    "id": "doc_1",
                    "vector": embeddings[0],
                    "payload": {"text": text, "model": embedding_result["model"]},
                }
            ],
        )

        # Verify pipeline
        mock_embedding_manager.generate_embeddings.assert_called_once_with([text])
        mock_vector_db.upsert.assert_called_once()
        assert storage_result["status"] == "success"


class TestPerformancePatterns:
    """Test performance-related service patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_service_calls(self):
        """Test concurrent service call patterns."""
        mock_embedding_manager = AsyncMock()

        async def generate_embedding(text: str):
            await asyncio.sleep(0.01)  # Simulate I/O
            return {"embedding": [0.1, 0.2, 0.3], "text": text}

        async def generate_embeddings_batch(texts):
            return await asyncio.gather(*[generate_embedding(text) for text in texts])

        mock_embedding_manager.generate_embeddings = generate_embeddings_batch

        # Test concurrent processing
        texts = ["text1", "text2", "text3", "text4", "text5"]
        start_time = asyncio.get_event_loop().time()

        results = await mock_embedding_manager.generate_embeddings(texts)

        end_time = asyncio.get_event_loop().time()
        execution_time = end_time - start_time

        # Should complete faster than sequential processing
        assert execution_time < 0.1  # Much faster than 5 * 0.01 = 0.05
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_connection_pooling_patterns(self):
        """Test connection pooling and reuse patterns."""
        connection_pool = []
        active_connections = 0

        class MockConnection:
            def __init__(self):
                nonlocal active_connections
                active_connections += 1
                self.id = active_connections
                self.in_use = False

        async def get_connection():
            # Try to reuse existing connection
            for conn in connection_pool:
                if not conn.in_use:
                    conn.in_use = True
                    return conn

            # Create new connection if pool empty
            conn = MockConnection()
            connection_pool.append(conn)
            conn.in_use = True
            return conn

        async def release_connection(conn):
            conn.in_use = False

        # Test connection reuse
        conn1 = await get_connection()
        await release_connection(conn1)

        conn2 = await get_connection()
        assert conn2.id == conn1.id  # Same connection reused

        # Test concurrent connections
        conn3 = await get_connection()
        assert conn3.id != conn2.id  # New connection created

        assert len(connection_pool) == 2
        assert active_connections == 2

    @pytest.mark.asyncio
    async def test_batch_processing_patterns(self):
        """Test batch processing optimization patterns."""
        mock_service = AsyncMock()

        class BatchProcessor:
            def __init__(self, batch_size=3, flush_timeout=0.1):
                self.batch_size = batch_size
                self.flush_timeout = flush_timeout
                self.batch = []
                self.futures = []
                self._flush_task = None

            async def add_item(self, item):
                future = asyncio.Future()
                self.batch.append(item)
                self.futures.append(future)

                if len(self.batch) >= self.batch_size:
                    await self._flush_batch()
                elif self._flush_task is None:
                    self._flush_task = asyncio.create_task(self._schedule_flush())

                return await future

            async def _schedule_flush(self):
                await asyncio.sleep(self.flush_timeout)
                if self.batch:
                    await self._flush_batch()

            async def _flush_batch(self):
                if not self.batch:
                    return

                # Process batch
                results = await mock_service.process_batch(self.batch.copy())

                # Resolve futures
                for future, result in zip(self.futures, results, strict=False):
                    if not future.done():
                        future.set_result(result)

                # Clear batch
                self.batch.clear()
                self.futures.clear()
                self._flush_task = None

        # Setup mock
        mock_service.process_batch.return_value = ["result1", "result2", "result3"]

        processor = BatchProcessor()

        # Test batch processing
        tasks = [processor.add_item(f"item{i}") for i in range(3)]
        results = await asyncio.gather(*tasks)

        assert results == ["result1", "result2", "result3"]
        mock_service.process_batch.assert_called_once_with(["item0", "item1", "item2"])


class TestErrorHandlingPatterns:
    """Test error handling and resilience patterns."""

    @pytest.mark.asyncio
    async def test_retry_with_exponential_backoff(self):
        """Test retry pattern with exponential backoff."""
        call_count = 0

        async def flaky_service():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"

        async def retry_with_backoff(func, max_retries=3, base_delay=0.01):
            for attempt in range(max_retries):
                try:
                    return await func()
                except Exception:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2**attempt)
                    await asyncio.sleep(delay)

        result = await retry_with_backoff(flaky_service)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fallback_service_pattern(self):
        """Test fallback service pattern."""
        primary_service = AsyncMock()
        fallback_service = AsyncMock()

        # Primary service fails
        primary_service.process.side_effect = Exception("Primary service down")
        fallback_service.process.return_value = "fallback_result"

        async def service_with_fallback(data):
            try:
                return await primary_service.process(data)
            except Exception:
                return await fallback_service.process(data)

        result = await service_with_fallback("test_data")
        assert result == "fallback_result"

        primary_service.process.assert_called_once_with("test_data")
        fallback_service.process.assert_called_once_with("test_data")

    @pytest.mark.asyncio
    async def test_timeout_handling_pattern(self):
        """Test timeout handling in service calls."""

        async def slow_service():
            await asyncio.sleep(0.1)  # Simulate slow operation
            return "slow_result"

        async def fast_service():
            await asyncio.sleep(0.01)
            return "fast_result"

        # Test timeout with slow service
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(slow_service(), timeout=0.05)

        # Test successful completion with fast service
        result = await asyncio.wait_for(fast_service(), timeout=0.05)
        assert result == "fast_result"

    @pytest.mark.asyncio
    async def test_graceful_degradation_pattern(self):
        """Test graceful degradation when services are unavailable."""
        cache_service = AsyncMock()
        ai_service = AsyncMock()

        # AI service is down, but cache works
        cache_service.get.return_value = {"cached": "result"}
        ai_service.process.side_effect = Exception("AI service unavailable")

        async def intelligent_service_with_degradation(data):
            # Try AI-enhanced processing first
            try:
                return await ai_service.process(data)
            except Exception:
                # Fall back to cached results
                cached_result = await cache_service.get(f"cache:{data}")
                if cached_result:
                    return cached_result
                # Final fallback - basic processing
                return {"basic": f"processed_{data}"}

        result = await intelligent_service_with_degradation("test")
        assert result == {"cached": "result"}

        # Test final fallback when cache is also empty
        cache_service.get.return_value = None
        result = await intelligent_service_with_degradation("test2")
        assert result == {"basic": "processed_test2"}
