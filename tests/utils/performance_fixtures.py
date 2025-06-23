"""Optimized performance fixtures for faster test execution."""

import asyncio
import gc
import time
from collections.abc import AsyncGenerator
from collections.abc import Generator
from typing import Any
from typing import Dict
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest


class FixtureCache:
    """Global fixture cache for expensive objects."""
    
    _cache: Dict[str, Any] = {}
    _creation_times: Dict[str, float] = {}
    
    @classmethod
    def get(cls, key: str, factory, ttl: float = 300.0):
        """Get cached fixture or create new one."""
        current_time = time.time()
        
        # Check if cached version is still valid
        if key in cls._cache:
            creation_time = cls._creation_times.get(key, 0)
            if current_time - creation_time < ttl:
                return cls._cache[key]
        
        # Create new fixture
        fixture = factory()
        cls._cache[key] = fixture
        cls._creation_times[key] = current_time
        return fixture
    
    @classmethod
    def clear(cls):
        """Clear all cached fixtures."""
        cls._cache.clear()
        cls._creation_times.clear()


@pytest.fixture(scope="session")
def optimized_async_loop():
    """Optimized async event loop for test performance."""
    # Create high-performance event loop
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    
    # Optimize loop settings
    loop.set_debug(False)
    asyncio.set_event_loop(loop)
    
    yield loop
    
    # Clean shutdown
    try:
        # Cancel all running tasks
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Wait for cancellation to complete
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        
        loop.close()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture(scope="session")
def fast_mock_factory():
    """Factory for creating optimized mock objects."""
    
    def create_async_mock(**kwargs):
        """Create fast async mock with pre-configured methods."""
        mock = AsyncMock(**kwargs)
        
        # Pre-configure common async patterns
        mock.__aenter__ = AsyncMock(return_value=mock)
        mock.__aexit__ = AsyncMock(return_value=None)
        
        # Fast return values for common methods
        mock.close = AsyncMock()
        mock.cleanup = AsyncMock()
        mock.start = AsyncMock()
        mock.stop = AsyncMock()
        
        return mock
    
    def create_sync_mock(**kwargs):
        """Create fast sync mock with pre-configured methods."""
        mock = MagicMock(**kwargs)
        
        # Pre-configure common sync patterns
        mock.__enter__ = MagicMock(return_value=mock)
        mock.__exit__ = MagicMock(return_value=None)
        
        return mock
    
    return {
        "async": create_async_mock,
        "sync": create_sync_mock,
    }


@pytest.fixture(scope="session")
def cached_database_pool(fast_mock_factory):
    """Cached database pool mock for reuse across tests."""
    return FixtureCache.get(
        "database_pool",
        lambda: fast_mock_factory["async"](
            execute=AsyncMock(return_value=MagicMock(rowcount=1)),
            fetch=AsyncMock(return_value=[]),
            fetchrow=AsyncMock(return_value=None),
            fetchone=AsyncMock(return_value=None),
            acquire=AsyncMock(),
            release=AsyncMock(),
        )
    )


@pytest.fixture(scope="session")
def cached_vector_db(fast_mock_factory):
    """Cached vector database mock for reuse across tests."""
    return FixtureCache.get(
        "vector_db",
        lambda: fast_mock_factory["async"](
            search=AsyncMock(return_value=[]),
            upsert=AsyncMock(return_value={"status": "success"}),
            delete=AsyncMock(return_value={"status": "success"}),
            create_collection=AsyncMock(return_value={"status": "success"}),
            get_collection=AsyncMock(return_value={"vectors_count": 100}),
        )
    )


@pytest.fixture(scope="session")
def cached_embedding_service(fast_mock_factory):
    """Cached embedding service mock for reuse across tests."""
    # Pre-generate embeddings to avoid computation
    sample_embedding = [0.1] * 1536
    batch_embeddings = [sample_embedding] * 10
    
    return FixtureCache.get(
        "embedding_service",
        lambda: fast_mock_factory["async"](
            embed_text=AsyncMock(return_value=sample_embedding),
            embed_batch=AsyncMock(return_value=batch_embeddings),
            embed_documents=AsyncMock(return_value=batch_embeddings),
        )
    )


@pytest.fixture(scope="session")
def cached_web_scraper(fast_mock_factory):
    """Cached web scraper mock for reuse across tests."""
    mock_result = {
        "success": True,
        "url": "https://example.com",
        "title": "Test Page",
        "content": "Test content for the page.",
        "markdown": "# Test Page\n\nTest content for the page.",
        "metadata": {"description": "Test page description"},
        "links": ["https://example.com/link1"],
        "timestamp": "2024-01-01T00:00:00Z",
    }
    
    return FixtureCache.get(
        "web_scraper",
        lambda: fast_mock_factory["async"](
            scrape=AsyncMock(return_value=mock_result),
            scrape_multiple=AsyncMock(return_value=[mock_result] * 3),
            __aenter__=AsyncMock(return_value=fast_mock_factory["async"]()),
            __aexit__=AsyncMock(return_value=None),
        )
    )


@pytest.fixture(scope="session")
def minimal_test_data():
    """Minimal test data for fast test execution."""
    return FixtureCache.get(
        "test_data",
        lambda: {
            "urls": [
                "https://example.com/doc1",
                "https://example.com/doc2",
            ],
            "documents": [
                {
                    "id": 1,
                    "url": "https://example.com/doc1",
                    "title": "Test Doc 1",
                    "content": "Short test content 1",
                },
                {
                    "id": 2,
                    "url": "https://example.com/doc2", 
                    "title": "Test Doc 2",
                    "content": "Short test content 2",
                },
            ],
            "embeddings": [[0.1] * 1536, [0.2] * 1536],
            "queries": ["test query", "another query"],
            "mock_responses": [
                {"status": 200, "data": {"result": "success"}},
                {"status": 404, "error": "Not found"},
            ],
        }
    )


@pytest.fixture
def performance_monitor():
    """Lightweight performance monitoring for tests."""
    
    class TestPerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.measurements = []
        
        def start(self):
            """Start performance monitoring."""
            gc.collect()  # Clean slate
            self.start_time = time.perf_counter()
        
        def checkpoint(self, label: str):
            """Record a performance checkpoint."""
            if self.start_time is None:
                return
            
            current_time = time.perf_counter()
            duration = current_time - self.start_time
            
            self.measurements.append({
                "label": label,
                "duration": duration,
                "timestamp": current_time,
            })
        
        def get_total_time(self) -> float:
            """Get total elapsed time."""
            if self.start_time is None:
                return 0.0
            return time.perf_counter() - self.start_time
        
        def assert_under(self, max_time: float, message: str = ""):
            """Assert that total time is under threshold."""
            total_time = self.get_total_time()
            assert total_time <= max_time, (
                f"Performance assertion failed: {total_time:.3f}s > {max_time:.3f}s. {message}"
            )
    
    return TestPerformanceMonitor()


@pytest.fixture
def fast_database_session(cached_database_pool):
    """Fast database session mock that reuses connection pool."""
    session = MagicMock()
    session.pool = cached_database_pool
    session.execute = cached_database_pool.execute
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def fast_vector_search(cached_vector_db, minimal_test_data):
    """Fast vector search service using cached components."""
    
    class FastVectorSearch:
        def __init__(self):
            self.db = cached_vector_db
            self.embeddings = minimal_test_data["embeddings"]
        
        async def search(self, query: str, limit: int = 5):
            """Fast mock search."""
            return [
                {
                    "id": i,
                    "score": 0.9 - (i * 0.1),
                    "payload": {"content": f"Result {i}"},
                }
                for i in range(min(limit, 3))
            ]
        
        async def upsert(self, documents: list):
            """Fast mock upsert."""
            return {"inserted": len(documents), "status": "success"}
    
    return FastVectorSearch()


@pytest.fixture
def memory_efficient_config():
    """Memory-efficient configuration for tests."""
    return {
        "database": {
            "pool_size": 2,  # Minimal pool
            "max_connections": 5,
            "timeout": 10,
        },
        "vector_db": {
            "batch_size": 10,  # Small batches
            "cache_size": 100,
        },
        "embedding": {
            "batch_size": 5,  # Small embedding batches
            "cache_enabled": False,  # Disable cache in tests
        },
        "scraping": {
            "concurrent_requests": 2,  # Limit concurrency
            "request_delay": 0,  # No delays in tests
        },
    }


# Cleanup fixture to run after test session
@pytest.fixture(scope="session", autouse=True)
def cleanup_fixture_cache():
    """Cleanup fixture cache after test session."""
    yield
    FixtureCache.clear()
    gc.collect()


# Fast async context manager for common test patterns
@pytest.fixture
async def fast_async_context():
    """Fast async context manager for test setup/teardown."""
    
    class FastAsyncContext:
        def __init__(self):
            self._resources = []
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            # Fast cleanup without waiting
            for resource in self._resources:
                if hasattr(resource, 'close'):
                    try:
                        if asyncio.iscoroutinefunction(resource.close):
                            await resource.close()
                        else:
                            resource.close()
                    except Exception:
                        pass  # Ignore cleanup errors
        
        def add_resource(self, resource):
            """Add resource for cleanup."""
            self._resources.append(resource)
    
    return FastAsyncContext()


# Performance assertion helpers
@pytest.fixture
def performance_assertions():
    """Performance assertion helpers."""
    
    def assert_fast_execution(func, max_time: float = 0.1):
        """Assert function executes within time limit."""
        start = time.perf_counter()
        result = func()
        duration = time.perf_counter() - start
        
        assert duration <= max_time, (
            f"Function took {duration:.3f}s, expected <= {max_time:.3f}s"
        )
        return result
    
    async def assert_fast_async_execution(coro, max_time: float = 0.1):
        """Assert async function executes within time limit."""
        start = time.perf_counter()
        result = await coro
        duration = time.perf_counter() - start
        
        assert duration <= max_time, (
            f"Async function took {duration:.3f}s, expected <= {max_time:.3f}s"
        )
        return result
    
    return {
        "fast_execution": assert_fast_execution,
        "fast_async_execution": assert_fast_async_execution,
    }


# Pre-configured test markers for performance optimization
def pytest_configure(config):
    """Configure performance-optimized test markers."""
    config.addinivalue_line(
        "markers", "fast_unit: marks test as fast unit test (< 0.1s)"
    )
    config.addinivalue_line(
        "markers", "cached_fixtures: marks test as using cached fixtures"
    )
    config.addinivalue_line(
        "markers", "memory_efficient: marks test as memory efficient"
    )
    config.addinivalue_line(
        "markers", "parallel_safe: marks test as safe for parallel execution"
    )