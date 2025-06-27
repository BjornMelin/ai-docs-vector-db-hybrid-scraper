"""Pytest configuration for comprehensive service layer testing.

Provides fixtures and configuration for function-based service testing,
circuit breaker patterns, database connection pooling, browser automation
monitoring, and performance benchmarking.
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Config
from src.services.functional.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    create_circuit_breaker,
)


# Async test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Configuration fixtures
@pytest.fixture
def mock_config():
    """Mock unified configuration for testing."""
    config = MagicMock(spec=Config)

    # Database configuration
    config.database.connection_pool_size = 10
    config.database.max_overflow = 20
    config.database.pool_timeout = 30
    config.database.pool_recycle = 3600
    config.database.enable_ml_scaling = True
    config.database.ml_prediction_interval = 60
    config.database.connection_affinity_enabled = True

    # Cache configuration
    config.cache.enable_caching = True
    config.cache.dragonfly_url = "redis://localhost:6379"
    config.cache.enable_local_cache = True
    config.cache.enable_dragonfly_cache = True
    config.cache.local_max_size = 1000
    config.cache.local_max_memory_mb = 100
    config.cache.embedding_cache_ttl = 3600

    # Browser configuration
    config.browser.enable_tier_monitoring = True
    config.browser.health_check_interval = 30
    config.browser.enable_unified_management = True
    config.browser.tier_priorities = [
        "lightweight",
        "playwright",
        "crawl4ai",
        "browser_use",
        "firecrawl",
    ]
    config.browser.enable_automatic_failover = True

    # Monitoring configuration
    config.monitoring.enable_database_monitoring = True
    config.monitoring.metrics_collection_interval = 10

    # Embedding configuration
    config.embedding_provider = MagicMock()

    return config


# Service mock fixtures
@pytest.fixture
def mock_client_manager():
    """Mock ClientManager for dependency injection testing."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.get_embedding_manager = AsyncMock()
    manager.get_cache_manager = AsyncMock()
    manager.get_crawl_manager = AsyncMock()
    manager.get_task_queue_manager = AsyncMock()
    manager.get_health_status = AsyncMock()
    return manager


@pytest.fixture
def mock_cache_manager():
    """Mock CacheManager for testing."""
    manager = AsyncMock()
    manager.get = AsyncMock()
    manager.set = AsyncMock()
    manager.delete = AsyncMock()
    manager.close = AsyncMock()
    manager.get_stats = AsyncMock()
    manager.get_performance_stats = AsyncMock()
    return manager


@pytest.fixture
def mock_embedding_manager():
    """Mock EmbeddingManager for testing."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.generate_embeddings = AsyncMock()
    manager.get_provider_info = MagicMock()
    manager.get_usage_report = MagicMock()
    return manager


@pytest.fixture
def mock_vector_db_manager():
    """Mock VectorDB manager for testing."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.upsert = AsyncMock()
    manager.search = AsyncMock()
    manager.delete = AsyncMock()
    return manager


@pytest.fixture
def mock_crawl_manager():
    """Mock CrawlManager for testing."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.scrape_url = AsyncMock()
    manager.crawl_site = AsyncMock()
    manager.scrape_with_tier = AsyncMock()
    manager.get_metrics = MagicMock()
    manager.get_tier_metrics = MagicMock()
    return manager


@pytest.fixture
def mock_browser_monitor():
    """Mock BrowserMonitor for testing."""
    monitor = AsyncMock()
    monitor.check_all_tiers_health = AsyncMock()
    monitor.collect_performance_metrics = AsyncMock()
    monitor.get_resource_utilization = AsyncMock()
    monitor.get_optimal_tier_for_request = AsyncMock()
    monitor.run_tier_benchmarks = AsyncMock()
    monitor.check_alert_conditions = AsyncMock()
    return monitor


@pytest.fixture
def mock_content_intelligence():
    """Mock ContentIntelligence service for testing."""
    service = AsyncMock()
    service.analyze_content = AsyncMock()
    service.detect_content_type = AsyncMock()
    service.assess_quality = AsyncMock()
    return service


# Circuit breaker fixtures
@pytest.fixture
def simple_circuit_breaker():
    """Simple circuit breaker for testing."""
    config = CircuitBreakerConfig.simple_mode()
    return CircuitBreaker(config)


@pytest.fixture
def enterprise_circuit_breaker():
    """Enterprise circuit breaker for testing."""
    config = CircuitBreakerConfig.enterprise_mode()
    return CircuitBreaker(config)


@pytest.fixture
def circuit_breaker_factory():
    """Factory for creating circuit breakers with custom config."""

    def _create_circuit_breaker(mode="simple", **kwargs):
        return create_circuit_breaker(mode, **kwargs)

    return _create_circuit_breaker


# Database testing fixtures
@pytest.fixture
def mock_database_manager():
    """Mock DatabaseManager for testing."""
    manager = AsyncMock()
    manager.initialize = AsyncMock()
    manager.cleanup = AsyncMock()
    manager.get_connection = AsyncMock()
    manager.get_pool_metrics = AsyncMock()
    manager.predict_load_requirements = AsyncMock()
    manager.adapt_pool_size = AsyncMock()
    manager.get_affinity_stats = AsyncMock()
    manager.check_connection_health = AsyncMock()
    manager.recover_connection = AsyncMock()

    # Mock properties
    manager.is_initialized = True
    manager.pool_size = 10
    manager.max_overflow = 20
    manager.connection_affinity_enabled = True

    return manager


@pytest.fixture
def mock_database_monitor():
    """Mock DatabaseMonitor for testing."""
    monitor = AsyncMock()
    monitor.collect_metrics = AsyncMock()
    monitor.analyze_performance_trends = AsyncMock()
    monitor.check_alert_conditions = AsyncMock()
    monitor.calculate_prediction_accuracy = AsyncMock()
    monitor.get_performance_metrics = AsyncMock()
    monitor.get_historical_metrics = AsyncMock()
    monitor.get_prediction_history = AsyncMock()
    return monitor


# Performance testing fixtures
@pytest.fixture
def performance_test_data():
    """Generate test data for performance testing."""
    return {
        "texts": [f"Test document {i}" for i in range(100)],
        "embeddings": [[0.1 * i, 0.2 * i, 0.3 * i] for i in range(100)],
        "urls": [f"https://example.com/page{i}" for i in range(50)],
        "queries": [f"search query {i}" for i in range(25)],
    }


@pytest.fixture
def benchmark_config():
    """Configuration for benchmark testing."""
    return {
        "rounds": 10,
        "iterations": 1,
        "warmup_rounds": 2,
        "min_time": 0.01,
        "max_time": 10.0,
        "timer": time.perf_counter,
    }


# Integration testing fixtures
@pytest.fixture
async def integrated_services(mock_config):
    """Setup integrated services for testing."""
    services = {
        "config": mock_config,
        "client_manager": AsyncMock(),
        "cache_manager": AsyncMock(),
        "embedding_manager": AsyncMock(),
        "vector_db_manager": AsyncMock(),
        "crawl_manager": AsyncMock(),
        "browser_monitor": AsyncMock(),
        "content_intelligence": AsyncMock(),
    }

    # Setup default return values
    services["cache_manager"].get.return_value = None
    services["embedding_manager"].generate_embeddings.return_value = {
        "embeddings": [[0.1, 0.2, 0.3]],
        "provider": "openai",
        "model": "text-embedding-3-small",
        "cost": 0.001,
        "latency_ms": 150.0,
    }
    services["vector_db_manager"].search.return_value = {
        "matches": [
            {"id": "doc_1", "score": 0.95, "payload": {"text": "Test document"}}
        ]
    }
    services["crawl_manager"].scrape_url.return_value = {
        "success": True,
        "content": "Scraped content",
        "url": "https://example.com",
        "tier_used": "playwright",
    }
    services["content_intelligence"].analyze_content.return_value = {
        "content_type": "article",
        "quality_score": 0.92,
        "extracted_text": "Analyzed content",
    }

    return services


# Health monitoring fixtures
@pytest.fixture
def tier_health_data():
    """Mock tier health data for testing."""
    return {
        "lightweight": {
            "status": "healthy",
            "success_rate": 0.98,
            "average_latency_ms": 150.0,
            "active_sessions": 5,
            "max_sessions": 50,
            "cpu_utilization": 0.3,
            "memory_utilization": 0.2,
            "error_count": 1,
            "uptime_percentage": 99.8,
        },
        "playwright": {
            "status": "healthy",
            "success_rate": 0.95,
            "average_latency_ms": 800.0,
            "active_sessions": 8,
            "max_sessions": 20,
            "cpu_utilization": 0.6,
            "memory_utilization": 0.5,
            "error_count": 5,
            "uptime_percentage": 98.5,
        },
        "crawl4ai": {
            "status": "degraded",
            "success_rate": 0.75,
            "average_latency_ms": 2500.0,
            "active_sessions": 18,
            "max_sessions": 20,
            "cpu_utilization": 0.9,
            "memory_utilization": 0.85,
            "error_count": 25,
            "uptime_percentage": 95.0,
        },
    }


# Mutation testing fixtures
@pytest.fixture
def mutation_test_cases():
    """Test cases for mutation testing."""
    return {
        "circuit_breaker_mutations": [
            {
                "original": "failure_count >= threshold",
                "mutation": "failure_count > threshold",
            },
            {
                "original": "time >= recovery_timeout",
                "mutation": "time > recovery_timeout",
            },
            {"original": "except Exception", "mutation": "except ValueError"},
        ],
        "dependency_injection_mutations": [
            {"original": "await initialize()", "mutation": "initialize()"},
            {"original": "finally: cleanup()", "mutation": "finally: pass"},
            {"original": "return Config()", "mutation": "return None"},
        ],
        "caching_mutations": [
            {"original": "time > expires", "mutation": "time >= expires"},
            {"original": "sorted(params)", "mutation": "params"},
            {"original": "del cache[key]", "mutation": "pass"},
        ],
    }


# Test data generators
@pytest.fixture
def generate_test_embeddings():
    """Generate test embeddings for various scenarios."""

    def _generate(count=10, dimension=384):
        import random

        embeddings = []
        for _i in range(count):
            embedding = [random.random() for _ in range(dimension)]
            embeddings.append(embedding)
        return embeddings

    return _generate


@pytest.fixture
def generate_test_documents():
    """Generate test documents for various scenarios."""

    def _generate(count=10):
        documents = []
        for i in range(count):
            doc = {
                "id": f"doc_{i}",
                "text": f"This is test document number {i} with some content for testing.",
                "metadata": {
                    "source": "test",
                    "category": f"category_{i % 3}",
                    "timestamp": time.time() - (i * 100),
                },
            }
            documents.append(doc)
        return documents

    return _generate


# Cleanup fixtures
@pytest.fixture(autouse=True)
async def cleanup_services():
    """Automatically cleanup services after tests."""
    yield
    # Cleanup code would go here
    # For mocks, no actual cleanup needed
    pass


# Marker definitions for test categorization
pytest_plugins = ["pytest_asyncio", "pytest_benchmark"]


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "service_integration: mark test as service integration test"
    )
    config.addinivalue_line(
        "markers", "circuit_breaker: mark test as circuit breaker test"
    )
    config.addinivalue_line(
        "markers", "database_pooling: mark test as database pooling test"
    )
    config.addinivalue_line(
        "markers", "browser_monitoring: mark test as browser monitoring test"
    )
    config.addinivalue_line(
        "markers", "performance_benchmark: mark test as performance benchmark"
    )
    config.addinivalue_line("markers", "mutation_testing: mark test as mutation test")
    config.addinivalue_line(
        "markers", "dependency_injection: mark test as dependency injection test"
    )


# Custom assertions for service testing
class ServiceTestAssertions:
    """Custom assertions for service testing."""

    @staticmethod
    def assert_circuit_breaker_state(circuit_breaker, expected_state):
        """Assert circuit breaker is in expected state."""
        assert circuit_breaker.state.value == expected_state, (
            f"Circuit breaker state is {circuit_breaker.state.value}, "
            f"expected {expected_state}"
        )

    @staticmethod
    def assert_metrics_within_range(metrics, metric_name, min_value, max_value):
        """Assert metric is within expected range."""
        value = metrics.get(metric_name)
        assert value is not None, f"Metric {metric_name} not found"
        assert min_value <= value <= max_value, (
            f"Metric {metric_name} value {value} not in range [{min_value}, {max_value}]"
        )

    @staticmethod
    def assert_service_health(health_data, service_name, expected_status):
        """Assert service health status."""
        service_health = health_data.get(service_name)
        assert service_health is not None, f"Health data for {service_name} not found"
        assert service_health["status"] == expected_status, (
            f"Service {service_name} status is {service_health['status']}, "
            f"expected {expected_status}"
        )


@pytest.fixture
def service_assertions():
    """Provide custom service testing assertions."""
    return ServiceTestAssertions()


# Configuration for different test environments
@pytest.fixture
def test_environment_config():
    """Configuration for different test environments."""
    return {
        "unit": {
            "timeout": 5,
            "mock_external_services": True,
            "enable_metrics": False,
        },
        "integration": {
            "timeout": 30,
            "mock_external_services": False,
            "enable_metrics": True,
        },
        "performance": {
            "timeout": 60,
            "mock_external_services": True,
            "enable_metrics": True,
            "benchmark_rounds": 10,
        },
    }
