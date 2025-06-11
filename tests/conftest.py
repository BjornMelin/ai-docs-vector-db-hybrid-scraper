"""Pytest configuration and shared fixtures."""

import os
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest

# Add src to path for all tests - eliminates need for import path manipulation
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from dotenv import load_dotenv  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for offline envs

    def load_dotenv(*args, **kwargs):
        """Fallback no-op load_dotenv implementation."""
        return False


try:
    from qdrant_client.models import PointStruct  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - basic fallback
    from dataclasses import dataclass

    @dataclass
    class PointStruct:  # type: ignore
        """Simplified stand-in for qdrant_client.models.PointStruct."""

        id: int
        vector: list[float]
        payload: dict


# Load test environment variables at module import
_test_env_path = Path(__file__).parent.parent / ".env.test"
if _test_env_path.exists():
    load_dotenv(_test_env_path, override=True)


# Remove the custom event_loop fixture to use pytest-asyncio's default
# The event loop scope is now configured in pyproject.toml


@pytest.fixture()
def temp_dir() -> Generator[Path]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture()
def mock_env_vars() -> Generator[None]:
    """Mock environment variables for testing."""
    # Save current values
    saved_vars = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "QDRANT_URL": os.environ.get("QDRANT_URL"),
    }

    # Set test values if not already set
    os.environ.setdefault("OPENAI_API_KEY", "test_key")
    os.environ.setdefault("QDRANT_URL", "http://localhost:6333")

    yield

    # Restore original values
    for key, value in saved_vars.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture()
def mock_qdrant_client() -> MagicMock:
    """Mock Qdrant client for testing."""
    client = MagicMock()
    client.create_collection = AsyncMock()
    client.delete_collection = AsyncMock()
    client.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    client.upsert = AsyncMock()
    client.search = AsyncMock(return_value=[])
    client.count = AsyncMock(return_value=MagicMock(count=0))
    client.close = AsyncMock()
    return client


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client for testing."""
    client = MagicMock()
    client.embeddings.create = AsyncMock(
        return_value=MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)],
        ),
    )
    return client


@pytest.fixture()
def mock_crawl4ai() -> MagicMock:
    """Mock Crawl4AI AsyncWebCrawler for testing."""
    crawler = MagicMock()
    crawler.__aenter__ = AsyncMock(return_value=crawler)
    crawler.__aexit__ = AsyncMock(return_value=None)
    crawler.arun = AsyncMock(
        return_value=MagicMock(
            success=True,
            cleaned_html="<p>Test content</p>",
            markdown="Test content",
            metadata={"title": "Test Page"},
            links={"internal": ["http://example.com/page1"]},
        ),
    )
    return crawler


@pytest.fixture()
def sample_documentation_site() -> dict:
    """Sample documentation site configuration."""
    return {
        "name": "test-docs",
        "url": "https://test.example.com",
        "max_depth": 2,
        "exclude_patterns": ["/api/", "/internal/"],
    }


@pytest.fixture()
def sample_crawl_result() -> dict:
    """Sample crawl result data."""
    return {
        "url": "https://test.example.com/docs/page1",
        "title": "Test Page",
        "content": "This is test content for the documentation page.",
        "markdown": "# Test Page\n\nThis is test content for the documentation page.",
        "metadata": {
            "description": "Test page description",
            "keywords": ["test", "documentation"],
        },
        "links": ["https://test.example.com/docs/page2"],
        "success": True,
        "error": None,
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.fixture()
def sample_vector_points() -> list[PointStruct]:
    """Sample vector points for testing."""
    return [
        PointStruct(
            id=1,
            vector=[0.1] * 1536,
            payload={
                "url": "https://test.example.com/docs/page1",
                "title": "Test Page 1",
                "content": "Test content 1",
                "chunk_index": 0,
            },
        ),
        PointStruct(
            id=2,
            vector=[0.2] * 1536,
            payload={
                "url": "https://test.example.com/docs/page2",
                "title": "Test Page 2",
                "content": "Test content 2",
                "chunk_index": 0,
            },
        ),
    ]


# Enhanced Database Testing Fixtures
@pytest.fixture()
def enhanced_db_config():
    """Enhanced SQLAlchemy configuration for testing."""
    from src.config.models import SQLAlchemyConfig
    
    return SQLAlchemyConfig(
        database_url="sqlite+aiosqlite:///:memory:",
        pool_size=5,
        min_pool_size=2,
        max_pool_size=15,
        max_overflow=10,
        pool_timeout=30.0,
        pool_recycle=3600,
        pool_pre_ping=True,
        adaptive_pool_sizing=True,
        enable_query_monitoring=True,
        slow_query_threshold_ms=100.0,
        pool_growth_factor=1.5,
        echo_queries=False,
    )


@pytest.fixture()
def mock_predictive_load_monitor():
    """Mock PredictiveLoadMonitor for testing."""
    from unittest.mock import Mock, AsyncMock
    from src.infrastructure.database.predictive_monitor import LoadPrediction
    
    monitor = Mock()
    monitor.start = AsyncMock()
    monitor.stop = AsyncMock()
    monitor.record_request_start = AsyncMock()
    monitor.record_request_end = AsyncMock()
    monitor.record_connection_error = AsyncMock()
    monitor.get_current_load = AsyncMock()
    monitor.calculate_load_factor = Mock(return_value=0.5)
    monitor.predict_future_load = AsyncMock(
        return_value=LoadPrediction(
            predicted_load=0.6,
            confidence_score=0.8,
            recommendation="Monitor - moderate load predicted",
            time_horizon_minutes=15,
            feature_importance={"avg_requests": 0.3, "memory_trend": 0.2},
            trend_direction="stable",
        )
    )
    monitor.train_prediction_model = AsyncMock(return_value=True)
    monitor.get_prediction_metrics = AsyncMock(
        return_value={
            "model_trained": True,
            "training_samples": 100,
            "prediction_accuracy_avg": 0.85,
        }
    )
    return monitor


@pytest.fixture()
def mock_multi_level_circuit_breaker():
    """Mock MultiLevelCircuitBreaker for testing."""
    from unittest.mock import Mock, AsyncMock
    from src.infrastructure.database.enhanced_circuit_breaker import CircuitState, FailureType
    
    breaker = Mock()
    breaker.state = CircuitState.CLOSED
    breaker.execute = AsyncMock()
    breaker.get_health_status = Mock(
        return_value={
            "state": "closed",
            "failure_metrics": {"total_failures": 0},
            "request_metrics": {"success_rate": 1.0},
        }
    )
    breaker.register_partial_failure_handler = Mock()
    breaker.register_fallback_handler = Mock()
    breaker.force_open = AsyncMock()
    breaker.force_close = AsyncMock()
    breaker.get_failure_analysis = AsyncMock(
        return_value={
            "status": "healthy",
            "recommendations": ["System is operating normally"],
        }
    )
    return breaker


@pytest.fixture()
def mock_connection_affinity_manager():
    """Mock ConnectionAffinityManager for testing."""
    from unittest.mock import Mock, AsyncMock
    
    manager = Mock()
    manager.get_optimal_connection = AsyncMock(return_value="conn_123")
    manager.register_connection = AsyncMock()
    manager.unregister_connection = AsyncMock()
    manager.track_query_performance = AsyncMock()
    manager.get_connection_recommendations = AsyncMock(
        return_value={
            "query_type": "read",
            "available_connections": 5,
            "recommendations": [],
        }
    )
    manager.get_performance_report = AsyncMock(
        return_value={
            "summary": {"total_patterns": 25, "total_connections": 5},
            "top_patterns": [],
            "connection_performance": {},
        }
    )
    return manager


@pytest.fixture()
def mock_adaptive_config_manager():
    """Mock AdaptiveConfigManager for testing."""
    from unittest.mock import Mock, AsyncMock
    from src.infrastructure.database.adaptive_config import AdaptationStrategy, SystemLoadLevel
    
    manager = Mock()
    manager.strategy = AdaptationStrategy.MODERATE
    manager.start_monitoring = AsyncMock()
    manager.stop_monitoring = AsyncMock()
    manager.get_current_configuration = AsyncMock(
        return_value={
            "strategy": "moderate",
            "current_load_level": "medium",
            "current_settings": {
                "pool_size": 8,
                "monitoring_interval": 5.0,
                "failure_threshold": 5,
                "timeout_ms": 30000.0,
            },
        }
    )
    manager.force_adaptation = AsyncMock()
    manager.get_adaptation_history = AsyncMock(return_value=[])
    manager.get_performance_analysis = AsyncMock(
        return_value={
            "current_load_level": "medium",
            "resource_utilization": {"avg_cpu_percent": 45.0},
            "recommendations": ["System operating within normal parameters"],
        }
    )
    return manager


@pytest.fixture()
def sample_load_metrics():
    """Sample load metrics for testing."""
    import time
    from src.infrastructure.database.load_monitor import LoadMetrics
    
    return [
        LoadMetrics(
            concurrent_requests=5,
            memory_usage_percent=60.0,
            cpu_usage_percent=40.0,
            avg_response_time_ms=150.0,
            connection_errors=0,
            timestamp=time.time() - 100,
        ),
        LoadMetrics(
            concurrent_requests=8,
            memory_usage_percent=70.0,
            cpu_usage_percent=55.0,
            avg_response_time_ms=200.0,
            connection_errors=1,
            timestamp=time.time() - 50,
        ),
        LoadMetrics(
            concurrent_requests=3,
            memory_usage_percent=45.0,
            cpu_usage_percent=30.0,
            avg_response_time_ms=100.0,
            connection_errors=0,
            timestamp=time.time(),
        ),
    ]


@pytest.fixture()
def sample_query_patterns():
    """Sample query patterns for connection affinity testing."""
    return [
        ("SELECT * FROM users WHERE id = ?", "read"),
        ("INSERT INTO users (name, email) VALUES (?, ?)", "write"),
        ("SELECT COUNT(*) FROM orders GROUP BY date", "analytics"),
        ("BEGIN; UPDATE accounts SET balance = ? WHERE id = ?; COMMIT;", "transaction"),
        ("ANALYZE TABLE performance_stats", "maintenance"),
    ]
