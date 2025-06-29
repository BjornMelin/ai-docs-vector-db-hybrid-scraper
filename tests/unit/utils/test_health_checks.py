"""Tests for centralized health check utilities."""

from unittest.mock import Mock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import Config
from src.utils.health_checks import ServiceHealthChecker


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    # Create a default config first
    config = Config()

    # Set providers and API keys
    config.embedding_provider = "openai"
    config.crawl_provider = "firecrawl"
    config.qdrant.url = "http://localhost:6333"
    config.qdrant.api_key = "test-key"
    config.openai.api_key = "sk-test-key"
    config.firecrawl.api_key = "test-firecrawl-key"
    config.cache.enable_dragonfly_cache = True
    config.cache.dragonfly_url = "redis://localhost:6380"
    return config


class TestServiceHealthChecker:
    """Test cases for ServiceHealthChecker."""

    def test_check_qdrant_connection_success(self, sample_config):
        """Test successful Qdrant connection check."""
        mock_collections = Mock()
        mock_collections.collections = [Mock(), Mock()]

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            result = ServiceHealthChecker.check_qdrant_connection(sample_config)

            assert result["service"] == "qdrant"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["collections_count"] == 2
            assert result["details"]["url"] == "http://localhost:6333"

    def test_check_qdrant_connection_auth_failure(self, sample_config):
        """Test Qdrant connection check with authentication failure."""

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = Mock()
            auth_error = UnexpectedResponse(
                status_code=401,
                reason_phrase="Unauthorized",
                content=b'{"error": "Unauthorized"}',
                headers={},
            )
            mock_client.get_collections.side_effect = auth_error
            mock_client_class.return_value = mock_client

            result = ServiceHealthChecker.check_qdrant_connection(sample_config)

            assert result["service"] == "qdrant"
            assert result["connected"] is False
            assert "Authentication failed" in result["error"]

    def test_check_qdrant_connection_general_error(self, sample_config):
        """Test Qdrant connection check with general error."""
        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.side_effect = Exception("Connection failed")
            mock_client_class.return_value = mock_client

            result = ServiceHealthChecker.check_qdrant_connection(sample_config)

            assert result["service"] == "qdrant"
            assert result["connected"] is False
            assert result["error"] == "Connection failed"

    def test_check_dragonfly_connection_success(self, sample_config):
        """Test successful DragonflyDB connection check."""
        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_redis.return_value = mock_client

            result = ServiceHealthChecker.check_dragonfly_connection(sample_config)

            assert result["service"] == "dragonfly"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["url"] == "redis://localhost:6380"

    def test_check_openai_connection_success(self, sample_config):
        """Test successful OpenAI connection check."""
        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_models = [Mock(), Mock(), Mock()]
            mock_client.models.list.return_value = mock_models
            mock_openai_class.return_value = mock_client

            result = ServiceHealthChecker.check_openai_connection(sample_config)

            assert result["service"] == "openai"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["available_models_count"] == 3

    def test_check_openai_connection_not_configured(self, sample_config):
        """Test OpenAI connection check when not configured."""
        sample_config.embedding_provider = "fastembed"

        result = ServiceHealthChecker.check_openai_connection(sample_config)

        assert result["service"] == "openai"
        assert result["connected"] is False
        assert "not configured" in result["error"]

    def test_check_firecrawl_connection_success(self, sample_config):
        """Test successful Firecrawl connection check."""
        with patch("httpx.get") as mock_httpx:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_httpx.return_value = mock_response

            result = ServiceHealthChecker.check_firecrawl_connection(sample_config)

            assert result["service"] == "firecrawl"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["status_code"] == 200

    def test_check_firecrawl_connection_api_error(self, sample_config):
        """Test Firecrawl connection check with API error."""
        with patch("httpx.get") as mock_httpx:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_httpx.return_value = mock_response

            result = ServiceHealthChecker.check_firecrawl_connection(sample_config)

            assert result["service"] == "firecrawl"
            assert result["connected"] is False
            assert "status 500" in result["error"]

    def test_perform_all_health_checks(self, sample_config):
        """Test performing all health checks."""
        with (
            patch.object(
                ServiceHealthChecker, "check_qdrant_connection"
            ) as mock_qdrant,
            patch.object(
                ServiceHealthChecker, "check_dragonfly_connection"
            ) as mock_dragonfly,
            patch.object(
                ServiceHealthChecker, "check_openai_connection"
            ) as mock_openai,
            patch.object(
                ServiceHealthChecker, "check_firecrawl_connection"
            ) as mock_firecrawl,
        ):
            # Configure mock returns
            mock_qdrant.return_value = {"service": "qdrant", "connected": True}
            mock_dragonfly.return_value = {"service": "dragonfly", "connected": True}
            mock_openai.return_value = {"service": "openai", "connected": True}
            mock_firecrawl.return_value = {"service": "firecrawl", "connected": True}

            result = ServiceHealthChecker.perform_all_health_checks(sample_config)

            # Verify all services were checked
            assert len(result) == 4
            assert "qdrant" in result
            assert "dragonfly" in result
            assert "openai" in result
            assert "firecrawl" in result

    def test_get_connection_summary_all_healthy(self, sample_config):
        """Test connection summary with all services healthy."""
        with patch.object(
            ServiceHealthChecker, "perform_all_health_checks"
        ) as mock_checks:
            mock_checks.return_value = {
                "qdrant": {"connected": True},
                "dragonfly": {"connected": True},
                "openai": {"connected": True},
            }

            result = ServiceHealthChecker.get_connection_summary(sample_config)

            assert result["overall_status"] == "healthy"
            assert result["_total_services"] == 3
            assert result["connected_count"] == 3
            assert result["failed_count"] == 0
            assert result["connected_services"] == ["qdrant", "dragonfly", "openai"]
            assert result["failed_services"] == []

    def test_get_connection_summary_some_unhealthy(self, sample_config):
        """Test connection summary with some services unhealthy."""
        with patch.object(
            ServiceHealthChecker, "perform_all_health_checks"
        ) as mock_checks:
            mock_checks.return_value = {
                "qdrant": {"connected": True},
                "dragonfly": {"connected": False},
                "openai": {"connected": True},
            }

            result = ServiceHealthChecker.get_connection_summary(sample_config)

            assert result["overall_status"] == "unhealthy"
            assert result["_total_services"] == 3
            assert result["connected_count"] == 2
            assert result["failed_count"] == 1
            assert result["connected_services"] == ["qdrant", "openai"]
            assert result["failed_services"] == ["dragonfly"]
