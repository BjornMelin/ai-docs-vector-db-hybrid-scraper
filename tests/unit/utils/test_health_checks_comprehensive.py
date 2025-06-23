"""Comprehensive tests for health check utilities.

This test suite provides complete coverage for the ServiceHealthChecker
class and all its health check methods.
"""

from unittest.mock import Mock, patch

from src.config import Config
from src.config.enums import CrawlProvider, EmbeddingProvider
from src.utils.health_checks import ServiceHealthChecker


class TestQdrantHealthCheck:
    """Test Qdrant connection health checks."""

    def test_qdrant_connection_success(self):
        """Test successful Qdrant connection."""
        config = Config()
        config.qdrant.url = "http://localhost:6333"
        config.qdrant.api_key = "test-key"

        # Mock successful Qdrant client
        mock_collections = Mock()
        mock_collections.collections = [Mock(), Mock()]  # 2 collections

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client = Mock()
            mock_client.get_collections.return_value = mock_collections
            mock_client_class.return_value = mock_client

            result = ServiceHealthChecker.check_qdrant_connection(config)

            assert result["service"] == "qdrant"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["collections_count"] == 2
            assert result["details"]["url"] == "http://localhost:6333"

            mock_client_class.assert_called_once_with(
                url="http://localhost:6333", api_key="test-key", timeout=5.0
            )

    def test_qdrant_authentication_error(self):
        """Test Qdrant authentication failure."""
        config = Config()

        from qdrant_client.http.exceptions import UnexpectedResponse

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.side_effect = UnexpectedResponse(
                status_code=401, reason_phrase="Unauthorized", content="", headers={}
            )

            result = ServiceHealthChecker.check_qdrant_connection(config)

            assert result["service"] == "qdrant"
            assert result["connected"] is False
            assert "Authentication failed - check API key" in result["error"]

    def test_qdrant_http_error(self):
        """Test Qdrant HTTP error responses."""
        config = Config()

        from qdrant_client.http.exceptions import UnexpectedResponse

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.side_effect = UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content="",
                headers={},
            )

            result = ServiceHealthChecker.check_qdrant_connection(config)

            assert result["service"] == "qdrant"
            assert result["connected"] is False
            assert "HTTP 500: Internal Server Error" in result["error"]

    def test_qdrant_generic_error(self):
        """Test Qdrant generic connection error."""
        config = Config()

        with patch("qdrant_client.QdrantClient") as mock_client_class:
            mock_client_class.side_effect = ConnectionError("Connection failed")

            result = ServiceHealthChecker.check_qdrant_connection(config)

            assert result["service"] == "qdrant"
            assert result["connected"] is False
            assert "Connection failed" in result["error"]


class TestDragonflyHealthCheck:
    """Test DragonflyDB connection health checks."""

    def test_dragonfly_connection_success(self):
        """Test successful DragonflyDB connection."""
        config = Config()
        config.cache.enable_dragonfly_cache = True
        config.cache.dragonfly_url = "redis://localhost:6379"

        with patch("redis.from_url") as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            result = ServiceHealthChecker.check_dragonfly_connection(config)

            assert result["service"] == "dragonfly"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["url"] == "redis://localhost:6379"

            mock_redis.assert_called_once_with(
                "redis://localhost:6379", socket_connect_timeout=5
            )

    def test_dragonfly_disabled(self):
        """Test when DragonflyDB cache is disabled."""
        config = Config()
        config.cache.enable_dragonfly_cache = False

        result = ServiceHealthChecker.check_dragonfly_connection(config)

        assert result["service"] == "dragonfly"
        assert result["connected"] is False
        assert "DragonflyDB cache not enabled" in result["error"]

    def test_dragonfly_connection_error(self):
        """Test DragonflyDB connection error."""
        config = Config()
        config.cache.enable_dragonfly_cache = True

        import redis

        with patch("redis.from_url") as mock_redis:
            mock_redis.side_effect = redis.ConnectionError("Connection refused")

            result = ServiceHealthChecker.check_dragonfly_connection(config)

            assert result["service"] == "dragonfly"
            assert result["connected"] is False
            assert "Connection refused - is DragonflyDB running?" in result["error"]

    def test_dragonfly_generic_error(self):
        """Test DragonflyDB generic error."""
        config = Config()
        config.cache.enable_dragonfly_cache = True

        with patch("redis.from_url") as mock_redis:
            mock_redis.side_effect = Exception("Generic error")

            result = ServiceHealthChecker.check_dragonfly_connection(config)

            assert result["service"] == "dragonfly"
            assert result["connected"] is False
            assert "Generic error" in result["error"]


class TestOpenAIHealthCheck:
    """Test OpenAI API connection health checks."""

    def test_openai_connection_success(self):
        """Test successful OpenAI API connection."""
        config = Config()
        config.embedding_provider = EmbeddingProvider.OPENAI
        config.openai.api_key = "sk-test-key"
        config.openai.model = "text-embedding-3-small"
        config.openai.dimensions = 1536

        with patch("openai.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_models = [Mock() for _ in range(50)]  # 50 available models
            mock_client.models.list.return_value = mock_models
            mock_openai_class.return_value = mock_client

            result = ServiceHealthChecker.check_openai_connection(config)

            assert result["service"] == "openai"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["model"] == "text-embedding-3-small"
            assert result["details"]["dimensions"] == 1536
            assert result["details"]["available_models_count"] == 50

            mock_openai_class.assert_called_once_with(
                api_key="sk-test-key", timeout=5.0
            )

    def test_openai_not_configured(self):
        """Test when OpenAI is not configured as embedding provider."""
        config = Config()
        config.embedding_provider = EmbeddingProvider.FASTEMBED

        result = ServiceHealthChecker.check_openai_connection(config)

        assert result["service"] == "openai"
        assert result["connected"] is False
        assert "OpenAI not configured as embedding provider" in result["error"]

    def test_openai_missing_api_key(self):
        """Test when OpenAI API key is missing."""
        config = Config()
        config.embedding_provider = EmbeddingProvider.OPENAI
        config.openai.api_key = None

        result = ServiceHealthChecker.check_openai_connection(config)

        assert result["service"] == "openai"
        assert result["connected"] is False
        assert "API key missing" in result["error"]

    def test_openai_api_error(self):
        """Test OpenAI API error."""
        config = Config()
        config.embedding_provider = EmbeddingProvider.OPENAI
        config.openai.api_key = "sk-invalid-key"

        with patch("openai.OpenAI") as mock_openai_class:
            mock_openai_class.side_effect = Exception("API error: Invalid API key")

            result = ServiceHealthChecker.check_openai_connection(config)

            assert result["service"] == "openai"
            assert result["connected"] is False
            assert "API error: Invalid API key" in result["error"]


class TestFirecrawlHealthCheck:
    """Test Firecrawl API connection health checks."""

    def test_firecrawl_connection_success(self):
        """Test successful Firecrawl API connection."""
        config = Config()
        config.crawl_provider = CrawlProvider.FIRECRAWL
        config.firecrawl.api_key = "fc-test-key"
        config.firecrawl.api_url = "https://api.firecrawl.dev"

        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            result = ServiceHealthChecker.check_firecrawl_connection(config)

            assert result["service"] == "firecrawl"
            assert result["connected"] is True
            assert result["error"] is None
            assert result["details"]["api_url"] == "https://api.firecrawl.dev"
            assert result["details"]["status_code"] == 200

            mock_get.assert_called_once_with(
                "https://api.firecrawl.dev/health",
                headers={"Authorization": "Bearer fc-test-key"},
                timeout=5.0,
            )

    def test_firecrawl_not_configured(self):
        """Test when Firecrawl is not configured as crawl provider."""
        config = Config()
        config.crawl_provider = CrawlProvider.CRAWL4AI

        result = ServiceHealthChecker.check_firecrawl_connection(config)

        assert result["service"] == "firecrawl"
        assert result["connected"] is False
        assert "Firecrawl not configured as crawl provider" in result["error"]

    def test_firecrawl_missing_api_key(self):
        """Test when Firecrawl API key is missing."""
        config = Config()
        config.crawl_provider = CrawlProvider.FIRECRAWL
        config.firecrawl.api_key = None

        result = ServiceHealthChecker.check_firecrawl_connection(config)

        assert result["service"] == "firecrawl"
        assert result["connected"] is False
        assert "API key missing" in result["error"]

    def test_firecrawl_api_error_status(self):
        """Test Firecrawl API error status response."""
        config = Config()
        config.crawl_provider = CrawlProvider.FIRECRAWL
        config.firecrawl.api_key = "fc-invalid-key"

        with patch("httpx.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_get.return_value = mock_response

            result = ServiceHealthChecker.check_firecrawl_connection(config)

            assert result["service"] == "firecrawl"
            assert result["connected"] is False
            assert "API returned status 401" in result["error"]

    def test_firecrawl_request_exception(self):
        """Test Firecrawl request exception."""
        config = Config()
        config.crawl_provider = CrawlProvider.FIRECRAWL
        config.firecrawl.api_key = "fc-test-key"

        with patch("httpx.get") as mock_get:
            mock_get.side_effect = Exception("Network error")

            result = ServiceHealthChecker.check_firecrawl_connection(config)

            assert result["service"] == "firecrawl"
            assert result["connected"] is False
            assert "Network error" in result["error"]


class TestAllHealthChecks:
    """Test comprehensive health check functionality."""

    def test_perform_all_health_checks_all_services(self):
        """Test performing health checks for all configured services."""
        config = Config()
        config.embedding_provider = EmbeddingProvider.OPENAI
        config.crawl_provider = CrawlProvider.FIRECRAWL
        config.cache.enable_dragonfly_cache = True
        config.openai.api_key = "sk-test"
        config.firecrawl.api_key = "fc-test"

        # Mock all health check methods
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
            mock_qdrant.return_value = {"service": "qdrant", "connected": True}
            mock_dragonfly.return_value = {"service": "dragonfly", "connected": True}
            mock_openai.return_value = {"service": "openai", "connected": True}
            mock_firecrawl.return_value = {"service": "firecrawl", "connected": True}

            results = ServiceHealthChecker.perform_all_health_checks(config)

            assert len(results) == 4
            assert "qdrant" in results
            assert "dragonfly" in results
            assert "openai" in results
            assert "firecrawl" in results

            mock_qdrant.assert_called_once_with(config)
            mock_dragonfly.assert_called_once_with(config)
            mock_openai.assert_called_once()
            mock_firecrawl.assert_called_once_with(config)

    def test_perform_all_health_checks_minimal_config(self):
        """Test health checks with minimal configuration (only Qdrant)."""
        config = Config()
        config.embedding_provider = EmbeddingProvider.FASTEMBED
        config.crawl_provider = CrawlProvider.CRAWL4AI
        config.cache.enable_dragonfly_cache = False

        with patch.object(
            ServiceHealthChecker, "check_qdrant_connection"
        ) as mock_qdrant:
            mock_qdrant.return_value = {"service": "qdrant", "connected": True}

            results = ServiceHealthChecker.perform_all_health_checks(config)

            assert len(results) == 1
            assert "qdrant" in results
            assert "dragonfly" not in results
            assert "openai" not in results
            assert "firecrawl" not in results

            mock_qdrant.assert_called_once_with(config)

    def test_get_connection_summary_all_healthy(self):
        """Test connection summary when all services are healthy."""
        config = Config()

        with patch.object(
            ServiceHealthChecker, "perform_all_health_checks"
        ) as mock_checks:
            mock_checks.return_value = {
                "qdrant": {"service": "qdrant", "connected": True},
                "dragonfly": {"service": "dragonfly", "connected": True},
                "openai": {"service": "openai", "connected": True},
            }

            summary = ServiceHealthChecker.get_connection_summary(config)

            assert summary["overall_status"] == "healthy"
            assert summary["total_services"] == 3
            assert summary["connected_count"] == 3
            assert summary["failed_count"] == 0
            assert set(summary["connected_services"]) == {
                "qdrant",
                "dragonfly",
                "openai",
            }
            assert summary["failed_services"] == []
            assert len(summary["detailed_results"]) == 3

    def test_get_connection_summary_some_failed(self):
        """Test connection summary when some services fail."""
        config = Config()

        with patch.object(
            ServiceHealthChecker, "perform_all_health_checks"
        ) as mock_checks:
            mock_checks.return_value = {
                "qdrant": {"service": "qdrant", "connected": True},
                "dragonfly": {"service": "dragonfly", "connected": False},
                "openai": {"service": "openai", "connected": False},
            }

            summary = ServiceHealthChecker.get_connection_summary(config)

            assert summary["overall_status"] == "unhealthy"
            assert summary["total_services"] == 3
            assert summary["connected_count"] == 1
            assert summary["failed_count"] == 2
            assert summary["connected_services"] == ["qdrant"]
            assert set(summary["failed_services"]) == {"dragonfly", "openai"}

    def test_get_connection_summary_all_failed(self):
        """Test connection summary when all services fail."""
        config = Config()

        with patch.object(
            ServiceHealthChecker, "perform_all_health_checks"
        ) as mock_checks:
            mock_checks.return_value = {
                "qdrant": {"service": "qdrant", "connected": False}
            }

            summary = ServiceHealthChecker.get_connection_summary(config)

            assert summary["overall_status"] == "unhealthy"
            assert summary["total_services"] == 1
            assert summary["connected_count"] == 0
            assert summary["failed_count"] == 1
            assert summary["connected_services"] == []
            assert summary["failed_services"] == ["qdrant"]
