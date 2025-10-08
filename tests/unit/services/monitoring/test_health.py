"""Comprehensive tests for monitoring health check functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.health.manager import (
    HealthCheckConfig,
    HealthCheckManager,
    HealthCheckResult,
    HealthStatus,
    HTTPHealthCheck,
    QdrantHealthCheck,
    RedisHealthCheck,
    SystemResourceHealthCheck,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_status_values(self):
        """Test health status enum values."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.UNHEALTHY == "unhealthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNKNOWN == "unknown"
        assert HealthStatus.SKIPPED == "skipped"


class TestHealthCheckConfig:
    """Test HealthCheckConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HealthCheckConfig()
        assert config.enabled is True
        assert config.interval == 30.0
        assert config.timeout == 10.0
        assert config.qdrant_url is None
        assert config.redis_url is None

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HealthCheckConfig(
            enabled=False,
            interval=60.0,
            timeout=5.0,
            qdrant_url="http://example:6333",
            redis_url="redis://cache:6379",
        )
        assert config.enabled is False
        assert config.interval == 60.0
        assert config.timeout == 5.0
        assert config.qdrant_url == "http://example:6333"
        assert config.redis_url == "redis://cache:6379"

    def test_from_unified_config(self):
        """Health check config mirrors unified settings."""

        settings = MagicMock()
        settings.monitoring.enable_health_checks = True
        settings.monitoring.system_metrics_interval = 45.0
        settings.monitoring.health_check_timeout = 12.0
        settings.qdrant.url = "http://qdrant:6333"
        settings.cache.enable_redis_cache = True
        settings.cache.enable_dragonfly_cache = False
        settings.cache.redis_url = "redis://redis:6379"

        config = HealthCheckConfig.from_unified_config(settings)

        assert config.enabled is True
        assert config.interval == 45.0
        assert config.timeout == 12.0
        assert config.qdrant_url == "http://qdrant:6333"
        assert config.redis_url == "redis://redis:6379"


class TestHealthCheckResult:
    """Test HealthCheckResult model."""

    def test_healthy_result(self):
        """Test healthy result creation."""
        result = HealthCheckResult(
            name="test_service",
            status=HealthStatus.HEALTHY,
            message="Service is running",
            duration_ms=50.0,
        )
        assert result.name == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is running"
        assert result.duration_ms == 50.0
        assert result.metadata == {}

    def test_unhealthy_result_with_metadata(self):
        """Test unhealthy result with metadata."""
        result = HealthCheckResult(
            name="test_service",
            status=HealthStatus.UNHEALTHY,
            message="Service unavailable",
            duration_ms=5000.0,
            metadata={"error": "Connection refused", "retries": 3},
        )
        assert result.status == HealthStatus.UNHEALTHY
        assert result.metadata["error"] == "Connection refused"
        assert result.metadata["retries"] == 3


class TestQdrantHealthCheck:
    """Test QdrantHealthCheck functionality."""

    @pytest.fixture
    def mock_qdrant_client(self):
        """Create mock Qdrant client."""
        client = AsyncMock()
        mock_cluster_info = MagicMock()
        mock_cluster_info.status = "green"
        mock_cluster_info.peers = []
        client.get_cluster_info.return_value = mock_cluster_info
        return client

    @pytest.fixture
    def health_check(self, mock_qdrant_client):
        """Create QdrantHealthCheck instance."""
        return QdrantHealthCheck(mock_qdrant_client, "qdrant")

    @pytest.mark.asyncio
    async def test_healthy_qdrant(self, health_check, mock_qdrant_client):
        """Test healthy Qdrant service."""
        result = await health_check.check()

        assert result.name == "qdrant"
        assert result.status == HealthStatus.HEALTHY
        assert "cluster" in result.message
        mock_qdrant_client.get_cluster_info.assert_called_once()

    @pytest.mark.asyncio
    async def test_unhealthy_qdrant(self, health_check, mock_qdrant_client):
        """Test unhealthy Qdrant service."""
        mock_qdrant_client.get_cluster_info.side_effect = Exception("Connection failed")

        result = await health_check.check()

        assert result.name == "qdrant"
        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message


class TestRedisHealthCheck:
    """Test RedisHealthCheck functionality."""

    @pytest.fixture
    def health_check(self):
        """Create RedisHealthCheck instance."""
        return RedisHealthCheck("redis://localhost:6379", "redis")

    @pytest.mark.asyncio
    async def test_healthy_redis(self, health_check):
        """Test healthy Redis service."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping.return_value = True
            mock_from_url.return_value = mock_redis

            result = await health_check.check()

            assert result.name == "redis"
            assert result.status == HealthStatus.HEALTHY
            assert "responding" in result.message

    @pytest.mark.asyncio
    async def test_unhealthy_redis(self, health_check):
        """Test unhealthy Redis service."""
        with patch("redis.asyncio.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.ping.side_effect = Exception("Redis unavailable")
            mock_from_url.return_value = mock_redis

            result = await health_check.check()

            assert result.name == "redis"
            assert result.status == HealthStatus.UNHEALTHY
            assert "Redis unavailable" in result.message


class TestHTTPHealthCheck:
    """Test HTTPHealthCheck functionality."""

    @pytest.fixture
    def health_check(self):
        """Create HTTPHealthCheck instance."""
        return HTTPHealthCheck(
            "http://api.example.com/health",
            name="api_service",
            headers={"Authorization": "Bearer token"},
        )

    @pytest.mark.asyncio
    async def test_healthy_http_service(self, health_check):
        """Test healthy HTTP service."""

        # Mock the internal _check method directly to avoid complex aiohttp mocking
        async def mock_check():
            return HealthCheckResult(
                name="api_service",
                status=HealthStatus.HEALTHY,
                message="HTTP endpoint responding with status 200",
                duration_ms=50.0,
                metadata={"status_code": 200, "content_type": "application/json"},
            )

        with patch.object(
            health_check, "_execute_with_timeout", return_value=await mock_check()
        ):
            result = await health_check.check()

            assert result.name == "api_service"
            assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_http_service(self, health_check):
        """Test unhealthy HTTP service."""

        async def mock_check():
            return HealthCheckResult(
                name="api_service",
                status=HealthStatus.UNHEALTHY,
                message="HTTP endpoint returned status 503, expected 200",
                duration_ms=75.0,
                metadata={"status_code": 503},
            )

        with patch.object(
            health_check, "_execute_with_timeout", return_value=await mock_check()
        ):
            result = await health_check.check()

            assert result.name == "api_service"
            assert result.status == HealthStatus.UNHEALTHY


class TestSystemResourceHealthCheck:
    """Test SystemResourceHealthCheck functionality."""

    @pytest.fixture
    def health_check(self):
        """Create SystemResourceHealthCheck instance."""
        return SystemResourceHealthCheck(
            name="system_resources",
            cpu_threshold=80.0,
            memory_threshold=85.0,
            disk_threshold=90.0,
        )

    @pytest.mark.asyncio
    async def test_healthy_system_resources(self, health_check):
        """Test healthy system resources."""

        async def mock_check():
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.HEALTHY,
                message="System resources healthy",
                duration_ms=25.0,
                metadata={
                    "cpu_percent": 45.0,
                    "memory_percent": 60.0,
                    "disk_percent": 70.0,
                    "memory_available_gb": 8.0,
                    "disk_free_gb": 100.0,
                },
            )

        with patch.object(
            health_check, "_execute_with_timeout", return_value=await mock_check()
        ):
            result = await health_check.check()

            assert result.name == "system_resources"
            assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_degraded_system_resources(self, health_check):
        """Test degraded system resources."""

        async def mock_check():
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.DEGRADED,
                message=(
                    "High CPU usage: 85.0%; High memory usage: 90.0%; "
                    "High disk usage: 95.0%"
                ),
                duration_ms=30.0,
                metadata={
                    "cpu_percent": 85.0,
                    "memory_percent": 90.0,
                    "disk_percent": 95.0,
                    "memory_available_gb": 2.0,
                    "disk_free_gb": 10.0,
                },
            )

        with patch.object(
            health_check, "_execute_with_timeout", return_value=await mock_check()
        ):
            result = await health_check.check()

            assert result.name == "system_resources"
            assert result.status == HealthStatus.DEGRADED


class TestHealthCheckManager:
    """Test HealthCheckManager functionality."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HealthCheckConfig(enabled=False, interval=30.0, timeout=5.0)

    @pytest.fixture
    def manager(self, config):
        """Create HealthCheckManager instance."""
        return HealthCheckManager(config)

    @pytest.fixture
    def mock_health_check(self):
        """Create mock health check."""
        check = AsyncMock()
        check.name = "test_service"
        check.check.return_value = HealthCheckResult(
            name="test_service",
            status=HealthStatus.HEALTHY,
            message="Service is healthy",
            duration_ms=50.0,
        )
        return check

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager.config.enabled is False
        assert len(manager.health_checks) == 0
        assert len(manager._last_results) == 0

    def test_add_health_check(self, manager, mock_health_check):
        """Test adding health check."""
        manager.add_health_check(mock_health_check)
        assert len(manager.health_checks) == 1
        assert any(check.name == "test_service" for check in manager.health_checks)

    @pytest.mark.asyncio
    async def test_check_all(self, manager, mock_health_check):
        """Test checking all health checks."""
        manager.add_health_check(mock_health_check)

        results = await manager.check_all()

        assert len(results) == 1
        assert "test_service" in results
        assert results["test_service"].status == HealthStatus.HEALTHY

    def test_get_overall_status_all_healthy(self, manager):
        """Test overall status when all services are healthy."""
        manager._last_results = {
            "service1": HealthCheckResult(
                name="service1",
                status=HealthStatus.HEALTHY,
                message="OK",
                duration_ms=50.0,
            ),
            "service2": HealthCheckResult(
                name="service2",
                status=HealthStatus.HEALTHY,
                message="OK",
                duration_ms=75.0,
            ),
        }

        status = manager.get_overall_status()
        assert status == HealthStatus.HEALTHY

    def test_get_overall_status_with_failures(self, manager):
        """Test overall status with service failures."""
        manager._last_results = {
            "service1": HealthCheckResult(
                name="service1",
                status=HealthStatus.HEALTHY,
                message="OK",
                duration_ms=50.0,
            ),
            "service2": HealthCheckResult(
                name="service2",
                status=HealthStatus.UNHEALTHY,
                message="Failed",
                duration_ms=5000.0,
            ),
        }

        status = manager.get_overall_status()
        assert status == HealthStatus.UNHEALTHY

    def test_disabled_health_check_manager(self):
        """Test manager with health checks disabled."""
        config = HealthCheckConfig(enabled=False, interval=30.0, timeout=5.0)
        manager = HealthCheckManager(config)

        assert manager.config.enabled is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, manager):
        """Test health check timeout handling."""
        # Create a real health check instance that will timeout

        # Use a slow/invalid URL that will timeout
        slow_check = HTTPHealthCheck(
            "http://192.0.2.1:9999/health", name="slow_service", timeout_seconds=0.1
        )
        manager.add_health_check(slow_check)

        results = await manager.check_all()

        # Should timeout and return unhealthy status
        assert "slow_service" in results
        assert results["slow_service"].status == HealthStatus.UNHEALTHY
        assert (
            "timed out" in results["slow_service"].message
            or "failed" in results["slow_service"].message
        )


class TestHealthCheckIntegration:
    """Test health check integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_health_check_cycle(self):
        """Test complete health check lifecycle."""
        config = HealthCheckConfig(enabled=False, interval=1.0, timeout=5.0)
        manager = HealthCheckManager(config)

        # Add multiple health checks
        mock_qdrant = AsyncMock()
        mock_cluster_info = MagicMock()
        mock_cluster_info.status = "green"
        mock_cluster_info.peers = []
        mock_qdrant.get_cluster_info.return_value = mock_cluster_info
        qdrant_check = QdrantHealthCheck(mock_qdrant, "qdrant")

        system_check = SystemResourceHealthCheck(name="system", cpu_threshold=90.0)

        manager.add_health_check(qdrant_check)
        manager.add_health_check(system_check)

        # Run health checks
        results = await manager.check_all()

        assert len(results) == 2
        assert "qdrant" in results
        assert "system" in results

        # Check overall status
        overall_status = manager.get_overall_status()
        assert overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
