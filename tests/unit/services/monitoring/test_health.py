"""Tests for health check functionality."""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import aiohttp
import pytest
from src.services.monitoring.health import HealthCheckConfig
from src.services.monitoring.health import HealthCheckManager
from src.services.monitoring.health import HealthCheckResult
from src.services.monitoring.health import HealthStatus
from src.services.monitoring.health import HTTPHealthCheck
from src.services.monitoring.health import QdrantHealthCheck
from src.services.monitoring.health import RedisHealthCheck
from src.services.monitoring.health import SystemResourceHealthCheck


class TestHealthCheckConfig:
    """Test HealthCheckConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HealthCheckConfig()
        assert config.enabled is True
        assert config.interval == 30.0
        assert config.timeout == 10.0
        assert config.max_retries == 3
        assert config.qdrant_url == "http://localhost:6333"
        assert config.redis_url == "redis://localhost:6379"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HealthCheckConfig(
            enabled=False,
            interval=60.0,
            timeout=5.0,
            max_retries=1,
            qdrant_url="http://qdrant:6333",
            redis_url="redis://redis:6379"
        )
        assert config.enabled is False
        assert config.interval == 60.0
        assert config.timeout == 5.0
        assert config.max_retries == 1
        assert config.qdrant_url == "http://qdrant:6333"
        assert config.redis_url == "redis://redis:6379"


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestHealthCheckResult:
    """Test HealthCheckResult model."""

    def test_healthy_result(self):
        """Test healthy result creation."""
        result = HealthCheckResult(
            name="test_service",
            status=HealthStatus.HEALTHY,
            message="Service is running"
        )
        assert result.name == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "Service is running"
        assert result.timestamp is not None
        assert result.details == {}

    def test_unhealthy_result_with_details(self):
        """Test unhealthy result with details."""
        details = {"error": "Connection refused", "code": 503}
        result = HealthCheckResult(
            name="failing_service",
            status=HealthStatus.UNHEALTHY,
            message="Service unavailable",
            details=details
        )
        assert result.status == HealthStatus.UNHEALTHY
        assert result.details == details


class TestQdrantHealthCheck:
    """Test Qdrant health check implementation."""

    @pytest.fixture
    def health_check(self):
        """Create Qdrant health check instance."""
        return QdrantHealthCheck("http://localhost:6333")

    @pytest.mark.asyncio
    async def test_healthy_qdrant(self, health_check):
        """Test healthy Qdrant response."""
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "ok"})

        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            result = await health_check.check()

        assert result.name == "qdrant"
        assert result.status == HealthStatus.HEALTHY
        assert "Qdrant is healthy" in result.message

    @pytest.mark.asyncio
    async def test_unhealthy_qdrant(self, health_check):
        """Test unhealthy Qdrant response."""
        mock_response = Mock()
        mock_response.status = 503

        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            result = await health_check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "503" in result.message

    @pytest.mark.asyncio
    async def test_qdrant_connection_error(self, health_check):
        """Test Qdrant connection error."""
        with patch('aiohttp.ClientSession.get', side_effect=aiohttp.ClientError("Connection failed")):
            result = await health_check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Connection failed" in result.message


class TestRedisHealthCheck:
    """Test Redis health check implementation."""

    @pytest.fixture
    def health_check(self):
        """Create Redis health check instance."""
        return RedisHealthCheck("redis://localhost:6379")

    @pytest.mark.asyncio
    async def test_healthy_redis(self, health_check):
        """Test healthy Redis response."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True

        with patch('redis.asyncio.from_url', return_value=mock_redis):
            result = await health_check.check()

        assert result.name == "redis"
        assert result.status == HealthStatus.HEALTHY
        assert "Redis is healthy" in result.message

    @pytest.mark.asyncio
    async def test_unhealthy_redis(self, health_check):
        """Test unhealthy Redis response."""
        mock_redis = AsyncMock()
        mock_redis.ping.side_effect = Exception("Redis connection failed")

        with patch('redis.asyncio.from_url', return_value=mock_redis):
            result = await health_check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Redis connection failed" in result.message


class TestHTTPHealthCheck:
    """Test HTTP health check implementation."""

    @pytest.fixture
    def health_check(self):
        """Create HTTP health check instance."""
        return HTTPHealthCheck("external_api", "https://api.example.com/health")

    @pytest.mark.asyncio
    async def test_healthy_http_service(self, health_check):
        """Test healthy HTTP service."""
        mock_response = Mock()
        mock_response.status = 200

        with patch('aiohttp.ClientSession.get', return_value=mock_response):
            result = await health_check.check()

        assert result.name == "external_api"
        assert result.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_unhealthy_http_service(self, health_check):
        """Test unhealthy HTTP service."""
        with patch('aiohttp.ClientSession.get', side_effect=aiohttp.ClientError("Timeout")):
            result = await health_check.check()

        assert result.status == HealthStatus.UNHEALTHY
        assert "Timeout" in result.message


class TestSystemResourceHealthCheck:
    """Test system resource health check."""

    @pytest.fixture
    def health_check(self):
        """Create system resource health check."""
        return SystemResourceHealthCheck()

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_healthy_system_resources(self, mock_disk, mock_memory, mock_cpu, health_check):
        """Test healthy system resources."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        mock_disk.return_value.percent = 70.0

        result = await health_check.check()

        assert result.name == "system_resources"
        assert result.status == HealthStatus.HEALTHY
        assert result.details["cpu_percent"] == 50.0
        assert result.details["memory_percent"] == 60.0
        assert result.details["disk_percent"] == 70.0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @pytest.mark.asyncio
    async def test_degraded_system_resources(self, mock_disk, mock_memory, mock_cpu, health_check):
        """Test degraded system resources."""
        mock_cpu.return_value = 85.0  # High CPU
        mock_memory.return_value.percent = 95.0  # High memory
        mock_disk.return_value.percent = 75.0

        result = await health_check.check()

        assert result.status == HealthStatus.DEGRADED
        assert "High resource usage" in result.message


class TestHealthCheckManager:
    """Test health check manager."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return HealthCheckConfig(
            enabled=True,
            interval=1.0,
            timeout=5.0,
            max_retries=1
        )

    @pytest.fixture
    def manager(self, config):
        """Create health check manager."""
        return HealthCheckManager(config)

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.health_checks) > 0
        assert any(check.name == "qdrant" for check in manager.health_checks)
        assert any(check.name == "redis" for check in manager.health_checks)
        assert any(check.name == "system_resources" for check in manager.health_checks)

    @pytest.mark.asyncio
    async def test_run_all_checks(self, manager):
        """Test running all health checks."""
        # Mock all health checks to return healthy
        for check in manager.health_checks:
            check.check = AsyncMock(return_value=HealthCheckResult(
                name=check.name,
                status=HealthStatus.HEALTHY,
                message="Healthy"
            ))

        results = await manager.run_all_checks()

        assert len(results) == len(manager.health_checks)
        assert all(result.status == HealthStatus.HEALTHY for result in results)

    @pytest.mark.asyncio
    async def test_get_overall_health_all_healthy(self, manager):
        """Test overall health when all services are healthy."""
        # Mock all health checks to return healthy
        for check in manager.health_checks:
            check.check = AsyncMock(return_value=HealthCheckResult(
                name=check.name,
                status=HealthStatus.HEALTHY,
                message="Healthy"
            ))

        status, details = await manager.get_overall_health()

        assert status == HealthStatus.HEALTHY
        assert details["overall_status"] == "healthy"
        assert all(result["status"] == "healthy" for result in details["services"].values())

    @pytest.mark.asyncio
    async def test_get_overall_health_with_failures(self, manager):
        """Test overall health with some service failures."""
        # Mock mixed health results
        healthy_result = HealthCheckResult(
            name="healthy_service",
            status=HealthStatus.HEALTHY,
            message="Healthy"
        )
        unhealthy_result = HealthCheckResult(
            name="unhealthy_service",
            status=HealthStatus.UNHEALTHY,
            message="Failed"
        )

        manager.health_checks[0].check = AsyncMock(return_value=healthy_result)
        manager.health_checks[1].check = AsyncMock(return_value=unhealthy_result)

        status, details = await manager.get_overall_health()

        assert status == HealthStatus.UNHEALTHY
        assert details["overall_status"] == "unhealthy"

    def test_add_custom_health_check(self, manager):
        """Test adding custom health check."""
        custom_check = HTTPHealthCheck("custom_api", "https://custom.api.com/health")
        initial_count = len(manager.health_checks)

        manager.add_health_check(custom_check)

        assert len(manager.health_checks) == initial_count + 1
        assert custom_check in manager.health_checks

    def test_disabled_health_check_manager(self):
        """Test disabled health check manager."""
        config = HealthCheckConfig(enabled=False)
        manager = HealthCheckManager(config)

        assert len(manager.health_checks) == 0

    @pytest.mark.asyncio
    async def test_health_check_timeout(self, manager):
        """Test health check timeout handling."""
        slow_check = Mock()
        slow_check.name = "slow_service"
        slow_check.check = AsyncMock(side_effect=TimeoutError())

        manager.add_health_check(slow_check)
        results = await manager.run_all_checks()

        # Should handle timeout gracefully
        slow_result = next(r for r in results if r.name == "slow_service")
        assert slow_result.status == HealthStatus.UNHEALTHY
        assert "timeout" in slow_result.message.lower()


class TestHealthCheckIntegration:
    """Integration tests for health check system."""

    @pytest.mark.asyncio
    async def test_full_health_check_cycle(self):
        """Test complete health check cycle."""
        config = HealthCheckConfig(enabled=True, interval=0.1)
        manager = HealthCheckManager(config)

        # Mock all dependencies as healthy
        for check in manager.health_checks:
            check.check = AsyncMock(return_value=HealthCheckResult(
                name=check.name,
                status=HealthStatus.HEALTHY,
                message="Service is healthy"
            ))

        # Run health checks
        status, details = await manager.get_overall_health()

        assert status == HealthStatus.HEALTHY
        assert isinstance(details, dict)
        assert "services" in details
        assert "overall_status" in details
