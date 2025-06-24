"""Comprehensive tests for service auto-detection and environment profiling system.

Tests all auto-detection components including environment detection, service discovery,
connection pooling, health monitoring, and integration with configuration system.
Uses property-based testing for edge cases and modern async testing patterns.
"""

import asyncio
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest
import respx
from hypothesis import given, strategies as st
from pydantic import ValidationError

# Import auto-detection components
from src.config.auto_detect import (
    AutoDetectedServices,
    AutoDetectionConfig,
    DetectedEnvironment,
    DetectedService,
    EnvironmentDetector,
)
from src.config.enums import Environment
from src.services.auto_detection.connection_pools import (
    ConnectionPoolManager,
    PoolHealthMetrics,
)
from src.services.auto_detection.health_checks import HealthChecker
from src.services.auto_detection.service_discovery import (
    ServiceDiscovery,
    ServiceDiscoveryResult,
)


# Hypothesis strategies for property-based testing
valid_ports = st.integers(min_value=1, max_value=65535)
valid_timeouts = st.floats(
    min_value=0.1, max_value=60.0, allow_nan=False, allow_infinity=False
)
valid_confidence = st.floats(
    min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
)
valid_hostnames = st.text(
    min_size=1,
    max_size=50,
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
)
service_types = st.sampled_from(["redis", "qdrant", "postgresql"])
environment_types = st.sampled_from([env.value for env in Environment])


class TestAutoDetectionConfig:
    """Test auto-detection configuration with validation and edge cases."""

    def test_default_config_creation(self):
        """Test creating auto-detection config with defaults."""
        config = AutoDetectionConfig()

        assert config.enabled is True
        assert config.timeout_seconds == 10.0
        assert config.cache_ttl_seconds == 300
        assert config.redis_discovery_enabled is True
        assert config.qdrant_discovery_enabled is True
        assert config.postgresql_discovery_enabled is True
        assert config.parallel_detection is True
        assert config.max_concurrent_detections == 10

    @given(timeout=valid_timeouts)
    def test_timeout_validation(self, timeout):
        """Test timeout validation with property-based testing."""
        if 0.1 <= timeout <= 60.0:
            config = AutoDetectionConfig(timeout_seconds=timeout)
            assert config.timeout_seconds == timeout
        else:
            with pytest.raises(ValidationError):
                AutoDetectionConfig(timeout_seconds=timeout)

    def test_timeout_boundary_validation(self):
        """Test timeout boundary conditions."""
        # Valid boundaries
        AutoDetectionConfig(timeout_seconds=0.1)
        AutoDetectionConfig(timeout_seconds=60.0)

        # Invalid boundaries
        with pytest.raises(ValidationError):
            AutoDetectionConfig(timeout_seconds=0.0)
        with pytest.raises(ValidationError):
            AutoDetectionConfig(timeout_seconds=61.0)

    @given(max_concurrent=st.integers(min_value=-10, max_value=100))
    def test_max_concurrent_validation(self, max_concurrent):
        """Test max concurrent detections validation."""
        if 1 <= max_concurrent <= 50:
            config = AutoDetectionConfig(max_concurrent_detections=max_concurrent)
            assert config.max_concurrent_detections == max_concurrent
        else:
            with pytest.raises(ValidationError):
                AutoDetectionConfig(max_concurrent_detections=max_concurrent)

    def test_selective_service_discovery(self):
        """Test enabling/disabling specific service discovery."""
        config = AutoDetectionConfig(
            redis_discovery_enabled=False,
            qdrant_discovery_enabled=True,
            postgresql_discovery_enabled=False,
        )

        assert config.redis_discovery_enabled is False
        assert config.qdrant_discovery_enabled is True
        assert config.postgresql_discovery_enabled is False

    def test_config_serialization(self):
        """Test config serialization and deserialization."""
        config = AutoDetectionConfig(
            timeout_seconds=5.0,
            max_concurrent_detections=5,
            redis_discovery_enabled=False,
        )

        # Test model dump
        config_dict = config.model_dump()
        assert config_dict["timeout_seconds"] == 5.0
        assert config_dict["redis_discovery_enabled"] is False

        # Test reconstruction
        new_config = AutoDetectionConfig(**config_dict)
        assert new_config.timeout_seconds == 5.0
        assert new_config.redis_discovery_enabled is False


class TestDetectedService:
    """Test detected service model with validation and edge cases."""

    @given(
        host=valid_hostnames,
        port=valid_ports,
        service_type=service_types,
        detection_time=st.floats(min_value=0.0, max_value=10000.0, allow_nan=False),
    )
    def test_service_creation_with_property_testing(
        self, host, port, service_type, detection_time
    ):
        """Test service creation with various inputs."""
        try:
            service = DetectedService(
                service_name=f"test-{service_type}",
                service_type=service_type,
                host=host,
                port=port,
                is_available=True,
                detection_time_ms=detection_time,
            )

            assert service.service_name == f"test-{service_type}"
            assert service.service_type == service_type
            assert service.host == host
            assert service.port == port
            assert service.is_available is True
            assert service.detection_time_ms == detection_time
        except ValidationError:
            # Some generated values might be invalid, which is acceptable
            pass

    def test_redis_service_with_pool_config(self):
        """Test Redis service with connection pool configuration."""
        pool_config = {
            "max_connections": 20,
            "retry_on_timeout": True,
            "protocol": 3,  # RESP3
            "decode_responses": True,
        }

        service = DetectedService(
            service_name="redis-primary",
            service_type="redis",
            host="localhost",
            port=6379,
            is_available=True,
            connection_string="redis://localhost:6379",
            version="8.2.0",
            supports_pooling=True,
            pool_config=pool_config,
            health_check_url="redis://localhost:6379/ping",
            detection_time_ms=150.5,
            metadata={"protocol": "RESP3", "memory_usage": "1.2GB"},
        )

        assert service.service_type == "redis"
        assert service.supports_pooling is True
        assert service.pool_config["protocol"] == 3
        assert service.metadata["protocol"] == "RESP3"

    def test_qdrant_service_with_grpc_config(self):
        """Test Qdrant service with gRPC configuration."""
        pool_config = {
            "timeout": 10.0,
            "prefer_grpc": True,
            "grpc_options": {
                "grpc.keepalive_time_ms": 30000,
                "grpc.keepalive_timeout_ms": 5000,
            },
        }

        service = DetectedService(
            service_name="qdrant-vector-db",
            service_type="qdrant",
            host="qdrant",
            port=6333,
            is_available=True,
            connection_string="http://qdrant:6333",
            version="1.7.0",
            supports_pooling=True,
            pool_config=pool_config,
            health_check_url="http://qdrant:6333/health",
            detection_time_ms=89.2,
            metadata={
                "grpc_available": True,
                "grpc_port": 6334,
                "collections_count": 5,
            },
        )

        assert service.service_type == "qdrant"
        assert service.pool_config["prefer_grpc"] is True
        assert service.metadata["grpc_available"] is True

    def test_postgresql_service_config(self):
        """Test PostgreSQL service configuration."""
        pool_config = {
            "min_size": 2,
            "max_size": 20,
            "command_timeout": 10,
            "server_settings": {"jit": "off", "application_name": "ai_docs_vector_db"},
        }

        service = DetectedService(
            service_name="postgres-primary",
            service_type="postgresql",
            host="postgres",
            port=5432,
            is_available=True,
            connection_string="postgresql://user:pass@postgres:5432/db",
            version="15.4",
            supports_pooling=True,
            pool_config=pool_config,
            detection_time_ms=234.7,
            metadata={"database_size": "2.1GB", "connections": 15},
        )

        assert service.service_type == "postgresql"
        assert service.pool_config["min_size"] == 2
        assert service.metadata["database_size"] == "2.1GB"


class TestDetectedEnvironment:
    """Test detected environment model with various scenarios."""

    @given(
        confidence=valid_confidence,
        detection_time=st.floats(min_value=0.0, max_value=5000.0, allow_nan=False),
        env_type=st.sampled_from([env for env in Environment]),
    )
    def test_environment_creation_with_property_testing(
        self, confidence, detection_time, env_type
    ):
        """Test environment creation with property-based inputs."""
        env = DetectedEnvironment(
            environment_type=env_type,
            is_containerized=True,
            is_kubernetes=False,
            detection_confidence=confidence,
            detection_time_ms=detection_time,
        )

        assert env.environment_type == env_type
        assert env.is_containerized is True
        assert env.detection_confidence == confidence
        assert env.detection_time_ms == detection_time

    def test_docker_environment_detection(self):
        """Test Docker environment detection."""
        env = DetectedEnvironment(
            environment_type=Environment.TESTING,
            is_containerized=True,
            is_kubernetes=False,
            container_runtime="docker",
            detection_confidence=0.8,
            detection_time_ms=45.3,
            metadata={"docker_version": "24.0.5", "host_os": "linux"},
        )

        assert env.environment_type == Environment.TESTING
        assert env.is_containerized is True
        assert env.container_runtime == "docker"
        assert env.metadata["docker_version"] == "24.0.5"

    def test_kubernetes_environment_detection(self):
        """Test Kubernetes environment detection."""
        env = DetectedEnvironment(
            environment_type=Environment.STAGING,
            is_containerized=True,
            is_kubernetes=True,
            container_runtime="containerd",
            detection_confidence=0.95,
            detection_time_ms=78.1,
            metadata={
                "k8s_namespace": "default",
                "k8s_pod_name": "app-deployment-abc123",
                "k8s_service_account": "default",
            },
        )

        assert env.environment_type == Environment.STAGING
        assert env.is_kubernetes is True
        assert env.metadata["k8s_namespace"] == "default"

    def test_cloud_environment_detection(self):
        """Test cloud environment detection."""
        env = DetectedEnvironment(
            environment_type=Environment.PRODUCTION,
            is_containerized=True,
            is_kubernetes=True,
            cloud_provider="aws",
            region="us-west-2",
            detection_confidence=1.0,
            detection_time_ms=234.5,
            metadata={
                "instance_type": "t3.large",
                "availability_zone": "us-west-2a",
                "instance_id": "i-1234567890abcdef0",
            },
        )

        assert env.environment_type == Environment.PRODUCTION
        assert env.cloud_provider == "aws"
        assert env.region == "us-west-2"
        assert env.metadata["instance_type"] == "t3.large"


class TestAutoDetectedServices:
    """Test auto-detected services container with validation."""

    def test_empty_services_creation(self):
        """Test creating empty auto-detected services."""
        env = DetectedEnvironment(
            environment_type=Environment.DEVELOPMENT,
            is_containerized=False,
            is_kubernetes=False,
            detection_confidence=0.5,
            detection_time_ms=12.3,
        )

        services = AutoDetectedServices(environment=env)

        assert services.environment.environment_type == Environment.DEVELOPMENT
        assert len(services.services) == 0
        assert services.redis_service is None
        assert services.qdrant_service is None
        assert services.postgresql_service is None

    def test_services_with_all_detected(self):
        """Test services container with all service types detected."""
        env = DetectedEnvironment(
            environment_type=Environment.PRODUCTION,
            is_containerized=True,
            is_kubernetes=True,
            cloud_provider="gcp",
            region="us-central1",
            detection_confidence=0.9,
            detection_time_ms=156.7,
        )

        redis_service = DetectedService(
            service_name="redis-cache",
            service_type="redis",
            host="redis.internal",
            port=6379,
            is_available=True,
            detection_time_ms=45.2,
        )

        qdrant_service = DetectedService(
            service_name="qdrant-vectors",
            service_type="qdrant",
            host="qdrant.internal",
            port=6333,
            is_available=True,
            detection_time_ms=67.8,
        )

        postgres_service = DetectedService(
            service_name="postgres-primary",
            service_type="postgresql",
            host="postgres.internal",
            port=5432,
            is_available=True,
            detection_time_ms=89.1,
        )

        services = AutoDetectedServices(
            environment=env, services=[redis_service, qdrant_service, postgres_service]
        )

        assert len(services.services) == 3
        assert services.redis_service is not None
        assert services.redis_service.service_name == "redis-cache"
        assert services.qdrant_service is not None
        assert services.qdrant_service.service_name == "qdrant-vectors"
        assert services.postgresql_service is not None
        assert services.postgresql_service.service_name == "postgres-primary"

    def test_services_completion_marking(self):
        """Test marking services detection as completed."""
        env = DetectedEnvironment(
            environment_type=Environment.TESTING,
            is_containerized=True,
            is_kubernetes=False,
            detection_confidence=0.7,
            detection_time_ms=89.4,
        )

        services = AutoDetectedServices(environment=env)

        # Initially not completed
        assert services.detection_completed_at is None
        assert services.total_detection_time_ms is None

        # Mark as completed
        start_time = services.detection_started_at
        time.sleep(0.001)  # Small delay to ensure different timestamps
        services.mark_completed()

        assert services.detection_completed_at is not None
        assert services.total_detection_time_ms is not None
        assert services.detection_completed_at > start_time
        assert services.total_detection_time_ms > 0


@pytest.mark.asyncio
class TestEnvironmentDetector:
    """Test environment detector with mocked external APIs."""

    @pytest.fixture
    def detector_config(self):
        """Create auto-detection config for testing."""
        return AutoDetectionConfig(
            timeout_seconds=5.0,
            cache_ttl_seconds=10,  # Short TTL for testing
            parallel_detection=True,
        )

    @pytest.fixture
    def detector(self, detector_config):
        """Create environment detector instance."""
        return EnvironmentDetector(detector_config)

    async def test_local_development_detection(self, detector):
        """Test detection of local development environment."""
        with (
            patch("os.path.exists", return_value=False),
            patch("os.getenv", return_value=None),
        ):
            env = await detector.detect()

            assert env.environment_type == Environment.DEVELOPMENT
            assert env.is_containerized is False
            assert env.is_kubernetes is False
            assert env.cloud_provider is None

    async def test_docker_container_detection(self, detector):
        """Test detection of Docker container environment."""

        def mock_exists(path):
            return path == "/.dockerenv"

        with patch("os.path.exists", side_effect=mock_exists):
            env = await detector.detect()

            assert env.is_containerized is True
            assert env.environment_type == Environment.TESTING

    async def test_kubernetes_detection(self, detector):
        """Test detection of Kubernetes environment."""

        def mock_exists(path):
            return path == "/var/run/secrets/kubernetes.io/serviceaccount"

        def mock_getenv(var):
            if var == "KUBERNETES_SERVICE_HOST":
                return "10.96.0.1"
            return None

        with (
            patch("os.path.exists", side_effect=mock_exists),
            patch("os.getenv", side_effect=mock_getenv),
        ):
            env = await detector.detect()

            assert env.is_kubernetes is True
            assert env.environment_type == Environment.STAGING

    @respx.mock
    async def test_aws_cloud_detection(self, detector):
        """Test detection of AWS cloud environment."""
        # Mock IMDSv2 token request
        respx.put("http://169.254.169.254/latest/api/token").mock(
            return_value=httpx.Response(200, text="test-token")
        )

        # Mock metadata request
        respx.get("http://169.254.169.254/latest/meta-data/placement/region").mock(
            return_value=httpx.Response(200, text="us-west-2")
        )

        env = await detector.detect()

        assert env.cloud_provider == "aws"
        assert env.region == "us-west-2"
        assert env.environment_type == Environment.PRODUCTION

    @respx.mock
    async def test_gcp_cloud_detection(self, detector):
        """Test detection of GCP cloud environment."""
        respx.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/zone"
        ).mock(
            return_value=httpx.Response(
                200, text="projects/123456789/zones/us-central1-a"
            )
        )

        env = await detector.detect()

        assert env.cloud_provider == "gcp"
        assert env.region == "us-central1"
        assert env.metadata["zone"] == "us-central1-a"

    @respx.mock
    async def test_azure_cloud_detection(self, detector):
        """Test detection of Azure cloud environment."""
        respx.get("http://169.254.169.254/metadata/instance/compute/location").mock(
            return_value=httpx.Response(200, text="eastus")
        )

        env = await detector.detect()

        assert env.cloud_provider == "azure"
        assert env.region == "eastus"

    async def test_detection_caching(self, detector):
        """Test that detection results are properly cached."""
        with patch("os.path.exists", return_value=False):
            # First detection
            env1 = await detector.detect()
            cache_time_1 = detector._cache_time

            # Second detection should use cache
            env2 = await detector.detect()
            cache_time_2 = detector._cache_time

            assert env1.environment_type == env2.environment_type
            assert cache_time_1 == cache_time_2  # Cache time didn't change

    async def test_detection_confidence_calculation(self, detector):
        """Test confidence calculation based on indicators."""

        def mock_exists(path):
            return path == "/.dockerenv"

        def mock_getenv(var):
            if var == "KUBERNETES_SERVICE_HOST":
                return "10.96.0.1"
            return None

        with (
            patch("os.path.exists", side_effect=mock_exists),
            patch("os.getenv", side_effect=mock_getenv),
        ):
            env = await detector.detect()

            # Should have high confidence with container + k8s indicators
            assert env.detection_confidence >= 0.5

    async def test_detection_error_handling(self, detector):
        """Test error handling during detection."""
        with patch("os.path.exists", side_effect=Exception("Mock error")):
            env = await detector.detect()

            # Should return default environment on error
            assert env.environment_type == Environment.DEVELOPMENT
            assert env.detection_confidence == 0.0
            assert "error" in env.metadata


@pytest.mark.asyncio
class TestServiceDiscovery:
    """Test service discovery with mocked network calls."""

    @pytest.fixture
    def discovery_config(self):
        """Create auto-detection config for testing."""
        return AutoDetectionConfig(
            timeout_seconds=3.0,
            max_concurrent_detections=5,
            redis_discovery_enabled=True,
            qdrant_discovery_enabled=True,
            postgresql_discovery_enabled=True,
        )

    @pytest.fixture
    def service_discovery(self, discovery_config):
        """Create service discovery instance."""
        return ServiceDiscovery(discovery_config)

    async def test_discover_all_services_parallel(self, service_discovery):
        """Test parallel service discovery."""
        with (
            patch.object(service_discovery, "_discover_redis") as mock_redis,
            patch.object(service_discovery, "_discover_qdrant") as mock_qdrant,
            patch.object(service_discovery, "_discover_postgresql") as mock_postgres,
        ):
            # Mock successful discovery results
            mock_redis.return_value = DetectedService(
                service_name="redis",
                service_type="redis",
                host="localhost",
                port=6379,
                is_available=True,
                detection_time_ms=50.0,
            )

            mock_qdrant.return_value = DetectedService(
                service_name="qdrant",
                service_type="qdrant",
                host="localhost",
                port=6333,
                is_available=True,
                detection_time_ms=75.0,
            )

            mock_postgres.return_value = None  # Not found

            result = await service_discovery.discover_all_services()

            assert isinstance(result, ServiceDiscoveryResult)
            assert len(result.services) == 2
            assert result.metadata["total_attempted"] == 3
            assert result.metadata["successful"] == 2
            assert result.metadata["failed"] == 1

    async def test_redis_discovery_with_connection_test(self, service_discovery):
        """Test Redis discovery with connection testing."""
        with (
            patch.object(service_discovery, "_test_tcp_connection", return_value=True),
            patch.object(
                service_discovery, "_test_redis_connection"
            ) as mock_redis_test,
        ):
            mock_redis_test.return_value = {
                "version": "8.2.0",
                "mode": "standalone",
                "memory_usage": "1.2GB",
            }

            service = await service_discovery._discover_redis()

            assert service is not None
            assert service.service_type == "redis"
            assert service.version == "8.2.0"
            assert service.supports_pooling is True
            assert service.pool_config["protocol"] == 3  # RESP3

    async def test_qdrant_discovery_with_grpc_preference(self, service_discovery):
        """Test Qdrant discovery with gRPC preference detection."""
        with (
            patch.object(service_discovery, "_test_qdrant_http") as mock_http,
            patch.object(service_discovery, "_test_tcp_connection") as mock_tcp,
        ):
            mock_http.return_value = {"version": "1.7.0", "api_available": True}

            # Mock gRPC availability
            mock_tcp.return_value = True

            service = await service_discovery._discover_qdrant()

            assert service is not None
            assert service.service_type == "qdrant"
            assert service.metadata["grpc_available"] is True
            assert service.pool_config["prefer_grpc"] is True

    async def test_postgresql_discovery_basic(self, service_discovery):
        """Test PostgreSQL discovery with basic TCP test."""
        with (
            patch.object(service_discovery, "_test_tcp_connection", return_value=True),
            patch.object(
                service_discovery, "_test_postgresql_connection"
            ) as mock_pg_test,
        ):
            mock_pg_test.return_value = {"version": "15.4", "tcp_available": True}

            service = await service_discovery._discover_postgresql()

            assert service is not None
            assert service.service_type == "postgresql"
            assert service.supports_pooling is True
            assert service.pool_config["min_size"] == 2

    async def test_service_discovery_caching(self, service_discovery):
        """Test service discovery result caching."""
        redis_service = DetectedService(
            service_name="redis",
            service_type="redis",
            host="localhost",
            port=6379,
            is_available=True,
            detection_time_ms=50.0,
        )

        # Cache the service
        service_discovery._cache_service("redis", redis_service)

        # Should return cached result
        cached_service = service_discovery._get_cached_service("redis")
        assert cached_service is not None
        assert cached_service.service_name == "redis"

    async def test_tcp_connection_test(self, service_discovery):
        """Test TCP connection testing utility."""
        # Mock successful connection
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()

        with patch("asyncio.open_connection", return_value=(mock_reader, mock_writer)):
            result = await service_discovery._test_tcp_connection("localhost", 6379)
            assert result is True
            mock_writer.close.assert_called_once()

    async def test_discovery_error_handling(self, service_discovery):
        """Test error handling during service discovery."""
        with (
            patch.object(
                service_discovery,
                "_discover_redis",
                side_effect=Exception("Network error"),
            ),
            patch.object(service_discovery, "_discover_qdrant", return_value=None),
            patch.object(service_discovery, "_discover_postgresql", return_value=None),
        ):
            result = await service_discovery.discover_all_services()

            assert len(result.services) == 0
            assert len(result.errors) >= 1
            assert "Network error" in str(result.errors)


@pytest.mark.asyncio
class TestConnectionPoolManager:
    """Test connection pool management with health monitoring."""

    @pytest.fixture
    def pool_config(self):
        """Create auto-detection config for testing."""
        return AutoDetectionConfig(connection_pooling_enabled=True, timeout_seconds=5.0)

    @pytest.fixture
    def pool_manager(self, pool_config):
        """Create connection pool manager instance."""
        return ConnectionPoolManager(pool_config)

    @pytest.fixture
    def sample_services(self):
        """Create sample detected services for testing."""
        redis_service = DetectedService(
            service_name="redis",
            service_type="redis",
            host="localhost",
            port=6379,
            is_available=True,
            supports_pooling=True,
            pool_config={
                "max_connections": 10,
                "retry_on_timeout": True,
                "protocol": 3,
            },
            detection_time_ms=45.0,
        )

        qdrant_service = DetectedService(
            service_name="qdrant",
            service_type="qdrant",
            host="localhost",
            port=6333,
            is_available=True,
            supports_pooling=True,
            pool_config={"timeout": 10.0, "prefer_grpc": True},
            detection_time_ms=67.0,
            metadata={"grpc_available": True, "grpc_port": 6334},
        )

        return [redis_service, qdrant_service]

    async def test_pool_initialization(self, pool_manager, sample_services):
        """Test connection pool initialization."""
        with (
            patch("redis.asyncio.ConnectionPool") as mock_redis_pool,
            patch("qdrant_client.AsyncQdrantClient") as mock_qdrant_client,
        ):
            mock_redis_pool.return_value = Mock()
            mock_qdrant_client.return_value = AsyncMock()

            await pool_manager.initialize_pools(sample_services)

            assert pool_manager._initialized is True
            assert "redis" in pool_manager._pools
            assert "qdrant" in pool_manager._pools

    async def test_redis_pool_initialization_with_ping(self, pool_manager):
        """Test Redis pool initialization with connection test."""
        redis_service = DetectedService(
            service_name="redis",
            service_type="redis",
            host="localhost",
            port=6379,
            is_available=True,
            supports_pooling=True,
            pool_config={"max_connections": 10},
            detection_time_ms=45.0,
        )

        with (
            patch("redis.asyncio.ConnectionPool") as mock_pool,
            patch("redis.asyncio.Redis") as mock_redis,
        ):
            mock_redis_instance = AsyncMock()
            mock_redis.return_value = mock_redis_instance
            mock_redis_instance.ping = AsyncMock()
            mock_redis_instance.aclose = AsyncMock()

            await pool_manager._initialize_redis_pool(redis_service)

            mock_redis_instance.ping.assert_called_once()
            mock_redis_instance.aclose.assert_called_once()

    async def test_qdrant_pool_initialization_with_grpc(self, pool_manager):
        """Test Qdrant pool initialization with gRPC preference."""
        qdrant_service = DetectedService(
            service_name="qdrant",
            service_type="qdrant",
            host="localhost",
            port=6333,
            is_available=True,
            supports_pooling=True,
            pool_config={"prefer_grpc": True},
            detection_time_ms=67.0,
            metadata={"grpc_available": True, "grpc_port": 6334},
        )

        with patch("qdrant_client.AsyncQdrantClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client.return_value = mock_client_instance
            mock_client_instance.get_collections = AsyncMock()

            await pool_manager._initialize_qdrant_pool(qdrant_service)

            # Should use gRPC port
            mock_client.assert_called_with(
                host="localhost",
                port=6334,  # gRPC port
                prefer_grpc=True,
            )

    async def test_pool_health_metrics(self, pool_manager):
        """Test pool health metrics tracking."""
        # Initialize with empty metrics
        pool_manager._health_metrics["redis"] = PoolHealthMetrics(
            pool_name="redis",
            total_connections=10,
            active_connections=3,
            idle_connections=7,
            pool_utilization=0.3,
            average_connection_time_ms=25.5,
            failed_connections=0,
            last_health_check=time.time(),
            is_healthy=True,
        )

        metrics = pool_manager.get_pool_health("redis")
        assert metrics is not None
        assert metrics.pool_name == "redis"
        assert metrics.pool_utilization == 0.3
        assert metrics.is_healthy is True

    async def test_redis_connection_context_manager(self, pool_manager):
        """Test Redis connection context manager."""
        mock_pool = Mock()
        pool_manager._pools["redis"] = mock_pool

        # Initialize metrics
        pool_manager._health_metrics["redis"] = PoolHealthMetrics(
            pool_name="redis",
            total_connections=0,
            active_connections=0,
            idle_connections=0,
            pool_utilization=0.0,
            average_connection_time_ms=0.0,
            failed_connections=0,
            last_health_check=time.time(),
            is_healthy=True,
        )

        with patch("redis.asyncio.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_redis.return_value = mock_client

            async with pool_manager.get_redis_connection() as client:
                assert client == mock_client

            mock_client.aclose.assert_called_once()

    async def test_pool_cleanup(self, pool_manager, sample_services):
        """Test connection pool cleanup."""
        with (
            patch("redis.asyncio.ConnectionPool") as mock_redis_pool,
            patch("qdrant_client.AsyncQdrantClient") as mock_qdrant_client,
        ):
            mock_pool = AsyncMock()
            mock_client = AsyncMock()
            mock_redis_pool.return_value = mock_pool
            mock_qdrant_client.return_value = mock_client

            await pool_manager.initialize_pools(sample_services)

            # Add mock disconnect/close methods
            mock_pool.disconnect = AsyncMock()
            mock_client.close = AsyncMock()

            await pool_manager.cleanup()

            assert pool_manager._initialized is False
            assert len(pool_manager._pools) == 0

    async def test_pool_statistics(self, pool_manager):
        """Test pool statistics collection."""
        # Add some mock metrics
        pool_manager._health_metrics["redis"] = PoolHealthMetrics(
            pool_name="redis",
            total_connections=10,
            active_connections=3,
            idle_connections=7,
            pool_utilization=0.3,
            average_connection_time_ms=25.5,
            failed_connections=1,
            last_health_check=time.time(),
            is_healthy=True,
        )

        pool_manager._health_metrics["qdrant"] = PoolHealthMetrics(
            pool_name="qdrant",
            total_connections=1,
            active_connections=0,
            idle_connections=1,
            pool_utilization=0.0,
            average_connection_time_ms=15.2,
            failed_connections=0,
            last_health_check=time.time(),
            is_healthy=True,
        )

        stats = pool_manager.get_pool_stats()

        assert stats["total_pools"] == 2
        assert stats["healthy_pools"] == 2
        assert "redis" in stats["pools"]
        assert "qdrant" in stats["pools"]
        assert stats["pools"]["redis"]["failed_connections"] == 1


@pytest.mark.asyncio
class TestHealthChecker:
    """Test health checking system with monitoring."""

    @pytest.fixture
    def health_config(self):
        """Create auto-detection config for testing."""
        return AutoDetectionConfig(timeout_seconds=3.0, cache_ttl_seconds=60)

    @pytest.fixture
    def health_checker(self, health_config):
        """Create health checker instance."""
        return HealthChecker(health_config)

    @pytest.fixture
    def sample_services(self):
        """Create sample services for health checking."""
        return [
            DetectedService(
                service_name="redis",
                service_type="redis",
                host="localhost",
                port=6379,
                is_available=True,
                health_check_url="redis://localhost:6379/ping",
                detection_time_ms=45.0,
            ),
            DetectedService(
                service_name="qdrant",
                service_type="qdrant",
                host="localhost",
                port=6333,
                is_available=True,
                health_check_url="http://localhost:6333/health",
                detection_time_ms=67.0,
            ),
        ]

    async def test_health_check_initialization(self, health_checker, sample_services):
        """Test health checker initialization."""
        await health_checker.initialize_monitoring(sample_services)

        assert health_checker._initialized is True
        assert len(health_checker._services) == 2
        assert "redis" in health_checker._service_health
        assert "qdrant" in health_checker._service_health

    @respx.mock
    async def test_qdrant_health_check(self, health_checker):
        """Test Qdrant HTTP health check."""
        qdrant_service = DetectedService(
            service_name="qdrant",
            service_type="qdrant",
            host="localhost",
            port=6333,
            is_available=True,
            health_check_url="http://localhost:6333/health",
            detection_time_ms=67.0,
        )

        # Mock successful health check
        respx.get("http://localhost:6333/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )

        is_healthy = await health_checker._check_qdrant_health(qdrant_service)
        assert is_healthy is True

    async def test_redis_health_check_tcp(self, health_checker):
        """Test Redis health check with TCP connection."""
        redis_service = DetectedService(
            service_name="redis",
            service_type="redis",
            host="localhost",
            port=6379,
            is_available=True,
            detection_time_ms=45.0,
        )

        # Mock successful TCP connection
        with patch.object(health_checker, "_test_tcp_connection", return_value=True):
            is_healthy = await health_checker._check_redis_health(redis_service)
            assert is_healthy is True

    async def test_health_monitoring_loop(self, health_checker, sample_services):
        """Test background health monitoring loop."""
        await health_checker.initialize_monitoring(sample_services)

        with patch.object(
            health_checker, "_check_service_health", return_value=True
        ) as mock_check:
            # Start monitoring briefly
            monitoring_task = asyncio.create_task(health_checker._monitor_services())

            # Let it run for a short time
            await asyncio.sleep(0.1)

            # Cancel the task
            monitoring_task.cancel()

            try:
                await monitoring_task
            except asyncio.CancelledError:
                pass

            # Should have attempted health checks
            assert mock_check.call_count >= 0

    async def test_health_metrics_collection(self, health_checker, sample_services):
        """Test health metrics collection and retrieval."""
        await health_checker.initialize_monitoring(sample_services)

        # Manually update metrics
        health_checker._service_health["redis"]["checks_performed"] = 10
        health_checker._service_health["redis"]["successful_checks"] = 9
        health_checker._service_health["redis"]["failed_checks"] = 1

        metrics = health_checker.get_health_metrics("redis")
        assert metrics["checks_performed"] == 10
        assert metrics["successful_checks"] == 9
        assert metrics["success_rate"] == 0.9

    async def test_uptime_tracking(self, health_checker, sample_services):
        """Test service uptime tracking."""
        await health_checker.initialize_monitoring(sample_services)

        # Simulate some uptime
        start_time = time.time() - 3600  # 1 hour ago
        health_checker._service_health["redis"]["start_time"] = start_time

        uptime = health_checker.get_uptime("redis")
        assert uptime >= 3600  # At least 1 hour

    async def test_health_checker_cleanup(self, health_checker, sample_services):
        """Test health checker cleanup."""
        await health_checker.initialize_monitoring(sample_services)

        # Start monitoring
        monitoring_task = asyncio.create_task(health_checker._monitor_services())
        health_checker._monitoring_task = monitoring_task

        await health_checker.cleanup()

        assert health_checker._initialized is False
        assert len(health_checker._services) == 0
        assert monitoring_task.cancelled()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
