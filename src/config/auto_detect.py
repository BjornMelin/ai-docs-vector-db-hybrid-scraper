"""Service auto-detection and environment profiling for 2025 production deployments.

This module provides intelligent service discovery and connection optimization:
- Environment detection (Docker, Kubernetes, AWS, GCP, Azure, local)
- Service discovery with connection pooling (Redis 8.2, Qdrant, PostgreSQL)
- GitOps-ready declarative configuration patterns
- Circuit breaker resilience patterns

Integrates with existing Pydantic V2 configuration system and deployment tiers.
"""

import asyncio  # noqa: PLC0415
import logging  # noqa: PLC0415
import os  # noqa: PLC0415
import time  # noqa: PLC0415
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field, computed_field

from src.config.enums import Environment


# Delayed import to avoid circular dependency
# from src.services.errors import circuit_breaker


logger = logging.getLogger(__name__)


class DetectedEnvironment(BaseModel):
    """Container for detected environment information."""

    environment_type: Environment = Field(description="Detected environment type")
    is_containerized: bool = Field(description="Running in container")
    is_kubernetes: bool = Field(description="Running in Kubernetes")
    cloud_provider: str | None = Field(None, description="Cloud provider if detected")
    region: str | None = Field(None, description="Cloud region if detected")
    container_runtime: str | None = Field(None, description="Container runtime")
    detection_confidence: float = Field(description="Confidence score 0.0-1.0")
    detection_time_ms: float = Field(description="Time taken to detect")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DetectedService(BaseModel):
    """Container for detected service information."""

    service_name: str = Field(description="Service name")
    service_type: str = Field(description="Service type (redis, qdrant, postgresql)")
    host: str = Field(description="Service host")
    port: int = Field(description="Service port")
    is_available: bool = Field(description="Service availability")
    connection_string: str | None = Field(None, description="Connection string")
    version: str | None = Field(None, description="Service version")
    supports_pooling: bool = Field(
        default=False, description="Supports connection pooling"
    )
    pool_config: dict[str, Any] = Field(
        default_factory=dict, description="Pool configuration"
    )
    health_check_url: str | None = Field(None, description="Health check endpoint")
    detection_time_ms: float = Field(description="Time taken to detect")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AutoDetectionConfig(BaseModel):
    """Configuration for auto-detection behavior."""

    enabled: bool = Field(default=True, description="Enable auto-detection")
    timeout_seconds: float = Field(
        default=10.0, gt=0, le=60, description="Detection timeout"
    )
    cache_ttl_seconds: int = Field(
        default=300, description="Cache TTL for detection results"
    )
    environment_detection_enabled: bool = Field(
        default=True, description="Enable environment detection"
    )
    service_discovery_enabled: bool = Field(
        default=True, description="Enable service discovery"
    )
    connection_pooling_enabled: bool = Field(
        default=True, description="Enable connection pooling"
    )

    # Service-specific settings
    redis_discovery_enabled: bool = Field(
        default=True, description="Enable Redis discovery"
    )
    qdrant_discovery_enabled: bool = Field(
        default=True, description="Enable Qdrant discovery"
    )
    postgresql_discovery_enabled: bool = Field(
        default=True, description="Enable PostgreSQL discovery"
    )

    # Environment-specific settings
    docker_detection_enabled: bool = Field(
        default=True, description="Enable Docker detection"
    )
    kubernetes_detection_enabled: bool = Field(
        default=True, description="Enable Kubernetes detection"
    )
    cloud_detection_enabled: bool = Field(
        default=True, description="Enable cloud detection"
    )

    # Performance settings
    parallel_detection: bool = Field(
        default=True, description="Run detections in parallel"
    )
    max_concurrent_detections: int = Field(
        default=10, gt=0, le=50, description="Max concurrent detections"
    )
    circuit_breaker_enabled: bool = Field(
        default=True, description="Enable circuit breaker"
    )


class AutoDetectedServices(BaseModel):
    """Container for all auto-detected services and environment."""

    environment: DetectedEnvironment = Field(description="Detected environment")
    services: list[DetectedService] = Field(
        default_factory=list, description="Detected services"
    )
    detection_started_at: float = Field(
        default_factory=time.time, description="Detection start time"
    )
    detection_completed_at: float | None = Field(
        None, description="Detection completion time"
    )
    total_detection_time_ms: float | None = Field(
        None, description="Total detection time"
    )
    errors: list[str] = Field(default_factory=list, description="Detection errors")

    @computed_field
    @property
    def redis_service(self) -> DetectedService | None:
        """Get Redis service if detected."""
        return next((s for s in self.services if s.service_type == "redis"), None)

    @computed_field
    @property
    def qdrant_service(self) -> DetectedService | None:
        """Get Qdrant service if detected."""
        return next((s for s in self.services if s.service_type == "qdrant"), None)

    @computed_field
    @property
    def postgresql_service(self) -> DetectedService | None:
        """Get PostgreSQL service if detected."""
        return next((s for s in self.services if s.service_type == "postgresql"), None)

    def mark_completed(self) -> None:
        """Mark detection as completed and calculate total time."""
        self.detection_completed_at = time.time()
        self.total_detection_time_ms = (
            self.detection_completed_at - self.detection_started_at
        ) * 1000


class EnvironmentDetector:
    """Detects current environment (Docker, Kubernetes, cloud, local)."""

    def __init__(self, config: AutoDetectionConfig):
        self.config = config
        self.logger = logger.getChild("environment")
        self._cache: DetectedEnvironment | None = None
        self._cache_time: float | None = None

    async def detect(self) -> DetectedEnvironment:
        """Detect current environment with caching."""
        # Check cache first
        if self._is_cache_valid():
            self.logger.debug("Using cached environment detection")
            return self._cache

        start_time = time.time()

        try:
            # Run detections in parallel if enabled
            if self.config.parallel_detection:
                results = await asyncio.gather(
                    self._detect_container(),
                    self._detect_kubernetes(),
                    self._detect_cloud_provider(),
                    return_exceptions=True,
                )

                is_containerized = (
                    results[0] if not isinstance(results[0], Exception) else False
                )
                is_kubernetes = (
                    results[1] if not isinstance(results[1], Exception) else False
                )
                cloud_info = results[2] if not isinstance(results[2], Exception) else {}
            else:
                # Sequential detection
                is_containerized = await self._detect_container()
                is_kubernetes = await self._detect_kubernetes()
                cloud_info = await self._detect_cloud_provider()

            # Determine environment type
            environment_type = self._determine_environment_type(
                is_containerized, is_kubernetes, cloud_info
            )

            detection_time_ms = (time.time() - start_time) * 1000

            detected_env = DetectedEnvironment(
                environment_type=environment_type,
                is_containerized=is_containerized,
                is_kubernetes=is_kubernetes,
                cloud_provider=cloud_info.get("provider"),
                region=cloud_info.get("region"),
                container_runtime=cloud_info.get("runtime"),
                detection_confidence=self._calculate_confidence(
                    is_containerized, is_kubernetes, cloud_info
                ),
                detection_time_ms=detection_time_ms,
                metadata=cloud_info,
            )

            # Cache the result
            self._cache = detected_env
            self._cache_time = time.time()

            self.logger.info(
                f"Environment detected: {environment_type.value} "
                f"(containerized: {is_containerized}, k8s: {is_kubernetes}, "
                f"cloud: {cloud_info.get('provider', 'none')}) "
                f"in {detection_time_ms:.1f}ms"
            )

        except Exception as e:
            self.logger.exception("Environment detection failed")
            # Return default environment on failure
            return DetectedEnvironment(
                environment_type=Environment.DEVELOPMENT,
                is_containerized=False,
                is_kubernetes=False,
                detection_confidence=0.0,
                detection_time_ms=(time.time() - start_time) * 1000,
                metadata={"error": str(e)},
            )
        else:
            return detected_env

    def _is_cache_valid(self) -> bool:
        """Check if cached result is still valid."""
        if not self._cache or not self._cache_time:
            return False

        cache_age = time.time() - self._cache_time
        return cache_age < self.config.cache_ttl_seconds

    async def _detect_container(self) -> bool:
        """Detect if running in a container."""
        if not self.config.docker_detection_enabled:
            return False

        try:
            # Check multiple container indicators
            indicators = [
                # Docker
                Path("/.dockerenv").exists(),
                # Generic container
                Path("/proc/1/cgroup").exists() and self._check_cgroup_container(),
                # Kubernetes
                Path("/var/run/secrets/kubernetes.io").exists(),
            ]

            return any(indicators)

        except Exception as e:
            self.logger.debug(f"Container detection failed: {e}")
            return False

    def _check_cgroup_container(self) -> bool:
        """Check cgroup for container indicators."""
        try:
            with Path("/proc/1/cgroup").open() as f:
                content = f.read()
                container_patterns = ["docker", "containerd", "lxc", "kubepods"]
                return any(pattern in content for pattern in container_patterns)
        except Exception:
            return False

    async def _detect_kubernetes(self) -> bool:
        """Detect if running in Kubernetes."""
        if not self.config.kubernetes_detection_enabled:
            return False

        try:
            # Check Kubernetes service account
            k8s_indicators = [
                Path("/var/run/secrets/kubernetes.io/serviceaccount").exists(),
                os.getenv("KUBERNETES_SERVICE_HOST") is not None,
                os.getenv("KUBERNETES_SERVICE_PORT") is not None,
            ]

            return any(k8s_indicators)

        except Exception as e:
            self.logger.debug(f"Kubernetes detection failed: {e}")
            return False

    async def _detect_cloud_provider(self) -> dict[str, Any]:
        """Detect cloud provider via metadata APIs."""
        if not self.config.cloud_detection_enabled:
            return {}

        cloud_info = {}

        # Try cloud providers in parallel with short timeouts
        async with httpx.AsyncClient(timeout=5.0) as client:
            # AWS metadata (IMDSv2)
            aws_info = await self._detect_aws(client)
            if aws_info:
                cloud_info.update(aws_info)
                return cloud_info

            # GCP metadata
            gcp_info = await self._detect_gcp(client)
            if gcp_info:
                cloud_info.update(gcp_info)
                return cloud_info

            # Azure metadata
            azure_info = await self._detect_azure(client)
            if azure_info:
                cloud_info.update(azure_info)
                return cloud_info

        return cloud_info

    async def _detect_aws(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Detect AWS via IMDSv2."""
        try:
            # Get IMDSv2 token first
            token_response = await client.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            )

            if token_response.status_code != 200:
                return {}

            token = token_response.text
            headers = {"X-aws-ec2-metadata-token": token}

            # Get instance metadata
            region_response = await client.get(
                "http://169.254.169.254/latest/meta-data/placement/region",
                headers=headers,
            )

            if region_response.status_code == 200:
                return {
                    "provider": "aws",
                    "region": region_response.text,
                    "runtime": "ec2",
                }

        except Exception as e:
            self.logger.debug(f"AWS detection failed: {e}")

        return {}

    async def _detect_gcp(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Detect GCP via metadata API."""
        try:
            response = await client.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/zone",
                headers={"Metadata-Flavor": "Google"},
            )

            if response.status_code == 200:
                zone = response.text.split("/")[-1]
                region = "-".join(zone.split("-")[:-1])

                return {
                    "provider": "gcp",
                    "region": region,
                    "zone": zone,
                    "runtime": "gce",
                }

        except Exception as e:
            self.logger.debug(f"GCP detection failed: {e}")

        return {}

    async def _detect_azure(self, client: httpx.AsyncClient) -> dict[str, Any]:
        """Detect Azure via IMDS."""
        try:
            response = await client.get(
                "http://169.254.169.254/metadata/instance/compute/location",
                headers={"Metadata": "true"},
                params={"api-version": "2021-02-01"},
            )

            if response.status_code == 200:
                return {"provider": "azure", "region": response.text, "runtime": "vm"}

        except Exception as e:
            self.logger.debug(f"Azure detection failed: {e}")

        return {}

    def _determine_environment_type(
        self, is_containerized: bool, is_kubernetes: bool, cloud_info: dict[str, Any]
    ) -> Environment:
        """Determine environment type based on detection results."""
        if cloud_info.get("provider"):
            return Environment.PRODUCTION
        elif is_kubernetes:
            return Environment.STAGING
        elif is_containerized:
            return Environment.TESTING
        else:
            return Environment.DEVELOPMENT

    def _calculate_confidence(
        self, is_containerized: bool, is_kubernetes: bool, cloud_info: dict[str, Any]
    ) -> float:
        """Calculate confidence score based on detection indicators."""
        confidence = 0.0

        if cloud_info.get("provider"):
            confidence += 0.4
        if is_kubernetes:
            confidence += 0.3
        if is_containerized:
            confidence += 0.2

        # Add bonus for multiple indicators
        indicators_count = sum(
            [bool(cloud_info.get("provider")), is_kubernetes, is_containerized]
        )

        if indicators_count >= 2:
            confidence += 0.1

        return min(confidence, 1.0)


# Service discovery will be implemented in subsequent files
# This is the foundation for the auto-detection system
