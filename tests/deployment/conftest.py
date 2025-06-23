import typing
"""Deployment testing configuration and fixtures.

This module provides shared configuration and fixtures for deployment testing,
including environment setup, pipeline validation, and infrastructure testing.
"""

import asyncio
import json
import os
import tempfile
from collections.abc import AsyncGenerator
from collections.abc import Generator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio


@dataclass
class DeploymentEnvironment:
    """Configuration for a deployment environment."""

    name: str
    tier: str  # development, staging, production
    infrastructure: str  # local, cloud, hybrid
    database_type: str  # sqlite, postgresql
    cache_type: str  # local, redis, dragonfly
    vector_db_type: str  # memory, qdrant
    monitoring_level: str  # basic, full, enterprise
    load_balancer: bool
    ssl_enabled: bool
    backup_enabled: bool

    @property
    def is_production(self) -> bool:
        """Check if this is a production environment."""
        return self.tier == "production"

    @property
    def requires_ssl(self) -> bool:
        """Check if SSL is required for this environment."""
        return self.tier in ("staging", "production")


@dataclass
class DeploymentTestConfig:
    """Configuration for deployment testing."""

    # Environment settings
    target_environment: str = "development"
    validate_infrastructure: bool = True
    test_rollback: bool = True
    test_scaling: bool = False
    test_disaster_recovery: bool = False

    # Pipeline settings
    simulate_build: bool = True
    run_integration_tests: bool = True
    validate_security: bool = True
    test_performance: bool = False

    # Blue-green deployment settings
    test_blue_green: bool = False
    test_traffic_switching: bool = False
    test_zero_downtime: bool = False

    # Timeouts
    deployment_timeout_seconds: int = 300
    health_check_timeout_seconds: int = 60
    rollback_timeout_seconds: int = 120

    # Retry settings
    max_retries: int = 3
    retry_delay_seconds: int = 5


class DeploymentTestFixtures:
    """Shared fixtures for deployment testing."""

    @staticmethod
    def get_environment_configs() -> dict[str, DeploymentEnvironment]:
        """Get predefined environment configurations."""
        return {
            "development": DeploymentEnvironment(
                name="development",
                tier="development",
                infrastructure="local",
                database_type="sqlite",
                cache_type="local",
                vector_db_type="memory",
                monitoring_level="basic",
                load_balancer=False,
                ssl_enabled=False,
                backup_enabled=False,
            ),
            "staging": DeploymentEnvironment(
                name="staging",
                tier="staging",
                infrastructure="cloud",
                database_type="postgresql",
                cache_type="redis",
                vector_db_type="qdrant",
                monitoring_level="full",
                load_balancer=True,
                ssl_enabled=True,
                backup_enabled=True,
            ),
            "production": DeploymentEnvironment(
                name="production",
                tier="production",
                infrastructure="cloud",
                database_type="postgresql",
                cache_type="dragonfly",
                vector_db_type="qdrant",
                monitoring_level="enterprise",
                load_balancer=True,
                ssl_enabled=True,
                backup_enabled=True,
            ),
        }


@pytest.fixture(scope="session")
def deployment_config() -> DeploymentTestConfig:
    """Deployment test configuration fixture."""
    config = DeploymentTestConfig()

    # Override from environment variables
    if target_env := os.getenv("DEPLOYMENT_TARGET_ENV"):
        config.target_environment = target_env

    if os.getenv("DEPLOYMENT_VALIDATE_INFRA", "").lower() == "true":
        config.validate_infrastructure = True

    if os.getenv("DEPLOYMENT_TEST_ROLLBACK", "").lower() == "true":
        config.test_rollback = True

    if os.getenv("DEPLOYMENT_TEST_BLUE_GREEN", "").lower() == "true":
        config.test_blue_green = True

    return config


@pytest.fixture(scope="session")
def environment_configs() -> dict[str, DeploymentEnvironment]:
    """Environment configurations fixture."""
    return DeploymentTestFixtures.get_environment_configs()


@pytest.fixture
def deployment_environment(
    deployment_config: DeploymentTestConfig,
    environment_configs: dict[str, DeploymentEnvironment],
) -> DeploymentEnvironment:
    """Current deployment environment fixture."""
    return environment_configs[deployment_config.target_environment]


@pytest.fixture
def temp_deployment_dir() -> Generator[Path, None, None]:
    """Temporary directory for deployment testing."""
    with tempfile.TemporaryDirectory(prefix="deployment-test-") as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_docker_registry() -> Generator[str, None, None]:
    """Mock Docker registry for testing."""
    # In real scenarios, this would point to a test registry
    registry_url = "localhost:5000"
    yield registry_url


@pytest.fixture
def mock_infrastructure_config(temp_deployment_dir: Path) -> dict[str, Any]:
    """Mock infrastructure configuration for testing."""
    config = {
        "terraform": {
            "backend": {
                "local": {"path": str(temp_deployment_dir / "terraform.tfstate")}
            },
            "provider": {"docker": {"host": "unix:///var/run/docker.sock"}},
        },
        "kubernetes": {
            "namespace": "ai-docs-test",
            "ingress": {"enabled": True, "host": "test.ai-docs.local"},
        },
        "monitoring": {
            "prometheus": {"enabled": True, "port": 9090},
            "grafana": {"enabled": True, "port": 3000},
        },
    }

    # Write config to file
    config_file = temp_deployment_dir / "infrastructure.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    return config


@pytest_asyncio.fixture
async def deployment_health_checker() -> AsyncGenerator[
    "DeploymentHealthChecker", None
]:
    """Health checker for deployment validation."""
    checker = DeploymentHealthChecker()
    await checker.initialize()
    try:
        yield checker
    finally:
        await checker.cleanup()


@pytest.fixture
def deployment_rollback_manager() -> "DeploymentRollbackManager":
    """Rollback manager for deployment testing."""
    return DeploymentRollbackManager()


@pytest.fixture
def blue_green_deployment_manager() -> "BlueGreenDeploymentManager":
    """Blue-green deployment manager for testing."""
    return BlueGreenDeploymentManager()


class DeploymentHealthChecker:
    """Health checker for deployment validation."""

    def __init__(self):
        self.health_endpoints = []
        self.initialized = False

    async def initialize(self) -> None:
        """Initialize health checker."""
        self.health_endpoints = [
            "http://localhost:8000/health",
            "http://localhost:6333/health",  # Qdrant
            "http://localhost:6379/ping",  # Redis/Dragonfly
        ]
        self.initialized = True

    async def check_health(self, endpoint: str, timeout: int = 30) -> dict[str, Any]:
        """Check health of a specific endpoint."""
        if not self.initialized:
            await self.initialize()

        try:
            # Simulate health check
            await asyncio.sleep(0.1)
            return {
                "endpoint": endpoint,
                "status": "healthy",
                "response_time_ms": 50.0,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat(),
            }

    async def check_all_health(self, timeout: int = 30) -> dict[str, Dict[str, Any]]:
        """Check health of all registered endpoints."""
        results = {}
        for endpoint in self.health_endpoints:
            results[endpoint] = await self.check_health(endpoint, timeout)
        return results

    async def wait_for_healthy(
        self, endpoint: str, timeout: int = 60, interval: int = 5
    ) -> bool:
        """Wait for an endpoint to become healthy."""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            health = await self.check_health(endpoint)
            if health["status"] == "healthy":
                return True
            await asyncio.sleep(interval)

        return False

    async def cleanup(self) -> None:
        """Cleanup health checker resources."""
        self.health_endpoints = []
        self.initialized = False


class DeploymentRollbackManager:
    """Manager for deployment rollback operations."""

    def __init__(self):
        self.deployment_history: list[dict[str, Any]] = []
        self.current_deployment: typing.Optional[dict[str, Any]] = None

    def record_deployment(self, deployment_info: dict[str, Any]) -> None:
        """Record a deployment for rollback capability."""
        deployment_info["timestamp"] = datetime.utcnow().isoformat()
        deployment_info["rollback_available"] = True

        self.deployment_history.append(deployment_info)
        self.current_deployment = deployment_info

    def get_rollback_target(self) -> typing.Optional[dict[str, Any]]:
        """Get the target deployment for rollback."""
        if len(self.deployment_history) < 2:
            return None

        # Return the second-to-last deployment
        return self.deployment_history[-2]

    async def execute_rollback(self) -> dict[str, Any]:
        """Execute deployment rollback."""
        rollback_target = self.get_rollback_target()

        if not rollback_target:
            return {
                "success": False,
                "error": "No rollback target available",
            }

        try:
            # Simulate rollback process
            await asyncio.sleep(2)

            rollback_info = {
                "success": True,
                "rolled_back_to": rollback_target["deployment_id"],
                "rollback_time": datetime.utcnow().isoformat(),
                "previous_deployment": self.current_deployment["deployment_id"]
                if self.current_deployment
                else None,
            }

            # Update current deployment
            self.current_deployment = rollback_target.copy()
            self.current_deployment["rollback_info"] = rollback_info

            return rollback_info

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "rollback_time": datetime.utcnow().isoformat(),
            }

    def get_deployment_history(self) -> list[dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_history.copy()


class BlueGreenDeploymentManager:
    """Manager for blue-green deployment testing."""

    def __init__(self):
        self.blue_env = {"name": "blue", "active": False, "healthy": False}
        self.green_env = {"name": "green", "active": True, "healthy": True}
        self.switch_in_progress = False

    async def deploy_to_inactive(
        self, deployment_info: dict[str, Any]
    ) -> dict[str, Any]:
        """Deploy to the inactive environment."""
        inactive_env = self.blue_env if self.green_env["active"] else self.green_env

        try:
            # Simulate deployment
            await asyncio.sleep(3)

            inactive_env.update(
                {
                    "healthy": True,
                    "deployment_id": deployment_info["deployment_id"],
                    "version": deployment_info["version"],
                    "deployment_time": datetime.utcnow().isoformat(),
                }
            )

            return {
                "success": True,
                "environment": inactive_env["name"],
                "deployment_id": deployment_info["deployment_id"],
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "environment": inactive_env["name"],
            }

    async def switch_traffic(self, force: bool = False) -> dict[str, Any]:
        """Switch traffic between blue and green environments."""
        if self.switch_in_progress:
            return {
                "success": False,
                "error": "Switch already in progress",
            }

        active_env = self.blue_env if self.blue_env["active"] else self.green_env
        inactive_env = self.green_env if self.blue_env["active"] else self.blue_env

        # Check if inactive environment is healthy
        if not force and not inactive_env["healthy"]:
            return {
                "success": False,
                "error": f"{inactive_env['name']} environment is not healthy",
            }

        try:
            self.switch_in_progress = True

            # Simulate traffic switch
            await asyncio.sleep(2)

            # Switch environments
            active_env["active"] = False
            inactive_env["active"] = True

            result = {
                "success": True,
                "switched_from": active_env["name"],
                "switched_to": inactive_env["name"],
                "switch_time": datetime.utcnow().isoformat(),
            }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
        finally:
            self.switch_in_progress = False

    def get_environment_status(self) -> dict[str, Any]:
        """Get current environment status."""
        return {
            "blue": self.blue_env.copy(),
            "green": self.green_env.copy(),
            "switch_in_progress": self.switch_in_progress,
        }


# Pytest markers for deployment testing
pytest.mark.deployment = pytest.mark.deployment
pytest.mark.environment = pytest.mark.environment
pytest.mark.pipeline = pytest.mark.pipeline
pytest.mark.infrastructure = pytest.mark.infrastructure
pytest.mark.blue_green = pytest.mark.blue_green
pytest.mark.disaster_recovery = pytest.mark.disaster_recovery
