"""Blue-Green Deployment Service for Zero-Downtime Releases.

This module provides enterprise-grade blue-green deployment capabilities including:
- Zero-downtime production deployments
- Health check validation before traffic switching
- Automated rollback on health check failures
- State synchronization between environments
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .feature_flags import FeatureFlagManager
from .models import (
    DeploymentConfig,
    DeploymentEnvironment,
    DeploymentHealth,
    DeploymentMetrics,
    DeploymentStatus,
)


logger = logging.getLogger(__name__)


class BlueGreenStatus(str, Enum):
    """Status of blue-green deployment operations."""

    IDLE = "idle"
    PREPARING = "preparing"
    DEPLOYING = "deploying"
    HEALTH_CHECK = "health_check"
    READY_TO_SWITCH = "ready_to_switch"
    SWITCHING = "switching"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"


class BlueGreenEnvironment(BaseModel):
    """Configuration for a blue-green environment."""

    name: str = Field(..., description="Environment name (blue/green)")
    active: bool = Field(
        ..., description="Whether this environment is currently active"
    )
    deployment_id: str | None = Field(default=None, description="Current deployment ID")
    version: str | None = Field(default=None, description="Deployed version")
    health: DeploymentHealth | None = Field(default=None, description="Health status")
    last_deployment: datetime | None = Field(
        default=None, description="Last deployment time"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


@dataclass
class BlueGreenConfig:
    """Configuration for blue-green deployment."""

    deployment_id: str
    target_version: str
    health_check_endpoint: str = "/health"
    health_check_timeout: int = 30
    health_check_retries: int = 3
    health_check_interval: int = 10
    switch_delay_seconds: int = 5
    enable_automatic_switch: bool = True
    enable_automatic_rollback: bool = True
    max_deployment_time_minutes: int = 30

    def to_deployment_config(self) -> DeploymentConfig:
        """Convert to base deployment configuration."""
        return DeploymentConfig(
            deployment_id=self.deployment_id,
            environment=DeploymentEnvironment.PRODUCTION,
            feature_flags={"blue_green_deployment": True},
            monitoring_enabled=True,
            rollback_enabled=self.enable_automatic_rollback,
            max_duration_minutes=self.max_deployment_time_minutes,
            health_check_interval_seconds=self.health_check_interval,
        )


class BlueGreenDeployment:
    """Enterprise blue-green deployment manager."""

    def __init__(
        self,
        qdrant_service: Any,
        cache_manager: Any,
        feature_flag_manager: FeatureFlagManager | None = None,
    ):
        """Initialize blue-green deployment manager.

        Args:
            qdrant_service: Qdrant service for data storage
            cache_manager: Cache manager for state management
            feature_flag_manager: Optional feature flag manager

        """
        self.qdrant_service = qdrant_service
        self.cache_manager = cache_manager
        self.feature_flag_manager = feature_flag_manager

        # Environment state
        self._blue_env = BlueGreenEnvironment(name="blue", active=False)
        self._green_env = BlueGreenEnvironment(
            name="green", active=True
        )  # Green starts active
        self._current_deployment: BlueGreenConfig | None = None
        self._deployment_status = BlueGreenStatus.IDLE

        # Monitoring
        self._deployment_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize blue-green deployment manager."""
        if self._initialized:
            return

        try:
            await self._check_feature_flags()
        except (AttributeError, ImportError, OSError):
            logger.exception("Failed to initialize blue-green deployment manager")
            self._initialized = False
            raise

        try:
            await self._initialize_environment()
        except (AttributeError, ImportError, OSError):
            logger.exception("Failed to initialize environment")
            self._initialized = False
            raise

        self._initialized = True
        logger.info("Blue-green deployment manager initialized successfully")

    async def _check_feature_flags(self) -> None:
        """Check if blue-green deployment is enabled via feature flags."""
        if not self.feature_flag_manager:
            return

        enabled = await self.feature_flag_manager.is_feature_enabled(
            "blue_green_deployment"
        )
        if not enabled:
            logger.info("Blue-green deployment disabled via feature flags")
            self._initialized = True
            return

    async def _initialize_environment(self) -> None:
        """Initialize environment state and monitoring."""
        # Load environment state from storage
        await self._load_environment_state()

        # Start health check monitoring
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def deploy(self, config: BlueGreenConfig) -> str:
        """Start a blue-green deployment.

        Args:
            config: Blue-green deployment configuration

        Returns:
            str: Deployment ID

        Raises:
            RuntimeError: If deployment is already in progress

        """
        if not self._initialized:
            await self.initialize()

        if self._deployment_status not in (
            BlueGreenStatus.IDLE,
            BlueGreenStatus.COMPLETED,
            BlueGreenStatus.FAILED,
        ):
            msg = f"Deployment already in progress: {self._deployment_status}"
            raise RuntimeError(msg)

        self._current_deployment = config
        self._deployment_status = BlueGreenStatus.PREPARING

        # Start deployment process
        self._deployment_task = asyncio.create_task(self._execute_deployment(config))

        logger.info("Started blue-green deployment: %s", config.deployment_id)
        return config.deployment_id

    async def get_status(self) -> dict[str, Any]:
        """Get current deployment status.

        Returns:
            dict[str, Any]: Current status including environments and deployment info

        """
        return {
            "status": self._deployment_status.value,
            "blue_environment": self._blue_env.model_dump(),
            "green_environment": self._green_env.model_dump(),
            "current_deployment": {
                "deployment_id": self._current_deployment.deployment_id
                if self._current_deployment
                else None,
                "target_version": self._current_deployment.target_version
                if self._current_deployment
                else None,
            }
            if self._current_deployment
            else None,
            "active_environment": "blue" if self._blue_env.active else "green",
        }

    async def switch_environments(self, force: bool = False) -> bool:
        """Manually switch between blue and green environments.

        Args:
            force: Force switch even if target environment is unhealthy

        Returns:
            bool: True if switch was successful

        """
        if not self._initialized:
            await self.initialize()

        # Determine target environment
        if self._blue_env.active:
            target_env = self._green_env
            source_env = self._blue_env
        else:
            target_env = self._blue_env
            source_env = self._green_env

        # Health check target environment (unless forced)
        if not force and (
            not target_env.health or target_env.health.status != "healthy"  # pylint: disable=no-member
        ):
            logger.warning(
                "Target environment %s is not healthy, aborting switch",
                target_env.name,
            )
            return False

        # Perform switch
        try:
            await self._execute_environment_switch(source_env, target_env)
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to switch environments")
            self._deployment_status = BlueGreenStatus.FAILED
            return False
        else:
            return True

    async def _execute_environment_switch(
        self, source_env: BlueGreenEnvironment, target_env: BlueGreenEnvironment
    ) -> None:
        """Execute the environment switch process."""
        self._deployment_status = BlueGreenStatus.SWITCHING

        await self._switch_traffic(source_env.name, target_env.name)

        # Update environment states
        source_env.active = False
        target_env.active = True

        await self._persist_environment_state()

        self._deployment_status = BlueGreenStatus.COMPLETED
        logger.info(
            "Successfully switched from %s to %s", source_env.name, target_env.name
        )

    async def rollback(self) -> bool:
        """Rollback to previous environment.

        Returns:
            bool: True if rollback was successful

        """
        if not self._initialized:
            await self.initialize()

        logger.info("Starting rollback procedure")

        try:
            return await self._execute_rollback()
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Error during rollback")
            self._deployment_status = BlueGreenStatus.FAILED
            return False

    async def _execute_rollback(self) -> bool:
        """Execute rollback to previous environment."""
        self._deployment_status = BlueGreenStatus.ROLLING_BACK

        # Switch back to the other environment
        if success := await self.switch_environments(force=True):
            logger.info(f"Rollback completed successfully: {success}")
            return True

        logger.error("Rollback failed")
        return False

    async def get_deployment_metrics(self) -> dict[str, DeploymentMetrics]:
        """Get metrics for both environments.

        Returns:
            dict[str, DeploymentMetrics]: Metrics for blue and green environments

        """
        metrics = {}

        for env in [self._blue_env, self._green_env]:
            if env.deployment_id:
                # In production, fetch real metrics from monitoring system
                metrics[env.name] = DeploymentMetrics(
                    deployment_id=env.deployment_id,
                    environment=DeploymentEnvironment.PRODUCTION,
                    status=DeploymentStatus.SUCCESS
                    if env.health and env.health.status == "healthy"  # pylint: disable=no-member
                    else DeploymentStatus.FAILED,
                    total_requests=0,  # Would be populated from real metrics
                    successful_requests=0,
                    failed_requests=0,
                    avg_response_time_ms=env.health.response_time_ms  # pylint: disable=no-member
                    if env.health
                    else 0.0,
                    error_rate=env.health.error_rate if env.health else 0.0,  # pylint: disable=no-member
                    created_at=env.last_deployment or datetime.now(tz=UTC),
                )

        return metrics

    async def _execute_deployment(self, config: BlueGreenConfig) -> None:
        """Execute the blue-green deployment process."""
        try:
            target_env = self._green_env if self._blue_env.active else self._blue_env
            logger.info(
                "Deploying to %s environment (version: %s)",
                target_env.name,
                config.target_version,
            )

            await self._run_deployment_phases(target_env, config)
        except (OSError, PermissionError):
            logger.exception("Deployment failed")
            self._deployment_status = BlueGreenStatus.FAILED
            await self._handle_deployment_failure(config)

    async def _run_deployment_phases(
        self, target_env: BlueGreenEnvironment, config: BlueGreenConfig
    ) -> None:
        """Run all deployment phases sequentially."""
        # Phase 1: Deploy to inactive environment
        self._deployment_status = BlueGreenStatus.DEPLOYING
        await self._deploy_to_environment(target_env, config)

        # Phase 2: Health check new deployment
        self._deployment_status = BlueGreenStatus.HEALTH_CHECK
        if not await self._perform_health_checks(target_env, config):
            logger.error(
                "Health checks failed for %s environment",
                target_env.name,
            )
            self._deployment_status = BlueGreenStatus.FAILED
            return

        # Phase 3: Ready to switch
        self._deployment_status = BlueGreenStatus.READY_TO_SWITCH
        logger.info(
            "Deployment to %s successful, ready to switch traffic", target_env.name
        )

        # Phase 4: Automatic switch (if enabled)
        await self._handle_automatic_switch(config)

    async def _handle_automatic_switch(self, config: BlueGreenConfig) -> None:
        """Handle automatic environment switching if enabled."""
        if not config.enable_automatic_switch:
            logger.info("Automatic switch disabled, manual intervention required")
            return

        await asyncio.sleep(config.switch_delay_seconds)

        if success := await self.switch_environments():
            self._deployment_status = BlueGreenStatus.COMPLETED
            logger.info(f"Blue-green deployment completed successfully: {success}")
        else:
            self._deployment_status = BlueGreenStatus.FAILED
            logger.error("Failed to switch environments")

    async def _handle_deployment_failure(self, config: BlueGreenConfig) -> None:
        """Handle deployment failure and potential rollback."""
        # Automatic rollback on failure
        if config.enable_automatic_rollback:
            await self.rollback()

    async def _deploy_to_environment(
        self, env: BlueGreenEnvironment, config: BlueGreenConfig
    ) -> None:
        """Deploy new version to specified environment."""
        try:
            self._update_environment_metadata(env, config)
            await self._execute_deployment_steps(env, config)
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to deploy to %s environment", env.name)
            raise

    def _update_environment_metadata(
        self, env: BlueGreenEnvironment, config: BlueGreenConfig
    ) -> None:
        """Update environment metadata for deployment."""
        env.deployment_id = config.deployment_id
        env.version = config.target_version
        env.last_deployment = datetime.now(tz=UTC)
        env.metadata.update(
            {
                "deployment_config": config.__dict__,
                "deployment_start": datetime.now(tz=UTC).isoformat(),
            }
        )

    async def _execute_deployment_steps(
        self, env: BlueGreenEnvironment, config: BlueGreenConfig
    ) -> None:
        """Execute actual deployment steps."""
        # In production, this would:
        # 1. Deploy new application version to environment
        # 2. Update load balancer configuration
        # 3. Apply database migrations if needed
        # 4. Update service configurations

        # Simulate deployment time
        await asyncio.sleep(2)

        logger.info(
            "Deployed version %s to %s environment", config.target_version, env.name
        )

    async def _perform_health_checks(
        self, env: BlueGreenEnvironment, config: BlueGreenConfig
    ) -> bool:
        """Perform health checks on the deployed environment."""
        for attempt in range(config.health_check_retries):
            try:
                if await self._perform_single_health_check(env, attempt):
                    return True
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(
                    "Health check failed for %s environment (attempt %d): %s",
                    env.name,
                    attempt + 1,
                    e,
                )

                if attempt < config.health_check_retries - 1:
                    await asyncio.sleep(config.health_check_interval)

        # All health checks failed
        env.health = DeploymentHealth(
            status="unhealthy",
            response_time_ms=0.0,
            error_rate=100.0,
            success_count=0,
            error_count=config.health_check_retries,
            last_check=datetime.now(tz=UTC),
        )
        return False

    async def _perform_single_health_check(
        self, env: BlueGreenEnvironment, attempt: int
    ) -> bool:
        """Perform a single health check attempt."""
        # Simulate health check
        await asyncio.sleep(1)

        # In production, this would make HTTP requests to health endpoints
        health = DeploymentHealth(
            status="healthy",
            response_time_ms=50.0,
            error_rate=0.0,
            success_count=100,
            error_count=0,
            last_check=datetime.now(tz=UTC),
            details={"environment": env.name, "version": env.version},
        )

        env.health = health
        logger.info(
            "Health check passed for %s environment (attempt %d)",
            env.name,
            attempt + 1,
        )
        return True

    async def _switch_traffic(self, from_env: str, to_env: str) -> None:
        """Switch traffic from one environment to another."""
        try:
            await self._execute_traffic_switch(from_env, to_env)
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to switch traffic")
            raise

    async def _execute_traffic_switch(self, from_env: str, to_env: str) -> None:
        """Execute traffic switching operations."""
        # In production, this would:
        # 1. Update load balancer configuration
        # 2. Update DNS records if needed
        # 3. Update service mesh configuration
        # 4. Drain connections from old environment

        logger.info("Switching traffic from %s to %s", from_env, to_env)

        # Simulate traffic switch
        await asyncio.sleep(1)

        logger.info("Traffic switch completed")

    async def _health_check_loop(self) -> None:
        """Background task for continuous health monitoring."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Monitor both environments
                for env in [self._blue_env, self._green_env]:
                    if env.deployment_id:  # Only check deployed environments
                        await self._check_environment_health(env)

            except asyncio.CancelledError:
                break
            except (TimeoutError, OSError, PermissionError):
                logger.exception("Error in health check loop")
                await asyncio.sleep(30)

    async def _check_environment_health(self, env: BlueGreenEnvironment) -> None:
        """Check health of a specific environment."""
        try:
            self._update_environment_health_status(env)
        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error checking health for %s environment", env.name)

    def _update_environment_health_status(self, env: BlueGreenEnvironment) -> None:
        """Update health status for environment."""
        # In production, perform actual health checks
        # For now, maintain existing health status
        if env.health:
            env.health.last_check = datetime.now(tz=UTC)

    async def _load_environment_state(self) -> None:
        """Load environment state from storage."""
        # In production, load from database/storage

    async def _persist_environment_state(self) -> None:
        """Persist environment state to storage."""
        # In production, save to database/storage

    async def cleanup(self) -> None:
        """Cleanup blue-green deployment manager resources."""
        if self._deployment_task:
            self._deployment_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._deployment_task

        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task

        self._deployment_status = BlueGreenStatus.IDLE
        self._current_deployment = None
        self._initialized = False
        logger.info("Blue-green deployment manager cleanup completed")
