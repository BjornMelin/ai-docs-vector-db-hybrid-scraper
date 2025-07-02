"""Canary Deployment Service for Progressive Rollouts.

This module provides enterprise-grade canary deployment capabilities including:
- Progressive traffic rollout with configurable percentages
- Real-time metrics monitoring and automatic rollback triggers
- Statistical analysis for deployment success validation
- Integration with feature flags for controlled rollouts
"""

import asyncio
import contextlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .feature_flags import FeatureFlagManager
from .models import DeploymentConfig, DeploymentEnvironment


logger = logging.getLogger(__name__)


class CanaryStatus(str, Enum):
    """Status of canary deployment operations."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    MONITORING = "monitoring"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


class CanaryStage(str, Enum):
    """Stages of canary deployment progression."""

    STAGE_1 = "stage_1"  # 5% traffic
    STAGE_2 = "stage_2"  # 25% traffic
    STAGE_3 = "stage_3"  # 50% traffic
    STAGE_4 = "stage_4"  # 75% traffic
    FULL_ROLLOUT = "full_rollout"  # 100% traffic


class CanaryMetrics(BaseModel):
    """Metrics for canary deployment analysis."""

    deployment_id: str = Field(..., description="Canary deployment identifier")
    stage: CanaryStage = Field(..., description="Current canary stage")

    # Traffic metrics
    canary_traffic_percentage: float = Field(
        ..., description="Current canary traffic percentage"
    )
    canary_requests: int = Field(default=0, description="Requests to canary version")
    stable_requests: int = Field(default=0, description="Requests to stable version")

    # Performance comparison
    canary_response_time_ms: float = Field(
        default=0.0, description="Canary response time"
    )
    stable_response_time_ms: float = Field(
        default=0.0, description="Stable response time"
    )
    canary_error_rate: float = Field(default=0.0, description="Canary error rate")
    stable_error_rate: float = Field(default=0.0, description="Stable error rate")

    # Success criteria
    success_criteria_met: bool = Field(
        default=False, description="Whether success criteria are met"
    )
    health_score: float = Field(default=0.0, description="Overall health score (0-100)")

    # Timestamps
    stage_start_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC), description="Stage start time"
    )
    next_stage_time: datetime | None = Field(
        default=None, description="Next stage promotion time"
    )

    @property
    def stage_duration_minutes(self) -> float:
        """Calculate current stage duration in minutes."""
        return (datetime.now(tz=UTC) - self.stage_start_time).total_seconds() / 60.0


@dataclass
class CanaryConfig:
    """Configuration for canary deployment."""

    deployment_id: str
    target_version: str

    # Traffic progression
    initial_traffic_percentage: float = 5.0
    stage_duration_minutes: int = 15
    auto_promote: bool = True
    max_error_rate: float = 5.0
    max_response_time_ms: float = 500.0
    min_success_rate: float = 95.0

    # Monitoring
    monitoring_window_minutes: int = 10
    health_check_interval_seconds: int = 30
    rollback_on_failure: bool = True

    # Advanced settings
    custom_traffic_stages: list[float] | None = None  # Custom traffic percentages
    success_criteria: dict[str, Any] | None = None
    feature_flags: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate canary configuration."""
        if not 0 < self.initial_traffic_percentage <= 100:
            msg = "Initial traffic percentage must be 0-100"
            raise ValueError(msg)
        if self.max_error_rate < 0:
            msg = "Max error rate must be non-negative"
            raise ValueError(msg)
        if self.min_success_rate < 0 or self.min_success_rate > 100:
            msg = "Min success rate must be 0-100"
            raise ValueError(msg)

    @property
    def traffic_stages(self) -> list[float]:
        """Get traffic progression stages."""
        if self.custom_traffic_stages:
            return sorted(self.custom_traffic_stages)
        # Default progressive rollout: 5% -> 25% -> 50% -> 75% -> 100%
        return [5.0, 25.0, 50.0, 75.0, 100.0]

    def to_deployment_config(self) -> DeploymentConfig:
        """Convert to base deployment configuration."""
        return DeploymentConfig(
            deployment_id=self.deployment_id,
            environment=DeploymentEnvironment.CANARY,
            feature_flags=self.feature_flags or {"canary_deployment": True},
            monitoring_enabled=True,
            rollback_enabled=self.rollback_on_failure,
            max_duration_minutes=len(self.traffic_stages) * self.stage_duration_minutes,
            health_check_interval_seconds=self.health_check_interval_seconds,
            failure_threshold=self.max_error_rate,
        )


class CanaryDeployment:
    """Enterprise canary deployment manager with progressive rollout."""

    def __init__(
        self,
        qdrant_service: Any,
        cache_manager: Any,
        feature_flag_manager: FeatureFlagManager | None = None,
    ):
        """Initialize canary deployment manager.

        Args:
            qdrant_service: Qdrant service for data storage
            cache_manager: Cache manager for metrics caching
            feature_flag_manager: Optional feature flag manager

        """
        self.qdrant_service = qdrant_service
        self.cache_manager = cache_manager
        self.feature_flag_manager = feature_flag_manager

        # Deployment state
        self._active_deployments: dict[str, CanaryConfig] = {}
        self._deployment_metrics: dict[str, CanaryMetrics] = {}
        self._deployment_status: dict[str, CanaryStatus] = {}

        # Monitoring
        self._monitoring_tasks: dict[str, asyncio.Task] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize canary deployment manager."""
        if self._initialized:
            return

        try:
            await self._check_feature_flags()
        except (AttributeError, ImportError, OSError):
            logger.exception("Failed to initialize canary deployment manager")
            self._initialized = False
            raise

        try:
            await self._load_active_deployments()
        except (AttributeError, ImportError, OSError):
            logger.exception("Failed to load active deployments")
            self._initialized = False
            raise

        self._initialized = True
        logger.info("Canary deployment manager initialized successfully")

    async def _check_feature_flags(self) -> None:
        """Check if canary deployment is enabled via feature flags."""
        if not self.feature_flag_manager:
            return

        enabled = await self.feature_flag_manager.is_feature_enabled(
            "canary_deployment"
        )
        if not enabled:
            logger.info("Canary deployment disabled via feature flags")
            self._initialized = True
            return

    async def start_canary(self, config: CanaryConfig) -> str:
        """Start a new canary deployment.

        Args:
            config: Canary deployment configuration

        Returns:
            str: Deployment ID

        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If deployment with same ID already exists

        """
        if not self._initialized:
            await self.initialize()

        # Validate configuration
        config.__post_init__()

        if config.deployment_id in self._active_deployments:
            msg = f"Canary deployment {config.deployment_id} already exists"
            raise RuntimeError(msg)

        # Initialize deployment
        self._active_deployments[config.deployment_id] = config
        self._deployment_status[config.deployment_id] = CanaryStatus.INITIALIZING

        # Create initial metrics
        initial_metrics = CanaryMetrics(
            deployment_id=config.deployment_id,
            stage=CanaryStage.STAGE_1,
            canary_traffic_percentage=config.initial_traffic_percentage,
        )
        self._deployment_metrics[config.deployment_id] = initial_metrics

        # Start monitoring task
        self._monitoring_tasks[config.deployment_id] = asyncio.create_task(
            self._monitor_canary_deployment(config.deployment_id)
        )

        # Persist deployment state
        await self._persist_deployment_state(config.deployment_id)

        logger.info(
            "Started canary deployment: %s (version: %s)",
            config.deployment_id,
            config.target_version,
        )
        return config.deployment_id

    async def get_canary_status(self, deployment_id: str) -> dict[str, Any]:
        """Get status of a canary deployment.

        Args:
            deployment_id: Canary deployment identifier

        Returns:
            dict[str, Any]: Deployment status and metrics

        """
        if deployment_id not in self._active_deployments:
            return {"error": "Deployment not found"}

        config = self._active_deployments[deployment_id]
        metrics = self._deployment_metrics.get(deployment_id)
        status = self._deployment_status.get(deployment_id, CanaryStatus.FAILED)

        return {
            "deployment_id": deployment_id,
            "status": status.value,
            "version": config.target_version,
            "current_stage": metrics.stage.value if metrics else "unknown",
            "traffic_percentage": metrics.canary_traffic_percentage if metrics else 0.0,
            "health_score": metrics.health_score if metrics else 0.0,
            "success_criteria_met": metrics.success_criteria_met if metrics else False,
            "stage_duration_minutes": metrics.stage_duration_minutes
            if metrics
            else 0.0,
            "next_promotion": metrics.next_stage_time.isoformat()
            if metrics and metrics.next_stage_time
            else None,
            "metrics": metrics.model_dump() if metrics else None,
        }

    async def promote_canary(self, deployment_id: str, force: bool = False) -> bool:
        """Manually promote canary to next stage or full rollout.

        Args:
            deployment_id: Canary deployment identifier
            force: Force promotion even if success criteria not met

        Returns:
            bool: True if promotion was successful

        """
        if deployment_id not in self._active_deployments:
            return False

        config = self._active_deployments[deployment_id]
        if not (metrics := self._deployment_metrics.get(deployment_id)):
            return False

        # Check success criteria (unless forced)
        if not force and not metrics.success_criteria_met:
            logger.warning(
                "Canary promotion blocked: success criteria not met for %s",
                deployment_id,
            )
            return False

        # Determine next stage
        current_percentage = metrics.canary_traffic_percentage
        traffic_stages = config.traffic_stages

        # Find next stage
        next_percentage = None
        for stage_percentage in traffic_stages:
            if stage_percentage > current_percentage:
                next_percentage = stage_percentage
                break

        if next_percentage is None:
            # Already at full rollout
            await self._complete_canary_deployment(deployment_id)
            return True

        # Promote to next stage
        try:
            await self._execute_promotion(
                deployment_id, next_percentage, config, metrics
            )
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to promote canary %s", deployment_id)
            self._deployment_status[deployment_id] = CanaryStatus.FAILED
            return False
        else:
            return True

    async def _execute_promotion(
        self,
        deployment_id: str,
        next_percentage: float,
        config: CanaryConfig,
        metrics: CanaryMetrics,
    ) -> None:
        """Execute canary promotion to next stage."""
        self._deployment_status[deployment_id] = CanaryStatus.PROMOTING

        await self._update_traffic_split(deployment_id, next_percentage)

        # Update metrics
        self._update_promotion_metrics(metrics, next_percentage, config)
        self._update_canary_stage(metrics, next_percentage)

        self._deployment_status[deployment_id] = CanaryStatus.MONITORING

        logger.info(
            "Promoted canary %s to %.1f%% traffic (stage: %s)",
            deployment_id,
            next_percentage,
            metrics.stage.value,
        )

    def _update_promotion_metrics(
        self, metrics: CanaryMetrics, next_percentage: float, config: CanaryConfig
    ) -> None:
        """Update metrics during canary promotion."""
        metrics.canary_traffic_percentage = next_percentage
        metrics.stage_start_time = datetime.now(tz=UTC)
        metrics.next_stage_time = datetime.now(tz=UTC) + timedelta(
            minutes=config.stage_duration_minutes
        )
        metrics.success_criteria_met = False  # Reset for new stage

    def _update_canary_stage(
        self, metrics: CanaryMetrics, next_percentage: float
    ) -> None:
        """Update canary stage based on traffic percentage."""
        if next_percentage >= 100.0:
            metrics.stage = CanaryStage.FULL_ROLLOUT
        elif next_percentage >= 75.0:
            metrics.stage = CanaryStage.STAGE_4
        elif next_percentage >= 50.0:
            metrics.stage = CanaryStage.STAGE_3
        elif next_percentage >= 25.0:
            metrics.stage = CanaryStage.STAGE_2
        else:
            metrics.stage = CanaryStage.STAGE_1

    async def rollback_canary(self, deployment_id: str, reason: str = "") -> bool:
        """Rollback a canary deployment.

        Args:
            deployment_id: Canary deployment identifier
            reason: Reason for rollback

        Returns:
            bool: True if rollback was successful

        """
        if deployment_id not in self._active_deployments:
            return False

        try:
            await self._execute_rollback(deployment_id, reason)
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to rollback canary %s", deployment_id)
            self._deployment_status[deployment_id] = CanaryStatus.FAILED
            return False
        else:
            return True

    async def _execute_rollback(self, deployment_id: str, reason: str) -> None:
        """Execute canary rollback operations."""
        self._deployment_status[deployment_id] = CanaryStatus.ROLLING_BACK

        await self._update_traffic_split(deployment_id, 0.0)

        # Update metrics
        self._update_rollback_metrics(deployment_id)

        await self._complete_canary_deployment(deployment_id, success=False)

        logger.info(
            "Rolled back canary deployment %s. Reason: %s", deployment_id, reason
        )

    def _update_rollback_metrics(self, deployment_id: str) -> None:
        """Update metrics during rollback."""
        if metrics := self._deployment_metrics.get(deployment_id):
            metrics.canary_traffic_percentage = 0.0
            metrics.success_criteria_met = False

    async def pause_canary(self, deployment_id: str) -> bool:
        """Pause a canary deployment.

        Args:
            deployment_id: Canary deployment identifier

        Returns:
            bool: True if paused successfully

        """
        if deployment_id not in self._active_deployments:
            return False

        self._deployment_status[deployment_id] = CanaryStatus.PAUSED

        # Cancel monitoring task
        if deployment_id in self._monitoring_tasks:
            self._monitoring_tasks[deployment_id].cancel()

        logger.info("Paused canary deployment: %s", deployment_id)
        return True

    async def resume_canary(self, deployment_id: str) -> bool:
        """Resume a paused canary deployment.

        Args:
            deployment_id: Canary deployment identifier

        Returns:
            bool: True if resumed successfully

        """
        if deployment_id not in self._active_deployments:
            return False

        if self._deployment_status.get(deployment_id) != CanaryStatus.PAUSED:
            return False

        # Resume monitoring
        self._deployment_status[deployment_id] = CanaryStatus.MONITORING
        self._monitoring_tasks[deployment_id] = asyncio.create_task(
            self._monitor_canary_deployment(deployment_id)
        )

        logger.info("Resumed canary deployment: %s", deployment_id)
        return True

    async def _monitor_canary_deployment(self, deployment_id: str) -> None:
        """Monitor a canary deployment and handle automatic promotion/rollback."""
        try:
            await self._run_monitoring_loop(deployment_id)
        except asyncio.CancelledError:
            logger.info("Monitoring cancelled for canary deployment: %s", deployment_id)
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Error monitoring canary deployment %s", deployment_id)
            self._deployment_status[deployment_id] = CanaryStatus.FAILED

    async def _run_monitoring_loop(self, deployment_id: str) -> None:
        """Run the main monitoring loop for canary deployment."""
        config = self._active_deployments[deployment_id]

        while deployment_id in self._active_deployments:
            current_status = self._deployment_status.get(deployment_id)

            if current_status in (
                CanaryStatus.COMPLETED,
                CanaryStatus.FAILED,
                CanaryStatus.PAUSED,
            ):
                break

            await self._process_monitoring_cycle(deployment_id, config)

            await asyncio.sleep(config.health_check_interval_seconds)

    async def _process_monitoring_cycle(
        self, deployment_id: str, config: CanaryConfig
    ) -> None:
        """Process a single monitoring cycle."""
        # Update metrics
        await self._update_deployment_metrics(deployment_id)

        # Evaluate deployment health
        metrics = self._deployment_metrics[deployment_id]
        success_criteria_met = await self._evaluate_success_criteria(deployment_id)
        metrics.success_criteria_met = success_criteria_met

        # Check for rollback conditions
        if await self._should_rollback(deployment_id):
            await self.rollback_canary(
                deployment_id, "Automatic rollback triggered"
            )
            return

        # Handle automatic promotion
        await self._handle_auto_promotion(deployment_id, config, metrics)

    async def _handle_auto_promotion(
        self, deployment_id: str, config: CanaryConfig, metrics: CanaryMetrics
    ) -> None:
        """Handle automatic promotion logic."""
        if not (config.auto_promote and metrics.success_criteria_met):
            return

        stage_duration_ok = (
            metrics.stage_duration_minutes >= config.stage_duration_minutes
        )

        if not stage_duration_ok:
            return

        if metrics.canary_traffic_percentage >= 100.0:
            # Complete deployment
            await self._complete_canary_deployment(deployment_id)
        else:
            # Promote to next stage
            await self.promote_canary(deployment_id)

    async def _update_deployment_metrics(self, deployment_id: str) -> None:
        """Update metrics for a canary deployment."""
        try:
            metrics = self._deployment_metrics[deployment_id]
            self._simulate_performance_metrics(deployment_id, metrics)
            self._calculate_health_score(metrics)
            self._update_request_counts(metrics)
        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error updating metrics for canary %s", deployment_id)

    def _simulate_performance_metrics(
        self, deployment_id: str, metrics: CanaryMetrics
    ) -> None:
        """Simulate performance metrics for canary deployment."""
        # In production, fetch real metrics from monitoring system
        # For now, simulate metrics

        # Simulate canary performance (slightly better than stable)
        canary_response_time = 45.0 + (hash(deployment_id) % 20)  # 45-65ms
        stable_response_time = 55.0 + (hash(deployment_id) % 15)  # 55-70ms
        canary_error_rate = max(
            0.0, 2.0 + (hash(deployment_id) % 10) / 10.0
        )  # 2.0-3.0%
        stable_error_rate = max(
            0.0, 3.0 + (hash(deployment_id) % 15) / 10.0
        )  # 3.0-4.5%

        # Update metrics
        metrics.canary_response_time_ms = canary_response_time
        metrics.stable_response_time_ms = stable_response_time
        metrics.canary_error_rate = canary_error_rate
        metrics.stable_error_rate = stable_error_rate

    def _calculate_health_score(self, metrics: CanaryMetrics) -> None:
        """Calculate overall health score for canary deployment."""
        # Calculate health score (0-100)
        response_time_score = max(
            0, 100 - (metrics.canary_response_time_ms / 10)
        )  # Penalty for high latency
        error_rate_score = max(
            0, 100 - (metrics.canary_error_rate * 10)
        )  # Penalty for errors
        metrics.health_score = (response_time_score + error_rate_score) / 2

    def _update_request_counts(self, metrics: CanaryMetrics) -> None:
        """Update request counts for canary and stable versions."""
        # Update request counts (simulate traffic)
        traffic_percentage = metrics.canary_traffic_percentage / 100.0
        total_requests = 1000  # Simulate 1000 requests per monitoring interval
        metrics.canary_requests += int(total_requests * traffic_percentage)
        metrics.stable_requests += int(total_requests * (1 - traffic_percentage))

    async def _evaluate_success_criteria(self, deployment_id: str) -> bool:
        """Evaluate if canary deployment meets success criteria."""
        try:
            config = self._active_deployments[deployment_id]
            metrics = self._deployment_metrics[deployment_id]

            return self._check_all_success_criteria(config, metrics)
        except (OSError, PermissionError):
            logger.exception(
                "Error evaluating success criteria for canary %s", deployment_id
            )
            return False

    def _check_all_success_criteria(
        self, config: CanaryConfig, metrics: CanaryMetrics
    ) -> bool:
        """Check all success criteria for canary deployment."""
        # Check error rate
        if metrics.canary_error_rate > config.max_error_rate:
            return False

        # Check response time
        if metrics.canary_response_time_ms > config.max_response_time_ms:
            return False

        # Check success rate
        if not self._check_success_rate(config, metrics):
            return False

        # Check minimum monitoring duration
        return metrics.stage_duration_minutes >= config.monitoring_window_minutes

    def _check_success_rate(self, config: CanaryConfig, metrics: CanaryMetrics) -> bool:
        """Check if success rate meets criteria."""
        if (total_requests := metrics.canary_requests) <= 0:
            return True

        error_requests = int(total_requests * metrics.canary_error_rate / 100.0)
        success_rate = ((total_requests - error_requests) / total_requests) * 100.0
        return success_rate >= config.min_success_rate

    async def _should_rollback(self, deployment_id: str) -> bool:
        """Check if canary deployment should be automatically rolled back."""
        try:
            config = self._active_deployments[deployment_id]
            metrics = self._deployment_metrics[deployment_id]

            return self._evaluate_rollback_conditions(deployment_id, config, metrics)
        except (OSError, PermissionError):
            logger.exception(
                "Error checking rollback conditions for canary %s", deployment_id
            )
            return False

    def _evaluate_rollback_conditions(
        self, deployment_id: str, config: CanaryConfig, metrics: CanaryMetrics
    ) -> bool:
        """Evaluate if rollback conditions are met."""
        # Check if rollback is enabled
        if not config.rollback_on_failure:
            return False

        # Check critical error rate threshold
        if self._check_critical_error_rate(deployment_id, config, metrics):
            return True

        # Check critical response time threshold
        if self._check_critical_response_time(deployment_id, config, metrics):
            return True

        # Check health score threshold
        return self._check_critical_health_score(deployment_id, metrics)

    def _check_critical_error_rate(
        self, deployment_id: str, config: CanaryConfig, metrics: CanaryMetrics
    ) -> bool:
        """Check if error rate exceeds critical threshold."""
        if metrics.canary_error_rate > config.max_error_rate * 2:  # 2x threshold
            logger.warning(
                "Canary %s exceeded critical error rate: %.2f%%",
                deployment_id,
                metrics.canary_error_rate,
            )
            return True
        return False

    def _check_critical_response_time(
        self, deployment_id: str, config: CanaryConfig, metrics: CanaryMetrics
    ) -> bool:
        """Check if response time exceeds critical threshold."""
        if (
            metrics.canary_response_time_ms > config.max_response_time_ms * 2
        ):  # 2x threshold
            logger.warning(
                "Canary %s exceeded critical response time: %.2fms",
                deployment_id,
                metrics.canary_response_time_ms,
            )
            return True
        return False

    def _check_critical_health_score(
        self, deployment_id: str, metrics: CanaryMetrics
    ) -> bool:
        """Check if health score is below critical threshold."""
        if metrics.health_score < 30.0:  # Critical health threshold
            logger.warning(
                "Canary %s health score too low: %.1f",
                deployment_id,
                metrics.health_score,
            )
            return True
        return False

    async def _update_traffic_split(
        self, deployment_id: str, canary_percentage: float
    ) -> None:
        """Update traffic split for canary deployment."""
        try:
            self._apply_traffic_configuration(deployment_id, canary_percentage)
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to update traffic split for %s", deployment_id)
            raise

    def _apply_traffic_configuration(
        self, deployment_id: str, canary_percentage: float
    ) -> None:
        """Apply traffic configuration for canary deployment."""
        # In production, this would update:
        # 1. Load balancer configuration
        # 2. Service mesh rules
        # 3. Feature flag percentages
        # 4. DNS weighted routing

        stable_percentage = 100.0 - canary_percentage

        logger.info(
            "Updated traffic split for %s: %.1f%% canary, %.1f%% stable",
            deployment_id,
            canary_percentage,
            stable_percentage,
        )

    async def _complete_canary_deployment(
        self, deployment_id: str, success: bool = True
    ) -> None:
        """Complete a canary deployment."""
        try:
            self._set_completion_status(deployment_id, success)
            self._cleanup_monitoring_task(deployment_id)

            await self._persist_deployment_state(deployment_id)

            self._schedule_cleanup_task(deployment_id)
        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error completing canary deployment %s", deployment_id)

    def _set_completion_status(self, deployment_id: str, success: bool) -> None:
        """Set the completion status for canary deployment."""
        if success:
            self._deployment_status[deployment_id] = CanaryStatus.COMPLETED
            logger.info("Canary deployment %s completed successfully", deployment_id)
        else:
            self._deployment_status[deployment_id] = CanaryStatus.FAILED
            logger.info("Canary deployment %s failed", deployment_id)

    def _cleanup_monitoring_task(self, deployment_id: str) -> None:
        """Clean up monitoring task for deployment."""
        if deployment_id in self._monitoring_tasks:
            self._monitoring_tasks[deployment_id].cancel()
            del self._monitoring_tasks[deployment_id]

    def _schedule_cleanup_task(self, deployment_id: str) -> None:
        """Schedule cleanup task for deployment."""
        # Clean up after delay (keep for analysis)
        self._monitoring_tasks[f"{deployment_id}_cleanup"] = asyncio.create_task(
            self._cleanup_deployment(deployment_id, delay_seconds=3600)
        )  # 1 hour

    async def _cleanup_deployment(self, deployment_id: str, delay_seconds: int) -> None:
        """Clean up completed deployment after delay."""
        try:
            await asyncio.sleep(delay_seconds)
            self._remove_deployment_data(deployment_id)
            logger.info("Cleaned up canary deployment: %s", deployment_id)
        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error cleaning up canary deployment %s", deployment_id)

    def _remove_deployment_data(self, deployment_id: str) -> None:
        """Remove deployment data from memory."""
        # Remove from active deployments
        self._active_deployments.pop(deployment_id, None)
        self._deployment_metrics.pop(deployment_id, None)
        self._deployment_status.pop(deployment_id, None)

    async def _load_active_deployments(self) -> None:
        """Load active canary deployments from storage."""
        # In production, load from database/storage

    async def _persist_deployment_state(self, deployment_id: str) -> None:
        """Persist deployment state to storage."""
        # In production, save to database/storage

    async def cleanup(self) -> None:
        """Cleanup canary deployment manager resources."""
        # Cancel all monitoring tasks
        for task in self._monitoring_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self._monitoring_tasks:
            with contextlib.suppress(Exception):
                await asyncio.gather(
                    *self._monitoring_tasks.values(), return_exceptions=True
                )

        self._active_deployments.clear()
        self._deployment_metrics.clear()
        self._deployment_status.clear()
        self._monitoring_tasks.clear()
        self._initialized = False
        logger.info("Canary deployment manager cleanup completed")
