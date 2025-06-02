"""Canary deployment for gradual rollout of new collections."""

import asyncio
import json
import logging
import time
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from ...config import UnifiedConfig
from ..base import BaseService
from ..core.qdrant_alias_manager import QdrantAliasManager
from ..errors import ServiceError
from ..vector_db.service import QdrantService

logger = logging.getLogger(__name__)


@dataclass
class CanaryStage:
    """Single stage in canary deployment."""

    percentage: float  # Traffic percentage
    duration_minutes: int  # How long to run at this percentage
    error_threshold: float = 0.05  # 5% error rate threshold
    latency_threshold: float = 200  # 200ms latency threshold


@dataclass
class CanaryMetrics:
    """Metrics collected during canary deployment."""

    latency: list[float] = field(default_factory=list)
    error_rate: list[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    stage_start_time: float | None = None


@dataclass
class CanaryDeploymentConfig:
    """Configuration for canary deployment."""

    alias: str
    old_collection: str
    new_collection: str
    stages: list[CanaryStage]
    current_stage: int = 0
    metrics: CanaryMetrics = field(default_factory=CanaryMetrics)
    start_time: float = field(default_factory=time.time)
    status: str = "pending"  # pending, running, completed, failed, rolled_back


class CanaryDeployment(BaseService):
    """Gradual rollout of new collections."""

    def __init__(
        self,
        config: UnifiedConfig,
        alias_manager: QdrantAliasManager,
        qdrant_service: QdrantService | None = None,
        client_manager=None,
    ):
        """Initialize canary deployment.

        Args:
            config: Unified configuration
            alias_manager: Alias manager instance
            qdrant_service: Optional Qdrant service for monitoring
            client_manager: Optional client manager for Redis access
        """
        super().__init__(config)
        self.aliases = alias_manager
        self.qdrant = qdrant_service
        self.client_manager = client_manager
        self.deployments: dict[str, CanaryDeploymentConfig] = {}
        self._state_file = self.config.data_dir / "canary_deployments.json"
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize canary deployment service."""
        # Load existing deployments from persistent storage
        await self._load_deployments()
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup canary deployment service."""
        # Save current state before cleanup
        await self._save_deployments()
        self._initialized = False

    async def start_canary(
        self,
        alias_name: str,
        new_collection: str,
        stages: list[dict] | None = None,
        auto_rollback: bool = True,
    ) -> str:
        """Start canary deployment with gradual traffic shift.

        Args:
            alias_name: Alias to update
            new_collection: New collection to deploy
            stages: Custom deployment stages
            auto_rollback: Whether to auto-rollback on failure

        Returns:
            Deployment ID

        Raises:
            ServiceError: If deployment cannot start
        """
        if not stages:
            # Default canary stages
            stages = [
                {"percentage": 5, "duration_minutes": 30},
                {"percentage": 25, "duration_minutes": 60},
                {"percentage": 50, "duration_minutes": 120},
                {"percentage": 100, "duration_minutes": 0},
            ]

        deployment_id = f"canary_{int(time.time())}"

        old_collection = await self.aliases.get_collection_for_alias(alias_name)
        if not old_collection:
            raise ServiceError(f"No collection found for alias {alias_name}")

        # Convert stage dicts to CanaryStage objects
        canary_stages = [
            CanaryStage(
                percentage=s["percentage"],
                duration_minutes=s["duration_minutes"],
                error_threshold=s.get("error_threshold", 0.05),
                latency_threshold=s.get("latency_threshold", 200),
            )
            for s in stages
        ]

        deployment_config = CanaryDeploymentConfig(
            alias=alias_name,
            old_collection=old_collection,
            new_collection=new_collection,
            stages=canary_stages,
        )

        self.deployments[deployment_id] = deployment_config

        # Persist to storage
        await self._save_deployments()

        # TODO: For production, offload canary deployment orchestration to a persistent task queue
        # (e.g., Celery, ARQ) to ensure deployment continues even if the server restarts.
        # Current asyncio.create_task is suitable for simpler deployments but lacks persistence.
        # This is critical for production deployments as interruptions could leave deployments in an inconsistent state.
        # Example with Celery:
        # canary_task_id = run_canary_deployment.apply_async(args=[deployment_id, auto_rollback])
        # self.deployments[deployment_id].task_id = canary_task_id

        # Start canary process
        # Store task reference to avoid RUF006 warning
        _ = asyncio.create_task(self._run_canary(deployment_id, auto_rollback))  # noqa: RUF006

        logger.info(f"Started canary deployment {deployment_id}")
        return deployment_id

    async def _run_canary(self, deployment_id: str, auto_rollback: bool = True) -> None:
        """Execute canary deployment stages.

        Args:
            deployment_id: ID of the deployment
            auto_rollback: Whether to auto-rollback on failure
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return

        deployment.status = "running"
        await self._save_deployments()

        try:
            for i, stage in enumerate(deployment.stages):
                deployment.current_stage = i
                deployment.metrics.stage_start_time = time.time()
                percentage = stage.percentage
                duration = stage.duration_minutes

                logger.info(
                    f"Canary stage {i}: {percentage}% traffic for {duration} minutes"
                )

                # TODO: Implement actual traffic shifting
                # Traffic shifting implementation depends on infrastructure:
                #
                # 1. API Gateway/Load Balancer based:
                #    - Update AWS ALB target group weights
                #    - Update NGINX upstream weights
                #    - Update Kong/Istio routing rules
                #
                # 2. Qdrant collection alias based:
                #    - Use weighted aliases if Qdrant supports them
                #    - Implement client-side traffic splitting
                #
                # 3. Application-level routing:
                #    - Update routing logic in vector search service
                #    - Use feature flags to control traffic distribution
                #
                # Example implementations:
                # await self._update_load_balancer_weights(percentage)
                # await self._update_alias_weights(deployment.alias, deployment.old_collection, deployment.new_collection, percentage)
                # await self._update_feature_flags(deployment_id, percentage)

                # Monitor during this stage
                if duration > 0:
                    await self._monitor_stage(
                        deployment_id,
                        duration * 60,
                        stage.error_threshold,
                        stage.latency_threshold,
                    )

                # Check metrics before proceeding
                if not self._check_health(deployment):
                    logger.error("Canary failed health check")
                    if auto_rollback:
                        await self._rollback_canary(deployment_id)
                    else:
                        deployment.status = "failed"
                    return

                # If 100%, complete the deployment
                if percentage == 100:
                    await self.aliases.switch_alias(
                        alias_name=deployment.alias,
                        new_collection=deployment.new_collection,
                    )
                    deployment.status = "completed"
                    await self._save_deployments()
                    logger.info("Canary deployment completed successfully")
                    return

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            deployment.status = "failed"
            await self._save_deployments()
            if auto_rollback:
                await self._rollback_canary(deployment_id)

    async def _monitor_stage(
        self,
        deployment_id: str,
        duration_seconds: int,
        error_threshold: float,
        latency_threshold: float,
    ) -> None:
        """Monitor metrics during canary stage.

        Args:
            deployment_id: ID of the deployment
            duration_seconds: How long to monitor
            error_threshold: Maximum allowed error rate
            latency_threshold: Maximum allowed latency
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return

        start_time = time.time()
        check_interval = 60  # Check every minute

        while time.time() - start_time < duration_seconds:
            # Collect metrics
            metrics = await self._collect_metrics(deployment)

            # Update deployment metrics
            deployment.metrics.latency.append(metrics["latency"])
            deployment.metrics.error_rate.append(metrics["error_rate"])

            # Check thresholds
            if metrics["error_rate"] > error_threshold:
                logger.error(
                    f"Error rate {metrics['error_rate']} exceeds threshold {error_threshold}"
                )
                raise ServiceError("Error rate threshold exceeded")

            if metrics["latency"] > latency_threshold:
                logger.error(
                    f"Latency {metrics['latency']}ms exceeds threshold {latency_threshold}ms"
                )
                raise ServiceError("Latency threshold exceeded")

            await asyncio.sleep(check_interval)

    async def _collect_metrics(
        self, deployment: CanaryDeploymentConfig
    ) -> dict[str, float]:
        """Collect current metrics for canary deployment.

        Args:
            deployment: Deployment configuration

        Returns:
            Current metrics
        """
        # TODO: Integrate with real monitoring systems
        # This method should be replaced with actual metrics collection from:
        # - Application Performance Monitoring (APM) tools (e.g., DataDog, New Relic)
        # - Qdrant collection metrics via self.qdrant.get_collection_stats()
        # - Load balancer metrics
        # - Application logs analysis
        # - Custom metrics endpoints

        try:
            # Example: Query actual Qdrant metrics for the new collection
            if self.qdrant and deployment.new_collection:
                # TODO: Extract actual latency and error metrics from collection stats
                # For now, we'll skip this and use simulated metrics
                pass

            # TODO: Query external monitoring systems
            # - APM metrics for latency percentiles
            # - Error rates from application logs
            # - Network metrics from load balancers

            # Temporary simulation for development/testing
            return await self._simulate_metrics(deployment)

        except Exception as e:
            logger.warning(f"Failed to collect real metrics, using simulated: {e}")
            return await self._simulate_metrics(deployment)

    async def _simulate_metrics(
        self, deployment: CanaryDeploymentConfig
    ) -> dict[str, float]:
        """Simulate metrics for development/testing purposes.

        TODO: Remove this method once real metrics integration is complete.
        """
        if deployment.metrics.stage_start_time:
            # Simulate improving metrics over time
            stage_duration = time.time() - deployment.metrics.stage_start_time

            # Base metrics that improve over time (first few minutes may be unstable)
            base_latency = (
                120 if stage_duration < 300 else 100
            )  # Higher latency in first 5 minutes
            base_error_rate = (
                0.03 if stage_duration < 300 else 0.02
            )  # Higher errors initially

            # Add some random variation
            import random

            latency = base_latency + random.uniform(-20, 20)
            error_rate = max(0, base_error_rate + random.uniform(-0.01, 0.01))

            return {
                "latency": latency,
                "error_rate": error_rate,
                "timestamp": time.time(),
            }

        return {"latency": 100.0, "error_rate": 0.01, "timestamp": time.time()}

    def _check_health(self, deployment: CanaryDeploymentConfig) -> bool:
        """Check if canary is healthy.

        Args:
            deployment: Deployment configuration

        Returns:
            True if healthy
        """
        metrics = deployment.metrics

        if not metrics.error_rate or not metrics.latency:
            return True

        # Check recent metrics (last 10 samples)
        recent_errors = metrics.error_rate[-10:]
        recent_latency = metrics.latency[-10:]

        if not recent_errors or not recent_latency:
            return True

        # Calculate averages
        avg_error_rate = sum(recent_errors) / len(recent_errors)
        avg_latency = sum(recent_latency) / len(recent_latency)

        # Get current stage thresholds
        current_stage = deployment.stages[deployment.current_stage]

        # Check against thresholds
        if avg_error_rate > current_stage.error_threshold:
            logger.error(
                f"Average error rate {avg_error_rate} exceeds threshold {current_stage.error_threshold}"
            )
            return False

        if avg_latency > current_stage.latency_threshold:
            logger.error(
                f"Average latency {avg_latency}ms exceeds threshold {current_stage.latency_threshold}ms"
            )
            return False

        return True

    async def _rollback_canary(self, deployment_id: str) -> None:
        """Rollback canary deployment.

        Args:
            deployment_id: ID of the deployment to rollback
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return

        logger.warning(f"Rolling back canary deployment {deployment_id}")

        try:
            # Ensure alias points to old collection
            current_collection = await self.aliases.get_collection_for_alias(
                deployment.alias
            )
            if current_collection != deployment.old_collection:
                await self.aliases.switch_alias(
                    alias_name=deployment.alias,
                    new_collection=deployment.old_collection,
                )

            deployment.status = "rolled_back"
            await self._save_deployments()
            logger.info(f"Canary deployment {deployment_id} rolled back")

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            deployment.status = "rollback_failed"
            await self._save_deployments()

    async def get_deployment_status(self, deployment_id: str) -> dict[str, Any]:
        """Get current status of canary deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            Status information
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return {"status": "not_found"}

        current_stage = deployment.stages[deployment.current_stage]

        status = {
            "deployment_id": deployment_id,
            "status": deployment.status,
            "alias": deployment.alias,
            "old_collection": deployment.old_collection,
            "new_collection": deployment.new_collection,
            "current_stage": deployment.current_stage,
            "total_stages": len(deployment.stages),
            "current_percentage": current_stage.percentage,
            "duration_minutes": (time.time() - deployment.start_time) / 60,
        }

        # Add metrics summary
        if deployment.metrics.latency:
            status["avg_latency"] = sum(deployment.metrics.latency) / len(
                deployment.metrics.latency
            )
            status["last_latency"] = deployment.metrics.latency[-1]

        if deployment.metrics.error_rate:
            status["avg_error_rate"] = sum(deployment.metrics.error_rate) / len(
                deployment.metrics.error_rate
            )
            status["last_error_rate"] = deployment.metrics.error_rate[-1]

        return status

    async def pause_deployment(self, deployment_id: str) -> bool:
        """Pause a canary deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            True if paused successfully
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment or deployment.status != "running":
            return False

        deployment.status = "paused"
        await self._save_deployments()
        logger.info(f"Paused canary deployment {deployment_id}")
        return True

    async def resume_deployment(self, deployment_id: str) -> bool:
        """Resume a paused canary deployment.

        Args:
            deployment_id: ID of the deployment

        Returns:
            True if resumed successfully
        """
        deployment = self.deployments.get(deployment_id)
        if not deployment or deployment.status != "paused":
            return False

        deployment.status = "running"
        await self._save_deployments()

        # TODO: For production, use persistent task queue to resume deployment monitoring
        # Same concerns as start_canary - deployments should survive server restarts

        # Restart monitoring from current stage
        # Store task reference to avoid RUF006 warning
        _ = asyncio.create_task(self._run_canary(deployment_id))  # noqa: RUF006
        logger.info(f"Resumed canary deployment {deployment_id}")
        return True

    def get_active_deployments(self) -> list[dict[str, Any]]:
        """Get list of active canary deployments.

        Returns:
            List of active deployment summaries
        """
        active = []
        for dep_id, deployment in self.deployments.items():
            if deployment.status in ["running", "paused"]:
                active.append(
                    {
                        "id": dep_id,
                        "alias": deployment.alias,
                        "status": deployment.status,
                        "current_stage": deployment.current_stage,
                        "current_percentage": deployment.stages[
                            deployment.current_stage
                        ].percentage,
                    }
                )
        return active

    async def _load_deployments(self) -> None:
        """Load deployments from persistent storage."""
        try:
            # First try Redis if available
            if self.client_manager and await self._load_from_redis():
                logger.info("Loaded canary deployments from Redis")
                return

            # Fall back to file storage
            if await self._load_from_file():
                logger.info("Loaded canary deployments from file")
                return

            logger.info("No existing canary deployments found")

        except Exception as e:
            logger.error(f"Failed to load canary deployments: {e}")
            # Continue with empty deployments dict

    async def _save_deployments(self) -> None:
        """Save deployments to persistent storage."""
        async with self._lock:
            try:
                # Try Redis first if available
                if self.client_manager and await self._save_to_redis():
                    logger.debug("Saved canary deployments to Redis")

                # Always save to file as backup
                await self._save_to_file()
                logger.debug("Saved canary deployments to file")

            except Exception as e:
                logger.error(f"Failed to save canary deployments: {e}")

    async def _load_from_redis(self) -> bool:
        """Load deployments from Redis."""
        try:
            redis_client = await self.client_manager.get_redis_client()
            data = await redis_client.get("canary_deployments")

            if not data:
                return False

            deployments_data = json.loads(data)
            self.deployments = self._deserialize_deployments(deployments_data)
            return True

        except Exception as e:
            logger.warning(f"Failed to load canary deployments from Redis: {e}")
            return False

    async def _save_to_redis(self) -> bool:
        """Save deployments to Redis."""
        try:
            redis_client = await self.client_manager.get_redis_client()
            deployments_data = self._serialize_deployments()

            # Save with TTL of 30 days
            await redis_client.setex(
                "canary_deployments", 30 * 24 * 3600, json.dumps(deployments_data)
            )
            return True

        except Exception as e:
            logger.warning(f"Failed to save canary deployments to Redis: {e}")
            return False

    async def _load_from_file(self) -> bool:
        """Load deployments from file."""
        try:
            if not self._state_file.exists():
                return False

            with open(self._state_file) as f:
                deployments_data = json.load(f)

            self.deployments = self._deserialize_deployments(deployments_data)
            return True

        except Exception as e:
            logger.warning(f"Failed to load canary deployments from file: {e}")
            return False

    async def _save_to_file(self) -> None:
        """Save deployments to file."""
        # Ensure parent directory exists
        self._state_file.parent.mkdir(parents=True, exist_ok=True)

        deployments_data = self._serialize_deployments()

        # Atomic write using temporary file
        temp_file = self._state_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(deployments_data, f, indent=2)

        # Atomic move
        temp_file.replace(self._state_file)

    def _serialize_deployments(self) -> dict[str, Any]:
        """Serialize deployments to JSON-compatible format."""
        serialized = {}

        for dep_id, deployment in self.deployments.items():
            # Convert dataclass to dict
            deployment_dict = asdict(deployment)

            # Handle Path objects by converting to strings
            if "stages" in deployment_dict:
                for _stage in deployment_dict["stages"]:
                    # Ensure all values are JSON serializable
                    pass

            serialized[dep_id] = deployment_dict

        return serialized

    def _deserialize_deployments(
        self, data: dict[str, Any]
    ) -> dict[str, CanaryDeploymentConfig]:
        """Deserialize deployments from JSON format."""
        deployments = {}

        for dep_id, dep_data in data.items():
            # Reconstruct stages
            stages = [CanaryStage(**stage_data) for stage_data in dep_data["stages"]]

            # Reconstruct metrics
            metrics_data = dep_data["metrics"]
            metrics = CanaryMetrics(
                latency=metrics_data["latency"],
                error_rate=metrics_data["error_rate"],
                success_count=metrics_data["success_count"],
                error_count=metrics_data["error_count"],
                stage_start_time=metrics_data["stage_start_time"],
            )

            # Reconstruct deployment config
            deployment = CanaryDeploymentConfig(
                alias=dep_data["alias"],
                old_collection=dep_data["old_collection"],
                new_collection=dep_data["new_collection"],
                stages=stages,
                current_stage=dep_data["current_stage"],
                metrics=metrics,
                start_time=dep_data["start_time"],
                status=dep_data["status"],
            )

            deployments[dep_id] = deployment

        return deployments
