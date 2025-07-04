"""Task queue manager for ARQ integration."""

import logging
from typing import Any

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from src.config import Config
from src.services.base import BaseService


logger = logging.getLogger(__name__)

# Import monitoring registry for Prometheus integration
try:
    from src.services.monitoring.metrics import get_metrics_registry

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False


class TaskQueueManager(BaseService):
    """Manages task queue operations using ARQ."""

    def __init__(self, config: Config):
        """Initialize task queue manager.

        Args:
            config: Unified configuration

        """
        super().__init__(config)
        self._redis_pool: ArqRedis | None = None
        self._redis_settings = self._create_redis_settings()

        # Initialize Prometheus metrics registry if available
        self.metrics_registry = None
        if MONITORING_AVAILABLE:
            try:
                self.metrics_registry = get_metrics_registry()
                logger.debug("Task queue monitoring enabled")
            except (AttributeError, ConnectionError, RuntimeError, TimeoutError):
                logger.debug("Task queue monitoring disabled")

    def _create_redis_settings(self) -> RedisSettings:
        """Create Redis settings from config."""
        # Parse redis URL to extract components
        url = self.config.task_queue.redis_url
        password = self.config.task_queue.redis_password
        database = self.config.task_queue.redis_database

        # Handle different URL formats
        url = url.removeprefix("redis://")  # Remove redis:// prefix

        # Extract host and port
        if "@" in url:
            # Has auth in URL
            auth_part, host_part = url.split("@", 1)
            if ":" in auth_part:
                _, url_password = auth_part.split(":", 1)
                # Config password takes precedence over URL password
                if not password:
                    password = url_password
            host, port = (
                host_part.split(":", 1) if ":" in host_part else (host_part, "6379")
            )
        else:
            # No auth in URL
            host, port = url.split(":", 1) if ":" in url else (url, "6379")

        return RedisSettings(
            host=host,
            port=int(port),
            password=password,
            database=database,
        )

    async def initialize(self) -> None:
        """Initialize Redis connection pool."""
        try:
            self._redis_pool = await create_pool(self._redis_settings)
            logger.info("Task queue manager initialized")
            self._initialized = True
        except (OSError, AttributeError, ConnectionError, ImportError):
            logger.exception("Failed to initialize task queue")
            raise

    async def cleanup(self) -> None:
        """Cleanup Redis connection pool."""
        if self._redis_pool:
            await self._redis_pool.close()
            self._redis_pool = None
        self._initialized = False

    async def enqueue(
        self,
        task_name: str,
        *args,
        _delay: int | None = None,
        _queue_name: str | None = None,
        **kwargs,
    ) -> str | None:
        """Enqueue a task for execution.

        Args:
            task_name: Name of the task function
            *args: Positional arguments for the task
            _delay: Delay in seconds before executing the task
            _queue_name: Queue name (defaults to config queue name)
            **kwargs: Keyword arguments for the task

        Returns:
            Job ID if successful, None otherwise

        """
        if not self._redis_pool:
            logger.error("Task queue not initialized")
            return None

        try:
            queue_name = _queue_name or self.config.task_queue.queue_name

            # Enqueue the job
            job = await self._redis_pool.enqueue_job(
                task_name, *args, _queue_name=queue_name, _defer_by=_delay, **kwargs
            )

            if job:
                logger.info(
                    f"Enqueued task {task_name} with job ID {job.job_id}"
                    f"{f' (delayed by {_delay}s)' if _delay else ''}"
                )

                # Record task enqueue metrics
                if self.metrics_registry:
                    self.metrics_registry.record_task_execution(
                        task_name, 0.0, True
                    )  # Just tracking enqueue for now

                return job.job_id
            logger.error(
                f"Failed to enqueue task {task_name}"
            )  # TODO: Convert f-string to logging format

            # Record failure metrics
            if self.metrics_registry:
                self.metrics_registry.record_task_execution(task_name, 0.0, False)

        except (TimeoutError, OSError, PermissionError):
            logger.exception("Error enqueueing task {task_name}")
            return None

        else:
            return None

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Get status of a job.

        Args:
            job_id: Job ID to check

        Returns:
            Job status information

        """
        if not self._redis_pool:
            return {"status": "error", "message": "Task queue not initialized"}

        try:
            job = await self._redis_pool.job(job_id)
            if not job:
                return {"status": "not_found", "job_id": job_id}

            return {
                "status": job.status,
                "job_id": job_id,
                "function": job.function,
                "args": job.args,
                "kwargs": job.kwargs,
                "enqueue_time": job.enqueue_time.isoformat()
                if job.enqueue_time
                else None,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "finish_time": job.finish_time.isoformat() if job.finish_time else None,
                "result": job.result if job.status == "complete" else None,
                "error": str(job.error) if job.error else None,
            }

        except Exception as e:
            logger.exception("Error getting job status")
            return {"status": "error", "message": str(e)}

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job.

        Args:
            job_id: Job ID to cancel

        Returns:
            True if cancelled successfully

        """
        if not self._redis_pool:
            return False

        try:
            job = await self._redis_pool.job(job_id)
            if job and job.status == "deferred":
                await job.abort()
                logger.info(
                    f"Cancelled job {job_id}"
                )  # TODO: Convert f-string to logging format
                return True

        except (TimeoutError, OSError, PermissionError):
            logger.exception("Error cancelling job")
            return False

        else:
            return False

    async def get_queue_stats(self, queue_name: str | None = None) -> dict[str, int]:
        """Get queue statistics.

        Args:
            queue_name: Queue name (defaults to config queue name)

        Returns:
            Queue statistics

        """
        if not self._redis_pool:
            return {"error": -1}

        queue_name = queue_name or self.config.task_queue.queue_name

        try:
            # Get job counts by status
            # ARQ stores jobs with different key patterns
            return {
                "pending": 0,
                "running": 0,
                "complete": 0,
                "failed": 0,
            }

            # This is a simplified version - in production you might want
            # to implement more detailed statistics gathering

        except (ConnectionError, OSError, PermissionError):
            logger.exception("Error getting queue stats")
            return {"error": -1}

    def get_redis_settings(self) -> RedisSettings:
        """Get Redis settings for worker configuration."""
        return self._redis_settings
