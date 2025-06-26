"""ARQ worker for processing background tasks."""

import logging
from typing import Any, ClassVar

from arq.connections import RedisSettings

from src.config import get_config

from .tasks import TASK_REGISTRY


logger = logging.getLogger(__name__)


class WorkerSettings:
    """ARQ worker settings."""

    # Task functions to register
    functions: ClassVar[list] = list(TASK_REGISTRY.values())

    # Cron jobs (if needed)
    cron_jobs: ClassVar[list] = [
        # Example: Run cleanup every day at 2 AM
        # cron(cleanup_old_jobs, hour=2, minute=0)
    ]

    # Worker configuration
    max_jobs = 10
    job_timeout = 3600  # 1 hour default
    max_tries = 3

    # Health check
    health_check_interval = 60
    health_check_key = "arq:health-check"

    # Queue configuration
    queue_name = "default"

    @classmethod
    def get_redis_settings(cls) -> RedisSettings:
        """Get Redis settings from config."""
        config = get_config()

        # Use the simplified config structure
        redis_url = config.get_redis_url()

        # Parse Redis URL to extract components
        # Format: redis://[password@]host:port/database
        if redis_url.startswith("redis://"):
            url_parts = redis_url[8:]  # Remove redis://
            password = None

            if "@" in url_parts:
                password, url_parts = url_parts.split("@", 1)

            if "/" in url_parts:
                host_port, database_str = url_parts.split("/", 1)
                database = int(database_str) if database_str else 0
            else:
                host_port = url_parts
                database = 0

            if ":" in host_port:
                host, port_str = host_port.split(":", 1)
                port = int(port_str)
            else:
                host = host_port
                port = 6379
        else:
            # Fallback defaults
            host = "localhost"
            port = 6379
            password = None
            database = 0

        return RedisSettings(
            host=host,
            port=port,
            password=password,
            database=database,
        )

    @classmethod
    def on_startup(cls, ctx: dict[str, Any]) -> None:
        """Called when worker starts."""
        logger.info("ARQ worker starting up")

    @classmethod
    def on_shutdown(cls, ctx: dict[str, Any]) -> None:
        """Called when worker shuts down."""
        logger.info("ARQ worker shutting down")

    @classmethod
    def on_job_start(cls, ctx: dict[str, Any]) -> None:
        """Called when a job starts."""
        job_id = ctx.get("job_id")
        function = ctx.get("job_try", {}).get("function")
        logger.info(f"Starting job {job_id}: {function}")

    @classmethod
    def on_job_end(cls, ctx: dict[str, Any]) -> None:
        """Called when a job ends."""
        job_id = ctx.get("job_id")
        function = ctx.get("job_try", {}).get("function")
        result = ctx.get("result")
        logger.info(f"Completed job {job_id}: {function} - Result: {result}")


# For ARQ to find the settings
redis_settings = WorkerSettings.get_redis_settings()
max_jobs = WorkerSettings.max_jobs
job_timeout = WorkerSettings.job_timeout
functions = WorkerSettings.functions
cron_jobs = WorkerSettings.cron_jobs
on_startup = WorkerSettings.on_startup
on_shutdown = WorkerSettings.on_shutdown
on_job_start = WorkerSettings.on_job_start
on_job_end = WorkerSettings.on_job_end
