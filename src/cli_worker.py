#!/usr/bin/env python3
"""CLI script to run the ARQ task queue worker."""

import logging
import sys

import click
from arq import run_worker

from src.services.task_queue.worker import WorkerSettings


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--workers", "-w", default=1, help="Number of worker processes", type=int)
@click.option(
    "--max-jobs", "-j", default=10, help="Maximum concurrent jobs per worker", type=int
)
@click.option(
    "--queue", "-q", default="default", help="Queue name to process", type=str
)
def main(workers: int, max_jobs: int, queue: str):
    """Run the ARQ task queue worker."""
    logger.info("Starting ARQ task queue worker...")

    # Get Redis settings
    redis_settings = WorkerSettings.get_redis_settings()

    logger.info(
        "Worker configuration: workers=%d, max_jobs=%d, queue=%s",
        workers,
        max_jobs,
        queue,
    )
    logger.info(
        "Connecting to Redis at %s:%d (database %d)",
        redis_settings.host,
        redis_settings.port,
        redis_settings.database,
    )

    # Prepare worker settings as a class with dynamic attributes
    class RuntimeWorkerSettings(WorkerSettings):
        """Worker settings with runtime overrides."""

    # Override class attributes for runtime configuration
    RuntimeWorkerSettings.max_jobs = max_jobs
    RuntimeWorkerSettings.queue_name = queue

    # Run the worker - run_worker blocks and runs the event loop internally
    # Type ignore needed because arq's type hints are overly restrictive
    run_worker(RuntimeWorkerSettings)  # type: ignore[arg-type]


if __name__ == "__main__":
    try:
        # Use default values for CLI worker
        main(workers=4, max_jobs=100, queue="default")
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
        sys.exit(0)
    except (AttributeError, OSError, PermissionError):
        logger.exception("Worker failed")
        sys.exit(1)
