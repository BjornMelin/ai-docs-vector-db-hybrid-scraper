#!/usr/bin/env python3
"""CLI script to run the ARQ task queue worker."""

import asyncio
import logging
import sys

import click
from arq import run_worker

from src.config.settings import get_config
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

    # Load config
    config = get_config()

    # Update worker settings
    WorkerSettings.max_jobs = max_jobs
    WorkerSettings.queue_name = queue

    # Get Redis settings
    redis_settings = WorkerSettings.get_redis_settings()

    logger.info(
        f"Worker configuration: workers={workers}, max_jobs={max_jobs}, queue={queue}"
    )
    logger.info(
        f"Connecting to Redis at {redis_settings.host}:{redis_settings.port} "
        f"(database {redis_settings.database})"
    )

    # Run the worker
    asyncio.run(
        run_worker(
            WorkerSettings,
            redis_settings=redis_settings,
            ctx={"config": config},
        )
    )


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
