#!/usr/bin/env python3
"""Start the Performance Optimization Agent service."""

import asyncio
import logging
import sys
from pathlib import Path


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import Settings
from src.infrastructure.clients.redis_client import RedisClientWrapper
from src.services.monitoring.performance_monitor import RealTimePerformanceMonitor
from src.services.performance.poa_service import PerformanceOptimizationAgent


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def main():
    """Start the POA service."""
    logger.info("Starting Performance Optimization Agent...")

    # Initialize components
    settings = Settings()
    monitor = RealTimePerformanceMonitor()

    # Try to initialize Redis client
    redis_client = None
    try:
        redis_client = RedisClientWrapper(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=True,
        )
        await redis_client.initialize()
        logger.info("Redis client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis: {e}")

    # Initialize POA
    poa = PerformanceOptimizationAgent(
        settings=settings, monitor=monitor, redis_client=redis_client
    )

    # Load baseline metrics
    baseline_path = Path("benchmarks/results/baseline.json")
    if baseline_path.exists():
        baseline_metrics = poa.benchmark.load_baseline(baseline_path)
        logger.info(f"Loaded {len(baseline_metrics)} baseline metrics")

    # Start POA
    await poa.start()

    # Get initial status
    status = await poa.get_status()
    logger.info(f"POA Status: {status['status']}")
    logger.info(f"Active optimizations: {status['active_optimizations']}")

    # Keep running
    try:
        logger.info("POA is running. Press Ctrl+C to stop...")
        while poa.running:
            await asyncio.sleep(10)

            # Log periodic status
            current_metrics = monitor.get_performance_summary()
            if current_metrics.get("status") != "no_data":
                logger.info(
                    f"Current P95: {current_metrics.get('p95_response_time', 0):.2f}ms, "
                    f"CPU: {current_metrics.get('cpu_percent', 0):.1f}%, "
                    f"Memory: {current_metrics.get('memory_percent', 0):.1f}%"
                )

    except KeyboardInterrupt:
        logger.info("Stopping POA...")
        await poa.stop()

    finally:
        # Cleanup
        if redis_client:
            await redis_client.close()
        monitor.cleanup()

        # Final status
        final_status = await poa.get_status()
        logger.info("\nFinal POA Status:")
        logger.info(f"Total optimizations: {final_status['total_optimizations']}")
        logger.info(f"Successful: {final_status['successful_optimizations']}")
        logger.info(f"Rolled back: {final_status['rolled_back']}")


if __name__ == "__main__":
    asyncio.run(main())
