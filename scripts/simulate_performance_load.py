#!/usr/bin/env python3
"""Simulate performance load to trigger POA optimizations."""

import asyncio
import logging
import random
import time
from datetime import datetime

import psutil


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoadSimulator:
    """Simulate various load patterns to trigger optimizations."""

    def __init__(self):
        self.running = False
        self.request_count = 0
        self.cache_hits = 0
        self.cache_total = 0

    async def simulate_high_latency_requests(self, duration: int = 30):
        """Simulate requests with high latency to trigger query optimization."""
        logger.info("Starting high latency simulation...")
        end_time = time.time() + duration

        while time.time() < end_time:
            # Simulate slow request (120-150ms to exceed 100ms P95)
            delay = random.uniform(0.12, 0.15)
            await asyncio.sleep(delay)
            self.request_count += 1

            if self.request_count % 10 == 0:
                logger.info(
                    f"High latency requests: {self.request_count} (avg: {delay * 1000:.0f}ms)"
                )

    async def simulate_low_cache_hits(self, duration: int = 30):
        """Simulate low cache hit rate to trigger cache optimization."""
        logger.info("Starting low cache hit simulation...")
        end_time = time.time() + duration

        while time.time() < end_time:
            # Simulate cache miss (only 30% hit rate)
            self.cache_total += 1
            if random.random() < 0.3:
                self.cache_hits += 1

            await asyncio.sleep(0.01)  # 10ms per cache check

            if self.cache_total % 100 == 0:
                hit_rate = self.cache_hits / self.cache_total
                logger.info(
                    f"Cache operations: {self.cache_total}, Hit rate: {hit_rate:.1%}"
                )

    async def simulate_high_memory_usage(self, duration: int = 30):
        """Simulate high memory usage to trigger memory optimization."""
        logger.info("Starting high memory simulation...")
        memory_hogs = []

        try:
            # Allocate large chunks of memory
            for i in range(10):
                # Allocate 100MB chunks
                data = bytearray(100 * 1024 * 1024)
                memory_hogs.append(data)

                memory = psutil.virtual_memory()
                logger.info(
                    f"Memory usage: {memory.percent:.1f}% ({memory.used / 1024 / 1024 / 1024:.1f}GB)"
                )

                if memory.percent > 85:
                    logger.warning("Memory usage exceeds 85%!")
                    break

                await asyncio.sleep(3)

            # Hold memory for duration
            await asyncio.sleep(duration)

        finally:
            # Clean up
            memory_hogs.clear()
            logger.info("Memory simulation completed, cleaning up...")

    async def simulate_connection_saturation(self, duration: int = 30):
        """Simulate high connection count to trigger connection pool optimization."""
        logger.info("Starting connection saturation simulation...")
        connections = []

        async def mock_connection(conn_id: int):
            """Simulate a long-lived connection."""
            logger.debug(f"Connection {conn_id} established")
            await asyncio.sleep(duration)
            logger.debug(f"Connection {conn_id} closed")

        # Create many concurrent connections
        for i in range(150):  # Exceed 100 connection threshold
            conn = asyncio.create_task(mock_connection(i))
            connections.append(conn)

            if i % 50 == 0:
                logger.info(f"Active connections: {i}")

            await asyncio.sleep(0.1)

        # Wait for all connections to complete
        await asyncio.gather(*connections)
        logger.info("Connection simulation completed")

    async def run_load_scenarios(self):
        """Run various load scenarios to trigger different optimizations."""
        logger.info("Starting load simulation scenarios...")

        scenarios = [
            ("High Latency", self.simulate_high_latency_requests),
            ("Low Cache Hits", self.simulate_low_cache_hits),
            ("High Memory", self.simulate_high_memory_usage),
            ("Connection Saturation", self.simulate_connection_saturation),
        ]

        for name, scenario_func in scenarios:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Starting scenario: {name}")
            logger.info(f"{'=' * 50}")

            await scenario_func(duration=30)

            # Cool down between scenarios
            logger.info("Cooling down for 30 seconds...")
            await asyncio.sleep(30)

        logger.info("\nAll load scenarios completed!")
        logger.info(f"Total requests: {self.request_count}")
        logger.info(
            f"Cache hit rate: {self.cache_hits / self.cache_total:.1%}"
            if self.cache_total > 0
            else "N/A"
        )


async def main():
    """Main entry point."""
    simulator = LoadSimulator()

    try:
        await simulator.run_load_scenarios()
    except KeyboardInterrupt:
        logger.info("Load simulation interrupted")
    except Exception as e:
        logger.error(f"Load simulation error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
