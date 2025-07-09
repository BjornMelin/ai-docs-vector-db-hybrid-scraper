#!/usr/bin/env python3
"""Test POA API endpoints."""

import asyncio
import logging
from typing import Any

import httpx


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/optimization"


async def test_endpoint(
    client: httpx.AsyncClient, method: str, path: str, **kwargs
) -> dict[str, Any]:
    """Test a single endpoint."""
    try:
        response = await client.request(method, f"{BASE_URL}{path}", **kwargs)
        response.raise_for_status()
        return {"status": "success", "data": response.json()}
    except httpx.HTTPStatusError as e:
        return {"status": "error", "error": str(e), "detail": e.response.text}
    except Exception as e:
        return {"status": "error", "error": str(e)}


async def main():
    """Test all POA API endpoints."""
    logger.info("Testing POA API endpoints...")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test GET /status
        logger.info("\n=== Testing GET /optimization/status ===")
        result = await test_endpoint(client, "GET", "/status")
        logger.info(f"Result: {result}")

        # Test POST /start
        logger.info("\n=== Testing POST /optimization/start ===")
        result = await test_endpoint(client, "POST", "/start")
        logger.info(f"Result: {result}")

        # Wait for POA to initialize
        await asyncio.sleep(2)

        # Test GET /metrics/current
        logger.info("\n=== Testing GET /optimization/metrics/current ===")
        result = await test_endpoint(client, "GET", "/metrics/current")
        logger.info(f"Result: {result}")

        # Test GET /metrics/trends
        logger.info("\n=== Testing GET /optimization/metrics/trends?minutes=5 ===")
        result = await test_endpoint(
            client, "GET", "/metrics/trends", params={"minutes": 5}
        )
        logger.info(f"Result: {result}")

        # Test GET /recommendations
        logger.info("\n=== Testing GET /optimization/recommendations ===")
        result = await test_endpoint(client, "GET", "/recommendations")
        logger.info(f"Result: {result}")

        # Test GET /history
        logger.info("\n=== Testing GET /optimization/history?limit=10 ===")
        result = await test_endpoint(client, "GET", "/history", params={"limit": 10})
        logger.info(f"Result: {result}")

        # Test POST /trigger/{optimization_type}
        logger.info("\n=== Testing POST /optimization/trigger/cache_ttl ===")
        result = await test_endpoint(client, "POST", "/trigger/cache_ttl")
        logger.info(f"Result: {result}")

        # Test streaming benchmark endpoint
        logger.info("\n=== Testing GET /optimization/benchmark/run (streaming) ===")
        try:
            async with client.stream("GET", f"{BASE_URL}/benchmark/run") as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        logger.info(f"Benchmark: {line}")
        except Exception as e:
            logger.error(f"Benchmark error: {e}")

        # Test POST /stop
        logger.info("\n=== Testing POST /optimization/stop ===")
        result = await test_endpoint(client, "POST", "/stop")
        logger.info(f"Result: {result}")

        # Final status check
        logger.info("\n=== Final Status Check ===")
        result = await test_endpoint(client, "GET", "/status")
        logger.info(f"Final status: {result}")


if __name__ == "__main__":
    asyncio.run(main())
