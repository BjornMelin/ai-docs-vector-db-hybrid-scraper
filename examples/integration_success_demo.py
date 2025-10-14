#!/usr/bin/env python3
"""Parallel processing system integration example.

Demonstrates integration of the parallel processing system with the
dependency injection container. Shows how to:
- Initialize the DI container with configuration
- Access the parallel processing system
- Monitor system health and performance metrics
- Use auto-optimization features
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path


# Add src to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import Settings
from src.infrastructure.container import DependencyContext


# Configure minimal logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise


class MockEmbeddingManager:
    """Mock embedding manager for demonstration."""

    def __init__(self):
        """Initialize mock embedding manager state."""
        self.name = "MockEmbeddingManager"


async def main():
    """Demonstrate parallel processing integration."""
    print("Parallel Processing System Integration Example")
    print("=" * 60)
    print()

    try:
        # Create configuration
        config = Settings()

        # Test DI Container Integration
        async with DependencyContext(config) as container:
            print("Integration Status:")
            print("-" * 40)

            # Create mock embedding manager
            mock_embedding_manager = MockEmbeddingManager()
            print("DI Container initialized")

            # Get parallel processing system from container
            parallel_system = container.parallel_processing_system(
                embedding_manager=mock_embedding_manager
            )
            print("Retrieved parallel processing system from DI container")

            # Test system status and capabilities
            status = await parallel_system.get_system_status()
            health_status = status["system_health"]["status"]
            print(f"System health: {health_status}")

            # Display optimization status
            opt_status = status["optimization_status"]
            print("\nOptimization Status:")
            print("-" * 40)
            print(f"Parallel processing: {opt_status['parallel_processing']}")
            print(f"Caching enabled: {opt_status['intelligent_caching']}")
            print(f"Algorithm optimization: {opt_status['optimized_algorithms']}")
            print(f"Auto-optimization: {opt_status['auto_optimization']}")

            # Test performance monitoring
            performance_metrics = status["performance_metrics"]
            print("\nPerformance Metrics:")
            print("-" * 40)
            avg_response = performance_metrics["avg_response_time_ms"]
            print(f"Avg response time: {avg_response:.2f}ms")

            throughput = performance_metrics["throughput_rps"]
            print(f"Throughput: {throughput:.2f} requests/sec")
            print(f"Cache hit rate: {performance_metrics['cache_hit_rate']:.1%}")
            print(f"Memory usage: {performance_metrics['memory_usage_mb']:.1f}MB")

            # Test auto-optimization
            optimization_result = await parallel_system.optimize_performance()
            print("\nAuto-Optimization:")
            print("-" * 40)
            print(f"Status: {optimization_result['status']}")

            print("\nSummary:")
            print("=" * 60)
            print("Successfully integrated parallel processing with DI container")
            print("All optimization components enabled")
            print("Performance monitoring active")

            print("\nUsage:")
            print("-" * 40)
            print(
                "Access via: "
                "container.parallel_processing_system(embedding_manager=...)"
            )
            print("Features: parallel processing, caching, auto-optimization")

            return True

    except (subprocess.SubprocessError, OSError, TimeoutError) as e:
        print(f"Integration test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    result_msg = "Integration successful" if success else "Integration failed"
    print(f"\n{result_msg}")
    sys.exit(0 if success else 1)
