"""Database Performance Optimization Module.

This module provides specific optimizations for Qdrant vector database
including index tuning, query optimization, and connection management.
"""

import asyncio
import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    HnswConfigDiff,
    OptimizersConfigDiff,
)

from src.config.settings import Settings


logger = logging.getLogger(__name__)


class QdrantOptimizer:
    """Qdrant-specific performance optimizer."""

    def __init__(self, client: QdrantClient, settings: Settings):
        """Initialize Qdrant optimizer.

        Args:
            client: Qdrant client instance
            settings: Application settings

        """
        self.client = client
        self.settings = settings
        self._optimization_history: list[dict[str, Any]] = []

    async def optimize_collection_index(
        self,
        collection_name: str,
        target_recall: float = 0.95,  # noqa: ARG002
        target_speed: str = "balanced",  # fast, balanced, accurate
    ) -> dict[str, Any]:
        """Optimize HNSW index parameters for a collection.

        Args:
            collection_name: Name of the collection to optimize
            target_recall: Target recall rate (0-1)
            target_speed: Speed vs accuracy tradeoff

        Returns:
            Dictionary with optimization results

        """
        logger.info(f"Optimizing index for collection: {collection_name}")

        # Get current collection info
        collection_info = await self.client.get_collection(collection_name)

        # Determine optimal HNSW parameters based on collection size and target
        vector_count = collection_info.vectors_count or 0

        # Optimization logic based on collection size and target speed
        if target_speed == "fast":
            m = 16  # Fewer connections for faster search
            ef_construct = 100
            ef = 100
        elif target_speed == "accurate":
            m = 32  # More connections for better accuracy
            ef_construct = 400
            ef = 200
        else:  # balanced
            m = 24
            ef_construct = 200
            ef = 150
        # Adjust based on vector count
        if vector_count > 1_000_000:
            m = min(m * 2, 64)  # Increase connections for large collections
            ef_construct = min(ef_construct * 2, 800)

        # Update HNSW config
        try:
            await self.client.update_collection(
                collection_name=collection_name,
                hnsw_config=HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct,
                    full_scan_threshold=10000,
                    max_indexing_threads=0,  # Use all available threads
                ),
            )

            optimization_result = {
                "collection": collection_name,
                "status": "success",
                "parameters": {
                    "m": m,
                    "ef_construct": ef_construct,
                    "ef": ef,
                },
                "vector_count": vector_count,
                "optimization_type": target_speed,
            }

            self._optimization_history.append(optimization_result)
            logger.info(
                f"Successfully optimized {collection_name} with params: {optimization_result['parameters']}"
            )

            return optimization_result  # noqa: TRY300

        except Exception as e:
            logger.exception(f"Failed to optimize collection {collection_name}")
            return {
                "collection": collection_name,
                "status": "failed",
                "error": str(e),
            }

    async def optimize_search_params(
        self,
        collection_name: str,
        query_vector: list[float],
        target_latency_ms: float = 100,
    ) -> dict[str, Any]:
        """Optimize search parameters for target latency.

        Args:
            collection_name: Collection to search
            query_vector: Query vector
            target_latency_ms: Target latency in milliseconds

        Returns:
            Optimized search parameters

        """
        # Start with conservative parameters
        limit = 10
        ef = 100

        # Test different ef values to find optimal setting
        best_params = {"limit": limit, "ef": ef}
        best_latency = float("inf")

        for test_ef in [50, 100, 150, 200]:
            start_time = asyncio.get_event_loop().time()

            try:
                await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    search_params={"hnsw_ef": test_ef},
                )

                latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000

                if latency_ms < target_latency_ms and latency_ms < best_latency:
                    best_latency = latency_ms
                    best_params["ef"] = test_ef

                if latency_ms > target_latency_ms:
                    break  # Don't test higher values

            except Exception as e:  # noqa: BLE001
                logger.warning(f"Search test failed with ef={test_ef}: {e}")

        return {
            "search_params": best_params,
            "expected_latency_ms": best_latency,
            "target_latency_ms": target_latency_ms,
        }

    async def enable_collection_optimizers(
        self, collection_name: str, indexing_threshold: int = 10000
    ) -> dict[str, Any]:
        """Enable automatic optimizers for a collection.

        Args:
            collection_name: Collection name
            indexing_threshold: Threshold for automatic indexing

        Returns:
            Optimizer configuration result

        """
        try:
            await self.client.update_collection(
                collection_name=collection_name,
                optimizers_config=OptimizersConfigDiff(
                    deleted_threshold=0.2,  # Vacuum when 20% deleted
                    vacuum_min_vector_number=1000,  # Min vectors before vacuum
                    default_segment_number=4,  # Parallel segments
                    max_segment_size=200_000,  # Max vectors per segment
                    memmap_threshold=50_000,  # Use memory mapping for large segments
                    indexing_threshold=indexing_threshold,
                    flush_interval_sec=60,  # Flush to disk every minute
                ),
            )

            return {  # noqa: TRY300
                "collection": collection_name,
                "status": "optimizers_enabled",
                "config": {
                    "deleted_threshold": 0.2,
                    "vacuum_min_vector_number": 1000,
                    "default_segment_number": 4,
                    "max_segment_size": 200_000,
                    "indexing_threshold": indexing_threshold,
                },
            }

        except Exception as e:
            logger.exception("Failed to enable optimizers")
            return {"status": "failed", "error": str(e)}

    async def get_optimization_report(self) -> dict[str, Any]:
        """Get report of all optimizations performed.

        Returns:
            Optimization history and recommendations

        """
        return {
            "optimization_history": self._optimization_history,
            "total_optimizations": len(self._optimization_history),
            "recommendations": await self._generate_recommendations(),
        }

    async def _generate_recommendations(self) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check all collections
        collections = await self.client.get_collections()

        for collection in collections.collections:
            info = await self.client.get_collection(collection.name)

            # Large collection recommendations
            if info.vectors_count and info.vectors_count > 100_000:
                recommendations.append(
                    f"Collection '{collection.name}' has {info.vectors_count} vectors. "
                    "Consider using quantization to reduce memory usage."
                )

            # Index recommendations
            if info.config.hnsw_config.m < 16:
                recommendations.append(
                    f"Collection '{collection.name}' has low HNSW m value ({info.config.hnsw_config.m}). "
                    "Consider increasing for better recall."
                )

        return recommendations
