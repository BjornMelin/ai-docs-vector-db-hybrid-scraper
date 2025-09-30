"""Targeted tests for agentic vector manager configuration models."""

from __future__ import annotations

from datetime import UTC, datetime

from src.services.vector_db.agentic_manager import (
    AgentCollectionConfig,
    CollectionPerformanceMetrics,
    CollectionType,
    OptimizationStrategy,
)


def test_agent_collection_config_defaults() -> None:
    """Default agentic collection settings prioritize balanced performance."""
    config = AgentCollectionConfig.model_validate(
        {"agent_id": "agent-123", "collection_type": CollectionType.MEMORY}
    )
    assert config.optimization_strategy is OptimizationStrategy.BALANCED
    assert config.enable_auto_optimization is True
    assert config.vector_dimension == 1536


def test_agent_collection_config_customization() -> None:
    """Agent collections should capture hardware-aware tuning hints."""
    config = AgentCollectionConfig.model_validate(
        {
            "agent_id": "agent-456",
            "collection_type": CollectionType.REASONING,
            "vector_dimension": 768,
            "optimization_strategy": OptimizationStrategy.SPEED_OPTIMIZED,
            "memory_limit_gb": 2.0,
            "enable_quantization": True,
        }
    )
    assert config.optimization_strategy is OptimizationStrategy.SPEED_OPTIMIZED
    assert config.memory_limit_gb == 2.0
    assert config.enable_quantization is True


def test_collection_performance_metrics_shape() -> None:
    """Performance metrics capture latency and throughput details."""
    metrics = CollectionPerformanceMetrics(
        collection_name="agent_memory",
        total_points=1000,
        avg_query_latency_ms=12.5,
        p95_query_latency_ms=20.0,
        throughput_qps=150.0,
        memory_usage_mb=512.0,
        disk_usage_mb=2048.0,
        accuracy_score=0.93,
        last_optimized=datetime.now(UTC),
    )
    assert metrics.collection_name == "agent_memory"
    assert metrics.avg_query_latency_ms < metrics.p95_query_latency_ms
    assert metrics.accuracy_score <= 1.0
