"""Performance regression checks for the metrics registry."""

from __future__ import annotations

import time

import pytest
from prometheus_client import CollectorRegistry

from src.services.monitoring.metrics import MetricsConfig, MetricsRegistry


@pytest.mark.performance
def test_metrics_registry_bulk_recording_is_fast() -> None:
    """Recording a large batch of metrics should remain fast and accurate."""

    registry = MetricsRegistry(
        MetricsConfig(namespace="perf_test", enabled=True),
        registry=CollectorRegistry(),
    )

    iterations = 500
    start = time.perf_counter()
    for _ in range(iterations):
        registry.record_rag_stage_latency("docs", "retrieve", 0.01)
        registry.record_rag_answer("docs", "generated")
        registry.record_rag_error("docs", "generate", "timeout")
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"Metric recording took {elapsed:.4f}s (expected < 50ms)"

    metrics = list(registry.registry.collect())

    stage_metric = next(
        metric
        for metric in metrics
        if metric.name == "perf_test_rag_stage_latency_seconds"
    )
    stage_count = next(
        sample.value
        for sample in stage_metric.samples
        if sample.name.endswith("_count")
    )
    assert stage_count == pytest.approx(iterations)

    answers_metric = next(
        metric for metric in metrics if metric.name == "perf_test_rag_answers"
    )
    answers_total = next(
        sample.value
        for sample in answers_metric.samples
        if sample.name.endswith("_total")
    )
    assert answers_total == pytest.approx(iterations)

    errors_metric = next(
        metric for metric in metrics if metric.name == "perf_test_rag_errors"
    )
    errors_total = next(
        sample.value
        for sample in errors_metric.samples
        if sample.name.endswith("_total")
    )
    assert errors_total == pytest.approx(iterations)
