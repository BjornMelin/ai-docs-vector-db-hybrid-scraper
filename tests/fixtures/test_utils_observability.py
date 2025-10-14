"""Test utilities for observability integration testing.

This module provides helper functions and classes for common test operations
in observability integration tests, including span manipulation, metric aggregation,
and log correlation utilities.
"""

import asyncio
import random
import time
import uuid
from typing import Any

from tests.fixtures.observability import LogEntry, Metric, TraceSpan


class TraceTestHelper:
    """Helper utilities for trace-related testing operations."""

    @staticmethod
    def create_child_span(
        parent_span: TraceSpan,
        service_name: str,
        operation_name: str,
        duration_ms: float = 50.0,
        tags: dict[str, Any] | None = None,
    ) -> TraceSpan:
        """Create a child span for a given parent span."""
        return TraceSpan(
            trace_id=parent_span.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span.span_id,
            service_name=service_name,
            operation_name=operation_name,
            start_time=parent_span.start_time + 0.01,  # Slight delay after parent
            end_time=parent_span.start_time + 0.01 + (duration_ms / 1000),
            tags=tags or {},
            logs=[],
        )

    @staticmethod
    def calculate_trace_duration(spans: list[TraceSpan]) -> float:
        """Calculate the total duration of a trace."""
        if not spans:
            return 0.0

        start_times = [s.start_time for s in spans if s.start_time is not None]
        end_times = [s.end_time for s in spans if s.end_time is not None]

        if not start_times or not end_times:
            return 0.0

        return max(end_times) - min(start_times)

    @staticmethod
    def get_trace_hierarchy(spans: list[TraceSpan]) -> dict[str, list[TraceSpan]]:
        """Organize spans by their hierarchical relationships."""
        hierarchy = {}
        root_spans = [s for s in spans if s.parent_span_id is None]

        for root_span in root_spans:
            hierarchy[root_span.span_id] = [root_span]
            TraceTestHelper._build_span_tree(spans, root_span.span_id, hierarchy)

        return hierarchy

    @staticmethod
    def _build_span_tree(
        all_spans: list[TraceSpan],
        parent_span_id: str,
        hierarchy: dict[str, list[TraceSpan]],
    ) -> None:
        """Recursively build the span tree for a given parent."""
        child_spans = [s for s in all_spans if s.parent_span_id == parent_span_id]
        if child_spans:
            hierarchy[parent_span_id].extend(child_spans)
            for child in child_spans:
                TraceTestHelper._build_span_tree(all_spans, child.span_id, hierarchy)

    @staticmethod
    def validate_trace_structure(spans: list[TraceSpan]) -> dict[str, Any]:
        """Validate the structure and relationships of a trace."""
        issues = []

        # Check for root spans
        root_spans = [s for s in spans if s.parent_span_id is None]
        if len(root_spans) != 1:
            issues.append(f"Expected 1 root span, found {len(root_spans)}")

        # Check span relationships
        span_ids = {s.span_id for s in spans}
        for span in spans:
            if span.parent_span_id and span.parent_span_id not in span_ids:
                issues.append(
                    f"Parent span {span.parent_span_id} not found for "
                    f"span {span.span_id}"
                )

        # Check timing
        for span in spans:
            if span.end_time and span.start_time and span.end_time < span.start_time:
                issues.append(
                    f"Invalid timing for span {span.span_id}: end before start"
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "span_count": len(spans),
            "root_spans": len(root_spans),
        }


class MetricsTestHelper:
    """Helper utilities for metrics-related testing operations."""

    @staticmethod
    def aggregate_metrics_by_name(metrics: list[Metric]) -> dict[str, list[Metric]]:
        """Group metrics by their names."""
        aggregated = {}
        for metric in metrics:
            if metric.name not in aggregated:
                aggregated[metric.name] = []
            aggregated[metric.name].append(metric)
        return aggregated

    @staticmethod
    def calculate_rate_counter(
        metrics: list[Metric], time_window_seconds: float = 60.0
    ) -> float:
        """Calculate the rate of change for counter metrics."""
        if len(metrics) < 2:
            return 0.0

        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

        # Get metrics within time window
        current_time = time.time()
        recent_metrics = [
            m
            for m in sorted_metrics
            if current_time - m.timestamp <= time_window_seconds
        ]

        if len(recent_metrics) < 2:
            return 0.0

        # Calculate rate
        first = recent_metrics[0]
        last = recent_metrics[-1]
        time_diff = last.timestamp - first.timestamp
        value_diff = last.value - first.value

        return value_diff / time_diff if time_diff > 0 else 0.0

    @staticmethod
    def aggregate_by_tag(metrics: list[Metric], tag_key: str) -> dict[str, float]:
        """Aggregate metric values by a specific tag."""
        aggregated = {}
        for metric in metrics:
            tag_value = metric.tags.get(tag_key)
            if tag_value is not None:
                if tag_value not in aggregated:
                    aggregated[tag_value] = 0.0
                aggregated[tag_value] += metric.value
        return aggregated

    @staticmethod
    def calculate_rate(metrics: list[Metric]) -> float:
        """Calculate rate of change between first and last metric in a series."""
        if len(metrics) < 2:
            return 0.0

        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        first = sorted_metrics[0]
        last = sorted_metrics[-1]

        time_diff = last.timestamp - first.timestamp
        value_diff = last.value - first.value

        return value_diff / time_diff if time_diff > 0 else 0.0

    @staticmethod
    def calculate_summary_stats(metrics: list[Metric]) -> dict[str, float]:
        """Calculate summary statistics for a list of metrics."""
        if not metrics:
            return {"count": 0, "min": 0, "max": 0, "sum": 0, "avg": 0}

        values = [m.value for m in metrics]
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "sum": sum(values),
            "avg": sum(values) / len(values),
        }

    @staticmethod
    def calculate_percentile(values: list[float], percentile: float) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = int(percentile * len(sorted_values))
        return sorted_values[min(index, len(sorted_values) - 1)]

    @staticmethod
    def extract_histogram_percentiles(
        metrics: list[Metric], percentiles: list[float] | None = None
    ) -> dict[float, float]:
        """Extract percentiles from histogram metrics."""
        if percentiles is None:
            percentiles = [0.5, 0.95, 0.99]
        if not metrics:
            return {}

        values = [m.value for m in metrics]
        return {
            p: MetricsTestHelper.calculate_percentile(values, p) for p in percentiles
        }


class LogTestHelper:
    """Helper utilities for log-related testing operations."""

    @staticmethod
    def correlate_logs_by_trace(logs: list[LogEntry], trace_id: str) -> list[LogEntry]:
        """Extract logs correlated with a specific trace."""
        return [log for log in logs if log.trace_id == trace_id]

    @staticmethod
    def correlate_logs_by_correlation_id(
        logs: list[LogEntry], correlation_id: str
    ) -> list[LogEntry]:
        """Extract logs correlated with a specific correlation ID."""
        return [log for log in logs if log.correlation_id == correlation_id]

    @staticmethod
    def group_logs_by_service(logs: list[LogEntry]) -> dict[str, list[LogEntry]]:
        """Group logs by service name."""
        grouped = {}
        for log in logs:
            if log.service not in grouped:
                grouped[log.service] = []
            grouped[log.service].append(log)
        return grouped

    @staticmethod
    def validate_log_sequence(
        logs: list[LogEntry], expected_sequence: list[str]
    ) -> dict[str, Any]:
        """Validate that logs appear in the expected sequence."""
        issues = []

        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda log_entry: log_entry.timestamp)

        # Check sequence
        log_messages = [log.message for log in sorted_logs]
        for i, expected_msg in enumerate(expected_sequence):
            if i >= len(log_messages) or expected_msg not in log_messages[i]:
                issues.append(
                    f"Expected '{expected_msg}' at position {i}, got "
                    f"'{log_messages[i] if i < len(log_messages) else 'EOF'}'"
                )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "actual_sequence": log_messages,
            "expected_sequence": expected_sequence,
        }


class AsyncTestHelper:
    """Helper utilities for async testing operations."""

    @staticmethod
    async def simulate_async_operation(
        duration_ms: float, operation_name: str = "async_operation"
    ) -> dict[str, Any]:
        """Simulate an async operation with timing."""
        start_time = time.time()
        await asyncio.sleep(duration_ms / 1000)
        end_time = time.time()

        return {
            "operation": operation_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": (end_time - start_time) * 1000,
        }

    @staticmethod
    async def run_concurrent_operations(
        operations: list[dict[str, Any]],
    ) -> list[dict[str, Any] | BaseException]:
        """Run multiple operations concurrently."""
        tasks = []
        for op in operations:
            duration = op.get("duration_ms", 100)
            name = op.get("name", "operation")
            tasks.append(AsyncTestHelper.simulate_async_operation(duration, name))

        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def simulate_service_interaction(
        service_name: str,
        operation: str,
        success_rate: float = 1.0,
        avg_duration_ms: float = 100,
        variance_ms: float = 20,
    ) -> dict[str, Any]:
        """Simulate a service interaction with configurable success rate and timing."""
        # Determine success
        success = random.random() < success_rate

        # Calculate duration with variance
        duration_ms = avg_duration_ms + random.uniform(-variance_ms, variance_ms)
        duration_ms = max(1, duration_ms)  # Minimum 1ms

        # Simulate the operation
        await asyncio.sleep(duration_ms / 1000)

        result = {
            "service": service_name,
            "operation": operation,
            "success": success,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }

        if not success:
            result["error"] = f"Simulated failure in {service_name}.{operation}"

        return result


class ObservabilityTestAssertions:
    """Common assertions for observability testing."""

    @staticmethod
    def assert_trace_hierarchy(
        spans: list[TraceSpan],
        expected_root_service: str,
        expected_child_services: list[str],
    ) -> None:
        """Assert that a trace has the expected hierarchical structure."""
        root_spans = [s for s in spans if s.parent_span_id is None]
        assert len(root_spans) == 1, f"Expected 1 root span, got {len(root_spans)}"

        root_span = root_spans[0]
        assert root_span.service_name == expected_root_service

        child_spans = [s for s in spans if s.parent_span_id == root_span.span_id]
        actual_child_services = {s.service_name for s in child_spans}

        assert actual_child_services == set(expected_child_services), (
            f"Expected child services {expected_child_services}, got "
            f"{list(actual_child_services)}"
        )

    @staticmethod
    def assert_metric_exists(
        metrics: list[Metric], name: str, metric_type: str, min_count: int = 1
    ) -> list[Metric]:
        """Assert that metrics with given name and type exist."""
        matching_metrics = [
            m for m in metrics if m.name == name and m.metric_type == metric_type
        ]

        assert len(matching_metrics) >= min_count, (
            f"Expected at least {min_count} {metric_type} metrics named "
            f"'{name}', got {len(matching_metrics)}"
        )

        return matching_metrics

    @staticmethod
    def assert_logs_correlated(
        logs: list[LogEntry], correlation_id: str, expected_services: list[str]
    ) -> None:
        """Assert that logs are properly correlated across services."""
        correlated_logs = LogTestHelper.correlate_logs_by_correlation_id(
            logs, correlation_id
        )

        assert len(correlated_logs) > 0, (
            f"No logs found for correlation ID {correlation_id}"
        )

        actual_services = {log.service for log in correlated_logs}
        assert actual_services == set(expected_services), (
            f"Expected logs from services {expected_services}, got "
            f"{list(actual_services)}"
        )

    @staticmethod
    def assert_no_error_logs(logs: list[LogEntry]) -> None:
        """Assert that no error logs exist."""
        error_logs = [log for log in logs if log.level == "ERROR"]
        assert len(error_logs) == 0, (
            f"Found {len(error_logs)} error logs: {[log.message for log in error_logs]}"
        )

    @staticmethod
    def assert_service_metrics_complete(
        metrics: list[Metric], service_name: str
    ) -> None:
        """Assert that a service has complete metrics coverage."""
        service_metrics = [m for m in metrics if m.tags.get("service") == service_name]

        # Should have at least one metric
        assert len(service_metrics) > 0, f"No metrics found for service {service_name}"

        # Should have different metric types
        metric_types = {m.metric_type for m in service_metrics}
        assert len(metric_types) >= 1, (
            f"Service {service_name} should have at least one metric type"
        )
