"""Enterprise Observability Platform for Unified Monitoring.

This module provides observability for all enterprise features,
implementing distributed tracing, metrics collection, log aggregation,
alerting, and anomaly detection.
"""

import asyncio
import contextlib
import json
import logging
import statistics
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from src.config import Config, create_enterprise_config


# Type alias for clarity
EnterpriseConfig = Config


logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Metric types for observability."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"
    RATE = "rate"


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TraceStatus(str, Enum):
    """Distributed trace status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class Metric:
    """Individual metric data point."""

    name: str
    value: float
    metric_type: MetricType
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class TraceSpan:
    """Individual span in a distributed trace."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    operation_name: str
    service_name: str
    start_time: datetime
    end_time: datetime | None = None
    status: TraceStatus = TraceStatus.SUCCESS
    tags: dict[str, Any] = field(default_factory=dict)
    logs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        """Calculate span duration in milliseconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return None

    def finish(self, status: TraceStatus = TraceStatus.SUCCESS) -> None:
        """Finish the span."""
        self.end_time = datetime.now(tz=UTC)
        self.status = status

    def log_event(self, level: str, message: str, **kwargs) -> None:
        """Add log event to span."""
        self.logs.append(
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "level": level,
                "message": message,
                **kwargs,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "service_name": self.service_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs,
        }


@dataclass
class Alert:
    """Alert definition and state."""

    name: str
    description: str
    condition: str
    severity: AlertSeverity
    threshold: float
    metric_name: str
    service_name: str | None = None

    # Alert state
    is_active: bool = False
    last_triggered: datetime | None = None
    trigger_count: int = 0
    acknowledged: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    def should_trigger(self, metric_value: float) -> bool:
        """Check if alert should trigger based on metric value."""
        if self.condition == "greater_than":
            return metric_value > self.threshold
        if self.condition == "less_than":
            return metric_value < self.threshold
        if self.condition == "equals":
            return abs(metric_value - self.threshold) < 0.001
        if self.condition == "not_equals":
            return abs(metric_value - self.threshold) >= 0.001
        return False

    def trigger(self) -> None:
        """Trigger the alert."""
        self.is_active = True
        self.last_triggered = datetime.now(tz=UTC)
        self.trigger_count += 1
        self.acknowledged = False

    def resolve(self) -> None:
        """Resolve the alert."""
        self.is_active = False
        self.acknowledged = False

    def acknowledge(self) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True


@dataclass
class Anomaly:
    """Detected anomaly in system behavior."""

    metric_name: str
    service_name: str
    expected_value: float
    actual_value: float
    deviation_score: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    confidence: float = 0.0
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> AlertSeverity:
        """Calculate severity based on deviation score."""
        if self.deviation_score > 3.0:
            return AlertSeverity.CRITICAL
        if self.deviation_score > 2.0:
            return AlertSeverity.HIGH
        if self.deviation_score > 1.5:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW


class MetricsCollector:
    """Collects and aggregates metrics from all services."""

    def __init__(self, retention_hours: int = 24):
        self.metrics: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.retention_hours = retention_hours
        self.metric_metadata: dict[str, dict[str, Any]] = {}

    def record_metric(self, metric: Metric) -> None:
        """Record a metric data point."""
        metric_key = f"{metric.name}:{json.dumps(metric.tags, sort_keys=True)}"
        self.metrics[metric_key].append(metric)

        # Store metadata
        if metric_key not in self.metric_metadata:
            self.metric_metadata[metric_key] = {
                "name": metric.name,
                "type": metric.metric_type.value,
                "tags": metric.tags,
                "first_seen": metric.timestamp.isoformat(),
                "sample_count": 0,
            }

        self.metric_metadata[metric_key]["sample_count"] += 1
        self.metric_metadata[metric_key]["last_seen"] = metric.timestamp.isoformat()

    def get_metric_values(
        self,
        metric_name: str,
        tags: dict[str, str] | None = None,
        since: datetime | None = None,
    ) -> list[float]:
        """Get metric values matching criteria."""
        if tags is None:
            tags = {}

        metric_key = f"{metric_name}:{json.dumps(tags, sort_keys=True)}"

        if metric_key not in self.metrics:
            return []

        metrics = self.metrics[metric_key]

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        return [m.value for m in metrics]

    def get_metric_statistics(
        self, metric_name: str, tags: dict[str, str] | None = None
    ) -> dict[str, float]:
        """Get statistical summary of metric."""
        values = self.get_metric_values(metric_name, tags)

        if not values:
            return {}

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "p95": np.percentile(values, 95) if values else 0.0,
            "p99": np.percentile(values, 99) if values else 0.0,
        }

    def cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = datetime.now(tz=UTC) - timedelta(hours=self.retention_hours)

        for metric_key in list(self.metrics.keys()):
            metric_deque = self.metrics[metric_key]

            # Remove old metrics
            while metric_deque and metric_deque[0].timestamp < cutoff_time:
                metric_deque.popleft()

            # Remove empty deques
            if not metric_deque:
                del self.metrics[metric_key]
                if metric_key in self.metric_metadata:
                    del self.metric_metadata[metric_key]


class DistributedTracer:
    """Distributed tracing for enterprise services."""

    def __init__(self, retention_hours: int = 24):
        self.active_traces: dict[str, list[TraceSpan]] = {}
        self.completed_traces: dict[str, list[TraceSpan]] = {}
        self.retention_hours = retention_hours

    def start_trace(
        self, trace_id: str, operation_name: str, service_name: str
    ) -> TraceSpan:
        """Start a new trace."""
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=None,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.now(tz=UTC),
        )

        if trace_id not in self.active_traces:
            self.active_traces[trace_id] = []

        self.active_traces[trace_id].append(span)
        return span

    def start_span(
        self,
        trace_id: str,
        operation_name: str,
        service_name: str,
        parent_span_id: str | None = None,
    ) -> TraceSpan:
        """Start a new span within an existing trace."""
        span = TraceSpan(
            trace_id=trace_id,
            span_id=self._generate_span_id(),
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=service_name,
            start_time=datetime.now(tz=UTC),
        )

        if trace_id not in self.active_traces:
            self.active_traces[trace_id] = []

        self.active_traces[trace_id].append(span)
        return span

    def finish_span(
        self, span: TraceSpan, status: TraceStatus = TraceStatus.SUCCESS
    ) -> None:
        """Finish a span."""
        span.finish(status)

        # Check if trace is complete
        if span.trace_id in self.active_traces:
            active_spans = self.active_traces[span.trace_id]
            if all(s.end_time is not None for s in active_spans):
                # Move to completed traces
                self.completed_traces[span.trace_id] = active_spans
                del self.active_traces[span.trace_id]

    def get_trace(self, trace_id: str) -> list[TraceSpan] | None:
        """Get trace by ID."""
        if trace_id in self.active_traces:
            return self.active_traces[trace_id]
        return self.completed_traces.get(trace_id)

    def get_trace_summary(self, trace_id: str) -> dict[str, Any] | None:
        """Get trace summary statistics."""
        spans = self.get_trace(trace_id)
        if not spans:
            return None

        completed_spans = [s for s in spans if s.end_time is not None]

        if not completed_spans:
            return {
                "trace_id": trace_id,
                "status": "in_progress",
                "span_count": len(spans),
                "services": list({s.service_name for s in spans}),
            }

        total_duration = max(s.end_time for s in completed_spans) - min(
            s.start_time for s in spans
        )

        return {
            "trace_id": trace_id,
            "status": "completed",
            "span_count": len(spans),
            "total_duration_ms": total_duration.total_seconds() * 1000,
            "services": list({s.service_name for s in spans}),
            "errors": sum(1 for s in completed_spans if s.status == TraceStatus.ERROR),
            "root_operation": spans[0].operation_name if spans else None,
        }

    def _generate_span_id(self) -> str:
        """Generate unique span ID."""
        return str(uuid.uuid4())[:16]

    def cleanup_old_traces(self) -> None:
        """Remove traces older than retention period."""
        cutoff_time = datetime.now(tz=UTC) - timedelta(hours=self.retention_hours)

        # Clean completed traces
        for trace_id in list(self.completed_traces.keys()):
            spans = self.completed_traces[trace_id]
            if all(s.end_time and s.end_time < cutoff_time for s in spans):
                del self.completed_traces[trace_id]

        # Clean stale active traces
        for trace_id in list(self.active_traces.keys()):
            spans = self.active_traces[trace_id]
            if all(s.start_time < cutoff_time for s in spans):
                del self.active_traces[trace_id]


class AnomalyDetector:
    """AI-powered anomaly detection for metrics."""

    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
        self.baselines: dict[str, dict[str, float]] = {}
        self.history_window = 100  # Number of samples for baseline

    def update_baseline(self, metric_name: str, values: list[float]) -> None:
        """Update baseline statistics for a metric."""
        if len(values) < 2:
            return

        self.baselines[metric_name] = {
            "mean": statistics.mean(values),
            "std_dev": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "sample_count": len(values),
        }

    def detect_anomaly(
        self, metric_name: str, value: float, service_name: str = "unknown"
    ) -> Anomaly | None:
        """Detect if a metric value is anomalous."""
        if metric_name not in self.baselines:
            return None

        baseline = self.baselines[metric_name]

        if baseline["std_dev"] == 0:
            # No variation in baseline
            if value != baseline["mean"]:
                return Anomaly(
                    metric_name=metric_name,
                    service_name=service_name,
                    expected_value=baseline["mean"],
                    actual_value=value,
                    deviation_score=float("inf"),
                    confidence=1.0,
                )
            return None

        # Calculate Z-score
        z_score = abs(value - baseline["mean"]) / baseline["std_dev"]

        if z_score > self.sensitivity:
            confidence = min(z_score / (self.sensitivity * 2), 1.0)

            return Anomaly(
                metric_name=metric_name,
                service_name=service_name,
                expected_value=baseline["mean"],
                actual_value=value,
                deviation_score=z_score,
                confidence=confidence,
                context={
                    "baseline_mean": baseline["mean"],
                    "baseline_std_dev": baseline["std_dev"],
                    "z_score": z_score,
                    "threshold": self.sensitivity,
                },
            )

        return None

    def get_baseline_info(self) -> dict[str, dict[str, float]]:
        """Get current baseline information."""
        return self.baselines.copy()


class AlertManager:
    """Manages alerts and notifications."""

    def __init__(self):
        self.alerts: dict[str, Alert] = {}
        self.alert_history: list[dict[str, Any]] = []
        self.notification_handlers: list[Callable[[Alert], None]] = []

    def add_alert(self, alert: Alert) -> None:
        """Add an alert definition."""
        self.alerts[alert.name] = alert
        logger.info("Added alert: %s", alert.name)

    def remove_alert(self, alert_name: str) -> None:
        """Remove an alert definition."""
        if alert_name in self.alerts:
            del self.alerts[alert_name]
            logger.info("Removed alert: %s", alert_name)

    def check_alerts(self, metrics_collector: MetricsCollector) -> list[Alert]:
        """Check all alerts against current metrics."""
        triggered_alerts = []

        for alert in self.alerts.values():
            try:
                # Get recent metric values
                recent_values = metrics_collector.get_metric_values(
                    alert.metric_name, since=datetime.now(tz=UTC) - timedelta(minutes=5)
                )

                if not recent_values:
                    continue

                # Check latest value
                latest_value = recent_values[-1]

                if alert.should_trigger(latest_value):
                    if not alert.is_active:
                        alert.trigger()
                        triggered_alerts.append(alert)
                        self._record_alert_event(alert, "triggered", latest_value)
                        self._notify_handlers(alert)
                elif alert.is_active:
                    alert.resolve()
                    self._record_alert_event(alert, "resolved", latest_value)

            except Exception:
                logger.exception("Error checking alert %s", alert.name)

        return triggered_alerts

    def add_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)

    def get_active_alerts(self) -> list[Alert]:
        """Get all currently active alerts."""
        return [alert for alert in self.alerts.values() if alert.is_active]

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert system summary."""
        active_alerts = self.get_active_alerts()

        return {
            "total_alerts": len(self.alerts),
            "active_alerts": len(active_alerts),
            "critical_alerts": sum(
                1 for a in active_alerts if a.severity == AlertSeverity.CRITICAL
            ),
            "unacknowledged_alerts": sum(
                1 for a in active_alerts if not a.acknowledged
            ),
            "alert_history_count": len(self.alert_history),
        }

    def _record_alert_event(
        self, alert: Alert, event_type: str, metric_value: float
    ) -> None:
        """Record alert event in history."""
        self.alert_history.append(
            {
                "alert_name": alert.name,
                "event_type": event_type,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "metric_value": metric_value,
                "severity": alert.severity.value,
                "service_name": alert.service_name,
            }
        )

        # Keep last 1000 events
        if len(self.alert_history) > 1000:
            self.alert_history.pop(0)

    def _notify_handlers(self, alert: Alert) -> None:
        """Notify all handlers about alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception:
                logger.exception("Error in alert notification handler")


class EnterpriseObservabilityPlatform:
    """Comprehensive observability platform for enterprise features."""

    def __init__(self, config: EnterpriseConfig):
        self.config = config

        # Core components
        self.metrics_collector = MetricsCollector(
            retention_hours=config.observability.retention_days * 24
        )
        self.distributed_tracer = DistributedTracer(
            retention_hours=config.observability.retention_days * 24
        )
        self.anomaly_detector = AnomalyDetector(sensitivity=2.0)
        self.alert_manager = AlertManager()

        # Platform state
        self.is_initialized = False
        self.cleanup_task: asyncio.Task | None = None
        self.monitoring_task: asyncio.Task | None = None

        # Performance tracking
        self.platform_metrics = {
            "metrics_processed": 0,
            "traces_completed": 0,
            "anomalies_detected": 0,
            "alerts_triggered": 0,
        }

        logger.info("Enterprise observability platform initialized")

    async def initialize(self) -> None:
        """Initialize the observability platform."""
        if self.is_initialized:
            return

        try:
            # Setup default alerts
            await self._setup_default_alerts()

            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.monitoring_task = asyncio.create_task(self._monitoring_loop())

            # Add alert notification handler
            self.alert_manager.add_notification_handler(self._handle_alert_notification)

            self.is_initialized = True
            logger.info("Enterprise observability platform started successfully")

        except Exception:
            logger.exception("Failed to initialize observability platform")
            raise

    async def cleanup(self) -> None:
        """Cleanup observability platform resources."""
        if self.cleanup_task:
            self.cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.cleanup_task

        if self.monitoring_task:
            self.monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.monitoring_task

        self.is_initialized = False
        logger.info("Enterprise observability platform cleanup completed")

    def record_metric(
        self,
        name: str,
        value: float,
        *,
        metric_type: MetricType = MetricType.GAUGE,
        tags: dict[str, str] | None = None,
        service_name: str = "unknown",
    ) -> None:
        """Record a metric."""
        if tags is None:
            tags = {}

        tags["service"] = service_name

        metric = Metric(name=name, value=value, metric_type=metric_type, tags=tags)

        self.metrics_collector.record_metric(metric)
        self.platform_metrics["metrics_processed"] += 1

        # Check for anomalies
        if self.config.observability.enabled:
            anomaly = self.anomaly_detector.detect_anomaly(name, value, service_name)
            if anomaly:
                self.platform_metrics["anomalies_detected"] += 1
                logger.warning(
                    f"Anomaly detected: {anomaly.metric_name} = {anomaly.actual_value}"
                )

    def start_trace(
        self, operation_name: str, service_name: str, trace_id: str | None = None
    ) -> TraceSpan:
        """Start a new distributed trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        return self.distributed_tracer.start_trace(
            trace_id, operation_name, service_name
        )

    def start_span(
        self,
        trace_id: str,
        operation_name: str,
        service_name: str,
        parent_span_id: str | None = None,
    ) -> TraceSpan:
        """Start a new span within a trace."""
        return self.distributed_tracer.start_span(
            trace_id, operation_name, service_name, parent_span_id
        )

    def finish_span(
        self, span: TraceSpan, status: TraceStatus = TraceStatus.SUCCESS
    ) -> None:
        """Finish a span."""
        self.distributed_tracer.finish_span(span, status)

        if status == TraceStatus.SUCCESS and span.duration_ms:
            # Record performance metrics
            self.record_metric(
                "span.duration",
                span.duration_ms,
                metric_type=MetricType.TIMER,
                tags={"operation": span.operation_name},
                service_name=span.service_name,
            )

        # Check if trace is complete
        trace = self.distributed_tracer.get_trace(span.trace_id)
        if trace and all(s.end_time for s in trace):
            self.platform_metrics["traces_completed"] += 1

    def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health status."""
        alert_summary = self.alert_manager.get_alert_summary()

        # Calculate overall health score
        health_score = 100.0

        if alert_summary["critical_alerts"] > 0:
            health_score -= alert_summary["critical_alerts"] * 20
        if alert_summary["active_alerts"] > 0:
            health_score -= alert_summary["active_alerts"] * 5

        health_score = max(0, min(100, health_score))

        return {
            "overall_health_score": health_score,
            "health_status": "healthy"
            if health_score > 80
            else "degraded"
            if health_score > 60
            else "unhealthy",
            "alerts": alert_summary,
            "platform_metrics": self.platform_metrics,
            "anomaly_baselines": len(self.anomaly_detector.get_baseline_info()),
            "active_traces": len(self.distributed_tracer.active_traces),
            "completed_traces": len(self.distributed_tracer.completed_traces),
        }

    def get_service_metrics(self, service_name: str) -> dict[str, Any]:
        """Get metrics for a specific service."""
        # Get all metrics for the service
        service_metrics = {}

        for metric_metadata in self.metrics_collector.metric_metadata.values():
            if metric_metadata["tags"].get("service") == service_name:
                stats = self.metrics_collector.get_metric_statistics(
                    metric_metadata["name"], metric_metadata["tags"]
                )
                service_metrics[metric_metadata["name"]] = stats

        return {
            "service_name": service_name,
            "metrics": service_metrics,
            "metric_count": len(service_metrics),
        }

    async def _setup_default_alerts(self) -> None:
        """Setup default enterprise alerts."""
        default_alerts = [
            Alert(
                name="high_error_rate",
                description="High error rate detected",
                condition="greater_than",
                severity=AlertSeverity.CRITICAL,
                threshold=0.05,  # 5% error rate
                metric_name="error_rate",
            ),
            Alert(
                name="high_response_time",
                description="High response time detected",
                condition="greater_than",
                severity=AlertSeverity.HIGH,
                threshold=1000.0,  # 1 second
                metric_name="response_time_ms",
            ),
            Alert(
                name="low_availability",
                description="Low service availability",
                condition="less_than",
                severity=AlertSeverity.CRITICAL,
                threshold=0.99,  # 99% availability
                metric_name="availability",
            ),
        ]

        for alert in default_alerts:
            self.alert_manager.add_alert(alert)

    async def _cleanup_loop(self) -> None:
        """Background cleanup of old data."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                self.metrics_collector.cleanup_old_metrics()
                self.distributed_tracer.cleanup_old_traces()

                logger.info("Observability platform cleanup completed")

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in cleanup loop")

    async def _monitoring_loop(self) -> None:
        """Background monitoring and alerting."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Update anomaly baselines
                await self._update_anomaly_baselines()

                # Check alerts
                triggered_alerts = self.alert_manager.check_alerts(
                    self.metrics_collector
                )
                self.platform_metrics["alerts_triggered"] += len(triggered_alerts)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in monitoring loop")

    async def _update_anomaly_baselines(self) -> None:
        """Update anomaly detection baselines."""
        cutoff_time = datetime.now(tz=UTC) - timedelta(hours=1)

        for metric_key, metadata in self.metrics_collector.metric_metadata.items():
            try:
                values = self.metrics_collector.get_metric_values(
                    metadata["name"], metadata["tags"], since=cutoff_time
                )

                if len(values) >= 10:  # Need minimum samples
                    self.anomaly_detector.update_baseline(metadata["name"], values)

            except Exception:
                logger.exception("Error updating baseline for %s", metric_key)

    def _handle_alert_notification(self, alert: Alert) -> None:
        """Handle alert notifications."""
        logger.warning(
            f"ALERT TRIGGERED: {alert.name} - {alert.description} "
            f"(Severity: {alert.severity.value})"
        )

        # Here you would integrate with notification systems:
        # - Email notifications
        # - Slack/Teams messages
        # - PagerDuty incidents
        # - SMS alerts
        # - Webhook calls


# Global observability platform instance
_observability_platform: EnterpriseObservabilityPlatform | None = None


async def get_observability_platform(
    config: EnterpriseConfig = None,
) -> EnterpriseObservabilityPlatform:
    """Get or create the global observability platform."""
    global _observability_platform

    if _observability_platform is None:
        if config is None:
            config = create_enterprise_config()

        _observability_platform = EnterpriseObservabilityPlatform(config)
        await _observability_platform.initialize()

    return _observability_platform


async def cleanup_observability_platform() -> None:
    """Cleanup the global observability platform."""
    global _observability_platform

    if _observability_platform:
        await _observability_platform.cleanup()
        _observability_platform = None
