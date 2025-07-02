"""Autonomous Health Monitor for Zero-Maintenance Infrastructure.

This module implements an AI-driven health monitoring system that provides intelligent
monitoring, validation, and optimization of configuration across all environments and
deployment modes with predictive failure detection capabilities.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

from src.services.circuit_breaker.modern import ModernCircuitBreakerManager
from src.services.monitoring.health import HealthCheckManager, HealthStatus
from src.services.observability.performance import get_performance_monitor


logger = logging.getLogger(__name__)


class PredictionConfidence(str, Enum):
    """Prediction confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FailureRiskLevel(str, Enum):
    """Failure risk levels."""

    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SystemMetrics:
    """Comprehensive system metrics for health analysis."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: dict[str, int]
    database_connections: int
    cache_hit_ratio: float
    response_time_p95: float
    error_rate: float
    circuit_breaker_states: dict[str, str]
    service_health_scores: dict[str, float]


@dataclass
class FailurePrediction:
    """Failure prediction result."""

    failure_type: str
    risk_level: FailureRiskLevel
    confidence: PredictionConfidence
    time_to_failure_minutes: int | None
    contributing_factors: list[str]
    recommended_actions: list[str]
    auto_remediation_available: bool
    business_impact_score: float


@dataclass
class HealthTrend:
    """Health trend analysis."""

    metric_name: str
    current_value: float
    trend_direction: str  # "improving", "stable", "degrading"
    trend_slope: float
    prediction_window_minutes: int
    predicted_value: float
    threshold_breach_probability: float


class FailurePredictionEngine:
    """ML-based failure prediction using time-series analysis and pattern
    recognition."""

    def __init__(self):
        self.metric_history: dict[str, list[float]] = {}
        self.failure_patterns: dict[str, list[dict[str, Any]]] = {}
        self.prediction_models = {
            "memory_exhaustion": self._predict_memory_exhaustion,
            "cpu_overload": self._predict_cpu_overload,
            "disk_space": self._predict_disk_exhaustion,
            "connection_pool": self._predict_connection_exhaustion,
            "response_time": self._predict_response_degradation,
            "error_spike": self._predict_error_spike,
        }

    async def predict_failures(self, metrics: SystemMetrics) -> list[FailurePrediction]:
        """Predict potential failures based on current metrics and trends."""
        self._update_metric_history(metrics)

        predictions = []
        for failure_type, predictor in self.prediction_models.items():
            try:
                prediction = await predictor(metrics)
                if prediction and prediction.risk_level != FailureRiskLevel.MINIMAL:
                    predictions.append(prediction)
            except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                logger.warning(f"Failed to generate prediction for {failure_type}: {e}")

        return sorted(
            predictions, key=lambda p: self._risk_priority(p.risk_level), reverse=True
        )

    def _update_metric_history(self, metrics: SystemMetrics):
        """Update metric history for trend analysis."""
        metric_values = {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "disk_percent": metrics.disk_percent,
            "database_connections": metrics.database_connections,
            "cache_hit_ratio": metrics.cache_hit_ratio,
            "response_time_p95": metrics.response_time_p95,
            "error_rate": metrics.error_rate,
        }

        for metric_name, value in metric_values.items():
            if metric_name not in self.metric_history:
                self.metric_history[metric_name] = []

            self.metric_history[metric_name].append(value)

            # Keep only last 100 readings for trend analysis
            if len(self.metric_history[metric_name]) > 100:
                self.metric_history[metric_name] = self.metric_history[metric_name][
                    -100:
                ]

    async def _predict_memory_exhaustion(
        self, metrics: SystemMetrics
    ) -> FailurePrediction | None:
        """Predict memory exhaustion based on usage trends."""
        if (
            "memory_percent" not in self.metric_history
            or len(self.metric_history["memory_percent"]) < 10
        ):
            return None

        memory_history = self.metric_history["memory_percent"]
        current_usage = metrics.memory_percent

        # Calculate trend
        trend = self._calculate_trend(memory_history[-10:])

        # Predict exhaustion
        if trend > 0.5 and current_usage > 70:  # Growing trend and high usage
            time_to_exhaustion = self._estimate_time_to_threshold(
                memory_history, current_usage, 95.0
            )

            risk_level = self._calculate_risk_level(
                current_usage, trend, time_to_exhaustion
            )

            return FailurePrediction(
                failure_type="memory_exhaustion",
                risk_level=risk_level,
                confidence=self._calculate_confidence(trend, len(memory_history)),
                time_to_failure_minutes=time_to_exhaustion,
                contributing_factors=self._identify_memory_factors(metrics),
                recommended_actions=[
                    "Clear cache segments",
                    "Restart memory-intensive services",
                    "Scale memory allocation",
                    "Enable memory monitoring alerts",
                ],
                auto_remediation_available=current_usage < 90,
                business_impact_score=self._calculate_business_impact(
                    "memory_exhaustion", risk_level
                ),
            )

        return None

    async def _predict_cpu_overload(
        self, metrics: SystemMetrics
    ) -> FailurePrediction | None:
        """Predict CPU overload based on usage patterns."""
        if (
            "cpu_percent" not in self.metric_history
            or len(self.metric_history["cpu_percent"]) < 10
        ):
            return None

        cpu_history = self.metric_history["cpu_percent"]
        current_usage = metrics.cpu_percent

        # Detect sustained high usage
        recent_avg = (
            sum(cpu_history[-5:]) / 5 if len(cpu_history) >= 5 else current_usage
        )

        if recent_avg > 80 or current_usage > 90:
            time_to_overload = self._estimate_time_to_threshold(
                cpu_history, current_usage, 95.0
            )

            risk_level = (
                FailureRiskLevel.HIGH if current_usage > 90 else FailureRiskLevel.MEDIUM
            )

            return FailurePrediction(
                failure_type="cpu_overload",
                risk_level=risk_level,
                confidence=PredictionConfidence.HIGH
                if current_usage > 90
                else PredictionConfidence.MEDIUM,
                time_to_failure_minutes=time_to_overload,
                contributing_factors=self._identify_cpu_factors(metrics),
                recommended_actions=[
                    "Scale CPU allocation",
                    "Optimize query performance",
                    "Enable request throttling",
                    "Review background processes",
                ],
                auto_remediation_available=True,
                business_impact_score=self._calculate_business_impact(
                    "cpu_overload", risk_level
                ),
            )

        return None

    async def _predict_disk_exhaustion(
        self, metrics: SystemMetrics
    ) -> FailurePrediction | None:
        """Predict disk space exhaustion."""
        current_usage = metrics.disk_percent

        if current_usage > 85:
            # Estimate time to full based on current growth rate
            time_to_full = self._estimate_disk_exhaustion_time(current_usage)

            risk_level = (
                FailureRiskLevel.CRITICAL
                if current_usage > 95
                else FailureRiskLevel.HIGH
                if current_usage > 90
                else FailureRiskLevel.MEDIUM
            )

            return FailurePrediction(
                failure_type="disk_space",
                risk_level=risk_level,
                confidence=PredictionConfidence.HIGH,
                time_to_failure_minutes=time_to_full,
                contributing_factors=[
                    "High disk usage",
                    "Log file growth",
                    "Cache accumulation",
                ],
                recommended_actions=[
                    "Clean up log files",
                    "Clear temporary files",
                    "Archive old data",
                    "Expand disk space",
                ],
                auto_remediation_available=current_usage < 98,
                business_impact_score=self._calculate_business_impact(
                    "disk_space", risk_level
                ),
            )

        return None

    async def _predict_connection_exhaustion(
        self, metrics: SystemMetrics
    ) -> FailurePrediction | None:
        """Predict database connection pool exhaustion."""
        connection_utilization = (
            metrics.database_connections / 100
        )  # Assuming 100 max connections

        if connection_utilization > 0.8:
            risk_level = (
                FailureRiskLevel.CRITICAL
                if connection_utilization > 0.95
                else FailureRiskLevel.HIGH
                if connection_utilization > 0.9
                else FailureRiskLevel.MEDIUM
            )

            return FailurePrediction(
                failure_type="connection_pool",
                risk_level=risk_level,
                confidence=PredictionConfidence.HIGH,
                time_to_failure_minutes=self._estimate_connection_exhaustion_time(
                    connection_utilization
                ),
                contributing_factors=[
                    "High connection usage",
                    "Connection leaks",
                    "Long-running queries",
                ],
                recommended_actions=[
                    "Increase connection pool size",
                    "Optimize connection usage",
                    "Enable connection timeout",
                    "Review long-running queries",
                ],
                auto_remediation_available=True,
                business_impact_score=self._calculate_business_impact(
                    "connection_pool", risk_level
                ),
            )

        return None

    async def _predict_response_degradation(
        self, metrics: SystemMetrics
    ) -> FailurePrediction | None:
        """Predict response time degradation."""
        current_response_time = metrics.response_time_p95

        if current_response_time > 5000:  # 5 seconds
            risk_level = (
                FailureRiskLevel.HIGH
                if current_response_time > 10000
                else FailureRiskLevel.MEDIUM
            )

            return FailurePrediction(
                failure_type="response_time",
                risk_level=risk_level,
                confidence=PredictionConfidence.HIGH,
                time_to_failure_minutes=None,  # Already degraded
                contributing_factors=self._identify_response_time_factors(metrics),
                recommended_actions=[
                    "Scale application instances",
                    "Optimize database queries",
                    "Enable caching",
                    "Review external dependencies",
                ],
                auto_remediation_available=True,
                business_impact_score=self._calculate_business_impact(
                    "response_time", risk_level
                ),
            )

        return None

    async def _predict_error_spike(
        self, metrics: SystemMetrics
    ) -> FailurePrediction | None:
        """Predict error rate spikes."""
        current_error_rate = metrics.error_rate

        if current_error_rate > 0.05:  # 5% error rate
            risk_level = (
                FailureRiskLevel.CRITICAL
                if current_error_rate > 0.2
                else FailureRiskLevel.HIGH
                if current_error_rate > 0.1
                else FailureRiskLevel.MEDIUM
            )

            return FailurePrediction(
                failure_type="error_spike",
                risk_level=risk_level,
                confidence=PredictionConfidence.HIGH,
                time_to_failure_minutes=None,  # Already occurring
                contributing_factors=self._identify_error_factors(metrics),
                recommended_actions=[
                    "Enable circuit breakers",
                    "Review error logs",
                    "Check external dependencies",
                    "Implement retry policies",
                ],
                auto_remediation_available=True,
                business_impact_score=self._calculate_business_impact(
                    "error_spike", risk_level
                ),
            )

        return None

    def _calculate_trend(self, values: list[float]) -> float:
        """Calculate trend slope for a series of values."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * values[i] for i in range(n))
        x2_sum = sum(i * i for i in range(n))

        if n * x2_sum - x_sum * x_sum == 0:
            return 0.0

        return (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum)

    def _estimate_time_to_threshold(
        self, history: list[float], current: float, threshold: float
    ) -> int | None:
        """Estimate time in minutes until threshold is reached."""
        if len(history) < 5:
            return None

        trend = self._calculate_trend(history[-10:])
        if trend <= 0:
            return None  # Not trending upward

        remaining = threshold - current
        if remaining <= 0:
            return 0  # Already at threshold

        # Estimate time based on trend (assuming trend per minute)
        time_minutes = remaining / (trend * 60)  # Convert to minutes
        return max(1, int(time_minutes))

    def _estimate_disk_exhaustion_time(self, current_usage: float) -> int | None:
        """Estimate time until disk is full."""
        # Simple heuristic based on current usage
        if current_usage > 98:
            return 5  # 5 minutes
        if current_usage > 95:
            return 30  # 30 minutes
        if current_usage > 90:
            return 120  # 2 hours
        return 480  # 8 hours

    def _estimate_connection_exhaustion_time(self, utilization: float) -> int | None:
        """Estimate time until connection pool exhaustion."""
        if utilization > 0.98:
            return 1  # 1 minute
        if utilization > 0.95:
            return 5  # 5 minutes
        if utilization > 0.9:
            return 15  # 15 minutes
        return 60  # 1 hour

    def _calculate_risk_level(
        self, current_value: float, _trend: float, time_to_failure: int | None
    ) -> FailureRiskLevel:
        """Calculate risk level based on current value, trend, and time to failure."""
        if current_value > 95 or (time_to_failure and time_to_failure < 5):
            return FailureRiskLevel.CRITICAL
        if current_value > 90 or (time_to_failure and time_to_failure < 15):
            return FailureRiskLevel.HIGH
        if current_value > 80 or (time_to_failure and time_to_failure < 60):
            return FailureRiskLevel.MEDIUM
        return FailureRiskLevel.LOW

    def _calculate_confidence(
        self, trend: float, history_length: int
    ) -> PredictionConfidence:
        """Calculate prediction confidence based on trend strength and history."""
        confidence_score = abs(trend) * 0.5 + (min(history_length, 50) / 50) * 0.5

        if confidence_score > 0.8:
            return PredictionConfidence.VERY_HIGH
        if confidence_score > 0.6:
            return PredictionConfidence.HIGH
        if confidence_score > 0.4:
            return PredictionConfidence.MEDIUM
        return PredictionConfidence.LOW

    def _identify_memory_factors(self, metrics: SystemMetrics) -> list[str]:
        """Identify contributing factors for memory issues."""
        factors = []

        if metrics.cache_hit_ratio < 0.7:
            factors.append("Low cache hit ratio causing memory pressure")

        if metrics.database_connections > 50:
            factors.append("High database connection count")

        if metrics.response_time_p95 > 2000:
            factors.append("Slow responses indicating memory pressure")

        return factors or ["General memory consumption increase"]

    def _identify_cpu_factors(self, metrics: SystemMetrics) -> list[str]:
        """Identify contributing factors for CPU issues."""
        factors = []

        if metrics.response_time_p95 > 3000:
            factors.append("High response times indicating CPU pressure")

        if metrics.database_connections > 70:
            factors.append("High database load")

        if metrics.error_rate > 0.02:
            factors.append("Error processing overhead")

        return factors or ["General CPU load increase"]

    def _identify_response_time_factors(self, metrics: SystemMetrics) -> list[str]:
        """Identify contributing factors for response time issues."""
        factors = []

        if metrics.cpu_percent > 80:
            factors.append("High CPU usage")

        if metrics.memory_percent > 80:
            factors.append("Memory pressure")

        if metrics.cache_hit_ratio < 0.8:
            factors.append("Poor cache performance")

        if metrics.database_connections > 60:
            factors.append("Database connection pressure")

        return factors or ["Unknown performance bottleneck"]

    def _identify_error_factors(self, metrics: SystemMetrics) -> list[str]:
        """Identify contributing factors for error spikes."""
        factors = []

        # Check circuit breaker states
        open_breakers = [
            name
            for name, state in metrics.circuit_breaker_states.items()
            if state == "open"
        ]
        if open_breakers:
            factors.append(f"Circuit breakers open: {', '.join(open_breakers)}")

        if metrics.response_time_p95 > 5000:
            factors.append("High response times causing timeouts")

        if metrics.cpu_percent > 90:
            factors.append("CPU overload causing errors")

        if metrics.memory_percent > 90:
            factors.append("Memory pressure causing errors")

        return factors or ["Unknown error source"]

    def _calculate_business_impact(
        self, failure_type: str, risk_level: FailureRiskLevel
    ) -> float:
        """Calculate business impact score."""
        base_scores = {
            "memory_exhaustion": 0.8,
            "cpu_overload": 0.7,
            "disk_space": 0.9,
            "connection_pool": 0.8,
            "response_time": 0.6,
            "error_spike": 0.9,
        }

        risk_multipliers = {
            FailureRiskLevel.MINIMAL: 0.1,
            FailureRiskLevel.LOW: 0.3,
            FailureRiskLevel.MEDIUM: 0.6,
            FailureRiskLevel.HIGH: 0.8,
            FailureRiskLevel.CRITICAL: 1.0,
        }

        base_score = base_scores.get(failure_type, 0.5)
        multiplier = risk_multipliers.get(risk_level, 0.5)

        return base_score * multiplier

    def _risk_priority(self, risk_level: FailureRiskLevel) -> int:
        """Convert risk level to priority number for sorting."""
        priority_map = {
            FailureRiskLevel.CRITICAL: 5,
            FailureRiskLevel.HIGH: 4,
            FailureRiskLevel.MEDIUM: 3,
            FailureRiskLevel.LOW: 2,
            FailureRiskLevel.MINIMAL: 1,
        }
        return priority_map.get(risk_level, 0)


class AutonomousHealthMonitor:
    """AI-driven health monitoring with predictive failure detection and autonomous
    remediation."""

    def __init__(
        self,
        health_manager: HealthCheckManager,
        circuit_breaker_manager: ModernCircuitBreakerManager,
    ):
        """Initialize autonomous health monitor.

        Args:
            health_manager: Health check manager for service monitoring
            circuit_breaker_manager: Circuit breaker manager for failure protection
        """
        self.health_manager = health_manager
        self.circuit_breaker_manager = circuit_breaker_manager
        self.prediction_engine = FailurePredictionEngine()

        # State tracking
        self.monitoring_active = False
        self.last_health_check = None
        self.prediction_history: list[FailurePrediction] = []
        self.remediation_history: list[dict[str, Any]] = []

        # Configuration
        self.monitoring_interval = 30  # seconds
        self.prediction_threshold = FailureRiskLevel.MEDIUM
        self.max_predictions_history = 1000
        self.max_remediation_history = 500

    async def start_monitoring(self):
        """Start continuous autonomous health monitoring."""
        if self.monitoring_active:
            logger.warning("Health monitoring already active")
            return

        self.monitoring_active = True
        logger.info("Starting autonomous health monitoring")

        try:
            await self.continuous_monitoring_loop()
        except (TimeoutError, OSError, PermissionError):
            logger.exception("Health monitoring loop failed")
            self.monitoring_active = False
            raise

    async def stop_monitoring(self):
        """Stop autonomous health monitoring."""
        self.monitoring_active = False
        logger.info("Stopped autonomous health monitoring")

    async def continuous_monitoring_loop(self):
        """Main monitoring loop with predictive capabilities."""
        while self.monitoring_active:
            try:
                # 1. Collect comprehensive health metrics
                health_metrics = await self.collect_comprehensive_health_metrics()

                # 2. Generate failure predictions
                predictions = await self.prediction_engine.predict_failures(
                    health_metrics
                )

                # 3. Store prediction history
                self._update_prediction_history(predictions)

                # 4. Handle current health issues
                await self.handle_current_health_issues(health_metrics)

                # 5. Process predictions for preemptive action
                await self.process_failure_predictions(predictions)

                # 6. Update circuit breaker configurations based on health
                await self.optimize_circuit_breakers(health_metrics, predictions)

                # 7. Generate health insights and recommendations
                insights = await self.generate_health_insights(
                    health_metrics, predictions
                )

                # Log health status summary
                await self.log_health_status_summary(
                    health_metrics, predictions, insights
                )

            except Exception:
                logger.exception("Error in monitoring loop")

            # Wait for next monitoring cycle
            await asyncio.sleep(self.monitoring_interval)

    async def collect_comprehensive_health_metrics(self) -> SystemMetrics:
        """Collect comprehensive system health metrics."""
        try:
            # Get performance monitor
            performance_monitor = get_performance_monitor()

            # Collect system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            network = psutil.net_io_counters()

            # Get health check results
            health_results = await self.health_manager.check_all()

            # Get circuit breaker states
            circuit_breaker_states = (
                await self.circuit_breaker_manager.get_all_statuses()
            )

            # Calculate service health scores
            service_health_scores = {}
            for service, result in health_results.items():
                health_scores = 1.0 if result.status == HealthStatus.HEALTHY else 0.0
                service_health_scores[service] = health_scores

            # Get performance metrics
            system_performance = performance_monitor.get_system_performance_summary()

            return SystemMetrics(
                timestamp=datetime.now(tz=datetime.UTC),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                disk_percent=(disk.used / disk.total) * 100,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                },
                database_connections=self._estimate_database_connections(),
                cache_hit_ratio=self._estimate_cache_hit_ratio(),
                response_time_p95=self._get_response_time_p95(system_performance),
                error_rate=self._calculate_current_error_rate(),
                circuit_breaker_states={
                    name: status.get("state", "unknown")
                    for name, status in circuit_breaker_states.items()
                },
                service_health_scores=service_health_scores,
            )

        except Exception:
            logger.exception("Failed to collect health metrics")
            # Return minimal metrics on failure
            return SystemMetrics(
                timestamp=datetime.now(tz=datetime.UTC),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                database_connections=0,
                cache_hit_ratio=0.0,
                response_time_p95=0.0,
                error_rate=0.0,
                circuit_breaker_states={},
                service_health_scores={},
            )

    async def handle_current_health_issues(self, metrics: SystemMetrics):
        """Handle current health issues detected in metrics."""
        issues_detected = []

        # Check for immediate health issues
        if metrics.cpu_percent > 95:
            issues_detected.append("Critical CPU usage")

        if metrics.memory_percent > 95:
            issues_detected.append("Critical memory usage")

        if metrics.disk_percent > 98:
            issues_detected.append("Critical disk usage")

        if metrics.error_rate > 0.1:
            issues_detected.append("High error rate")

        if metrics.response_time_p95 > 10000:  # 10 seconds
            issues_detected.append("Critical response time")

        # Check service health
        unhealthy_services = [
            service
            for service, score in metrics.service_health_scores.items()
            if score < 0.5
        ]
        if unhealthy_services:
            issues_detected.append(
                f"Unhealthy services: {', '.join(unhealthy_services)}"
            )

        if issues_detected:
            logger.warning(
                f"Current health issues detected: {', '.join(issues_detected)}"
            )

            # Record in remediation history
            self.remediation_history.append(
                {
                    "timestamp": datetime.now(tz=datetime.UTC),
                    "type": "current_issues",
                    "issues": issues_detected,
                    "metrics_snapshot": metrics,
                    "action_taken": "logged_and_monitored",
                }
            )

    async def process_failure_predictions(self, predictions: list[FailurePrediction]):
        """Process failure predictions for preemptive action."""
        high_risk_predictions = [
            p
            for p in predictions
            if p.risk_level in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL]
        ]

        if not high_risk_predictions:
            return

        logger.warning(
            f"High-risk failure predictions detected: "
            f"{len(high_risk_predictions)} predictions"
        )

        for prediction in high_risk_predictions:
            await self.handle_prediction(prediction)

    async def handle_prediction(self, prediction: FailurePrediction):
        """Handle individual failure prediction."""
        logger.warning(
            f"Prediction: {prediction.failure_type} - "
            f"Risk: {prediction.risk_level.value} - "
            f"Confidence: {prediction.confidence.value} - "
            f"Time to failure: {prediction.time_to_failure_minutes} minutes"
        )

        # Log recommended actions
        logger.info(
            f"Recommended actions for {prediction.failure_type}: "
            f"{', '.join(prediction.recommended_actions)}"
        )

        # For now, we log the prediction and recommendations
        # In a full implementation, this would trigger automated remediation
        # based on the prediction.auto_remediation_available flag

        self.remediation_history.append(
            {
                "timestamp": datetime.now(tz=datetime.UTC),
                "type": "prediction",
                "prediction": prediction,
                "action_taken": "logged_recommendations",
            }
        )

    async def optimize_circuit_breakers(
        self, metrics: SystemMetrics, predictions: list[FailurePrediction]
    ):
        """Optimize circuit breaker configurations based on health and predictions."""
        # Identify services under stress
        stressed_services = []

        if metrics.response_time_p95 > 5000:
            stressed_services.append("api_gateway")

        if metrics.database_connections > 80:
            stressed_services.append("database")

        if metrics.cache_hit_ratio < 0.7:
            stressed_services.append("cache")

        # Check for error spike predictions
        error_predictions = [p for p in predictions if p.failure_type == "error_spike"]
        if error_predictions:
            stressed_services.extend(["external_api", "search_service"])

        # Log circuit breaker optimization recommendations
        if stressed_services:
            logger.info(
                f"Recommend reviewing circuit breaker configurations for: "
                f"{', '.join(set(stressed_services))}"
            )

    async def generate_health_insights(
        self, metrics: SystemMetrics, predictions: list[FailurePrediction]
    ) -> dict[str, Any]:
        """Generate health insights and recommendations."""
        return {
            "overall_health_score": self._calculate_overall_health_score(metrics),
            "critical_metrics": self._identify_critical_metrics(metrics),
            "trending_issues": self._identify_trending_issues(predictions),
            "recommendations": self._generate_recommendations(metrics, predictions),
            "system_stability": self._assess_system_stability(metrics, predictions),
        }

    async def log_health_status_summary(
        self,
        metrics: SystemMetrics,
        predictions: list[FailurePrediction],
        insights: dict[str, Any],
    ):
        """Log a comprehensive health status summary."""
        summary_lines = [
            "=== Health Status Summary ===",
            f"Overall Health Score: {insights['overall_health_score']:.2f}/1.0",
            (
                f"CPU: {metrics.cpu_percent:.1f}% | "
                f"Memory: {metrics.memory_percent:.1f}% | "
                f"Disk: {metrics.disk_percent:.1f}%"
            ),
            f"Response Time P95: {metrics.response_time_p95:.0f}ms | "
            f"Error Rate: {metrics.error_rate:.3f}",
            f"Cache Hit Ratio: {metrics.cache_hit_ratio:.2f} | "
            f"DB Connections: {metrics.database_connections}",
            (
                f"Active Predictions: {len(predictions)} | "
                f"High Risk: {len([p for p in predictions if p.risk_level == FailureRiskLevel.HIGH])}"
            ),
            f"System Stability: {insights['system_stability']}",
            "=== End Summary ===",
        ]

        for line in summary_lines:
            logger.info(line)

    def _update_prediction_history(self, predictions: list[FailurePrediction]):
        """Update prediction history with latest predictions."""
        self.prediction_history.extend(predictions)

        # Keep only recent predictions
        if len(self.prediction_history) > self.max_predictions_history:
            self.prediction_history = self.prediction_history[
                -self.max_predictions_history :
            ]

    def _estimate_database_connections(self) -> int:
        """Estimate current database connections."""
        # This would integrate with actual database monitoring
        # For now, return a mock value
        return 45

    def _estimate_cache_hit_ratio(self) -> float:
        """Estimate cache hit ratio."""
        # This would integrate with actual cache monitoring
        # For now, return a mock value
        return 0.85

    def _get_response_time_p95(self, performance_summary: dict[str, Any]) -> float:
        """Get 95th percentile response time from performance data."""
        # Extract from performance monitor data
        return (
            performance_summary.get("avg_response_time", 0) * 1.5
        )  # Rough P95 estimate

    def _calculate_current_error_rate(self) -> float:
        """Calculate current error rate."""
        # This would integrate with actual error tracking
        # For now, return a mock value
        return 0.01  # 1% error rate

    def _calculate_overall_health_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall system health score."""
        # Weight different metrics
        cpu_score = max(0, (100 - metrics.cpu_percent) / 100)
        memory_score = max(0, (100 - metrics.memory_percent) / 100)
        disk_score = max(0, (100 - metrics.disk_percent) / 100)
        response_score = max(
            0, 1 - (metrics.response_time_p95 / 10000)
        )  # Normalize to 10s max
        error_score = max(0, 1 - (metrics.error_rate * 10))  # Penalize errors heavily
        cache_score = metrics.cache_hit_ratio

        # Calculate weighted average
        weights = [0.2, 0.2, 0.15, 0.2, 0.15, 0.1]
        scores = [
            cpu_score,
            memory_score,
            disk_score,
            response_score,
            error_score,
            cache_score,
        ]

        overall_score = sum(
            score * weight for score, weight in zip(scores, weights, strict=False)
        )
        return max(0, min(1, overall_score))

    def _identify_critical_metrics(self, metrics: SystemMetrics) -> list[str]:
        """Identify metrics that are in critical state."""
        critical = []

        if metrics.cpu_percent > 90:
            critical.append(f"CPU: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > 90:
            critical.append(f"Memory: {metrics.memory_percent:.1f}%")

        if metrics.disk_percent > 95:
            critical.append(f"Disk: {metrics.disk_percent:.1f}%")

        if metrics.response_time_p95 > 5000:
            critical.append(f"Response Time: {metrics.response_time_p95:.0f}ms")

        if metrics.error_rate > 0.05:
            critical.append(f"Error Rate: {metrics.error_rate:.3f}")

        if metrics.cache_hit_ratio < 0.5:
            critical.append(f"Cache Hit Ratio: {metrics.cache_hit_ratio:.2f}")

        return critical

    def _identify_trending_issues(
        self, predictions: list[FailurePrediction]
    ) -> list[str]:
        """Identify trending issues from predictions."""
        trending = []

        high_risk_predictions = [
            p
            for p in predictions
            if p.risk_level in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL]
        ]

        trending.extend(
            f"{prediction.failure_type} ({prediction.risk_level.value})"
            for prediction in high_risk_predictions
        )

        return trending

    def _generate_recommendations(
        self, metrics: SystemMetrics, predictions: list[FailurePrediction]
    ) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Resource-based recommendations
        if metrics.cpu_percent > 80:
            recommendations.append(
                "Consider scaling CPU resources or optimizing CPU-intensive operations"
            )

        if metrics.memory_percent > 80:
            recommendations.append(
                "Monitor memory usage and consider clearing caches or scaling memory"
            )

        if metrics.disk_percent > 85:
            recommendations.append("Clean up disk space or expand storage capacity")

        if metrics.response_time_p95 > 3000:
            recommendations.append(
                "Investigate performance bottlenecks and optimize response times"
            )

        if metrics.error_rate > 0.02:
            recommendations.append(
                "Review error logs and implement error rate reduction strategies"
            )

        if metrics.cache_hit_ratio < 0.8:
            recommendations.append("Optimize caching strategy to improve hit ratios")

        # Prediction-based recommendations
        for prediction in predictions:
            if prediction.risk_level in [
                FailureRiskLevel.HIGH,
                FailureRiskLevel.CRITICAL,
            ]:
                recommendations.extend(prediction.recommended_actions)

        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]  # Limit to top 10 recommendations

    def _assess_system_stability(
        self, metrics: SystemMetrics, predictions: list[FailurePrediction]
    ) -> str:
        """Assess overall system stability."""
        health_score = self._calculate_overall_health_score(metrics)
        critical_predictions = len(
            [p for p in predictions if p.risk_level == FailureRiskLevel.CRITICAL]
        )
        high_predictions = len(
            [p for p in predictions if p.risk_level == FailureRiskLevel.HIGH]
        )

        if health_score > 0.9 and critical_predictions == 0 and high_predictions == 0:
            return "Excellent"
        if health_score > 0.8 and critical_predictions == 0 and high_predictions <= 1:
            return "Good"
        if health_score > 0.7 and critical_predictions == 0:
            return "Fair"
        if health_score > 0.5 and critical_predictions <= 1:
            return "Poor"
        return "Critical"

    async def get_monitoring_status(self) -> dict[str, Any]:
        """Get current monitoring status and statistics."""
        return {
            "monitoring_active": self.monitoring_active,
            "monitoring_interval_seconds": self.monitoring_interval,
            "last_health_check": self.last_health_check.isoformat()
            if self.last_health_check
            else None,
            "prediction_history_count": len(self.prediction_history),
            "remediation_history_count": len(self.remediation_history),
            "recent_predictions": len(
                [
                    p
                    for p in self.prediction_history
                    if (
                        datetime.now(tz=datetime.UTC) - datetime.now(tz=datetime.UTC)
                    ).total_seconds()
                    < 3600
                ]
            ),
            "high_risk_predictions_last_hour": len(
                [
                    p
                    for p in self.prediction_history
                    if p.risk_level
                    in [FailureRiskLevel.HIGH, FailureRiskLevel.CRITICAL]
                    and (
                        datetime.now(tz=datetime.UTC) - datetime.now(tz=datetime.UTC)
                    ).total_seconds()
                    < 3600
                ]
            ),
        }
