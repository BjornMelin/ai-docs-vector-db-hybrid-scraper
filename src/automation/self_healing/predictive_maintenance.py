"""Predictive Maintenance System for Zero-Maintenance Infrastructure.

This module implements ML-based predictive maintenance using time-series analysis,
anomaly detection, and intelligent scheduling for proactive system maintenance.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from src.automation.self_healing.auto_remediation_engine import (
    AutoRemediationEngine,
    DetectedIssue,
    RemediationSeverity,
)
from src.automation.self_healing.autonomous_health_monitor import (
    AutonomousHealthMonitor,
    SystemMetrics,
)


logger = logging.getLogger(__name__)


class MaintenanceUrgency(str, Enum):
    """Maintenance urgency levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MaintenanceType(str, Enum):
    """Types of maintenance operations."""

    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"


class ComponentHealth(str, Enum):
    """Component health status."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class TimeSeriesData:
    """Time series data point for analysis."""

    timestamp: datetime
    metric_name: str
    value: float
    component: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""

    timestamp: datetime
    metric_name: str
    current_value: float
    expected_range: tuple[float, float]
    anomaly_score: float
    severity: str
    contributing_factors: list[str]
    recommendation: str


@dataclass
class ComponentHealthAssessment:
    """Health assessment for a system component."""

    component_name: str
    health_status: ComponentHealth
    health_score: float  # 0.0 to 1.0
    degradation_rate: float
    time_to_maintenance: int | None  # minutes
    risk_factors: list[str]
    maintenance_recommendations: list[str]
    historical_patterns: dict[str, Any]


@dataclass
class MaintenanceRecommendation:
    """Maintenance recommendation based on predictive analysis."""

    recommendation_id: str
    component: str
    maintenance_type: MaintenanceType
    urgency: MaintenanceUrgency
    description: str
    estimated_duration_minutes: int
    optimal_execution_window: tuple[datetime, datetime]
    prerequisites: list[str]
    expected_benefits: list[str]
    risk_if_delayed: str
    confidence_score: float


@dataclass
class MaintenanceExecution:
    """Maintenance execution tracking."""

    execution_id: str
    recommendation: MaintenanceRecommendation
    status: str
    start_time: datetime | None = None
    end_time: datetime | None = None
    results: list[str] = field(default_factory=list)
    metrics_before: dict[str, float] | None = None
    metrics_after: dict[str, float] | None = None
    success: bool = False
    notes: str = ""


class TimeSeriesPredictor:
    """Time series prediction and anomaly detection using ML models."""

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.historical_data: dict[str, list[TimeSeriesData]] = {}
        self.anomaly_detectors: dict[str, IsolationForest] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.prediction_models: dict[str, Any] = {}

    def add_data_point(self, data_point: TimeSeriesData):
        """Add a new data point to the time series."""
        metric_key = f"{data_point.component}_{data_point.metric_name}"

        if metric_key not in self.historical_data:
            self.historical_data[metric_key] = []

        self.historical_data[metric_key].append(data_point)

        # Keep only recent data points
        if len(self.historical_data[metric_key]) > 1000:
            self.historical_data[metric_key] = self.historical_data[metric_key][-1000:]

        # Retrain model if we have enough data
        if len(self.historical_data[metric_key]) >= self.window_size:
            self._train_anomaly_detector(metric_key)

    def _train_anomaly_detector(self, metric_key: str):
        """Train anomaly detection model for a specific metric."""
        data_points = self.historical_data[metric_key]

        if len(data_points) < self.window_size:
            return

        # Prepare training data
        values = np.array([dp.value for dp in data_points[-200:]]).reshape(-1, 1)

        # Initialize scaler if not exists
        if metric_key not in self.scalers:
            self.scalers[metric_key] = StandardScaler()

        # Scale data
        scaled_values = self.scalers[metric_key].fit_transform(values)

        # Train isolation forest
        self.anomaly_detectors[metric_key] = IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42,
        )
        self.anomaly_detectors[metric_key].fit(scaled_values)

    def detect_anomalies(self, data_point: TimeSeriesData) -> AnomalyDetection | None:
        """Detect if a data point is anomalous."""
        metric_key = f"{data_point.component}_{data_point.metric_name}"

        if metric_key not in self.anomaly_detectors:
            return None

        # Scale the new data point
        scaled_value = self.scalers[metric_key].transform([[data_point.value]])

        # Predict anomaly
        anomaly_score = self.anomaly_detectors[metric_key].decision_function(
            scaled_value
        )[0]
        is_anomaly = self.anomaly_detectors[metric_key].predict(scaled_value)[0] == -1

        if is_anomaly:
            # Calculate expected range
            recent_values = [dp.value for dp in self.historical_data[metric_key][-50:]]
            expected_range = (min(recent_values), max(recent_values))

            # Determine severity
            severity = self._calculate_anomaly_severity(
                anomaly_score, data_point.value, expected_range
            )

            return AnomalyDetection(
                timestamp=data_point.timestamp,
                metric_name=data_point.metric_name,
                current_value=data_point.value,
                expected_range=expected_range,
                anomaly_score=anomaly_score,
                severity=severity,
                contributing_factors=self._identify_anomaly_factors(
                    data_point, recent_values
                ),
                recommendation=self._generate_anomaly_recommendation(
                    data_point, severity
                ),
            )

        return None

    def predict_future_values(
        self, metric_key: str, steps_ahead: int = 10
    ) -> list[float]:
        """Predict future values for a metric."""
        if metric_key not in self.historical_data:
            return []

        data_points = self.historical_data[metric_key]
        if len(data_points) < self.window_size:
            return []

        # Simple linear extrapolation (in real implementation, use more sophisticated models)
        recent_values = [dp.value for dp in data_points[-20:]]

        # Calculate trend
        x = np.arange(len(recent_values))
        coefficients = np.polyfit(x, recent_values, 1)

        # Predict future values
        future_x = np.arange(len(recent_values), len(recent_values) + steps_ahead)
        predictions = np.polyval(coefficients, future_x)

        return predictions.tolist()

    def _calculate_anomaly_severity(
        self,
        anomaly_score: float,
        current_value: float,
        expected_range: tuple[float, float],
    ) -> str:
        """Calculate severity of detected anomaly."""
        # Normalize anomaly score (isolation forest returns negative scores for anomalies)
        normalized_score = abs(anomaly_score)

        # Calculate deviation from expected range
        range_size = expected_range[1] - expected_range[0]
        if range_size == 0:
            deviation_ratio = 0
        elif current_value < expected_range[0]:
            deviation_ratio = (expected_range[0] - current_value) / range_size
        elif current_value > expected_range[1]:
            deviation_ratio = (current_value - expected_range[1]) / range_size
        else:
            deviation_ratio = 0

        # Combine score and deviation to determine severity
        severity_score = normalized_score + deviation_ratio

        if severity_score > 2.0:
            return "critical"
        if severity_score > 1.5:
            return "high"
        if severity_score > 1.0:
            return "medium"
        return "low"

    def _identify_anomaly_factors(
        self, data_point: TimeSeriesData, recent_values: list[float]
    ) -> list[str]:
        """Identify contributing factors for the anomaly."""
        factors = []

        current_value = data_point.value
        avg_recent = sum(recent_values) / len(recent_values) if recent_values else 0

        if current_value > avg_recent * 2:
            factors.append("Value significantly higher than recent average")
        elif current_value < avg_recent * 0.5:
            factors.append("Value significantly lower than recent average")

        # Check for rapid changes
        if len(recent_values) >= 2:
            recent_change = abs(recent_values[-1] - recent_values[-2])
            avg_change = sum(
                abs(recent_values[i] - recent_values[i - 1])
                for i in range(1, len(recent_values))
            ) / (len(recent_values) - 1)

            if recent_change > avg_change * 3:
                factors.append("Rapid change detected")

        if not factors:
            factors.append("Statistical anomaly detected")

        return factors

    def _generate_anomaly_recommendation(
        self, data_point: TimeSeriesData, severity: str
    ) -> str:
        """Generate recommendation for handling the anomaly."""
        metric_name = data_point.metric_name.lower()

        if severity in ["critical", "high"]:
            if "memory" in metric_name:
                return "Immediate memory cleanup and monitoring required"
            if "cpu" in metric_name:
                return "CPU load investigation and optimization needed"
            if "error" in metric_name:
                return "Error rate spike - investigate root cause immediately"
            if "response" in metric_name:
                return "Performance degradation - optimize response times"
            return "High-severity anomaly - investigate immediately"
        return "Monitor metric closely and investigate if pattern continues"


class ComponentHealthAnalyzer:
    """Analyzes component health and predicts maintenance needs."""

    def __init__(self):
        self.component_baselines: dict[str, dict[str, float]] = {}
        self.health_history: dict[str, list[ComponentHealthAssessment]] = {}
        self.degradation_patterns: dict[str, dict[str, Any]] = {}

    async def assess_component_health(
        self, component: str, metrics: dict[str, float]
    ) -> ComponentHealthAssessment:
        """Assess health of a system component based on current metrics."""

        # Initialize baseline if not exists
        if component not in self.component_baselines:
            self.component_baselines[component] = metrics.copy()

        # Calculate health score
        health_score = self._calculate_health_score(component, metrics)

        # Determine health status
        health_status = self._determine_health_status(health_score)

        # Calculate degradation rate
        degradation_rate = self._calculate_degradation_rate(component, health_score)

        # Estimate time to maintenance
        time_to_maintenance = self._estimate_maintenance_time(
            health_score, degradation_rate
        )

        # Identify risk factors
        risk_factors = self._identify_risk_factors(component, metrics)

        # Generate maintenance recommendations
        maintenance_recommendations = self._generate_maintenance_recommendations(
            component, health_status, risk_factors
        )

        # Analyze historical patterns
        historical_patterns = self._analyze_historical_patterns(component)

        assessment = ComponentHealthAssessment(
            component_name=component,
            health_status=health_status,
            health_score=health_score,
            degradation_rate=degradation_rate,
            time_to_maintenance=time_to_maintenance,
            risk_factors=risk_factors,
            maintenance_recommendations=maintenance_recommendations,
            historical_patterns=historical_patterns,
        )

        # Store in history
        if component not in self.health_history:
            self.health_history[component] = []

        self.health_history[component].append(assessment)

        # Keep only recent history
        if len(self.health_history[component]) > 100:
            self.health_history[component] = self.health_history[component][-100:]

        return assessment

    def _calculate_health_score(
        self, component: str, metrics: dict[str, float]
    ) -> float:
        """Calculate overall health score for a component."""
        if component not in self.component_baselines:
            return 1.0  # Assume healthy if no baseline

        baseline = self.component_baselines[component]

        # Calculate normalized deviations
        deviations = []
        for metric_name, current_value in metrics.items():
            if metric_name in baseline:
                baseline_value = baseline[metric_name]
                if baseline_value != 0:
                    deviation = abs(current_value - baseline_value) / baseline_value
                    deviations.append(min(deviation, 2.0))  # Cap at 200% deviation

        if not deviations:
            return 1.0

        # Calculate average deviation
        avg_deviation = sum(deviations) / len(deviations)

        # Convert to health score (1.0 = perfect health, 0.0 = critical)
        return max(0.0, 1.0 - avg_deviation)

    def _determine_health_status(self, health_score: float) -> ComponentHealth:
        """Determine health status based on health score."""
        if health_score >= 0.9:
            return ComponentHealth.EXCELLENT
        if health_score >= 0.8:
            return ComponentHealth.GOOD
        if health_score >= 0.6:
            return ComponentHealth.FAIR
        if health_score >= 0.4:
            return ComponentHealth.POOR
        return ComponentHealth.CRITICAL

    def _calculate_degradation_rate(
        self, component: str, current_health_score: float
    ) -> float:
        """Calculate health degradation rate."""
        if (
            component not in self.health_history
            or len(self.health_history[component]) < 2
        ):
            return 0.0

        recent_assessments = self.health_history[component][-10:]  # Last 10 assessments

        if len(recent_assessments) < 2:
            return 0.0

        # Calculate average degradation rate
        degradation_rates = []
        for i in range(1, len(recent_assessments)):
            time_diff = (
                recent_assessments[i].timestamp - recent_assessments[i - 1].timestamp
            ).total_seconds() / 3600  # hours
            if time_diff > 0:
                health_diff = (
                    recent_assessments[i - 1].health_score
                    - recent_assessments[i].health_score
                )
                degradation_rate = health_diff / time_diff  # health points per hour
                degradation_rates.append(degradation_rate)

        if degradation_rates:
            return sum(degradation_rates) / len(degradation_rates)

        return 0.0

    def _estimate_maintenance_time(
        self, health_score: float, degradation_rate: float
    ) -> int | None:
        """Estimate time until maintenance is needed (in minutes)."""
        if degradation_rate <= 0:
            return None  # No degradation detected

        # Define maintenance threshold
        maintenance_threshold = 0.6  # Maintenance needed when health drops below 60%

        if health_score <= maintenance_threshold:
            return 0  # Maintenance needed now

        # Calculate time to reach threshold
        health_margin = health_score - maintenance_threshold
        time_to_threshold_hours = health_margin / degradation_rate

        # Convert to minutes and add safety margin
        time_to_maintenance_minutes = int(
            time_to_threshold_hours * 60 * 0.8
        )  # 20% safety margin

        # Cap at reasonable values
        return min(
            max(time_to_maintenance_minutes, 60), 10080
        )  # Between 1 hour and 1 week

    def _identify_risk_factors(
        self, component: str, metrics: dict[str, float]
    ) -> list[str]:
        """Identify risk factors for the component."""
        risk_factors = []

        # Component-specific risk analysis
        if component == "database":
            if metrics.get("connection_count", 0) > 80:
                risk_factors.append("High connection count")
            if metrics.get("query_time_avg", 0) > 1000:  # ms
                risk_factors.append("Slow query performance")

        elif component == "cache":
            if metrics.get("hit_ratio", 1.0) < 0.7:
                risk_factors.append("Low cache hit ratio")
            if metrics.get("memory_usage", 0) > 0.9:
                risk_factors.append("High cache memory usage")

        elif component == "api_gateway":
            if metrics.get("error_rate", 0) > 0.05:
                risk_factors.append("Elevated error rate")
            if metrics.get("response_time_p95", 0) > 5000:  # ms
                risk_factors.append("High response times")

        # General system risks
        if metrics.get("cpu_percent", 0) > 80:
            risk_factors.append("High CPU usage")

        if metrics.get("memory_percent", 0) > 85:
            risk_factors.append("High memory usage")

        if metrics.get("disk_percent", 0) > 90:
            risk_factors.append("Low disk space")

        return risk_factors

    def _generate_maintenance_recommendations(
        self, component: str, health_status: ComponentHealth, risk_factors: list[str]
    ) -> list[str]:
        """Generate maintenance recommendations based on health assessment."""
        recommendations = []

        if health_status == ComponentHealth.CRITICAL:
            recommendations.append("Immediate maintenance required")
            recommendations.append("Consider component replacement or major repair")

        elif health_status == ComponentHealth.POOR:
            recommendations.append("Schedule maintenance within 24 hours")
            recommendations.append("Monitor closely for further degradation")

        elif health_status == ComponentHealth.FAIR:
            recommendations.append("Schedule preventive maintenance within a week")
            recommendations.append("Investigate root causes of degradation")

        # Risk-specific recommendations
        for risk_factor in risk_factors:
            if "connection" in risk_factor.lower():
                recommendations.append("Optimize database connection pooling")
            elif "cache" in risk_factor.lower():
                recommendations.append("Review cache configuration and sizing")
            elif "error" in risk_factor.lower():
                recommendations.append("Investigate and fix error sources")
            elif "response" in risk_factor.lower():
                recommendations.append("Optimize application performance")
            elif "cpu" in risk_factor.lower():
                recommendations.append("Analyze CPU usage patterns and optimize")
            elif "memory" in risk_factor.lower():
                recommendations.append("Review memory usage and potential leaks")
            elif "disk" in risk_factor.lower():
                recommendations.append("Clean up disk space and archive old data")

        # Component-specific recommendations
        if component == "database":
            recommendations.append("Review query performance and indexes")
            recommendations.append("Check database statistics and maintenance jobs")
        elif component == "cache":
            recommendations.append("Analyze cache usage patterns")
            recommendations.append("Consider cache eviction policy adjustments")
        elif component == "api_gateway":
            recommendations.append("Review API endpoint performance")
            recommendations.append(
                "Check rate limiting and circuit breaker configurations"
            )

        return list(set(recommendations))  # Remove duplicates

    def _analyze_historical_patterns(self, component: str) -> dict[str, Any]:
        """Analyze historical patterns for the component."""
        if component not in self.health_history:
            return {}

        history = self.health_history[component]
        if len(history) < 5:
            return {"insufficient_data": True}

        # Calculate trends
        health_scores = [assessment.health_score for assessment in history[-20:]]

        return {
            "average_health_score": sum(health_scores) / len(health_scores),
            "health_trend": "improving"
            if health_scores[-1] > health_scores[0]
            else "degrading",
            "volatility": np.std(health_scores) if len(health_scores) > 1 else 0,
            "maintenance_frequency": len(
                [
                    a
                    for a in history
                    if a.time_to_maintenance and a.time_to_maintenance < 1440
                ]
            ),  # Last day
            "common_risk_factors": self._find_common_risk_factors(history),
        }

    def _find_common_risk_factors(
        self, history: list[ComponentHealthAssessment]
    ) -> list[str]:
        """Find most common risk factors in component history."""
        risk_factor_counts = {}

        for assessment in history[-10:]:  # Last 10 assessments
            for risk_factor in assessment.risk_factors:
                risk_factor_counts[risk_factor] = (
                    risk_factor_counts.get(risk_factor, 0) + 1
                )

        # Return risk factors that appear in at least 30% of assessments
        threshold = len(history[-10:]) * 0.3
        common_factors = [
            factor for factor, count in risk_factor_counts.items() if count >= threshold
        ]

        return sorted(common_factors, key=lambda x: risk_factor_counts[x], reverse=True)


class PredictiveMaintenanceScheduler:
    """Schedules predictive maintenance based on ML predictions and business constraints."""

    def __init__(
        self,
        health_monitor: AutonomousHealthMonitor,
        remediation_engine: AutoRemediationEngine,
    ):
        self.health_monitor = health_monitor
        self.remediation_engine = remediation_engine
        self.time_series_predictor = TimeSeriesPredictor()
        self.health_analyzer = ComponentHealthAnalyzer()

        # Scheduling state
        self.pending_recommendations: list[MaintenanceRecommendation] = []
        self.scheduled_maintenance: list[MaintenanceExecution] = []
        self.execution_history: list[MaintenanceExecution] = []

        # Configuration
        self.monitoring_enabled = False
        self.prediction_interval = 300  # 5 minutes
        self.max_concurrent_maintenance = 2

    async def start_predictive_monitoring(self):
        """Start continuous predictive maintenance monitoring."""
        if self.monitoring_enabled:
            logger.warning("Predictive maintenance monitoring already active")
            return

        self.monitoring_enabled = True
        logger.info("Starting predictive maintenance monitoring")

        try:
            await self.continuous_prediction_loop()
        except Exception as e:
            logger.exception("Predictive maintenance monitoring failed")
            self.monitoring_enabled = False
            raise

    async def stop_predictive_monitoring(self):
        """Stop predictive maintenance monitoring."""
        self.monitoring_enabled = False
        logger.info("Stopped predictive maintenance monitoring")

    async def continuous_prediction_loop(self):
        """Main predictive maintenance loop."""
        while self.monitoring_enabled:
            try:
                # 1. Collect current system metrics
                current_metrics = (
                    await self.health_monitor.collect_comprehensive_health_metrics()
                )

                # 2. Add to time series for prediction
                await self.update_time_series_data(current_metrics)

                # 3. Detect anomalies
                anomalies = await self.detect_metric_anomalies(current_metrics)

                # 4. Assess component health
                health_assessments = await self.assess_all_component_health(
                    current_metrics
                )

                # 5. Generate maintenance recommendations
                new_recommendations = await self.generate_maintenance_recommendations(
                    health_assessments, anomalies
                )

                # 6. Update recommendation list
                self.pending_recommendations.extend(new_recommendations)

                # 7. Execute scheduled maintenance
                await self.execute_scheduled_maintenance()

                # 8. Clean up completed maintenance
                await self.cleanup_completed_maintenance()

                # Log status
                if new_recommendations or anomalies:
                    logger.info(
                        f"Predictive maintenance cycle: {len(new_recommendations)} new recommendations, "
                        f"{len(anomalies)} anomalies detected"
                    )

            except Exception as e:
                logger.exception("Error in predictive maintenance loop")

            # Wait for next cycle
            await asyncio.sleep(self.prediction_interval)

    async def update_time_series_data(self, metrics: SystemMetrics):
        """Update time series data with current metrics."""
        timestamp = datetime.utcnow()

        # Add system-level metrics
        system_metrics = [
            TimeSeriesData(timestamp, "cpu_percent", metrics.cpu_percent, "system"),
            TimeSeriesData(
                timestamp, "memory_percent", metrics.memory_percent, "system"
            ),
            TimeSeriesData(timestamp, "disk_percent", metrics.disk_percent, "system"),
            TimeSeriesData(
                timestamp, "response_time_p95", metrics.response_time_p95, "api_gateway"
            ),
            TimeSeriesData(timestamp, "error_rate", metrics.error_rate, "api_gateway"),
            TimeSeriesData(
                timestamp, "cache_hit_ratio", metrics.cache_hit_ratio, "cache"
            ),
            TimeSeriesData(
                timestamp,
                "database_connections",
                metrics.database_connections,
                "database",
            ),
        ]

        for metric_data in system_metrics:
            self.time_series_predictor.add_data_point(metric_data)

    async def detect_metric_anomalies(
        self, metrics: SystemMetrics
    ) -> list[AnomalyDetection]:
        """Detect anomalies in current metrics."""
        timestamp = datetime.utcnow()
        anomalies = []

        # Check each metric for anomalies
        metric_checks = [
            TimeSeriesData(timestamp, "cpu_percent", metrics.cpu_percent, "system"),
            TimeSeriesData(
                timestamp, "memory_percent", metrics.memory_percent, "system"
            ),
            TimeSeriesData(
                timestamp, "response_time_p95", metrics.response_time_p95, "api_gateway"
            ),
            TimeSeriesData(timestamp, "error_rate", metrics.error_rate, "api_gateway"),
            TimeSeriesData(
                timestamp, "cache_hit_ratio", metrics.cache_hit_ratio, "cache"
            ),
        ]

        for metric_data in metric_checks:
            anomaly = self.time_series_predictor.detect_anomalies(metric_data)
            if anomaly:
                anomalies.append(anomaly)
                logger.warning(
                    f"Anomaly detected: {anomaly.metric_name} = {anomaly.current_value} "
                    f"(expected: {anomaly.expected_range[0]:.2f}-{anomaly.expected_range[1]:.2f}) "
                    f"- {anomaly.severity} severity"
                )

        return anomalies

    async def assess_all_component_health(
        self, metrics: SystemMetrics
    ) -> list[ComponentHealthAssessment]:
        """Assess health of all system components."""
        components = {
            "system": {
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
                "disk_percent": metrics.disk_percent,
            },
            "database": {
                "connection_count": metrics.database_connections,
                "cpu_percent": metrics.cpu_percent,
                "memory_percent": metrics.memory_percent,
            },
            "cache": {
                "hit_ratio": metrics.cache_hit_ratio,
                "memory_percent": metrics.memory_percent,
            },
            "api_gateway": {
                "response_time_p95": metrics.response_time_p95,
                "error_rate": metrics.error_rate,
                "cpu_percent": metrics.cpu_percent,
            },
        }

        assessments = []
        for component, component_metrics in components.items():
            assessment = await self.health_analyzer.assess_component_health(
                component, component_metrics
            )
            assessments.append(assessment)

        return assessments

    async def generate_maintenance_recommendations(
        self,
        health_assessments: list[ComponentHealthAssessment],
        anomalies: list[AnomalyDetection],
    ) -> list[MaintenanceRecommendation]:
        """Generate maintenance recommendations based on health assessments and anomalies."""
        recommendations = []
        current_time = datetime.utcnow()

        # Generate recommendations based on health assessments
        for assessment in health_assessments:
            if assessment.health_status in [
                ComponentHealth.POOR,
                ComponentHealth.CRITICAL,
            ]:
                urgency = (
                    MaintenanceUrgency.HIGH
                    if assessment.health_status == ComponentHealth.CRITICAL
                    else MaintenanceUrgency.MEDIUM
                )

                # Determine maintenance type
                if (
                    assessment.time_to_maintenance
                    and assessment.time_to_maintenance < 60
                ):  # Less than 1 hour
                    maintenance_type = MaintenanceType.EMERGENCY
                    urgency = MaintenanceUrgency.CRITICAL
                elif assessment.degradation_rate > 0.01:  # Significant degradation
                    maintenance_type = MaintenanceType.PREDICTIVE
                else:
                    maintenance_type = MaintenanceType.PREVENTIVE

                # Calculate optimal execution window
                if urgency == MaintenanceUrgency.CRITICAL:
                    window_start = current_time
                    window_end = current_time + timedelta(hours=1)
                elif urgency == MaintenanceUrgency.HIGH:
                    window_start = current_time + timedelta(hours=1)
                    window_end = current_time + timedelta(hours=6)
                else:
                    window_start = current_time + timedelta(hours=6)
                    window_end = current_time + timedelta(days=1)

                recommendation = MaintenanceRecommendation(
                    recommendation_id=f"maint_{assessment.component_name}_{int(time.time())}",
                    component=assessment.component_name,
                    maintenance_type=maintenance_type,
                    urgency=urgency,
                    description=f"Maintenance required for {assessment.component_name} (health: {assessment.health_status.value})",
                    estimated_duration_minutes=self._estimate_maintenance_duration(
                        assessment.component_name, maintenance_type
                    ),
                    optimal_execution_window=(window_start, window_end),
                    prerequisites=self._get_maintenance_prerequisites(
                        assessment.component_name
                    ),
                    expected_benefits=assessment.maintenance_recommendations,
                    risk_if_delayed=self._assess_delay_risk(assessment),
                    confidence_score=min(1.0, assessment.health_score + 0.3),
                )

                recommendations.append(recommendation)

        # Generate recommendations based on anomalies
        for anomaly in anomalies:
            if anomaly.severity in ["high", "critical"]:
                urgency = (
                    MaintenanceUrgency.HIGH
                    if anomaly.severity == "critical"
                    else MaintenanceUrgency.MEDIUM
                )

                recommendation = MaintenanceRecommendation(
                    recommendation_id=f"anomaly_{anomaly.metric_name}_{int(time.time())}",
                    component=self._map_metric_to_component(anomaly.metric_name),
                    maintenance_type=MaintenanceType.CORRECTIVE,
                    urgency=urgency,
                    description=f"Address {anomaly.metric_name} anomaly: {anomaly.recommendation}",
                    estimated_duration_minutes=30,
                    optimal_execution_window=(
                        current_time,
                        current_time + timedelta(hours=2),
                    ),
                    prerequisites=["Validate anomaly persistence"],
                    expected_benefits=[f"Resolve {anomaly.metric_name} anomaly"],
                    risk_if_delayed="Potential system degradation or failure",
                    confidence_score=min(1.0, abs(anomaly.anomaly_score) / 2.0),
                )

                recommendations.append(recommendation)

        return recommendations

    async def execute_scheduled_maintenance(self):
        """Execute scheduled maintenance tasks."""
        if len(self.scheduled_maintenance) >= self.max_concurrent_maintenance:
            return

        # Find highest priority pending recommendations
        urgent_recommendations = [
            rec
            for rec in self.pending_recommendations
            if rec.urgency in [MaintenanceUrgency.CRITICAL, MaintenanceUrgency.HIGH]
            and datetime.utcnow() >= rec.optimal_execution_window[0]
        ]

        # Sort by urgency and confidence
        urgent_recommendations.sort(
            key=lambda r: (self._urgency_priority(r.urgency), r.confidence_score),
            reverse=True,
        )

        # Execute top recommendations
        for recommendation in urgent_recommendations[
            : self.max_concurrent_maintenance - len(self.scheduled_maintenance)
        ]:
            execution = await self.start_maintenance_execution(recommendation)
            if execution:
                self.scheduled_maintenance.append(execution)
                self.pending_recommendations.remove(recommendation)

    async def start_maintenance_execution(
        self, recommendation: MaintenanceRecommendation
    ) -> MaintenanceExecution | None:
        """Start executing a maintenance recommendation."""
        execution = MaintenanceExecution(
            execution_id=f"exec_{recommendation.recommendation_id}",
            recommendation=recommendation,
            status="starting",
            start_time=datetime.utcnow(),
        )

        try:
            logger.info(f"Starting maintenance execution: {recommendation.description}")

            # Capture metrics before maintenance
            current_metrics = (
                await self.health_monitor.collect_comprehensive_health_metrics()
            )
            execution.metrics_before = {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "response_time_p95": current_metrics.response_time_p95,
                "error_rate": current_metrics.error_rate,
            }

            # Execute maintenance based on type
            if recommendation.maintenance_type == MaintenanceType.PREDICTIVE:
                success = await self._execute_predictive_maintenance(recommendation)
            elif recommendation.maintenance_type == MaintenanceType.CORRECTIVE:
                success = await self._execute_corrective_maintenance(recommendation)
            elif recommendation.maintenance_type == MaintenanceType.EMERGENCY:
                success = await self._execute_emergency_maintenance(recommendation)
            else:
                success = await self._execute_preventive_maintenance(recommendation)

            execution.success = success
            execution.status = "completed" if success else "failed"
            execution.end_time = datetime.utcnow()

            # Capture metrics after maintenance
            post_metrics = (
                await self.health_monitor.collect_comprehensive_health_metrics()
            )
            execution.metrics_after = {
                "cpu_percent": post_metrics.cpu_percent,
                "memory_percent": post_metrics.memory_percent,
                "response_time_p95": post_metrics.response_time_p95,
                "error_rate": post_metrics.error_rate,
            }

            logger.info(
                f"Maintenance execution completed: {execution.execution_id} - Success: {success}"
            )

            return execution

        except Exception as e:
            logger.exception("Maintenance execution failed")
            execution.status = "failed"
            execution.success = False
            execution.end_time = datetime.utcnow()
            execution.notes = f"Execution failed: {e!s}"
            return execution

    async def _execute_predictive_maintenance(
        self, recommendation: MaintenanceRecommendation
    ) -> bool:
        """Execute predictive maintenance tasks."""
        component = recommendation.component

        if component == "system":
            # System-level predictive maintenance
            await asyncio.sleep(2)  # Simulate system optimization
            return True
        if component == "database":
            # Database predictive maintenance
            await asyncio.sleep(3)  # Simulate database optimization
            return True
        if component == "cache":
            # Cache predictive maintenance
            await asyncio.sleep(1)  # Simulate cache optimization
            return True
        if component == "api_gateway":
            # API gateway predictive maintenance
            await asyncio.sleep(2)  # Simulate API optimization
            return True

        return False

    async def _execute_corrective_maintenance(
        self, recommendation: MaintenanceRecommendation
    ) -> bool:
        """Execute corrective maintenance for detected issues."""
        # Create a detected issue for the remediation engine
        issue = DetectedIssue(
            issue_id=f"corrective_{recommendation.recommendation_id}",
            issue_type="anomaly_correction",
            severity=RemediationSeverity.MODERATE,
            description=recommendation.description,
            affected_components=[recommendation.component],
            metrics_snapshot={},
            detection_time=datetime.utcnow(),
            contributing_factors=["Predictive maintenance detected anomaly"],
            business_impact_score=0.5,
            auto_remediation_eligible=True,
        )

        # Use remediation engine for corrective action
        result = await self.remediation_engine.process_issue(issue)
        return result is not None and result.success

    async def _execute_emergency_maintenance(
        self, recommendation: MaintenanceRecommendation
    ) -> bool:
        """Execute emergency maintenance procedures."""
        logger.warning(
            f"Executing emergency maintenance for {recommendation.component}"
        )

        # Emergency procedures are more aggressive
        if recommendation.component == "system":
            # Emergency system recovery
            await asyncio.sleep(1)  # Simulate emergency procedures
            return True
        if recommendation.component == "database":
            # Emergency database recovery
            await asyncio.sleep(2)  # Simulate emergency DB procedures
            return True
        if recommendation.component == "cache":
            # Emergency cache recovery
            await asyncio.sleep(0.5)  # Simulate emergency cache procedures
            return True

        return False

    async def _execute_preventive_maintenance(
        self, recommendation: MaintenanceRecommendation
    ) -> bool:
        """Execute preventive maintenance tasks."""
        logger.info(f"Executing preventive maintenance for {recommendation.component}")

        # Preventive maintenance is less urgent
        await asyncio.sleep(5)  # Simulate longer preventive procedures
        return True

    async def cleanup_completed_maintenance(self):
        """Clean up completed maintenance executions."""
        completed_executions = [
            exec
            for exec in self.scheduled_maintenance
            if exec.status in ["completed", "failed"]
        ]

        for execution in completed_executions:
            # Move to history
            self.execution_history.append(execution)

            # Remove from scheduled
            self.scheduled_maintenance.remove(execution)

        # Keep only recent history
        if len(self.execution_history) > 500:
            self.execution_history = self.execution_history[-500:]

    def _estimate_maintenance_duration(
        self, component: str, maintenance_type: MaintenanceType
    ) -> int:
        """Estimate maintenance duration in minutes."""
        base_durations = {"system": 30, "database": 45, "cache": 15, "api_gateway": 20}

        type_multipliers = {
            MaintenanceType.PREVENTIVE: 1.5,
            MaintenanceType.PREDICTIVE: 1.0,
            MaintenanceType.CORRECTIVE: 0.8,
            MaintenanceType.EMERGENCY: 0.5,
        }

        base_duration = base_durations.get(component, 30)
        multiplier = type_multipliers.get(maintenance_type, 1.0)

        return int(base_duration * multiplier)

    def _get_maintenance_prerequisites(self, component: str) -> list[str]:
        """Get prerequisites for component maintenance."""
        prerequisites = {
            "system": ["Backup system state", "Notify operations team"],
            "database": [
                "Backup database",
                "Verify replication status",
                "Check active connections",
            ],
            "cache": ["Warm cache backup", "Check cache dependencies"],
            "api_gateway": ["Check traffic patterns", "Verify downstream services"],
        }

        return prerequisites.get(component, ["Verify system health"])

    def _assess_delay_risk(self, assessment: ComponentHealthAssessment) -> str:
        """Assess risk of delaying maintenance."""
        if assessment.health_status == ComponentHealth.CRITICAL:
            return "High risk of system failure if delayed"
        if assessment.health_status == ComponentHealth.POOR:
            return "Moderate risk of service degradation"
        if assessment.degradation_rate > 0.02:
            return "Accelerating degradation if not addressed"
        return "Low risk if delayed within recommended window"

    def _map_metric_to_component(self, metric_name: str) -> str:
        """Map metric name to component."""
        if "cpu" in metric_name or "memory" in metric_name or "disk" in metric_name:
            return "system"
        if "response" in metric_name or "error" in metric_name:
            return "api_gateway"
        if "cache" in metric_name:
            return "cache"
        if "database" in metric_name or "connection" in metric_name:
            return "database"
        return "system"

    def _urgency_priority(self, urgency: MaintenanceUrgency) -> int:
        """Convert urgency to priority number."""
        priorities = {
            MaintenanceUrgency.CRITICAL: 4,
            MaintenanceUrgency.HIGH: 3,
            MaintenanceUrgency.MEDIUM: 2,
            MaintenanceUrgency.LOW: 1,
        }
        return priorities.get(urgency, 0)

    async def get_predictive_maintenance_status(self) -> dict[str, Any]:
        """Get current predictive maintenance system status."""
        return {
            "monitoring_enabled": self.monitoring_enabled,
            "prediction_interval_seconds": self.prediction_interval,
            "pending_recommendations": len(self.pending_recommendations),
            "scheduled_maintenance": len(self.scheduled_maintenance),
            "execution_history_count": len(self.execution_history),
            "max_concurrent_maintenance": self.max_concurrent_maintenance,
            "recent_executions_successful": len(
                [e for e in self.execution_history[-20:] if e.success]
            ),
            "recent_executions_failed": len(
                [e for e in self.execution_history[-20:] if not e.success]
            ),
            "components_monitored": len(self.health_analyzer.component_baselines),
            "anomaly_detectors_active": len(
                self.time_series_predictor.anomaly_detectors
            ),
        }
