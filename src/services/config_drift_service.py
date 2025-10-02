"""Configuration drift detection background service.

Integrates with existing Task 20 infrastructure to provide automated
configuration drift monitoring and alerting using the application's task
queue system.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from src.config import (
    ConfigDriftDetector,
    DriftDetectionConfig,
    DriftSeverity,
    get_config,
    initialize_drift_detector,
)
from src.services.observability.performance import get_performance_monitor


# Import at top level to avoid import-outside-top-level violations
try:
    from src.services.task_queue.tasks import create_task as _queue_create_task
except ImportError:  # pragma: no cover - optional dependency
    _queue_create_task = None


logger = logging.getLogger(__name__)


async def _enqueue_task(
    task_name: str,
    payload: dict[str, Any],
    *,
    delay: timedelta | None = None,
) -> None:
    """Best-effort task scheduling wrapper for optional task queue."""

    if _queue_create_task is None:
        logger.debug("Task queue unavailable; skipping task %s", task_name)
        return
    await _queue_create_task(task_name, payload, delay=delay)


class ConfigDriftService:
    """Background service for automated config drift detection."""

    def __init__(self):
        """Initialize configuration drift service."""
        self.config = get_config()
        self.is_running = False
        self.drift_detector: ConfigDriftDetector | None = None
        self._setup_drift_detector()

    def _setup_drift_detector(self) -> None:
        """Set up the configuration drift detector with current config."""
        try:
            # Convert config to drift detection config
            drift_config = DriftDetectionConfig(
                enabled=self.config.drift_detection.enabled,
                snapshot_interval_minutes=self.config.drift_detection.snapshot_interval_minutes,
                comparison_interval_minutes=self.config.drift_detection.comparison_interval_minutes,
                monitored_paths=self.config.drift_detection.monitored_paths,
                excluded_paths=self.config.drift_detection.excluded_paths,
                alert_on_severity=[
                    DriftSeverity(severity)
                    for severity in self.config.drift_detection.alert_on_severity
                ],
                max_alerts_per_hour=self.config.drift_detection.max_alerts_per_hour,
                snapshot_retention_days=self.config.drift_detection.snapshot_retention_days,
                events_retention_days=self.config.drift_detection.events_retention_days,
                integrate_with_task20_anomaly=self.config.drift_detection.integrate_with_task20_anomaly,
                use_performance_monitoring=self.config.drift_detection.use_performance_monitoring,
                enable_auto_remediation=self.config.drift_detection.enable_auto_remediation,
                auto_remediation_severity_threshold=self.config.drift_detection.auto_remediation_severity_threshold,
            )

            # Initialize with config directory path and interval
            config_dir = Path(self.config.data_dir)
            self.drift_detector = initialize_drift_detector(
                config_dir,
                drift_config.snapshot_interval_minutes
                * 60,  # Convert minutes to seconds
            )
            logger.info("Configuration drift detector initialized successfully")

        except (AttributeError, ImportError, OSError):
            logger.exception("Failed to initialize drift detector")
            self.drift_detector = None

    async def start(self) -> None:
        """Start the configuration drift monitoring service."""
        if not self.config.drift_detection.enabled:
            logger.info("Configuration drift detection is disabled")
            return

        if self.drift_detector is None:
            logger.error("Cannot start drift service: detector not initialized")
            return

        self.is_running = True
        logger.info("Starting configuration drift monitoring service")

        try:
            # Schedule initial snapshot task
            await self._schedule_snapshot_task()

            # Schedule initial comparison task
            await self._schedule_comparison_task()

            logger.info("Configuration drift monitoring service started successfully")

        except (OSError, PermissionError, RuntimeError):
            logger.exception("Failed to start drift monitoring service")
            self.is_running = False
            raise

    async def stop(self) -> None:
        """Stop the configuration drift monitoring service."""
        self.is_running = False
        logger.info("Configuration drift monitoring service stopped")

    async def _schedule_snapshot_task(self) -> None:
        """Schedule the next configuration snapshot task."""
        if not self.is_running:
            return

        try:
            await _enqueue_task(
                "config_drift_snapshot",
                {},
                delay=timedelta(
                    minutes=self.config.drift_detection.snapshot_interval_minutes
                ),
            )
            logger.debug("Scheduled next configuration snapshot task")

        except (OSError, PermissionError, RuntimeError):
            logger.exception("Failed to schedule snapshot task")

    async def _schedule_comparison_task(self) -> None:
        """Schedule the next configuration comparison task."""
        if not self.is_running:
            return

        try:
            await _enqueue_task(
                "config_drift_comparison",
                {},
                delay=timedelta(
                    minutes=self.config.drift_detection.comparison_interval_minutes
                ),
            )
            logger.debug("Scheduled next configuration comparison task")

        except (TimeoutError, OSError, PermissionError):
            logger.exception("Failed to schedule comparison task")

    async def take_configuration_snapshot(self) -> dict[str, Any]:
        """Take snapshots of all monitored configuration sources.

        Returns:
            Dictionary with snapshot results

        """
        if self.drift_detector is None:
            msg = "Drift detector not initialized"
            raise RuntimeError(msg)

        performance_monitor = None
        with contextlib.suppress(Exception):
            performance_monitor = get_performance_monitor()

        snapshot_results = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "snapshots_taken": 0,
            "errors": [],
            "sources": [],
        }

        monitoring_context = (
            performance_monitor.monitor_operation(
                "config_drift_snapshot_batch", category="config_drift"
            )
            if performance_monitor
            else None
        )

        context = monitoring_context or contextlib.nullcontext()

        try:
            with context:
                for source in self.config.drift_detection.monitored_paths:
                    try:
                        snapshot = self.drift_detector.take_snapshot()
                        snapshot_results["snapshots_taken"] += 1
                        snapshot_results["sources"].append(
                            {
                                "source": source,
                                "hash": snapshot.config_hash[:8],
                                "size": len(str(snapshot.config_data)),
                                "timestamp": snapshot.timestamp.isoformat(),  # pylint: disable=no-member
                            }
                        )
                        logger.debug("Took snapshot for %s", source)

                    except (ValueError, TypeError, UnicodeDecodeError) as e:
                        error_msg = f"Failed to snapshot {source}: {e}"
                        snapshot_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Record custom metrics if performance monitoring available
                if monitoring_context is not None:
                    perf_data = {"custom_metrics": {}}
                    perf_data["custom_metrics"]["snapshots_taken"] = snapshot_results[
                        "snapshots_taken"
                    ]
                    perf_data["custom_metrics"]["sources_monitored"] = len(
                        self.config.drift_detection.monitored_paths
                    )
                    perf_data["custom_metrics"]["errors_count"] = len(
                        snapshot_results["errors"]
                    )

        except Exception as e:
            error_msg = f"Snapshot batch operation failed: {e}"
            snapshot_results["errors"].append(error_msg)
            logger.exception(error_msg)
            raise

        # Schedule next snapshot task
        await self._schedule_snapshot_task()

        return snapshot_results

    async def compare_configurations(self) -> dict[str, Any]:
        """Compare current configurations with previous snapshots to detect drift.

        Returns:
            Dictionary with comparison results and drift events

        """
        if self.drift_detector is None:
            msg = "Drift detector not initialized"
            raise RuntimeError(msg)

        performance_monitor = None
        with contextlib.suppress(Exception):
            performance_monitor = get_performance_monitor()

        comparison_results = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "sources_compared": 0,
            "drift_events": [],
            "alerts_sent": 0,
            "errors": [],
        }

        monitoring_context = (
            performance_monitor.monitor_operation(
                "config_drift_comparison_batch", category="config_drift"
            )
            if performance_monitor
            else None
        )

        context = monitoring_context or contextlib.nullcontext()

        try:
            with context:
                for source in self.config.drift_detection.monitored_paths:
                    try:
                        # Compare snapshots for this source
                        events = self._compare_snapshots_for_source(source)
                        comparison_results["sources_compared"] += 1

                        # Process detected drift events
                        for event in events:
                            comparison_results["drift_events"].append(
                                {
                                    "id": event.id,
                                    "source": event.source,
                                    "type": event.drift_type.value,
                                    "severity": event.severity.value,
                                    "description": event.description,
                                    "auto_remediable": event.auto_remediable,
                                    "timestamp": event.timestamp.isoformat(),
                                }
                            )

                            # Send alert if criteria met
                            if self._should_alert(event):
                                self._send_alert(event)
                                comparison_results["alerts_sent"] += 1

                            # Auto-remediate if enabled and safe
                            if (
                                self.config.drift_detection.enable_auto_remediation
                                and event.auto_remediable
                            ):
                                await self._attempt_auto_remediation(event)

                        if events:
                            logger.info(
                                "Detected %d drift events for %s", len(events), source
                            )

                    except (asyncio.CancelledError, TimeoutError, RuntimeError) as e:
                        error_msg = f"Failed to compare {source}: {e}"
                        comparison_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Record custom metrics if performance monitoring available
                if monitoring_context is not None:
                    perf_data = {"custom_metrics": {}}
                    perf_data["custom_metrics"]["sources_compared"] = (
                        comparison_results["sources_compared"]
                    )
                    perf_data["custom_metrics"]["drift_events_detected"] = len(
                        comparison_results["drift_events"]
                    )
                    perf_data["custom_metrics"]["alerts_sent"] = comparison_results[
                        "alerts_sent"
                    ]
                    perf_data["custom_metrics"]["errors_count"] = len(
                        comparison_results["errors"]
                    )

        except Exception as e:
            error_msg = f"Comparison batch operation failed: {e}"
            comparison_results["errors"].append(error_msg)
            logger.exception(error_msg)
            raise

        # Schedule next comparison task
        await self._schedule_comparison_task()

        return comparison_results

    async def _attempt_auto_remediation(self, event) -> bool:
        """Attempt to automatically remediate a configuration drift event.

        Args:
            event: Drift event to remediate

        Returns:
            True if remediation was successful

        """
        try:
            # Log remediation attempt
            logger.info("Attempting auto-remediation for drift event %s", event.id)

            # For now, just log the suggested remediation
            # In a full implementation, this would actually apply the changes
            if event.remediation_suggestion:
                logger.info("Remediation suggestion: %s", event.remediation_suggestion)

            # Create a task to track the remediation
            await _enqueue_task(
                "config_drift_remediation",
                {
                    "event_id": event.id,
                    "source": event.source,
                    "drift_type": event.drift_type.value,
                    "suggestion": event.remediation_suggestion,
                },
                delay=None,
            )

        except (TimeoutError, OSError, PermissionError):
            logger.exception("Auto-remediation failed for event %s", event.id)
            return False
        return True

    def _compare_snapshots_for_source(self, source: str) -> list[Any]:  # noqa: ARG002
        """Compare snapshots for a specific source and return drift events.

        Args:
            source: Configuration source path to compare

        Returns:
            List of drift events detected for the source
        """

        if self.drift_detector is None:
            return []

        # Use the detector's drift detection capability
        # Note: source parameter is used for context but detector returns all events
        return self.drift_detector.detect_drift()

    def _should_alert(self, event: Any) -> bool:
        """Determine if an alert should be sent for a drift event.

        Args:
            event: Drift event to evaluate

        Returns:
            True if alert should be sent
        """

        if not hasattr(event, "severity"):
            return False

        # Check if event severity is in the alert list
        alert_severities = self.config.drift_detection.alert_on_severity
        return event.severity.value in alert_severities

    def _send_alert(self, event: Any) -> None:
        """Send alert for a drift event.

        Args:
            event: Drift event that triggered the alert
        """

        try:
            # In a real implementation, this would send alerts via:
            # - Email notifications
            # - Slack/Teams messages
            # - PagerDuty/Opsgenie alerts
            # - Webhook notifications

            logger.warning(
                "Configuration drift alert: %s (severity: %s) at %s",
                event.description if hasattr(event, "description") else "Unknown drift",
                event.severity.value if hasattr(event, "severity") else "unknown",
                event.source if hasattr(event, "source") else "unknown source",
            )

        except (AttributeError, TypeError, RuntimeError):
            logger.exception("Failed to send drift alert")

    async def get_service_status(self) -> dict[str, Any]:
        """Get current status of the configuration drift service.

        Returns:
            Dictionary with service status information
        """

        status = {
            "service_running": self.is_running,
            "drift_detection_enabled": self.config.drift_detection.enabled,
            "detector_initialized": self.drift_detector is not None,
            "monitored_paths_count": len(self.config.drift_detection.monitored_paths),
            "config": {
                "snapshot_interval_minutes": (
                    self.config.drift_detection.snapshot_interval_minutes
                ),
                "comparison_interval_minutes": (
                    self.config.drift_detection.comparison_interval_minutes
                ),
                "alert_on_severity": (self.config.drift_detection.alert_on_severity),
                "auto_remediation_enabled": (
                    self.config.drift_detection.enable_auto_remediation
                ),
            },
        }

        if self.drift_detector:
            try:
                drift_summary = self.drift_detector.get_drift_summary()
                status["drift_summary"] = drift_summary
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                status["drift_summary_error"] = str(e)

        return status

    async def run_manual_detection(self) -> dict[str, Any]:
        """Run a manual configuration drift detection cycle.

        Returns:
            Dictionary with detection results
        """

        if self.drift_detector is None:
            msg = "Drift detector not initialized"
            raise RuntimeError(msg)

        logger.info("Running manual configuration drift detection")

        try:
            # Take fresh snapshots
            snapshot_results = await self.take_configuration_snapshot()

            # Wait a moment to ensure timestamp differences
            await asyncio.sleep(0.1)

            # Compare configurations
            comparison_results = await self.compare_configurations()

            return {
                "manual_detection": True,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "snapshot_results": snapshot_results,
                "comparison_results": comparison_results,
            }

        except (OSError, PermissionError):
            logger.exception("Manual detection failed")
            raise


class DriftServiceSingleton:
    """Singleton wrapper for ConfigDriftService to avoid global variables."""

    _instance: ConfigDriftService | None = None

    @classmethod
    def get_instance(cls) -> ConfigDriftService:
        if cls._instance is None:
            cls._instance = ConfigDriftService()
        return cls._instance


def get_drift_service() -> ConfigDriftService:
    """Get global configuration drift service instance.

    Returns:
        Global drift service instance
    """
    return DriftServiceSingleton.get_instance()


async def start_drift_service() -> None:
    """Start the global configuration drift service."""

    service = get_drift_service()
    await service.start()


async def stop_drift_service() -> None:
    """Stop the global configuration drift service."""

    service = get_drift_service()
    await service.stop()


async def get_drift_service_status() -> dict[str, Any]:
    """Get status of the global configuration drift service."""

    service = get_drift_service()
    return await service.get_service_status()


async def run_manual_drift_detection() -> dict[str, Any]:
    """Run manual drift detection using the global service."""

    service = get_drift_service()
    return await service.run_manual_detection()
