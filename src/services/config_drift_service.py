"""Configuration drift detection background service.

Integrates with existing Task 20 infrastructure to provide automated configuration
drift monitoring and alerting using the application's task queue system.
"""

import asyncio
import contextlib
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ..config.core import get_config
from ..config.drift_detection import (
    ConfigDriftDetector,
    DriftDetectionConfig as CoreDriftDetectionConfig,
    DriftSeverity,
    initialize_drift_detector,
)
from ..services.observability.performance import get_performance_monitor
from ..services.task_queue.tasks import create_task


logger = logging.getLogger(__name__)


class ConfigDriftService:
    """Background service for automated configuration drift detection."""

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
            drift_config = CoreDriftDetectionConfig(
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
                auto_remediation_severity_threshold=DriftSeverity(
                    self.config.drift_detection.auto_remediation_severity_threshold
                ),
            )

            self.drift_detector = initialize_drift_detector(drift_config)
            logger.info("Configuration drift detector initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize drift detector: {e}")
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

        except Exception as e:
            logger.exception(f"Failed to start drift monitoring service: {e}")
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
            # Create task for taking configuration snapshots
            await create_task(
                "config_drift_snapshot",
                {},
                delay=timedelta(
                    minutes=self.config.drift_detection.snapshot_interval_minutes
                ),
            )
            logger.debug("Scheduled next configuration snapshot task")

        except Exception as e:
            logger.exception(f"Failed to schedule snapshot task: {e}")

    async def _schedule_comparison_task(self) -> None:
        """Schedule the next configuration comparison task."""
        if not self.is_running:
            return

        try:
            # Create task for comparing configurations
            await create_task(
                "config_drift_comparison",
                {},
                delay=timedelta(
                    minutes=self.config.drift_detection.comparison_interval_minutes
                ),
            )
            logger.debug("Scheduled next configuration comparison task")

        except Exception as e:
            logger.exception(f"Failed to schedule comparison task: {e}")

    async def take_configuration_snapshot(self) -> dict[str, Any]:
        """Take snapshots of all monitored configuration sources.

        Returns:
            Dictionary with snapshot results
        """
        if self.drift_detector is None:
            raise RuntimeError("Drift detector not initialized")

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

        try:
            async with (
                monitoring_context if monitoring_context else asyncio.nullcontext()
            ):
                for source in self.config.drift_detection.monitored_paths:
                    try:
                        snapshot = self.drift_detector.take_snapshot(source)
                        snapshot_results["snapshots_taken"] += 1
                        snapshot_results["sources"].append(
                            {
                                "source": source,
                                "hash": snapshot.config_hash[:8],
                                "size": len(str(snapshot.config_data)),
                                "timestamp": snapshot.timestamp.isoformat(),
                            }
                        )
                        logger.debug(f"Took snapshot for {source}")

                    except Exception as e:
                        error_msg = f"Failed to snapshot {source}: {e}"
                        snapshot_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Record custom metrics if performance monitoring available
                if monitoring_context and hasattr(monitoring_context, "__enter__"):
                    with monitoring_context as perf_data:
                        perf_data["custom_metrics"]["snapshots_taken"] = (
                            snapshot_results["snapshots_taken"]
                        )
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
            raise RuntimeError("Drift detector not initialized")

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

        try:
            async with (
                monitoring_context if monitoring_context else asyncio.nullcontext()
            ):
                for source in self.config.drift_detection.monitored_paths:
                    try:
                        # Compare snapshots for this source
                        events = self.drift_detector.compare_snapshots(source)
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
                            if self.drift_detector.should_alert(event):
                                self.drift_detector.send_alert(event)
                                comparison_results["alerts_sent"] += 1

                            # Auto-remediate if enabled and safe
                            if (
                                self.config.drift_detection.enable_auto_remediation
                                and event.auto_remediable
                            ):
                                await self._attempt_auto_remediation(event)

                        if events:
                            logger.info(
                                f"Detected {len(events)} drift events for {source}"
                            )

                    except Exception as e:
                        error_msg = f"Failed to compare {source}: {e}"
                        comparison_results["errors"].append(error_msg)
                        logger.warning(error_msg)

                # Record custom metrics if performance monitoring available
                if monitoring_context and hasattr(monitoring_context, "__enter__"):
                    with monitoring_context as perf_data:
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
            logger.info(f"Attempting auto-remediation for drift event {event.id}")

            # For now, just log the suggested remediation
            # In a full implementation, this would actually apply the changes
            if event.remediation_suggestion:
                logger.info(f"Remediation suggestion: {event.remediation_suggestion}")

            # Create a task to track the remediation
            await create_task(
                "config_drift_remediation",
                {
                    "event_id": event.id,
                    "source": event.source,
                    "drift_type": event.drift_type.value,
                    "suggestion": event.remediation_suggestion,
                },
            )

            return True

        except Exception as e:
            logger.exception(f"Auto-remediation failed for event {event.id}: {e}")
            return False

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
                "snapshot_interval_minutes": self.config.drift_detection.snapshot_interval_minutes,
                "comparison_interval_minutes": self.config.drift_detection.comparison_interval_minutes,
                "alert_on_severity": self.config.drift_detection.alert_on_severity,
                "auto_remediation_enabled": self.config.drift_detection.enable_auto_remediation,
            },
        }

        if self.drift_detector:
            try:
                drift_summary = self.drift_detector.get_drift_summary()
                status["drift_summary"] = drift_summary
            except Exception as e:
                status["drift_summary_error"] = str(e)

        return status

    async def run_manual_detection(self) -> dict[str, Any]:
        """Run a manual configuration drift detection cycle.

        Returns:
            Dictionary with detection results
        """
        if self.drift_detector is None:
            raise RuntimeError("Drift detector not initialized")

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

        except Exception as e:
            logger.exception(f"Manual detection failed: {e}")
            raise


# Global service instance
_drift_service: ConfigDriftService | None = None


def get_drift_service() -> ConfigDriftService:
    """Get global configuration drift service instance.

    Returns:
        Global drift service instance
    """
    global _drift_service
    if _drift_service is None:
        _drift_service = ConfigDriftService()
    return _drift_service


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
