"""Simple monitoring integration tests."""

import pytest
from src.config.models import MonitoringConfig


class TestMonitoringE2E:
    """Simple monitoring system integration tests."""

    def test_monitoring_config_creation(self):
        """Test that MonitoringConfig can be created with valid defaults."""
        config = MonitoringConfig()

        assert config.enabled is True
        assert config.metrics_port == 8001
        assert config.metrics_path == "/metrics"
        assert config.health_path == "/health"
        assert config.include_system_metrics is True
        assert config.system_metrics_interval == 30.0

    def test_monitoring_config_custom_values(self):
        """Test MonitoringConfig with custom values."""
        config = MonitoringConfig(
            enabled=False,
            metrics_port=9090,
            metrics_path="/custom-metrics",
            health_path="/custom-health",
            include_system_metrics=False,
            system_metrics_interval=60.0,
        )

        assert config.enabled is False
        assert config.metrics_port == 9090
        assert config.metrics_path == "/custom-metrics"
        assert config.health_path == "/custom-health"
        assert config.include_system_metrics is False
        assert config.system_metrics_interval == 60.0

    def test_monitoring_config_validation(self):
        """Test MonitoringConfig field validation."""
        # Test invalid port
        with pytest.raises(ValueError):
            MonitoringConfig(metrics_port=0)

        with pytest.raises(ValueError):
            MonitoringConfig(metrics_port=70000)

        # Test invalid interval
        with pytest.raises(ValueError):
            MonitoringConfig(system_metrics_interval=0)

        with pytest.raises(ValueError):
            MonitoringConfig(health_check_timeout=0)
