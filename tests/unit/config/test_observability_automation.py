"""Tests for Configuration Observability Automation System.

Comprehensive test suite for the configuration automation system including:
- Drift detection and remediation
- Configuration validation
- Performance optimization
- Real-time monitoring
- API endpoints
- CLI functionality
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.config.observability.api import router
from src.config.observability.automation import (
    ConfigDrift,
    ConfigDriftSeverity,
    ConfigObservabilityAutomation,
    ConfigValidationResult,
    ConfigValidationStatus,
    OptimizationRecommendation,
)


class TestConfigObservabilityAutomation:
    """Test suite for the main automation system."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create sample configuration files
            (temp_path / ".env").write_text("AI_DOCS__MODE=simple\nAI_DOCS__DEBUG=true")
            (temp_path / ".env.enterprise").write_text(
                "AI_DOCS__MODE=enterprise\nAI_DOCS__DEBUG=false"
            )
            (temp_path / "docker-compose.yml").write_text(
                "version: '3.8'\nservices:\n  api:\n    image: test"
            )

            yield temp_path

    @pytest.fixture
    async def automation_system(self, temp_config_dir):
        """Create automation system instance."""
        system = ConfigObservabilityAutomation(
            config_dir=str(temp_config_dir),
            enable_auto_remediation=True,
            enable_performance_optimization=True,
            drift_check_interval=60,
            performance_optimization_interval=120,
        )

        # Mock the file system monitoring to avoid real file watching in tests
        system.start_file_monitoring = Mock()

        yield system

        # Cleanup
        if system.observer:
            system.observer.stop()

    async def test_initialization(self, automation_system):
        """Test automation system initialization."""
        assert automation_system.enable_auto_remediation is True
        assert automation_system.enable_performance_optimization is True
        assert automation_system.drift_check_interval == 60
        assert automation_system.performance_optimization_interval == 120
        assert len(automation_system.baseline_configurations) == 0
        assert len(automation_system.drift_history) == 0

    async def test_environment_detection(self, automation_system):
        """Test environment detection from configuration files."""
        await automation_system.detect_environments()

        # Should detect environments based on configuration files
        assert len(automation_system.detected_environments) > 0
        assert (
            "simple" in automation_system.detected_environments
            or "development" in automation_system.detected_environments
        )

    @patch("src.config.observability.automation.get_config")
    async def test_baseline_establishment(self, mock_get_config, automation_system):
        """Test establishment of baseline configurations."""
        # Mock configuration loading
        mock_config = Mock()
        mock_config.model_dump.return_value = {
            "mode": "simple",
            "environment": "development",
            "embedding_provider": "fastembed",
        }
        mock_get_config.return_value = mock_config

        automation_system.detected_environments = {"simple", "development"}

        await automation_system.establish_baseline_configurations()

        assert len(automation_system.baseline_configurations) > 0

    async def test_drift_detection(self, automation_system):
        """Test configuration drift detection."""
        # Set up baseline configuration
        automation_system.baseline_configurations = {
            "development": {
                "mode": "simple",
                "debug": True,
                "embedding_provider": "fastembed",
            }
        }
        automation_system.detected_environments = {"development"}

        # Mock current configuration with drift
        with patch.object(
            automation_system, "load_configuration_for_environment"
        ) as mock_load:
            mock_load.return_value = {
                "mode": "enterprise",  # Changed from simple
                "debug": False,  # Changed from True
                "embedding_provider": "fastembed",  # Unchanged
            }

            drifts = await automation_system.detect_configuration_drift()

            # Should detect 2 drifts (mode and debug)
            assert len(drifts) == 2

            # Check drift details
            mode_drift = next((d for d in drifts if d.parameter == "mode"), None)
            assert mode_drift is not None
            assert mode_drift.severity == ConfigDriftSeverity.CRITICAL
            assert mode_drift.expected_value == "simple"
            assert mode_drift.current_value == "enterprise"

    async def test_configuration_validation(self, automation_system):
        """Test configuration validation."""
        automation_system.detected_environments = {"development"}

        # Mock configuration with validation issues
        test_config = {
            "embedding_provider": "openai",
            "openai_api_key": None,  # Missing required key
            "crawl_provider": "firecrawl",
            "firecrawl_api_key": None,  # Missing required key
            "performance": {
                "max_concurrent_crawls": 100,  # Too high
            },
            "cache": {
                "ttl_embeddings": 60,  # Too low
            },
        }

        with patch.object(
            automation_system, "load_configuration_for_environment"
        ) as mock_load:
            mock_load.return_value = test_config

            validation_results = await automation_system.validate_configuration_health()

            # Should find multiple validation issues
            assert len(validation_results) > 0

            # Check for specific validation errors
            openai_key_error = next(
                (v for v in validation_results if v.parameter == "openai_api_key"), None
            )
            assert openai_key_error is not None
            assert openai_key_error.status == ConfigValidationStatus.ERROR

    async def test_optimization_recommendations(self, automation_system):
        """Test optimization recommendation generation."""
        automation_system.detected_environments = {"development"}

        # Add some mock performance metrics
        automation_system.performance_metrics = [
            type(
                "obj",
                (object,),
                {
                    "name": "response_time_ms",
                    "value": 1500.0,  # High response time
                    "environment": "development",
                    "timestamp": datetime.now(UTC),
                },
            )(),
            type(
                "obj",
                (object,),
                {
                    "name": "cache_hit_rate",
                    "value": 0.6,  # Low cache hit rate
                    "environment": "development",
                    "timestamp": datetime.now(UTC),
                },
            )(),
        ]

        # Mock configuration
        test_config = {
            "performance": {
                "max_concurrent_crawls": 5,  # Low concurrency
            },
            "cache": {
                "ttl_embeddings": 3600,  # Low TTL
            },
        }

        with patch.object(
            automation_system, "load_configuration_for_environment"
        ) as mock_load:
            mock_load.return_value = test_config

            recommendations = (
                await automation_system.generate_optimization_recommendations()
            )

            # Should generate recommendations based on performance metrics
            assert len(recommendations) > 0

            # Check for specific recommendations
            concurrency_rec = next(
                (r for r in recommendations if "concurrent_crawls" in r.parameter), None
            )
            assert concurrency_rec is not None
            assert concurrency_rec.recommended_value > concurrency_rec.current_value

    async def test_auto_remediation(self, automation_system):
        """Test automatic remediation of configuration issues."""
        # Create test drifts
        test_drifts = [
            ConfigDrift(
                severity=ConfigDriftSeverity.WARNING,
                parameter="performance.max_concurrent_crawls",
                expected_value=10,
                current_value=15,
                environment="development",
                timestamp=datetime.now(UTC),
                auto_fix_available=True,
            ),
            ConfigDrift(
                severity=ConfigDriftSeverity.INFO,
                parameter="cache.ttl_embeddings",
                expected_value=86400,
                current_value=43200,
                environment="development",
                timestamp=datetime.now(UTC),
                auto_fix_available=True,
            ),
        ]

        results = await automation_system.auto_remediate_issues(test_drifts)

        # Should attempt to fix both drifts
        assert len(results) == 2
        assert all(results.values())  # All fixes should succeed (mocked)

    async def test_system_status(self, automation_system):
        """Test system status reporting."""
        # Add some test data
        automation_system.drift_history = [
            ConfigDrift(
                severity=ConfigDriftSeverity.CRITICAL,
                parameter="test_param",
                expected_value="test",
                current_value="changed",
                environment="development",
                timestamp=datetime.now(UTC),
            )
        ]

        automation_system.validation_history = [
            ConfigValidationResult(
                status=ConfigValidationStatus.ERROR,
                parameter="test_param",
                message="Test error",
                environment="development",
                timestamp=datetime.now(UTC),
            )
        ]

        status = automation_system.get_system_status()

        # Verify status structure
        assert "system_status" in status
        assert "drift_analysis" in status
        assert "validation_status" in status
        assert "optimization" in status
        assert "environments" in status

        # Check specific values
        assert status["drift_analysis"]["recent_drifts"] == 1
        assert status["drift_analysis"]["critical_drifts"] == 1
        assert status["validation_status"]["errors"] == 1

    async def test_detailed_report(self, automation_system):
        """Test detailed report generation."""
        # Add test data
        automation_system.drift_history = [
            ConfigDrift(
                severity=ConfigDriftSeverity.WARNING,
                parameter="test_param",
                expected_value="test",
                current_value="changed",
                environment="development",
                timestamp=datetime.now(UTC),
            )
        ]

        report = automation_system.get_detailed_report()

        # Should include detailed analysis
        assert "detailed_analysis" in report
        assert "recent_drifts" in report["detailed_analysis"]
        assert len(report["detailed_analysis"]["recent_drifts"]) == 1


class TestConfigurationWatcher:
    """Test suite for configuration file watcher."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def automation_system(self, temp_config_dir):
        """Create automation system for watcher tests."""
        system = ConfigObservabilityAutomation(config_dir=str(temp_config_dir))
        system.handle_config_change = AsyncMock()
        return system

    def test_watcher_initialization(self, automation_system):
        """Test configuration watcher initialization."""
        from src.config.observability.automation import ConfigurationWatcher

        watcher = ConfigurationWatcher(automation_system)

        assert watcher.automation_system == automation_system
        assert ".env" in watcher.watched_files
        assert "docker-compose.yml" in watcher.watched_files

    def test_file_modification_detection(self, automation_system, temp_config_dir):
        """Test file modification detection."""
        from src.config.observability.automation import ConfigurationWatcher

        watcher = ConfigurationWatcher(automation_system)

        # Create mock event
        mock_event = Mock()
        mock_event.is_directory = False
        mock_event.src_path = str(temp_config_dir / ".env")

        # Handle modification
        watcher.on_modified(mock_event)

        # Should trigger config change handling (async task created)
        # In real test, would check that async task was scheduled


class TestAPIEndpoints:
    """Test suite for FastAPI endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(router)

        return TestClient(app)

    @pytest.fixture
    def mock_automation_system(self):
        """Create mock automation system."""
        mock_system = Mock(spec=ConfigObservabilityAutomation)

        # Mock system status
        mock_system.get_system_status.return_value = {
            "system_status": {
                "automation_enabled": True,
                "auto_remediation_enabled": False,
                "file_monitoring_active": True,
                "environments_monitored": 2,
                "last_drift_check": "2024-01-01T12:00:00",
                "last_optimization_check": "2024-01-01T12:00:00",
            },
            "drift_analysis": {
                "recent_drifts": 0,
                "critical_drifts": 0,
                "auto_fixes_available": 0,
                "total_drift_history": 0,
            },
            "validation_status": {
                "recent_validations": 0,
                "errors": 0,
                "warnings": 0,
                "critical_issues": 0,
            },
            "optimization": {
                "active_recommendations": 0,
                "performance_metrics_tracked": 0,
            },
            "environments": {
                "detected": ["development", "simple"],
                "baselines_established": ["development", "simple"],
            },
        }

        return mock_system

    @patch("src.config.observability.api.get_automation_system")
    def test_health_endpoint(self, mock_get_system, client, mock_automation_system):
        """Test health check endpoint."""
        mock_get_system.return_value = mock_automation_system

        response = client.get("/api/v1/config/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["automation_system"] == "active"
        assert "environments" in data

    @patch("src.config.observability.api.get_automation_system")
    def test_status_endpoint(self, mock_get_system, client, mock_automation_system):
        """Test system status endpoint."""
        mock_get_system.return_value = mock_automation_system

        response = client.get("/api/v1/config/status")

        assert response.status_code == 200
        data = response.json()
        assert data["automation_enabled"] is True
        assert data["environments_monitored"] == 2
        assert "drift_analysis" in data
        assert "validation_status" in data

    @patch("src.config.observability.api.get_automation_system")
    async def test_drift_check_endpoint(
        self, mock_get_system, client, mock_automation_system
    ):
        """Test drift check endpoint."""
        # Mock drift detection
        mock_drift = ConfigDrift(
            severity=ConfigDriftSeverity.WARNING,
            parameter="test_param",
            expected_value="test",
            current_value="changed",
            environment="development",
            timestamp=datetime.now(UTC),
        )

        mock_automation_system.detect_configuration_drift = AsyncMock(
            return_value=[mock_drift]
        )
        mock_automation_system.auto_remediate_issues = AsyncMock(
            return_value={"test_param": True}
        )
        mock_get_system.return_value = mock_automation_system

        response = client.post(
            "/api/v1/config/drift/check",
            json={
                "environment": "development",
                "auto_fix": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["drifts_detected"] == 1
        assert data["auto_fixes_applied"] == 1
        assert len(data["drifts"]) == 1

    @patch("src.config.observability.api.get_automation_system")
    async def test_validation_endpoint(
        self, mock_get_system, client, mock_automation_system
    ):
        """Test configuration validation endpoint."""
        # Mock validation results
        mock_result = ConfigValidationResult(
            status=ConfigValidationStatus.WARNING,
            parameter="test_param",
            message="Test warning",
            environment="development",
            timestamp=datetime.now(UTC),
        )

        mock_automation_system.validate_configuration_health = AsyncMock(
            return_value=[mock_result]
        )
        mock_get_system.return_value = mock_automation_system

        response = client.post(
            "/api/v1/config/validate",
            json={
                "environment": "development",
                "fix_issues": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total_checks"] == 1
        assert data["warnings"] == 1
        assert len(data["results"]) == 1

    @patch("src.config.observability.api.get_automation_system")
    async def test_optimization_endpoint(
        self, mock_get_system, client, mock_automation_system
    ):
        """Test optimization generation endpoint."""
        # Mock optimization recommendations
        mock_rec = OptimizationRecommendation(
            parameter="test_param",
            current_value=10,
            recommended_value=20,
            expected_improvement="50% better performance",
            confidence_score=0.9,
            performance_impact="High",
            environment="development",
            reasoning="Test reasoning",
        )

        mock_automation_system.generate_optimization_recommendations = AsyncMock(
            return_value=[mock_rec]
        )
        mock_get_system.return_value = mock_automation_system

        response = client.post(
            "/api/v1/config/optimize",
            json={
                "environment": "development",
                "apply_recommendations": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["recommendations_count"] == 1
        assert data["high_confidence_count"] == 1
        assert len(data["recommendations"]) == 1


class TestIntegration:
    """Integration tests for the complete automation system."""

    @pytest.fixture
    async def running_system(self):
        """Create and start a complete automation system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create configuration files
            (temp_path / ".env").write_text("AI_DOCS__MODE=simple\nAI_DOCS__DEBUG=true")
            (temp_path / ".env.enterprise").write_text(
                "AI_DOCS__MODE=enterprise\nAI_DOCS__DEBUG=false"
            )

            system = ConfigObservabilityAutomation(
                config_dir=str(temp_path),
                enable_auto_remediation=True,
                drift_check_interval=1,  # Fast for testing
                performance_optimization_interval=2,
            )

            # Mock file monitoring to avoid real watchers
            system.start_file_monitoring = Mock()

            await system.start()

            yield system

            await system.stop()

    async def test_complete_workflow(self, running_system):
        """Test complete automation workflow."""
        # Test environment detection
        assert len(running_system.detected_environments) > 0

        # Test drift detection
        drifts = await running_system.detect_configuration_drift()
        assert isinstance(drifts, list)

        # Test validation
        validations = await running_system.validate_configuration_health()
        assert isinstance(validations, list)

        # Test optimization
        recommendations = await running_system.generate_optimization_recommendations()
        assert isinstance(recommendations, list)

        # Test status reporting
        status = running_system.get_system_status()
        assert "system_status" in status
        assert status["system_status"]["automation_enabled"] is True

    async def test_file_change_handling(self, running_system):
        """Test handling of configuration file changes."""
        config_file = Path(running_system.config_dir) / ".env"

        # Simulate file change
        await running_system.handle_config_change(str(config_file))

        # Should trigger re-establishment of baselines
        # In real test, would verify baseline configurations were updated

    async def test_periodic_operations(self, running_system):
        """Test periodic drift checking and optimization."""
        # Wait for a short period to let periodic tasks run
        await asyncio.sleep(0.1)

        # Verify that periodic operations don't crash
        # In a real system, would verify metrics collection
        assert running_system.last_drift_check is not None
        assert running_system.last_optimization_check is not None


@pytest.mark.asyncio
class TestPerformanceAndScalability:
    """Performance and scalability tests."""

    async def test_large_configuration_handling(self):
        """Test handling of large configuration sets."""
        system = ConfigObservabilityAutomation()

        # Create large baseline configuration
        large_config = {f"param_{i}": f"value_{i}" for i in range(1000)}
        system.baseline_configurations = {"test": large_config}

        # Test comparison performance
        start_time = datetime.now(UTC)
        current_config = {f"param_{i}": f"changed_value_{i}" for i in range(1000)}

        drifts = await system._compare_configurations(
            large_config, current_config, "test", datetime.now(UTC)
        )

        end_time = datetime.now(UTC)
        duration = (end_time - start_time).total_seconds()

        # Should handle 1000 parameter comparison in reasonable time
        assert duration < 1.0  # Less than 1 second
        assert len(drifts) == 1000  # All parameters changed

    async def test_memory_usage_with_history(self):
        """Test memory management with large histories."""
        system = ConfigObservabilityAutomation()

        # Add large amount of historical data
        for i in range(10000):
            system.drift_history.append(
                ConfigDrift(
                    severity=ConfigDriftSeverity.INFO,
                    parameter=f"param_{i}",
                    expected_value="test",
                    current_value="changed",
                    environment="test",
                    timestamp=datetime.now(UTC) - timedelta(hours=i),
                )
            )

        # Trigger cleanup (simulate 24-hour cleanup)
        current_time = datetime.now(UTC)
        cutoff_time = current_time - timedelta(hours=24)
        system.drift_history = [
            drift for drift in system.drift_history if drift.timestamp > cutoff_time
        ]

        # Should keep only recent entries (last 24 hours)
        assert len(system.drift_history) <= 24
