"""Tests for observability initialization and setup."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.services.observability.config import ObservabilityConfig
from src.services.observability.init import (
    initialize_observability,
    shutdown_observability,
    is_observability_enabled,
    _setup_auto_instrumentation
)


class TestObservabilityInitialization:
    """Test observability system initialization."""

    def test_initialize_observability_disabled(self):
        """Test observability initialization when disabled."""
        config = ObservabilityConfig(enabled=False)
        
        result = initialize_observability(config)
        
        assert result is False

    @patch('src.services.observability.init.logger')
    def test_initialize_observability_import_error(self, mock_logger):
        """Test observability initialization with import errors."""
        config = ObservabilityConfig(enabled=True)
        
        # Mock the imports to cause ImportError
        with patch.dict('sys.modules', {'opentelemetry.metrics': None}):
            result = initialize_observability(config)
            
            assert result is False
            mock_logger.warning.assert_called()

    @patch('src.services.observability.init.logger')
    def test_initialize_observability_general_exception(self, mock_logger):
        """Test observability initialization with general exceptions."""
        config = ObservabilityConfig(enabled=True)
        
        # Mock builtins to cause an exception during import
        with patch('builtins.__import__', side_effect=Exception("General error")):
            result = initialize_observability(config)
            
            assert result is False
            mock_logger.error.assert_called()

    def test_initialize_observability_no_config(self):
        """Test observability initialization with no config provided."""
        with patch('src.services.observability.config.get_observability_config') as mock_get_config:
            mock_config = ObservabilityConfig(enabled=False)
            mock_get_config.return_value = mock_config
            
            result = initialize_observability()
            
            assert result is False
            mock_get_config.assert_called_once()


class TestAutoInstrumentation:
    """Test auto-instrumentation setup."""

    @patch('src.services.observability.init.logger')
    def test_setup_auto_instrumentation_all_disabled(self, mock_logger):
        """Test auto-instrumentation setup with all features disabled."""
        config = ObservabilityConfig(
            enabled=True,
            instrument_fastapi=False,
            instrument_httpx=False,
            instrument_redis=False,
            instrument_sqlalchemy=False
        )
        
        _setup_auto_instrumentation(config)
        
        # Should not log any instrumentation enabled messages
        info_calls = [call[0][0] for call in mock_logger.info.call_args_list if call[0]]
        instrumentation_calls = [call for call in info_calls if "instrumentation enabled" in call]
        assert len(instrumentation_calls) == 0

    @patch('src.services.observability.init.logger')
    def test_setup_auto_instrumentation_general_exception(self, mock_logger):
        """Test auto-instrumentation setup with general exceptions."""
        config = ObservabilityConfig(enabled=True, instrument_fastapi=True)
        
        with patch('builtins.__import__', side_effect=Exception("General error")):
            _setup_auto_instrumentation(config)
            
            mock_logger.warning.assert_called()


class TestObservabilityShutdown:
    """Test observability shutdown functionality."""

    @patch('src.services.observability.init.logger')
    def test_shutdown_observability_success(self, mock_logger):
        """Test successful observability shutdown."""
        # Setup mock providers
        mock_tracer_provider_instance = Mock()
        mock_meter_provider_instance = Mock()
        
        # Patch the module-level variables directly
        with patch('src.services.observability.init._tracer_provider', mock_tracer_provider_instance), \
             patch('src.services.observability.init._meter_provider', mock_meter_provider_instance):
            
            shutdown_observability()
            
            # Verify shutdown calls
            mock_tracer_provider_instance.shutdown.assert_called_once()
            mock_meter_provider_instance.shutdown.assert_called_once()
            mock_logger.info.assert_any_call("Shutting down OpenTelemetry tracer provider...")
            mock_logger.info.assert_any_call("Shutting down OpenTelemetry meter provider...")
            mock_logger.info.assert_any_call("OpenTelemetry shutdown completed")

    @patch('src.services.observability.init.logger')
    def test_shutdown_observability_tracer_error(self, mock_logger):
        """Test observability shutdown with tracer provider error."""
        mock_tracer_provider_instance = Mock()
        mock_tracer_provider_instance.shutdown.side_effect = Exception("Shutdown failed")
        
        with patch('src.services.observability.init._tracer_provider', mock_tracer_provider_instance), \
             patch('src.services.observability.init._meter_provider', None):
            
            shutdown_observability()
            
            mock_logger.error.assert_called_with("Error during tracer provider shutdown: Shutdown failed")

    @patch('src.services.observability.init.logger')
    def test_shutdown_observability_meter_error(self, mock_logger):
        """Test observability shutdown with meter provider error."""
        mock_meter_provider_instance = Mock()
        mock_meter_provider_instance.shutdown.side_effect = Exception("Meter shutdown failed")
        
        with patch('src.services.observability.init._tracer_provider', None), \
             patch('src.services.observability.init._meter_provider', mock_meter_provider_instance):
            
            shutdown_observability()
            
            mock_logger.error.assert_called_with("Error during meter provider shutdown: Meter shutdown failed")

    def test_shutdown_observability_no_providers(self):
        """Test observability shutdown with no active providers."""
        with patch('src.services.observability.init._tracer_provider', None), \
             patch('src.services.observability.init._meter_provider', None):
            
            # Should not raise exception
            shutdown_observability()


class TestObservabilityStatus:
    """Test observability status checking."""

    def test_is_observability_enabled_true(self):
        """Test observability enabled status when tracer provider exists."""
        mock_tracer_provider_instance = Mock()
        
        with patch('src.services.observability.init._tracer_provider', mock_tracer_provider_instance):
            result = is_observability_enabled()
            assert result is True

    def test_is_observability_enabled_false(self):
        """Test observability disabled status when no tracer provider."""
        with patch('src.services.observability.init._tracer_provider', None):
            result = is_observability_enabled()
            assert result is False


class TestObservabilityIntegration:
    """Integration tests for observability initialization."""

    def test_error_propagation_in_initialization(self):
        """Test that errors in setup are properly handled."""
        config = ObservabilityConfig(enabled=True)
        
        with patch('builtins.__import__', side_effect=ImportError("Missing dependency")):
            result = initialize_observability(config)
            
            # Should handle error gracefully
            assert result is False

    def test_disabled_config_propagation(self):
        """Test that disabled configuration is handled correctly."""
        # Configuration with observability disabled
        disabled_config = ObservabilityConfig(enabled=False)
        
        result = initialize_observability(disabled_config)
        assert result is False

    def test_default_config_initialization(self):
        """Test initialization with default config from get_observability_config."""
        with patch('src.services.observability.config.get_observability_config') as mock_get_config:
            mock_config = ObservabilityConfig(enabled=False)
            mock_get_config.return_value = mock_config
            
            result = initialize_observability()
            
            assert result is False
            mock_get_config.assert_called_once()

    def test_lifecycle_integration(self):
        """Test the full lifecycle: init -> shutdown -> status check."""
        # Test shutdown without initialization
        shutdown_observability()
        
        # Test status check
        assert is_observability_enabled() is False
        
        # Test initialization with disabled config
        config = ObservabilityConfig(enabled=False)
        result = initialize_observability(config)
        assert result is False
        
        # Test status check after failed initialization
        assert is_observability_enabled() is False


class TestObservabilityConfigHandling:
    """Test observability configuration handling."""

    def test_configuration_validation(self):
        """Test configuration validation in initialization."""
        # Valid configuration
        valid_config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            track_ai_operations=True
        )
        
        # Test with valid config (will fail due to missing dependencies, but config is valid)
        with patch('builtins.__import__', side_effect=ImportError("No OpenTelemetry")):
            result = initialize_observability(valid_config)
            assert result is False

    def test_configuration_attributes_access(self):
        """Test that configuration attributes are accessible."""
        config = ObservabilityConfig(
            enabled=True,
            service_name="test-service",
            otlp_endpoint="http://localhost:4317",
            console_exporter=True,
            instrument_fastapi=True,
            instrument_httpx=False
        )
        
        # Test that all config attributes are accessible
        assert config.enabled is True
        assert config.service_name == "test-service"
        assert config.otlp_endpoint == "http://localhost:4317"
        assert config.console_exporter is True
        assert config.instrument_fastapi is True
        assert config.instrument_httpx is False

    def test_auto_instrumentation_configuration(self):
        """Test auto-instrumentation configuration handling."""
        config = ObservabilityConfig(
            enabled=True,
            instrument_fastapi=True,
            instrument_httpx=True,
            instrument_redis=False,
            instrument_sqlalchemy=False
        )
        
        # Test that _setup_auto_instrumentation doesn't crash with valid config
        with patch('src.services.observability.init.logger'):
            _setup_auto_instrumentation(config)
            # Should complete without exceptions


class TestObservabilityErrorHandling:
    """Test error handling in observability system."""

    @patch('src.services.observability.init.logger')
    def test_robust_error_handling_in_shutdown(self, mock_logger):
        """Test that shutdown handles various error conditions gracefully."""
        # Test with provider that raises exception on shutdown
        mock_provider = Mock()
        mock_provider.shutdown.side_effect = RuntimeError("Critical shutdown error")
        
        with patch('src.services.observability.init._tracer_provider', mock_provider):
            # Should not raise exception despite provider error
            shutdown_observability()
            
            # Should log the error
            mock_logger.error.assert_called()

    def test_robust_error_handling_in_status_check(self):
        """Test that status check handles various conditions gracefully."""
        # Test with valid provider
        mock_provider = Mock()
        with patch('src.services.observability.init._tracer_provider', mock_provider):
            assert is_observability_enabled() is True
        
        # Test with None provider
        with patch('src.services.observability.init._tracer_provider', None):
            assert is_observability_enabled() is False
        
        # Test with provider that has no shutdown method (shouldn't affect status check)
        broken_provider = "not_a_provider_object"
        with patch('src.services.observability.init._tracer_provider', broken_provider):
            assert is_observability_enabled() is True  # Any non-None value should return True

    def test_initialization_with_partial_failures(self):
        """Test initialization behavior with partial component failures."""
        config = ObservabilityConfig(enabled=True)
        
        # Simulate scenario where some imports succeed but others fail
        def selective_import_failure(name, *args, **kwargs):
            if 'opentelemetry.metrics' in name:
                raise ImportError("Metrics not available")
            # Allow other imports to succeed by calling original import
            return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=selective_import_failure):
            result = initialize_observability(config)
            # Should fail gracefully
            assert result is False