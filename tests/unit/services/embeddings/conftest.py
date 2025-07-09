"""Configuration and fixtures for embedding service tests.

This module provides test-specific fixtures for embedding service testing,
including metrics registry initialization and service mocks.
"""

import pytest

from src.services.monitoring.metrics import MetricsConfig, initialize_metrics


@pytest.fixture(scope="session", autouse=True)
def initialize_test_metrics():
    """Initialize metrics registry for embedding service tests.

    This fixture automatically initializes the metrics registry singleton
    before any tests run that might use embedding services, preventing
    RuntimeError: Metrics registry not initialized.
    """
    # Create test-specific metrics configuration
    test_config = MetricsConfig(
        enabled=False,  # Disable metrics collection in tests
        export_port=0,  # No HTTP server
        namespace="test_ml_app",
        include_system_metrics=False,  # Skip system metrics in tests
        collection_interval=1.0,
    )

    # Initialize the metrics registry singleton
    initialize_metrics(test_config)

    return test_config
