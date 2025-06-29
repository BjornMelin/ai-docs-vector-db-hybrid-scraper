#!/usr/bin/env python3
"""Centralized timeout configuration for the application.

This module provides configurable timeout settings that were previously hardcoded
throughout the application, making them easier to manage and modify based on
deployment environment needs.
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class TimeoutSettings(BaseSettings):
    """Centralized timeout configuration settings."""

    # Configuration validation timeouts
    config_validation_timeout: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Configuration validation timeout in seconds",
    )
    config_reload_timeout: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="Configuration reload timeout in seconds",
    )

    # Deployment and CI/CD timeouts
    deployment_timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Deployment operation timeout in seconds",
    )
    cicd_pipeline_timeout: int = Field(
        default=1200,
        ge=300,
        le=7200,
        description="CI/CD pipeline timeout in seconds",
    )
    rollback_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Rollback operation timeout in seconds",
    )

    # Service operation timeouts
    operation_timeout: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="General operation timeout in seconds",
    )
    api_request_timeout: int = Field(
        default=60,
        ge=5,
        le=300,
        description="API request timeout in seconds",
    )
    database_query_timeout: int = Field(
        default=30,
        ge=5,
        le=180,
        description="Database query timeout in seconds",
    )

    # Browser and scraping timeouts
    browser_global_timeout_ms: int = Field(
        default=120000,
        ge=10000,
        le=600000,
        description="Browser global timeout in milliseconds",
    )
    browser_navigation_timeout_ms: int = Field(
        default=60000,
        ge=5000,
        le=300000,
        description="Browser navigation timeout in milliseconds",
    )
    scraping_timeout: int = Field(
        default=180,
        ge=30,
        le=900,
        description="Web scraping timeout in seconds",
    )

    # Task queue timeouts
    job_timeout: int = Field(
        default=3600,
        ge=300,
        le=14400,
        description="Background job timeout in seconds (default: 1 hour)",
    )
    task_execution_timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Task execution timeout in seconds",
    )

    # Test and benchmark timeouts
    test_timeout: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Test execution timeout in seconds",
    )
    benchmark_timeout: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="Benchmark execution timeout in seconds",
    )

    # Security test timeouts
    security_scan_timeout: int = Field(
        default=600,
        ge=120,
        le=3600,
        description="Security scan timeout in seconds",
    )
    vulnerability_scan_timeout: int = Field(
        default=120,
        ge=30,
        le=600,
        description="Vulnerability scan timeout in seconds",
    )

    # Chaos engineering timeouts
    chaos_test_timeout: int = Field(
        default=3600,
        ge=600,
        le=7200,
        description="Chaos test timeout in seconds (default: 1 hour)",
    )

    class Config:
        env_prefix = "TIMEOUT_"
        case_sensitive = False


# Global timeout settings instance
_timeout_settings: TimeoutSettings | None = None


def get_timeout_settings() -> TimeoutSettings:
    """Get the global timeout settings instance.

    Returns:
        TimeoutSettings instance with all timeout configurations

    """
    global _timeout_settings
    if _timeout_settings is None:
        _timeout_settings = TimeoutSettings()
    return _timeout_settings


def reset_timeout_settings() -> None:
    """Reset timeout settings (mainly for testing)."""
    global _timeout_settings
    _timeout_settings = None


class TimeoutConfig(BaseModel):
    """Timeout configuration for specific operations."""

    operation_name: str
    timeout_seconds: float
    warning_threshold_seconds: float | None = None
    critical_threshold_seconds: float | None = None

    def should_warn(self, elapsed_seconds: float) -> bool:
        """Check if elapsed time exceeds warning threshold."""
        if self.warning_threshold_seconds is None:
            return elapsed_seconds > (self.timeout_seconds * 0.75)
        return elapsed_seconds > self.warning_threshold_seconds

    def should_alert_critical(self, elapsed_seconds: float) -> bool:
        """Check if elapsed time exceeds critical threshold."""
        if self.critical_threshold_seconds is None:
            return elapsed_seconds > (self.timeout_seconds * 0.9)
        return elapsed_seconds > self.critical_threshold_seconds


def get_timeout_config(operation_name: str) -> TimeoutConfig:
    """Get timeout configuration for a specific operation.

    Args:
        operation_name: Name of the operation

    Returns:
        TimeoutConfig instance with appropriate timeout settings

    """
    settings = get_timeout_settings()

    # Map operation names to timeout settings
    timeout_map = {
        "config_validation": settings.config_validation_timeout,
        "config_reload": settings.config_reload_timeout,
        "deployment": settings.deployment_timeout,
        "cicd_pipeline": settings.cicd_pipeline_timeout,
        "rollback": settings.rollback_timeout,
        "api_request": settings.api_request_timeout,
        "database_query": settings.database_query_timeout,
        "browser_navigation": settings.browser_navigation_timeout_ms / 1000,
        "scraping": settings.scraping_timeout,
        "job_execution": settings.job_timeout,
        "task_execution": settings.task_execution_timeout,
        "test_execution": settings.test_timeout,
        "benchmark": settings.benchmark_timeout,
        "security_scan": settings.security_scan_timeout,
        "vulnerability_scan": settings.vulnerability_scan_timeout,
        "chaos_test": settings.chaos_test_timeout,
    }

    # Get timeout or use default operation timeout
    timeout = timeout_map.get(operation_name, settings.operation_timeout)

    return TimeoutConfig(
        operation_name=operation_name,
        timeout_seconds=float(timeout),
    )


# Export all public items
__all__ = [
    "TimeoutConfig",
    "TimeoutSettings",
    "get_timeout_config",
    "get_timeout_settings",
    "reset_timeout_settings",
]
