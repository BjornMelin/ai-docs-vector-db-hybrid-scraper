"""Core deployment models and data structures.

This module defines the fundamental data models used across all deployment services,
providing type-safe interfaces for deployment operations and metrics.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DeploymentStatus(str, Enum):
    """Status of deployment operations."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"


class DeploymentEnvironment(str, Enum):
    """Deployment environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    CANARY = "canary"
    BLUE = "blue"
    GREEN = "green"


class DeploymentHealth(BaseModel):
    """Health metrics for deployment monitoring."""

    status: str = Field(..., description="Overall health status")
    response_time_ms: float = Field(
        ..., description="Average response time in milliseconds"
    )
    error_rate: float = Field(..., description="Error rate as percentage (0-100)")
    success_count: int = Field(default=0, description="Number of successful requests")
    error_count: int = Field(default=0, description="Number of failed requests")
    last_check: datetime = Field(
        default_factory=datetime.utcnow, description="Last health check timestamp"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional health details"
    )


class DeploymentMetrics(BaseModel):
    """Comprehensive deployment metrics for analysis."""

    deployment_id: str = Field(..., description="Unique deployment identifier")
    environment: DeploymentEnvironment = Field(..., description="Target environment")
    status: DeploymentStatus = Field(..., description="Current deployment status")

    # Traffic metrics
    total_requests: int = Field(default=0, description="Total requests processed")
    successful_requests: int = Field(
        default=0, description="Number of successful requests"
    )
    failed_requests: int = Field(default=0, description="Number of failed requests")

    # Performance metrics
    avg_response_time_ms: float = Field(
        default=0.0, description="Average response time"
    )
    p95_response_time_ms: float = Field(
        default=0.0, description="95th percentile response time"
    )
    p99_response_time_ms: float = Field(
        default=0.0, description="99th percentile response time"
    )

    # Business metrics
    conversion_rate: float = Field(
        default=0.0, description="Conversion rate as percentage"
    )
    error_rate: float = Field(default=0.0, description="Error rate as percentage")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Deployment creation time"
    )
    started_at: datetime | None = Field(
        default=None, description="Deployment start time"
    )
    completed_at: datetime | None = Field(
        default=None, description="Deployment completion time"
    )

    # Additional context
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional deployment metadata"
    )

    @property
    def duration_seconds(self) -> float | None:
        """Calculate deployment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0


@dataclass
class TrafficSplit:
    """Traffic allocation configuration for A/B testing and canary deployments."""

    control_percentage: float  # Percentage of traffic to control/stable version
    variant_percentage: float  # Percentage of traffic to variant/canary version

    def __post_init__(self):
        """Validate traffic split percentages."""
        total = self.control_percentage + self.variant_percentage
        if not (99.9 <= total <= 100.1):  # Allow for floating point precision
            raise ValueError(f"Traffic split must total 100%, got {total}%")

        if self.control_percentage < 0 or self.variant_percentage < 0:
            raise ValueError("Traffic percentages must be non-negative")


@dataclass
class DeploymentConfig:
    """Base configuration for deployment operations."""

    deployment_id: str
    environment: DeploymentEnvironment
    feature_flags: dict[str, Any]
    monitoring_enabled: bool = True
    rollback_enabled: bool = True
    max_duration_minutes: int = 60
    health_check_interval_seconds: int = 30
    failure_threshold: float = 5.0  # Error rate threshold for automatic rollback

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "environment": self.environment.value,
            "feature_flags": self.feature_flags,
            "monitoring_enabled": self.monitoring_enabled,
            "rollback_enabled": self.rollback_enabled,
            "max_duration_minutes": self.max_duration_minutes,
            "health_check_interval_seconds": self.health_check_interval_seconds,
            "failure_threshold": self.failure_threshold,
        }
