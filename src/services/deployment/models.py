"""Core deployment models and data structures.

This module defines the fundamental data models used across all deployment services,
providing type-safe interfaces for deployment operations and metrics.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


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
    """Health metrics for deployment monitoring with advanced validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "examples": [
                {
                    "status": "healthy",
                    "response_time_ms": 125.5,
                    "error_rate": 0.5,
                    "success_count": 9950,
                    "error_count": 50,
                    "details": {"database": "connected", "cache": "healthy"},
                }
            ]
        },
    )

    status: str = Field(
        ...,
        description="Overall health status",
        pattern="^(healthy|degraded|unhealthy|unknown)$",
    )
    response_time_ms: float = Field(
        ..., ge=0.0, description="Average response time in milliseconds"
    )
    error_rate: float = Field(
        ..., ge=0.0, le=100.0, description="Error rate as percentage (0-100)"
    )
    success_count: int = Field(
        default=0, ge=0, description="Number of successful requests"
    )
    error_count: int = Field(default=0, ge=0, description="Number of failed requests")
    last_check: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Last health check timestamp",
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Additional health details"
    )

    @property
    @computed_field
    def total_requests(self) -> int:
        """Calculate total requests from success and error counts."""
        return self.success_count + self.error_count

    @property
    @computed_field
    def calculated_error_rate(self) -> float:
        """Calculate error rate from counts for validation."""
        if self.total_requests == 0:
            return 0.0
        return (self.error_count / self.total_requests) * 100.0

    @model_validator(mode="after")
    def validate_health_consistency(self) -> "DeploymentHealth":
        """Validate consistency between error rate and counts."""
        if self.total_requests > 0:
            calculated_rate = self.calculated_error_rate
            # Allow 1% tolerance for floating point precision
            if abs(self.error_rate - calculated_rate) > 1.0:
                msg = f"Error rate {self.error_rate}% doesn't match calculated rate {calculated_rate:.1f}%"
                raise ValueError(msg)
        return self


class DeploymentMetrics(BaseModel):
    """Comprehensive deployment metrics for analysis with security-first validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        frozen=False,
        json_schema_extra={
            "examples": [
                {
                    "deployment_id": "deploy-12345-abcde",
                    "environment": "production",
                    "status": "success",
                    "total_requests": 10000,
                    "successful_requests": 9850,
                    "failed_requests": 150,
                    "avg_response_time_ms": 245.7,
                    "p95_response_time_ms": 580.2,
                    "p99_response_time_ms": 1250.8,
                    "conversion_rate": 12.5,
                    "error_rate": 1.5,
                }
            ]
        },
    )

    deployment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9-_]+$",
        description="Unique deployment identifier",
    )
    environment: DeploymentEnvironment = Field(..., description="Target environment")
    status: DeploymentStatus = Field(..., description="Current deployment status")

    # Traffic metrics with validation
    total_requests: int = Field(
        default=0,
        ge=0,
        le=1_000_000_000,  # Reasonable upper bound
        description="Total requests processed",
    )
    successful_requests: int = Field(
        default=0, ge=0, description="Number of successful requests"
    )
    failed_requests: int = Field(
        default=0, ge=0, description="Number of failed requests"
    )

    # Performance metrics with realistic bounds
    avg_response_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        le=300_000.0,  # 5 minutes max reasonable response time
        description="Average response time",
    )
    p95_response_time_ms: float = Field(
        default=0.0, ge=0.0, le=300_000.0, description="95th percentile response time"
    )
    p99_response_time_ms: float = Field(
        default=0.0, ge=0.0, le=300_000.0, description="99th percentile response time"
    )

    # Business metrics
    conversion_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Conversion rate as percentage"
    )
    error_rate: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Error rate as percentage"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Deployment creation time",
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

    @field_validator("successful_requests", "failed_requests")
    @classmethod
    def validate_request_counts(cls, v: int, info: Any) -> int:
        """Validate request counts don't exceed total."""
        if info.data and "total_requests" in info.data:
            total = info.data["total_requests"]
            if v > total:
                msg = f"Request count {v} cannot exceed total requests {total}"
                raise ValueError(msg)
        return v

    @computed_field
    @property
    def duration_seconds(self) -> float | None:
        """Calculate deployment duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100.0

    @computed_field
    @property
    def calculated_error_rate(self) -> float:
        """Calculate error rate from request counts."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100.0

    @model_validator(mode="after")
    def validate_metrics_consistency(self) -> "DeploymentMetrics":
        """Validate consistency across all metrics."""
        # Validate request counts sum correctly
        calculated_total = self.successful_requests + self.failed_requests
        if calculated_total != self.total_requests:
            msg = f"Request counts don't sum correctly: {self.successful_requests} + {self.failed_requests} ≠ {self.total_requests}"
            raise ValueError(msg)

        # Validate error rate consistency
        if self.total_requests > 0:
            calculated_rate = self.calculated_error_rate
            if abs(self.error_rate - calculated_rate) > 0.1:  # 0.1% tolerance
                msg = f"Error rate {self.error_rate}% doesn't match calculated rate {calculated_rate:.1f}%"
                raise ValueError(msg)

        # Validate response time ordering (avg ≤ p95 ≤ p99)
        if not (
            self.avg_response_time_ms
            <= self.p95_response_time_ms
            <= self.p99_response_time_ms
        ):
            msg = "Response times must be ordered: avg ≤ p95 ≤ p99"
            raise ValueError(msg)

        # Validate timestamp ordering
        if (
            self.started_at
            and self.completed_at
            and self.started_at > self.completed_at
        ):
            msg = "Start time cannot be after completion time"
            raise ValueError(msg)

        if self.started_at and self.started_at < self.created_at:
            msg = "Start time cannot be before creation time"
            raise ValueError(msg)

        return self


class TrafficSplit(BaseModel):
    """Traffic allocation configuration for A/B testing and canary deployments with robust validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        frozen=True,  # Immutable configuration
        json_schema_extra={
            "examples": [
                {"control_percentage": 90.0, "variant_percentage": 10.0},
                {"control_percentage": 50.0, "variant_percentage": 50.0},
            ]
        },
    )

    control_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of traffic to control/stable version (0-100)",
    )
    variant_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of traffic to variant/canary version (0-100)",
    )

    @computed_field
    @property
    def total_percentage(self) -> float:
        """Calculate total traffic allocation percentage."""
        return self.control_percentage + self.variant_percentage

    @model_validator(mode="after")
    def validate_traffic_split(self) -> "TrafficSplit":
        """Validate traffic split percentages sum to 100%."""
        total = self.total_percentage
        if not (99.9 <= total <= 100.1):  # Allow for floating point precision
            msg = f"Traffic split must total 100%, got {total}%"
            raise ValueError(msg)
        return self


class DeploymentConfig(BaseModel):
    """Base configuration for deployment operations with comprehensive validation."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        frozen=False,
        json_schema_extra={
            "examples": [
                {
                    "deployment_id": "deploy-prod-2024-001",
                    "environment": "production",
                    "feature_flags": {"new_search": True, "beta_ui": False},
                    "monitoring_enabled": True,
                    "rollback_enabled": True,
                    "max_duration_minutes": 60,
                    "health_check_interval_seconds": 30,
                    "failure_threshold": 5.0,
                }
            ]
        },
    )

    deployment_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern="^[a-zA-Z0-9-_]+$",
        description="Unique deployment identifier",
    )
    environment: DeploymentEnvironment = Field(
        ..., description="Target deployment environment"
    )
    feature_flags: dict[str, Any] = Field(
        ..., description="Feature flag configuration for this deployment"
    )
    monitoring_enabled: bool = Field(
        default=True, description="Enable deployment monitoring and health checks"
    )
    rollback_enabled: bool = Field(
        default=True, description="Enable automatic rollback on failure"
    )
    max_duration_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,  # Max 24 hours
        description="Maximum deployment duration in minutes",
    )
    health_check_interval_seconds: int = Field(
        default=30,
        ge=1,
        le=3600,  # Max 1 hour
        description="Health check interval in seconds",
    )
    failure_threshold: float = Field(
        default=5.0,
        ge=0.0,
        le=100.0,
        description="Error rate threshold for automatic rollback (percentage)",
    )

    @field_validator("feature_flags")
    @classmethod
    def validate_feature_flags(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Validate feature flags have reasonable names and values."""
        if not v:
            return v

        for flag_name, flag_value in v.items():
            # Validate flag names
            if not isinstance(flag_name, str) or len(flag_name) == 0:
                msg = "Feature flag names must be non-empty strings"
                raise ValueError(msg)

            if len(flag_name) > 100:
                msg = f"Feature flag name too long: {flag_name[:50]}..."
                raise ValueError(msg)

            # Validate flag values are JSON-serializable basic types
            if not isinstance(flag_value, bool | str | int | float | type(None)):
                msg = f"Feature flag '{flag_name}' has invalid value type: {type(flag_value).__name__}"
                raise ValueError(msg)

        return v

    @computed_field
    @property
    def max_duration_seconds(self) -> int:
        """Calculate maximum duration in seconds."""
        return self.max_duration_minutes * 60

    @computed_field
    @property
    def configuration_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
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

    @model_validator(mode="after")
    def validate_deployment_configuration(self) -> "DeploymentConfig":
        """Validate deployment configuration consistency."""
        # Ensure health checks are reasonable for max duration
        max_checks = self.max_duration_seconds / self.health_check_interval_seconds
        if max_checks > 1000:  # Prevent excessive health checking
            msg = f"Configuration would result in {max_checks:.0f} health checks - reduce frequency or duration"
            raise ValueError(msg)

        # Production environments should have stricter settings
        if self.environment == DeploymentEnvironment.PRODUCTION:
            if not self.monitoring_enabled:
                msg = "Monitoring must be enabled for production deployments"
                raise ValueError(msg)

            if not self.rollback_enabled:
                msg = "Rollback must be enabled for production deployments"
                raise ValueError(msg)

            if self.failure_threshold > 10.0:
                msg = "Failure threshold too high for production deployment"
                raise ValueError(msg)

        return self
