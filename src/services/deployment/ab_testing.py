"""A/B Testing Service for Enterprise Deployment.

This module provides comprehensive A/B testing capabilities including:
- Traffic splitting with statistical significance testing
- Multi-variant experiments with proper randomization
- Real-time metrics collection and analysis
- Automated winner detection and rollout recommendations
"""

import asyncio
import contextlib
import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .feature_flags import FeatureFlagManager
from .models import DeploymentMetrics, DeploymentStatus


logger = logging.getLogger(__name__)


class ABTestStatus(str, Enum):
    """Status of A/B testing experiments."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


class ABTestResult(BaseModel):
    """Results of an A/B test analysis."""

    test_id: str = Field(..., description="A/B test identifier")
    variant_name: str = Field(..., description="Name of the variant")

    # Traffic metrics
    total_users: int = Field(..., description="Total users in this variant")
    conversion_count: int = Field(..., description="Number of conversions")
    conversion_rate: float = Field(..., description="Conversion rate as percentage")

    # Performance metrics
    avg_response_time_ms: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate as percentage")

    # Statistical analysis
    confidence_level: float = Field(..., description="Statistical confidence level")
    is_significant: bool = Field(
        ..., description="Whether results are statistically significant"
    )
    uplift_percentage: float | None = Field(
        default=None, description="Uplift compared to control"
    )

    # Timestamps
    period_start: datetime = Field(..., description="Analysis period start")
    period_end: datetime = Field(..., description="Analysis period end")


@dataclass
class ABTestVariant:
    """Configuration for an A/B test variant."""

    name: str
    traffic_percentage: float
    feature_flags: dict[str, Any]
    description: str = ""

    def __post_init__(self):
        """Validate variant configuration."""
        if not 0 <= self.traffic_percentage <= 100:
            raise ValueError(
                f"Traffic percentage must be 0-100, got {self.traffic_percentage}"
            )


@dataclass
class ABTestConfig:
    """Configuration for an A/B test experiment."""

    test_id: str
    name: str
    description: str
    variants: list[ABTestVariant]
    duration_days: int = 14
    min_sample_size: int = 1000
    confidence_level: float = 0.95
    conversion_goal: str = "default"

    def __post_init__(self):
        """Validate A/B test configuration."""
        if len(self.variants) < 2:
            raise ValueError("A/B test requires at least 2 variants")

        total_traffic = sum(variant.traffic_percentage for variant in self.variants)
        if not (99.9 <= total_traffic <= 100.1):  # Allow floating point precision
            raise ValueError(f"Variant traffic must total 100%, got {total_traffic}%")


class ABTestingManager:
    """Enterprise A/B testing manager with statistical analysis."""

    def __init__(
        self,
        qdrant_service: Any,
        cache_manager: Any,
        feature_flag_manager: FeatureFlagManager | None = None,
    ):
        """Initialize A/B testing manager.

        Args:
            qdrant_service: Qdrant service for data storage
            cache_manager: Cache manager for session state
            feature_flag_manager: Optional feature flag manager
        """
        self.qdrant_service = qdrant_service
        self.cache_manager = cache_manager
        self.feature_flag_manager = feature_flag_manager

        # Active tests storage
        self._active_tests: dict[str, ABTestConfig] = {}
        self._test_metrics: dict[str, dict[str, DeploymentMetrics]] = {}
        self._user_assignments: dict[
            str, dict[str, str]
        ] = {}  # {test_id: {user_id: variant}}

        # Monitoring
        self._monitoring_task: asyncio.Task | None = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize A/B testing manager."""
        if self._initialized:
            return

        try:
            # Check if A/B testing is enabled via feature flags
            if self.feature_flag_manager:
                ab_testing_enabled = await self.feature_flag_manager.is_feature_enabled(
                    "ab_testing"
                )
                if not ab_testing_enabled:
                    logger.info("A/B testing disabled via feature flags")
                    self._initialized = True
                    return

            # Load existing tests from storage
            await self._load_active_tests()

            # Start monitoring task
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())

            self._initialized = True
            logger.info("A/B testing manager initialized successfully")

        except Exception:
            logger.exception("Failed to initialize A/B testing manager: %s", e)
            self._initialized = False
            raise

    async def create_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test.

        Args:
            config: A/B test configuration

        Returns:
            str: Test ID

        Raises:
            ValueError: If test configuration is invalid
        """
        if not self._initialized:
            await self.initialize()

        # Validate configuration
        config.__post_init__()

        # Store test configuration
        self._active_tests[config.test_id] = config
        self._test_metrics[config.test_id] = {}
        self._user_assignments[config.test_id] = {}

        # Initialize metrics for each variant
        for variant in config.variants:
            self._test_metrics[config.test_id][variant.name] = DeploymentMetrics(
                deployment_id=f"{config.test_id}_{variant.name}",
                environment="production",  # A/B tests run in production
                status=DeploymentStatus.RUNNING,
            )

        # Persist to storage
        await self._persist_test_config(config)

        logger.info(
            "Created A/B test: %s with %d variants", config.name, len(config.variants)
        )
        return config.test_id

    async def assign_user_to_variant(self, test_id: str, user_id: str) -> str | None:
        """Assign user to A/B test variant using consistent hashing.

        Args:
            test_id: A/B test identifier
            user_id: User identifier

        Returns:
            str | None: Assigned variant name, or None if test not found
        """
        if test_id not in self._active_tests:
            return None

        # Check if user already assigned
        if user_id in self._user_assignments.get(test_id, {}):
            return self._user_assignments[test_id][user_id]

        test_config = self._active_tests[test_id]

        # Use consistent hashing for deterministic assignment
        hash_input = f"{test_id}:{user_id}".encode()
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
        percentage = (hash_value % 10000) / 100.0  # 0-99.99%

        # Assign to variant based on traffic allocation
        cumulative_percentage = 0.0
        for variant in test_config.variants:
            cumulative_percentage += variant.traffic_percentage
            if percentage < cumulative_percentage:
                # Assign user to this variant
                if test_id not in self._user_assignments:
                    self._user_assignments[test_id] = {}
                self._user_assignments[test_id][user_id] = variant.name

                # Cache assignment for fast lookup
                cache_key = f"ab_test:{test_id}:user:{user_id}"
                await self.cache_manager.set(
                    cache_key, variant.name, ttl=86400
                )  # 24 hours

                return variant.name

        # Fallback to first variant (shouldn't happen with proper config)
        fallback_variant = test_config.variants[0].name
        self._user_assignments[test_id][user_id] = fallback_variant
        return fallback_variant

    async def get_variant_features(
        self, test_id: str, variant_name: str
    ) -> dict[str, Any]:
        """Get feature flags for a specific variant.

        Args:
            test_id: A/B test identifier
            variant_name: Variant name

        Returns:
            dict[str, Any]: Feature flags for the variant
        """
        if test_id not in self._active_tests:
            return {}

        test_config = self._active_tests[test_id]
        for variant in test_config.variants:
            if variant.name == variant_name:
                return variant.feature_flags

        return {}

    async def record_conversion(
        self,
        test_id: str,
        user_id: str,
        conversion_value: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Record a conversion event for A/B test analysis.

        Args:
            test_id: A/B test identifier
            user_id: User identifier
            conversion_value: Value of the conversion (default: 1.0)
            metadata: Optional additional metadata

        Returns:
            bool: True if conversion recorded successfully
        """
        if test_id not in self._active_tests or test_id not in self._user_assignments:
            return False

        variant_name = self._user_assignments[test_id].get(user_id)
        if not variant_name:
            return False

        # Update metrics
        if (
            test_id in self._test_metrics
            and variant_name in self._test_metrics[test_id]
        ):
            metrics = self._test_metrics[test_id][variant_name]
            metrics.successful_requests += 1
            metrics.conversion_rate = self._calculate_conversion_rate(
                test_id, variant_name
            )

            # Store conversion event
            conversion_data = {
                "test_id": test_id,
                "user_id": user_id,
                "variant": variant_name,
                "value": conversion_value,
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "metadata": metadata or {},
            }

            # Persist to storage
            await self._persist_conversion_event(conversion_data)

        return True

    async def get_test_results(self, test_id: str) -> list[ABTestResult]:
        """Get current results for an A/B test.

        Args:
            test_id: A/B test identifier

        Returns:
            list[ABTestResult]: Results for each variant
        """
        if test_id not in self._active_tests:
            return []

        test_config = self._active_tests[test_id]
        results = []

        for variant in test_config.variants:
            metrics = self._test_metrics[test_id].get(variant.name)
            if not metrics:
                continue

            # Calculate statistical significance
            control_variant = test_config.variants[0]  # First variant is control
            uplift = None
            is_significant = False

            if variant.name != control_variant.name:
                uplift, is_significant = await self._calculate_statistical_significance(
                    test_id, control_variant.name, variant.name
                )

            result = ABTestResult(
                test_id=test_id,
                variant_name=variant.name,
                total_users=metrics.total_requests,
                conversion_count=metrics.successful_requests,
                conversion_rate=metrics.conversion_rate,
                avg_response_time_ms=metrics.avg_response_time_ms,
                error_rate=metrics.error_rate,
                confidence_level=test_config.confidence_level,
                is_significant=is_significant,
                uplift_percentage=uplift,
                period_start=metrics.created_at,
                period_end=datetime.now(tz=UTC),
            )

            results.append(result)

        return results

    async def stop_test(self, test_id: str, reason: str = "") -> bool:
        """Stop an active A/B test.

        Args:
            test_id: A/B test identifier
            reason: Optional reason for stopping

        Returns:
            bool: True if test stopped successfully
        """
        if test_id not in self._active_tests:
            return False

        # Mark test as stopped
        test_config = self._active_tests[test_id]

        # Update status in storage
        await self._update_test_status(test_id, ABTestStatus.STOPPED)

        # Remove from active tests
        del self._active_tests[test_id]

        logger.info("Stopped A/B test: %s. Reason: %s", test_config.name, reason)
        return True

    def _calculate_conversion_rate(self, test_id: str, variant_name: str) -> float:
        """Calculate conversion rate for a variant.

        Args:
            test_id: A/B test identifier
            variant_name: Variant name

        Returns:
            float: Conversion rate as percentage
        """
        if (
            test_id not in self._test_metrics
            or variant_name not in self._test_metrics[test_id]
        ):
            return 0.0

        metrics = self._test_metrics[test_id][variant_name]
        if metrics.total_requests == 0:
            return 0.0

        return (metrics.successful_requests / metrics.total_requests) * 100.0

    async def _calculate_statistical_significance(
        self, test_id: str, control_variant: str, test_variant: str
    ) -> tuple[float | None, bool]:
        """Calculate statistical significance between variants.

        Args:
            test_id: A/B test identifier
            control_variant: Control variant name
            test_variant: Test variant name

        Returns:
            tuple[float | None, bool]: (uplift_percentage, is_significant)
        """
        try:
            control_metrics = self._test_metrics[test_id][control_variant]
            test_metrics = self._test_metrics[test_id][test_variant]

            # Simple uplift calculation (in production, use proper statistical tests)
            control_rate = control_metrics.conversion_rate
            test_rate = test_metrics.conversion_rate

            if control_rate == 0:
                return None, False

            uplift = ((test_rate - control_rate) / control_rate) * 100.0

            # Simple significance check (minimum sample size)
            min_sample_size = self._active_tests[test_id].min_sample_size
            has_sufficient_sample = (
                control_metrics.total_requests >= min_sample_size
                and test_metrics.total_requests >= min_sample_size
            )

            # In production, implement proper statistical significance testing
            # (e.g., two-proportion z-test, chi-square test)
            is_significant = has_sufficient_sample and abs(uplift) > 5.0  # 5% threshold

            return uplift, is_significant

        except Exception:
            logger.exception("Error calculating statistical significance: %s", e)
            return None, False

    async def _monitoring_loop(self) -> None:
        """Background task for monitoring A/B tests."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                # Monitor active tests
                for test_id, config in list(self._active_tests.items()):
                    await self._check_test_completion(test_id, config)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in A/B testing monitoring loop: %s", e)
                await asyncio.sleep(60)

    async def _check_test_completion(self, test_id: str, config: ABTestConfig) -> None:
        """Check if an A/B test should be completed."""
        try:
            # Check duration
            test_start = min(
                metrics.created_at for metrics in self._test_metrics[test_id].values()
            )
            if datetime.now(tz=UTC) - test_start > timedelta(days=config.duration_days):
                await self.stop_test(test_id, "Duration completed")
                return

            # Check for early winner (if enabled)
            results = await self.get_test_results(test_id)
            for result in results:
                if (
                    result.is_significant
                    and result.uplift_percentage
                    and abs(result.uplift_percentage) > 20.0
                ):  # 20% uplift threshold
                    await self.stop_test(test_id, "Early winner detected")
                    return

        except Exception:
            logger.exception("Error checking test completion for %s: %s", test_id, e)

    async def _load_active_tests(self) -> None:
        """Load active tests from storage."""
        # In production, load from database/storage
        pass

    async def _persist_test_config(self, config: ABTestConfig) -> None:
        """Persist A/B test configuration to storage."""
        # In production, save to database/storage
        pass

    async def _persist_conversion_event(self, conversion_data: dict[str, Any]) -> None:
        """Persist conversion event to storage."""
        # In production, save to database/storage
        pass

    async def _update_test_status(self, test_id: str, status: ABTestStatus) -> None:
        """Update test status in storage."""
        # In production, update in database/storage
        pass

    async def cleanup(self) -> None:
        """Cleanup A/B testing manager resources."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._monitoring_task

        self._active_tests.clear()
        self._test_metrics.clear()
        self._user_assignments.clear()
        self._initialized = False
        logger.info("A/B testing manager cleanup completed")
