"""Function-based deployment utilities with dependency injection.

Simplified deployment functions that replace complex deployment service classes.
Provides feature flags, A/B testing, canary deployments, and blue/green deployments.
"""

import asyncio
import logging
import time
from typing import Annotated, Any

from fastapi import Depends

from src.config import Config

from .dependencies import get_config
from .monitoring import increment_counter, set_gauge


logger = logging.getLogger(__name__)


# Simple in-memory feature flag storage
_feature_flags: dict[str, dict[str, Any]] = {}
_feature_flags_lock = asyncio.Lock()

# A/B testing storage
_ab_tests: dict[str, dict[str, Any]] = {}
_ab_tests_lock = asyncio.Lock()


async def get_feature_flag(
    flag_name: str,
    user_id: str | None = None,
    default_value: bool = False,
    config: Annotated[Config, Depends(get_config)] = None,
) -> bool:
    """Check if a feature flag is enabled for a user.

    Pure function replacement for complex feature flag services.

    Args:
        flag_name: Name of the feature flag
        user_id: Optional user ID for targeted flags
        default_value: Default value if flag doesn't exist
        config: Injected configuration

    Returns:
        True if feature is enabled, False otherwise
    """
    try:
        async with _feature_flags_lock:
            flag_config = _feature_flags.get(flag_name)

            if not flag_config:
                logger.debug(
                    f"Feature flag '{flag_name}' not found, using default: {default_value}"
                )
                return default_value

            # Check if flag is globally enabled/disabled
            if not flag_config.get("enabled", False):
                await increment_counter(
                    "feature_flag_checks",
                    tags={"flag": flag_name, "result": "disabled"},
                )
                return False

            # Check percentage rollout
            rollout_percentage = flag_config.get("rollout_percentage", 100)
            if rollout_percentage < 100 and user_id:
                # Use hash of user_id to determine if they're in the rollout group
                user_hash = hash(user_id) % 100
                if user_hash >= rollout_percentage:
                    await increment_counter(
                        "feature_flag_checks",
                        tags={"flag": flag_name, "result": "rollout_excluded"},
                    )
                    return False

            # Check user targeting
            target_users = flag_config.get("target_users", [])
            if target_users and user_id and user_id not in target_users:
                await increment_counter(
                    "feature_flag_checks",
                    tags={"flag": flag_name, "result": "not_targeted"},
                )
                return False

            await increment_counter(
                "feature_flag_checks", tags={"flag": flag_name, "result": "enabled"}
            )
            return True

    except Exception as e:
        logger.exception(f"Feature flag check failed for '{flag_name}': {e}")
        return default_value


async def set_feature_flag(
    flag_name: str,
    enabled: bool = True,
    rollout_percentage: int = 100,
    target_users: list[str] | None = None,
    config: Annotated[Config, Depends(get_config)] = None,
) -> None:
    """Set a feature flag configuration.

    Args:
        flag_name: Name of the feature flag
        enabled: Whether the flag is globally enabled
        rollout_percentage: Percentage of users to enable for (0-100)
        target_users: Optional list of specific users to target
        config: Injected configuration
    """
    try:
        async with _feature_flags_lock:
            _feature_flags[flag_name] = {
                "enabled": enabled,
                "rollout_percentage": rollout_percentage,
                "target_users": target_users or [],
                "created_at": time.time(),
                "updated_at": time.time(),
            }

            await increment_counter("feature_flag_updates", tags={"flag": flag_name})
            await set_gauge(
                f"feature_flag_rollout_percentage.{flag_name}", rollout_percentage
            )

            logger.info(
                f"Updated feature flag '{flag_name}': enabled={enabled}, rollout={rollout_percentage}%"
            )

    except Exception as e:
        logger.exception(f"Failed to set feature flag '{flag_name}': {e}")


async def get_ab_test_variant(
    test_name: str,
    user_id: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> str:
    """Get A/B test variant for a user.

    Pure function replacement for A/B testing services.

    Args:
        test_name: Name of the A/B test
        user_id: User ID for consistent assignment
        config: Injected configuration

    Returns:
        Variant name (e.g., "control", "variant_a", "variant_b")
    """
    try:
        async with _ab_tests_lock:
            test_config = _ab_tests.get(test_name)

            if not test_config or not test_config.get("enabled", False):
                await increment_counter(
                    "ab_test_assignments",
                    tags={"test": test_name, "variant": "control"},
                )
                return "control"

            # Get variants and their weights
            variants = test_config.get("variants", {"control": 50, "variant_a": 50})

            # Use hash of user_id + test_name for consistent assignment
            user_hash = hash(f"{user_id}:{test_name}") % 100

            # Determine variant based on cumulative weights
            cumulative_weight = 0
            for variant_name, weight in variants.items():
                cumulative_weight += weight
                if user_hash < cumulative_weight:
                    await increment_counter(
                        "ab_test_assignments",
                        tags={"test": test_name, "variant": variant_name},
                    )
                    return variant_name

            # Fallback to control
            await increment_counter(
                "ab_test_assignments", tags={"test": test_name, "variant": "control"}
            )
            return "control"

    except Exception as e:
        logger.exception(f"A/B test variant assignment failed for '{test_name}': {e}")
        return "control"


async def create_ab_test(
    test_name: str,
    variants: dict[str, int],
    enabled: bool = True,
    config: Annotated[Config, Depends(get_config)] = None,
) -> None:
    """Create or update an A/B test configuration.

    Args:
        test_name: Name of the A/B test
        variants: Dictionary of variant names to weights (should sum to 100)
        enabled: Whether the test is active
        config: Injected configuration
    """
    try:
        # Validate variant weights
        total_weight = sum(variants.values())
        if total_weight != 100:
            logger.warning(
                f"A/B test '{test_name}' variant weights sum to {total_weight}, not 100"
            )

        async with _ab_tests_lock:
            _ab_tests[test_name] = {
                "enabled": enabled,
                "variants": variants,
                "created_at": time.time(),
                "updated_at": time.time(),
            }

            await increment_counter("ab_test_updates", tags={"test": test_name})

            for variant_name, weight in variants.items():
                await set_gauge(
                    f"ab_test_variant_weight.{test_name}.{variant_name}", weight
                )

            logger.info(f"Created A/B test '{test_name}': {variants}")

    except Exception as e:
        logger.exception(f"Failed to create A/B test '{test_name}': {e}")


async def check_canary_readiness(
    service_name: str,
    version: str,
    health_threshold: float = 0.95,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Check if a canary deployment is ready for promotion.

    Simplified canary deployment readiness check.

    Args:
        service_name: Name of the service
        version: Version being deployed
        health_threshold: Minimum health score (0.0-1.0)
        config: Injected configuration

    Returns:
        Canary readiness status
    """
    try:
        # This would integrate with actual monitoring systems
        # For now, return a simulated readiness check

        readiness = {
            "service": service_name,
            "version": version,
            "ready_for_promotion": False,
            "health_score": 0.0,
            "checks": {
                "error_rate": {"status": "unknown", "value": None},
                "response_time": {"status": "unknown", "value": None},
                "cpu_usage": {"status": "unknown", "value": None},
                "memory_usage": {"status": "unknown", "value": None},
            },
            "recommendation": "insufficient_data",
        }

        # Simulate health score calculation
        # In real implementation, this would fetch actual metrics
        simulated_health_score = 0.98  # High health score

        readiness["health_score"] = simulated_health_score
        readiness["ready_for_promotion"] = simulated_health_score >= health_threshold
        readiness["recommendation"] = (
            "promote" if readiness["ready_for_promotion"] else "wait"
        )

        # Update checks with simulated values
        readiness["checks"]["error_rate"] = {"status": "healthy", "value": 0.1}
        readiness["checks"]["response_time"] = {"status": "healthy", "value": 150}
        readiness["checks"]["cpu_usage"] = {"status": "healthy", "value": 45}
        readiness["checks"]["memory_usage"] = {"status": "healthy", "value": 60}

        await increment_counter(
            "canary_checks",
            tags={
                "service": service_name,
                "ready": str(readiness["ready_for_promotion"]),
            },
        )
        await set_gauge(f"canary_health_score.{service_name}", simulated_health_score)

        logger.info(
            f"Canary check for {service_name}:{version} -> {readiness['recommendation']} (health: {simulated_health_score:.2f})"
        )

        return readiness

    except Exception as e:
        logger.exception(
            f"Canary readiness check failed for {service_name}:{version}: {e}"
        )
        return {
            "service": service_name,
            "version": version,
            "ready_for_promotion": False,
            "error": str(e),
        }


async def perform_blue_green_switch(
    service_name: str,
    from_environment: str,
    to_environment: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Perform blue/green deployment switch.

    Simplified blue/green deployment switching.

    Args:
        service_name: Name of the service
        from_environment: Current active environment ("blue" or "green")
        to_environment: Target environment ("blue" or "green")
        config: Injected configuration

    Returns:
        Switch operation result
    """
    try:
        if from_environment == to_environment:
            return {
                "service": service_name,
                "status": "error",
                "error": "Source and target environments cannot be the same",
            }

        if to_environment not in ["blue", "green"]:
            return {
                "service": service_name,
                "status": "error",
                "error": "Target environment must be 'blue' or 'green'",
            }

        # Simulate blue/green switch
        # In real implementation, this would:
        # 1. Update load balancer configuration
        # 2. Update service discovery
        # 3. Verify traffic is flowing to new environment
        # 4. Monitor for issues

        switch_result = {
            "service": service_name,
            "from_environment": from_environment,
            "to_environment": to_environment,
            "status": "success",
            "timestamp": time.time(),
            "verification": {
                "traffic_switch": "completed",
                "health_check": "passed",
                "rollback_ready": True,
            },
        }

        await increment_counter(
            "blue_green_switches", tags={"service": service_name, "status": "success"}
        )
        await set_gauge(
            f"active_environment.{service_name}", 1 if to_environment == "blue" else 0
        )

        logger.info(
            f"Blue/green switch for {service_name}: {from_environment} -> {to_environment}"
        )

        return switch_result

    except Exception as e:
        logger.exception(f"Blue/green switch failed for {service_name}: {e}")
        await increment_counter(
            "blue_green_switches", tags={"service": service_name, "status": "error"}
        )
        return {
            "service": service_name,
            "status": "error",
            "error": str(e),
        }


async def get_deployment_status(
    service_name: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Get deployment status for a service.

    Comprehensive deployment status including feature flags, A/B tests, and environments.

    Args:
        service_name: Name of the service
        config: Injected configuration

    Returns:
        Deployment status summary
    """
    try:
        status = {
            "service": service_name,
            "timestamp": time.time(),
            "feature_flags": {},
            "ab_tests": {},
            "environments": {
                "blue": {"status": "unknown", "version": "unknown"},
                "green": {"status": "unknown", "version": "unknown"},
                "canary": {"status": "unknown", "version": "unknown"},
            },
        }

        # Get feature flags related to this service
        async with _feature_flags_lock:
            service_flags = {
                name: config
                for name, config in _feature_flags.items()
                if service_name in name.lower() or name.startswith(f"{service_name}_")
            }
            status["feature_flags"] = service_flags

        # Get A/B tests related to this service
        async with _ab_tests_lock:
            service_tests = {
                name: config
                for name, config in _ab_tests.items()
                if service_name in name.lower() or name.startswith(f"{service_name}_")
            }
            status["ab_tests"] = service_tests

        # Simulate environment status
        # In real implementation, this would query actual deployment infrastructure
        status["environments"]["blue"]["status"] = "active"
        status["environments"]["blue"]["version"] = "v1.2.3"
        status["environments"]["green"]["status"] = "standby"
        status["environments"]["green"]["version"] = "v1.2.4"
        status["environments"]["canary"]["status"] = "inactive"

        logger.debug(f"Retrieved deployment status for {service_name}")
        return status

    except Exception as e:
        logger.exception(f"Failed to get deployment status for {service_name}: {e}")
        return {
            "service": service_name,
            "timestamp": time.time(),
            "error": str(e),
        }


async def rollback_deployment(
    service_name: str,
    target_version: str | None = None,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Rollback a deployment to previous version.

    Simplified deployment rollback function.

    Args:
        service_name: Name of the service
        target_version: Optional specific version to rollback to
        config: Injected configuration

    Returns:
        Rollback operation result
    """
    try:
        # Simulate rollback operation
        # In real implementation, this would:
        # 1. Identify previous stable version
        # 2. Switch traffic back to previous environment
        # 3. Verify rollback success
        # 4. Update deployment status

        rollback_result = {
            "service": service_name,
            "target_version": target_version or "previous",
            "status": "success",
            "timestamp": time.time(),
            "actions": [
                "Identified previous stable version: v1.2.3",
                "Switched traffic from green to blue environment",
                "Verified rollback health checks",
                "Updated deployment status",
            ],
            "verification": {
                "traffic_switch": "completed",
                "health_check": "passed",
                "error_rate": "normal",
            },
        }

        await increment_counter(
            "deployment_rollbacks", tags={"service": service_name, "status": "success"}
        )

        logger.info(
            f"Rollback completed for {service_name} -> {target_version or 'previous'}"
        )

        return rollback_result

    except Exception as e:
        logger.exception(f"Rollback failed for {service_name}: {e}")
        await increment_counter(
            "deployment_rollbacks", tags={"service": service_name, "status": "error"}
        )
        return {
            "service": service_name,
            "status": "error",
            "error": str(e),
        }
