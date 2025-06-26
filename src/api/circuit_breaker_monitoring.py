"""Circuit breaker monitoring endpoints.

Provides API endpoints for monitoring circuit breaker health and metrics.
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.config.core import Config, get_config
from src.services.functional.circuit_breaker_factory import (
    get_circuit_breaker_factory,
)
from src.services.functional.enhanced_circuit_breaker import (
    get_all_circuit_breaker_metrics,
    get_circuit_breaker_registry,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/circuit-breakers", tags=["circuit-breakers"])


@router.get("/status")
async def get_circuit_breaker_status(
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Get overall circuit breaker system status.

    Returns:
        System status with counts and health information
    """
    try:
        factory = get_circuit_breaker_factory(config)
        all_breakers = factory.get_all_circuit_breakers()

        # Count breakers by state
        state_counts = {"closed": 0, "open": 0, "half_open": 0, "unknown": 0}

        for breaker in all_breakers.values():
            try:
                state = breaker.state.value if hasattr(breaker, "state") else "unknown"
                state_counts[state] = state_counts.get(state, 0) + 1
            except Exception:
                state_counts["unknown"] += 1

        # Calculate health percentage
        total_breakers = len(all_breakers)
        healthy_breakers = state_counts["closed"] + state_counts["half_open"]
        health_percentage = (
            (healthy_breakers / total_breakers * 100) if total_breakers > 0 else 100
        )

        return {
            "status": "healthy"
            if health_percentage >= 80
            else "degraded"
            if health_percentage >= 50
            else "unhealthy",
            "total_circuit_breakers": total_breakers,
            "health_percentage": round(health_percentage, 1),
            "state_counts": state_counts,
            "enhanced_circuit_breakers_enabled": config.circuit_breaker.use_enhanced_circuit_breaker,
        }

    except Exception as e:
        logger.exception("Failed to get circuit breaker status")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {e}")


@router.get("/metrics")
async def get_circuit_breaker_metrics(
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Get detailed metrics for all circuit breakers.

    Returns:
        Detailed metrics for each circuit breaker
    """
    try:
        factory = get_circuit_breaker_factory(config)
        metrics = factory.get_circuit_breaker_metrics()

        # Also get enhanced circuit breaker registry metrics
        if config.circuit_breaker.use_enhanced_circuit_breaker:
            enhanced_metrics = get_all_circuit_breaker_metrics()
            metrics.update(enhanced_metrics)

        return {
            "timestamp": time.time(),
            "metrics": metrics,
            "summary": {
                "total_services": len(metrics),
                "total_requests": sum(
                    m.get("total_requests", 0)
                    for m in metrics.values()
                    if isinstance(m, dict)
                ),
                "total_failures": sum(
                    m.get("failed_requests", 0)
                    for m in metrics.values()
                    if isinstance(m, dict)
                ),
            },
        }

    except Exception as e:
        logger.exception("Failed to get circuit breaker metrics")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {e}")


@router.get("/metrics/{service_name}")
async def get_service_circuit_breaker_metrics(
    service_name: str,
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Get metrics for a specific service's circuit breaker.

    Args:
        service_name: Name of the service

    Returns:
        Circuit breaker metrics for the specified service
    """
    try:
        factory = get_circuit_breaker_factory(config)
        breaker = factory.get_service_circuit_breaker(service_name)

        if not breaker:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit breaker not found for service: {service_name}",
            )

        metrics = breaker.get_metrics()
        return {
            "service_name": service_name,
            "timestamp": time.time(),
            "metrics": metrics,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get metrics for service: {service_name}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics for service {service_name}: {e}",
        )


@router.post("/reset/{service_name}")
async def reset_service_circuit_breaker(
    service_name: str,
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Reset a specific service's circuit breaker to closed state.

    Args:
        service_name: Name of the service

    Returns:
        Reset confirmation
    """
    try:
        factory = get_circuit_breaker_factory(config)
        breaker = factory.get_service_circuit_breaker(service_name)

        if not breaker:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit breaker not found for service: {service_name}",
            )

        breaker.reset()
        logger.info(f"Reset circuit breaker for service: {service_name}")

        return {
            "service_name": service_name,
            "status": "reset",
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to reset circuit breaker for service: {service_name}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset circuit breaker for service {service_name}: {e}",
        )


@router.post("/reset-all")
async def reset_all_circuit_breakers(
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Reset all circuit breakers to closed state.

    Returns:
        Reset confirmation with count
    """
    try:
        factory = get_circuit_breaker_factory(config)
        all_breakers = factory.get_all_circuit_breakers()

        factory.reset_all_circuit_breakers()

        logger.info(f"Reset {len(all_breakers)} circuit breakers")

        return {
            "status": "reset_all",
            "count": len(all_breakers),
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.exception("Failed to reset all circuit breakers")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset all circuit breakers: {e}",
        )


@router.get("/health")
async def get_circuit_breaker_health(
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Get health check information for circuit breaker system.

    Returns:
        Health check response
    """
    try:
        factory = get_circuit_breaker_factory(config)
        all_breakers = factory.get_all_circuit_breakers()

        # Count open circuit breakers
        open_breakers = []
        for service_name, breaker in all_breakers.items():
            try:
                if hasattr(breaker, "state") and breaker.state.value == "open":
                    open_breakers.append(service_name)
            except Exception:
                pass

        is_healthy = len(open_breakers) == 0

        return {
            "healthy": is_healthy,
            "total_circuit_breakers": len(all_breakers),
            "open_circuit_breakers": open_breakers,
            "open_count": len(open_breakers),
            "timestamp": time.time(),
        }

    except Exception as e:
        logger.exception("Failed to get circuit breaker health")
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": time.time(),
        }


@router.get("/config")
async def get_circuit_breaker_config(
    config: Config = Depends(get_config),
) -> dict[str, Any]:
    """Get circuit breaker configuration.

    Returns:
        Current circuit breaker configuration
    """
    try:
        return {
            "enhanced_circuit_breakers_enabled": config.circuit_breaker.use_enhanced_circuit_breaker,
            "default_failure_threshold": config.circuit_breaker.failure_threshold,
            "default_recovery_timeout": config.circuit_breaker.recovery_timeout,
            "metrics_enabled": config.circuit_breaker.enable_metrics_collection,
            "detailed_metrics_enabled": config.circuit_breaker.enable_detailed_metrics,
            "fallback_mechanisms_enabled": config.circuit_breaker.enable_fallback_mechanisms,
            "service_overrides": config.circuit_breaker.service_overrides,
        }

    except Exception as e:
        logger.exception("Failed to get circuit breaker configuration")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get configuration: {e}",
        )
