"""Function-based auto-detection with dependency injection.

Simplified auto-detection functions that replace complex service discovery classes.
Provides environment detection, service discovery, and health checking.
"""

import asyncio
import logging
import urllib.parse
from typing import Annotated, Any

from fastapi import Depends

from src.config import Config
from src.config.auto_detect import (
    AutoDetectionConfig,
    DetectedEnvironment,
    DetectedService,
    EnvironmentDetector,
)

from .dependencies import get_config
from .monitoring import increment_counter, record_timer


logger = logging.getLogger(__name__)


async def detect_environment(
    config: Annotated[Config, Depends(get_config)] = None,
) -> DetectedEnvironment:
    """Detect current runtime environment.

    Pure function replacement for EnvironmentDetector.detect().

    Args:
        config: Injected configuration

    Returns:
        DetectedEnvironment with platform and service info
    """
    try:
        # Create minimal auto-detection config for the detector
        auto_config = AutoDetectionConfig()

        detector = EnvironmentDetector(auto_config)
        environment = detector.detect()

        await increment_counter(
            "environment_detections", tags={"platform": environment.platform}
        )
        logger.info(f"Detected environment: {environment.platform}")

        return environment

    except Exception as e:
        logger.exception(f"Environment detection failed: {e}")
        # Return default environment on error
        return DetectedEnvironment(
            platform="unknown",
            container_runtime="unknown",
            orchestrator=None,
            cloud_provider=None,
            services={},
            environment_type="development",
            is_containerized=False,
            is_kubernetes=False,
            detection_confidence=0.0,
            detection_time_ms=0.0,
        )


async def check_service_availability(
    host: str,
    port: int,
    timeout: float = 5.0,
    service_name: str | None = None,
) -> dict[str, Any]:
    """Check if a service is available at host:port.

    Simple function for testing service connectivity.

    Args:
        host: Service hostname or IP
        port: Service port
        timeout: Connection timeout in seconds
        service_name: Optional service name for logging

    Returns:
        Availability check result
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Test TCP connection
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()

        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        result = {
            "host": host,
            "port": port,
            "available": True,
            "response_time_ms": duration_ms,
            "error": None,
        }

        if service_name:
            await increment_counter(
                "service_checks", tags={"service": service_name, "status": "available"}
            )
            await record_timer(
                "service_check_duration", duration_ms, tags={"service": service_name}
            )

        logger.debug(f"Service check {host}:{port} -> available ({duration_ms:.1f}ms)")
        return result

    except Exception as e:
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        result = {
            "host": host,
            "port": port,
            "available": False,
            "response_time_ms": duration_ms,
            "error": str(e),
        }

        if service_name:
            await increment_counter(
                "service_checks",
                tags={"service": service_name, "status": "unavailable"},
            )

        logger.debug(f"Service check {host}:{port} -> unavailable: {e}")
        return result


async def discover_services(
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, DetectedService]:
    """Discover available services in the environment.

    Pure function replacement for ServiceDiscovery.discover().

    Args:
        config: Injected configuration

    Returns:
        Dictionary of discovered services
    """
    try:
        services = {}

        # Check Qdrant
        qdrant_result = await check_service_availability(
            config.vector_db.host, config.vector_db.port, service_name="qdrant"
        )

        if qdrant_result["available"]:
            services["qdrant"] = DetectedService(
                name="qdrant",
                host=config.vector_db.host,
                port=config.vector_db.port,
                protocol="http",
                health_endpoint="/health",
                available=True,
                response_time_ms=qdrant_result["response_time_ms"],
            )

        # Check Dragonfly cache
        if hasattr(config.cache, "dragonfly_url") and config.cache.dragonfly_url:
            # Parse URL for host/port

            parsed = urllib.parse.urlparse(config.cache.dragonfly_url)
            if parsed.hostname and parsed.port:
                cache_result = await check_service_availability(
                    parsed.hostname, parsed.port, service_name="dragonfly"
                )

                if cache_result["available"]:
                    services["dragonfly"] = DetectedService(
                        name="dragonfly",
                        host=parsed.hostname,
                        port=parsed.port,
                        protocol="redis",
                        health_endpoint=None,
                        available=True,
                        response_time_ms=cache_result["response_time_ms"],
                    )

        await increment_counter(
            "service_discovery", tags={"discovered_count": str(len(services))}
        )
        logger.info(f"Discovered {len(services)} services: {list(services.keys())}")

        return services

    except Exception as e:
        logger.exception(f"Service discovery failed: {e}")
        return {}


async def get_connection_info(
    service_name: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Get connection information for a service.

    Simplified function to get service connection details.

    Args:
        service_name: Name of the service
        config: Injected configuration

    Returns:
        Connection information dictionary
    """
    try:
        if service_name == "qdrant":
            return {
                "service": service_name,
                "host": config.vector_db.host,
                "port": config.vector_db.port,
                "url": f"http://{config.vector_db.host}:{config.vector_db.port}",
                "protocol": "http",
                "connection_pool": {
                    "max_connections": 20,
                    "timeout": 30,
                },
            }

        elif service_name == "dragonfly" and hasattr(config.cache, "dragonfly_url"):
            return {
                "service": service_name,
                "url": config.cache.dragonfly_url,
                "protocol": "redis",
                "connection_pool": {
                    "max_connections": 10,
                    "timeout": 5,
                },
            }

        else:
            logger.warning(f"Unknown service: {service_name}")
            return {
                "service": service_name,
                "error": "Service not configured",
            }

    except Exception as e:
        logger.exception(f"Failed to get connection info for {service_name}: {e}")
        return {
            "service": service_name,
            "error": str(e),
        }


async def test_service_endpoints(
    service_name: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Test service endpoints for functionality.

    Enhanced health checking with endpoint-specific tests.

    Args:
        service_name: Name of the service to test
        config: Injected configuration

    Returns:
        Endpoint test results
    """
    try:
        results = {
            "service": service_name,
            "endpoints": {},
            "overall_status": "healthy",
        }

        if service_name == "qdrant":
            # Test basic connectivity
            connectivity = await check_service_availability(
                config.vector_db.host, config.vector_db.port, service_name="qdrant"
            )
            results["endpoints"]["connectivity"] = connectivity

            # Could add more specific endpoint tests here
            # e.g., GET /collections, POST /collections/test/points/search

        elif service_name == "dragonfly":
            if hasattr(config.cache, "dragonfly_url") and config.cache.dragonfly_url:
                parsed = urllib.parse.urlparse(config.cache.dragonfly_url)
                if parsed.hostname and parsed.port:
                    connectivity = await check_service_availability(
                        parsed.hostname, parsed.port, service_name="dragonfly"
                    )
                    results["endpoints"]["connectivity"] = connectivity

        # Determine overall status
        all_available = all(
            endpoint.get("available", False)
            for endpoint in results["endpoints"].values()
        )
        results["overall_status"] = "healthy" if all_available else "unhealthy"

        await increment_counter(
            "endpoint_tests",
            tags={"service": service_name, "status": results["overall_status"]},
        )

        return results

    except Exception as e:
        logger.exception(f"Endpoint testing failed for {service_name}: {e}")
        return {
            "service": service_name,
            "overall_status": "error",
            "error": str(e),
        }


async def get_service_metrics(
    service_name: str,
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Get performance metrics for a service.

    Simplified metrics collection for service monitoring.

    Args:
        service_name: Name of the service
        config: Injected configuration

    Returns:
        Service metrics dictionary
    """
    try:
        # Run availability check to get current metrics
        conn_info = await get_connection_info(service_name, config)

        if "error" in conn_info:
            return {
                "service": service_name,
                "status": "error",
                "error": conn_info["error"],
            }

        # Test endpoint if we have connection info
        if "host" in conn_info and "port" in conn_info:
            availability = await check_service_availability(
                conn_info["host"], conn_info["port"], service_name=service_name
            )

            return {
                "service": service_name,
                "status": "available" if availability["available"] else "unavailable",
                "response_time_ms": availability["response_time_ms"],
                "connection_info": conn_info,
                "last_check": asyncio.get_event_loop().time(),
            }

        return {
            "service": service_name,
            "status": "configured",
            "connection_info": conn_info,
        }

    except Exception as e:
        logger.exception(f"Failed to get metrics for {service_name}: {e}")
        return {
            "service": service_name,
            "status": "error",
            "error": str(e),
        }


async def auto_configure_services(
    config: Annotated[Config, Depends(get_config)] = None,
) -> dict[str, Any]:
    """Auto-configure services based on environment detection.

    Comprehensive function that combines detection and configuration.

    Args:
        config: Injected configuration

    Returns:
        Auto-configuration results
    """
    try:
        # Detect environment
        environment = await detect_environment(config)

        # Discover services
        services = await discover_services(config)

        # Test discovered services
        service_tests = {}
        for service_name in services.keys():
            service_tests[service_name] = await test_service_endpoints(
                service_name, config
            )

        result = {
            "environment": environment,
            "discovered_services": services,
            "service_tests": service_tests,
            "auto_configuration": {
                "successful": len(
                    [
                        t
                        for t in service_tests.values()
                        if t.get("overall_status") == "healthy"
                    ]
                ),
                "failed": len(
                    [
                        t
                        for t in service_tests.values()
                        if t.get("overall_status") != "healthy"
                    ]
                ),
            },
        }

        await increment_counter(
            "auto_configurations",
            tags={"success": str(result["auto_configuration"]["successful"])},
        )

        logger.info(
            f"Auto-configuration complete: {result['auto_configuration']['successful']} successful, "
            f"{result['auto_configuration']['failed']} failed"
        )

        return result

    except Exception as e:
        logger.exception(f"Auto-configuration failed: {e}")
        return {
            "error": str(e),
            "auto_configuration": {
                "successful": 0,
                "failed": 1,
            },
        }
