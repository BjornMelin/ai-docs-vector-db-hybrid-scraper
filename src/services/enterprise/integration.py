"""Enterprise Integration Framework for Unified Advanced Features.

This module provides the core integration framework that coordinates all enterprise
features into a cohesive, scalable, and maintainable system. It implements the
enterprise integration strategy with service orchestration, configuration management,
security framework, and observability platform.
"""

import asyncio
import contextlib
import logging
import time
# Callable import removed (unused)
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

# BaseModel and Field imports removed (unused)

from src.architecture.service_factory import BaseService
from src.config.modern import Config
from src.services.deployment.feature_flags import FeatureFlagManager
from src.services.observability.performance import PerformanceMonitor
from src.services.security.integration import SecurityManager


if TYPE_CHECKING:
    from src.services.deployment.blue_green import BlueGreenDeployment


logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service lifecycle status."""

    UNREGISTERED = "unregistered"
    REGISTERED = "registered"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class IntegrationPhase(str, Enum):
    """Enterprise integration deployment phases."""

    PLANNING = "planning"
    PREPARATION = "preparation"
    DEPLOYMENT = "deployment"
    VALIDATION = "validation"
    MONITORING = "monitoring"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class ServiceDescriptor:
    """Comprehensive service description for enterprise integration."""

    name: str
    service: BaseService
    dependencies: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    health_check_url: str | None = None
    startup_timeout: int = 60
    shutdown_timeout: int = 30
    criticality: str = "normal"  # critical, high, normal, low
    tags: set[str] = field(default_factory=set)

    # Runtime state
    status: ServiceStatus = ServiceStatus.UNREGISTERED
    last_health_check: datetime | None = None
    startup_time: float | None = None
    error_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Advanced health monitoring for enterprise services."""

    def __init__(self, service_name: str, descriptor: ServiceDescriptor):
        self.service_name = service_name
        self.descriptor = descriptor
        self.health_history: list[dict[str, Any]] = []
        self.alert_thresholds = {
            "error_rate": 0.05,  # 5% error rate
            "response_time": 1000,  # 1 second
            "availability": 0.99,  # 99% availability
        }

    async def check_health(self) -> dict[str, Any]:
        """Perform comprehensive health check."""
        start_time = time.time()
        health_result = {
            "timestamp": datetime.now(tz=UTC),
            "service": self.service_name,
            "status": "unknown",
            "response_time_ms": 0.0,
            "details": {},
        }

        try:
            # Check if service is properly initialized
            if not self.descriptor.service.is_initialized():
                health_result.update(
                    {
                        "status": "unhealthy",
                        "details": {"reason": "service_not_initialized"},
                    }
                )
                return health_result

            # Perform service-specific health check
            if hasattr(self.descriptor.service, "health_check"):
                service_health = await self.descriptor.service.health_check()
                health_result["details"].update(service_health)

            # Check dependencies
            dependency_health = await self._check_dependencies()
            health_result["details"]["dependencies"] = dependency_health

            # Calculate overall status
            response_time = (time.time() - start_time) * 1000
            health_result["response_time_ms"] = response_time

            if all(dep["status"] == "healthy" for dep in dependency_health.values()):
                if response_time < self.alert_thresholds["response_time"]:
                    health_result["status"] = "healthy"
                else:
                    health_result["status"] = "degraded"
                    health_result["details"]["reason"] = "slow_response"
            else:
                health_result["status"] = "unhealthy"
                health_result["details"]["reason"] = "dependency_failure"

        except Exception as e:
            health_result.update(
                {
                    "status": "unhealthy",
                    "details": {"reason": "health_check_exception", "error": str(e)},
                }
            )
            logger.exception("Health check failed for {self.service_name}")

        # Update descriptor status
        self.descriptor.status = ServiceStatus(health_result["status"])
        self.descriptor.last_health_check = health_result["timestamp"]

        # Store health history
        self.health_history.append(health_result)
        if len(self.health_history) > 100:  # Keep last 100 checks
            self.health_history.pop(0)

        return health_result

    async def _check_dependencies(self) -> dict[str, dict[str, Any]]:
        """Check health of service dependencies."""
        dependency_health = {}

        for dep_name in self.descriptor.dependencies:
            # This would integrate with the service registry
            # For now, return healthy status
            dependency_health[dep_name] = {
                "status": "healthy",
                "response_time_ms": 10.0,
            }

        return dependency_health

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary and statistics."""
        if not self.health_history:
            return {"status": "no_data", "checks": 0}

        recent_checks = self.health_history[-10:]  # Last 10 checks
        healthy_count = sum(
            1 for check in recent_checks if check["status"] == "healthy"
        )
        availability = healthy_count / len(recent_checks)

        avg_response_time = sum(
            check["response_time_ms"] for check in recent_checks
        ) / len(recent_checks)

        return {
            "current_status": recent_checks[-1]["status"],
            "availability_percent": availability * 100,
            "avg_response_time_ms": avg_response_time,
            "total_checks": len(self.health_history),
            "error_count": self.descriptor.error_count,
            "last_check": recent_checks[-1]["timestamp"].isoformat(),
        }


class ServiceDependencyGraph:
    """Manages service dependencies and orchestration order."""

    def __init__(self):
        self.graph: dict[str, set[str]] = {}
        self.reverse_graph: dict[str, set[str]] = {}

    def add_service(self, service_name: str, dependencies: list[str]) -> None:
        """Add service and its dependencies to the graph."""
        self.graph[service_name] = set(dependencies)

        # Build reverse graph for dependents
        if service_name not in self.reverse_graph:
            self.reverse_graph[service_name] = set()

        for dep in dependencies:
            if dep not in self.reverse_graph:
                self.reverse_graph[dep] = set()
            self.reverse_graph[dep].add(service_name)

    def resolve_startup_order(self) -> list[str]:
        """Resolve service startup order using topological sort."""
        visited = set()
        temp_visited = set()
        result = []

        def visit(node: str) -> None:
            if node in temp_visited:
                msg = f"Circular dependency detected involving {node}"
                raise ValueError(msg)
            if node in visited:
                return

            temp_visited.add(node)

            # Visit dependencies first
            for dep in self.graph.get(node, set()):
                visit(dep)

            temp_visited.remove(node)
            visited.add(node)
            result.append(node)

        # Visit all nodes
        for service in self.graph:
            if service not in visited:
                visit(service)

        return result

    def resolve_shutdown_order(self) -> list[str]:
        """Resolve service shutdown order (reverse of startup)."""
        return list(reversed(self.resolve_startup_order()))

    def detect_circular_dependencies(self) -> list[list[str]]:
        """Detect circular dependencies in the service graph."""
        cycles = []
        visited = set()
        path = []

        def dfs(node: str) -> None:
            if node in path:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = [*path[cycle_start:], node]
                cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            path.append(node)

            for dep in self.graph.get(node, set()):
                dfs(dep)

            path.remove(node)

        for service in self.graph:
            if service not in visited:
                dfs(service)

        return cycles

    def get_dependents(self, service_name: str) -> set[str]:
        """Get services that depend on the given service."""
        return self.reverse_graph.get(service_name, set())

    def validate_dependencies(
        self, available_services: set[str]
    ) -> dict[str, list[str]]:
        """Validate that all dependencies are available."""
        missing_deps = {}

        for service, deps in self.graph.items():
            missing = [dep for dep in deps if dep not in available_services]
            if missing:
                missing_deps[service] = missing

        return missing_deps


class EnterpriseServiceRegistry:
    """Centralized service discovery and orchestration for enterprise features."""

    def __init__(self, config: Config):
        self.config = config
        self.services: dict[str, ServiceDescriptor] = {}
        self.health_monitors: dict[str, HealthMonitor] = {}
        self.dependency_graph = ServiceDependencyGraph()

        # Integration state
        self.startup_order: list[str] = []
        self.shutdown_order: list[str] = []
        self.is_orchestrating = False

        # Monitoring
        self.health_check_interval = 30  # seconds
        self.health_check_task: asyncio.Task | None = None

        logger.info("Enterprise service registry initialized")

    async def register_service(
        self,
        service: BaseService,
        dependencies: list[str] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Register enterprise service with dependency tracking."""
        service_name = service.get_service_name()
        dependencies = dependencies or []
        config = config or {}

        # Create service descriptor
        descriptor = ServiceDescriptor(
            name=service_name,
            service=service,
            dependencies=dependencies,
            config=config,
            **kwargs,
        )

        # Register service
        self.services[service_name] = descriptor

        # Create health monitor
        self.health_monitors[service_name] = HealthMonitor(service_name, descriptor)

        # Update dependency graph
        self.dependency_graph.add_service(service_name, dependencies)

        logger.info("Registered enterprise service: %s", service_name)

    async def orchestrate_startup(self) -> None:
        """Start services in dependency order with health validation."""
        if self.is_orchestrating:
            msg = "Service orchestration already in progress"
            raise RuntimeError(msg)

        self.is_orchestrating = True

        try:
            # Validate dependencies
            available_services = set(self.services.keys())
            missing_deps = self.dependency_graph.validate_dependencies(
                available_services
            )

            if missing_deps:
                msg = f"Missing service dependencies: {missing_deps}"
                raise ValueError(msg)

            # Detect circular dependencies
            cycles = self.dependency_graph.detect_circular_dependencies()
            if cycles:
                msg = f"Circular dependencies detected: {cycles}"
                raise ValueError(msg)

            # Resolve startup order
            self.startup_order = self.dependency_graph.resolve_startup_order()
            self.shutdown_order = self.dependency_graph.resolve_shutdown_order()

            logger.info("Starting services in order: %s", self.startup_order)

            # Start services
            for service_name in self.startup_order:
                await self._start_service(service_name)

            # Start health monitoring
            self.health_check_task = asyncio.create_task(self._health_check_loop())

            logger.info("Enterprise service orchestration completed successfully")

        except Exception as e:
            logger.exception("Service orchestration failed")
            await self.coordinate_shutdown()
            raise
        finally:
            self.is_orchestrating = False

    async def coordinate_shutdown(self) -> None:
        """Graceful shutdown with dependency awareness."""
        logger.info("Starting enterprise service shutdown")

        # Stop health monitoring
        if self.health_check_task:
            self.health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.health_check_task

        # Shutdown services in reverse order
        for service_name in self.shutdown_order:
            await self._stop_service(service_name)

        logger.info("Enterprise service shutdown completed")

    async def get_system_status(self) -> dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "total_services": len(self.services),
            "orchestration_active": self.is_orchestrating,
            "services": {},
            "health_summary": {},
        }

        # Get individual service status
        for service_name, descriptor in self.services.items():
            health_monitor = self.health_monitors[service_name]
            status["services"][service_name] = {
                "status": descriptor.status.value,
                "dependencies": descriptor.dependencies,
                "criticality": descriptor.criticality,
                "uptime_seconds": (
                    time.time() - descriptor.startup_time
                    if descriptor.startup_time
                    else 0
                ),
                "error_count": descriptor.error_count,
                "health": health_monitor.get_health_summary(),
            }

        # Generate overall health summary
        healthy_services = sum(
            1 for desc in self.services.values() if desc.status == ServiceStatus.HEALTHY
        )

        status["health_summary"] = {
            "healthy_services": healthy_services,
            "total_services": len(self.services),
            "health_percentage": (healthy_services / len(self.services) * 100)
            if self.services
            else 0,
            "overall_status": "healthy"
            if healthy_services == len(self.services)
            else "degraded",
        }

        return status

    async def validate_service_health(self, service_name: str) -> dict[str, Any]:
        """Perform detailed health validation for specific service."""
        if service_name not in self.health_monitors:
            msg = f"Service {service_name} not registered"
            raise ValueError(msg)

        health_monitor = self.health_monitors[service_name]
        return await health_monitor.check_health()

    async def _start_service(self, service_name: str) -> None:
        """Start individual service with health validation."""
        descriptor = self.services[service_name]

        try:
            descriptor.status = ServiceStatus.STARTING
            start_time = time.time()

            logger.info("Starting service: %s", service_name)

            # Initialize service
            await descriptor.service.initialize()

            # Validate health
            health_result = await self.health_monitors[service_name].check_health()

            if health_result["status"] == "healthy":
                descriptor.status = ServiceStatus.HEALTHY
                descriptor.startup_time = start_time
                logger.info("Service %s started successfully", service_name)
            else:
                descriptor.status = ServiceStatus.UNHEALTHY
                msg = f"Service {service_name} failed health check: {health_result}"
                raise RuntimeError(msg)

        except Exception as e:
            descriptor.status = ServiceStatus.FAILED
            descriptor.error_count += 1
            logger.exception("Failed to start service {service_name}")
            raise

    async def _stop_service(self, service_name: str) -> None:
        """Stop individual service gracefully."""
        descriptor = self.services[service_name]

        try:
            descriptor.status = ServiceStatus.STOPPING

            logger.info("Stopping service: %s", service_name)

            # Cleanup service
            await descriptor.service.cleanup()

            descriptor.status = ServiceStatus.STOPPED
            logger.info("Service %s stopped successfully", service_name)

        except Exception as e:
            descriptor.status = ServiceStatus.FAILED
            descriptor.error_count += 1
            logger.exception("Failed to stop service {service_name}")

    async def _health_check_loop(self) -> None:
        """Background health monitoring for all services."""
        while True:
            try:
                # Check health of all services
                for service_name in self.services:
                    try:
                        await self.health_monitors[service_name].check_health()
                    except Exception as e:
                        logger.exception("Health check failed for {service_name}")

                await asyncio.sleep(self.health_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in health check loop")
                await asyncio.sleep(self.health_check_interval)


class EnterpriseIntegrationManager:
    """Main orchestrator for enterprise feature integration."""

    def __init__(self, config: Config):
        self.config = config
        self.service_registry = EnterpriseServiceRegistry(config)

        # Integration components
        self.security_manager: SecurityManager | None = None
        self.performance_monitor: PerformanceMonitor | None = None
        self.feature_flag_manager: FeatureFlagManager | None = None
        self.deployment_manager: BlueGreenDeployment | None = None

        # Integration state
        self.integration_phase = IntegrationPhase.PLANNING
        self.integration_start_time: float | None = None

        logger.info("Enterprise integration manager initialized")

    async def initialize_enterprise_features(self) -> None:
        """Initialize all enterprise features with coordinated integration."""
        self.integration_phase = IntegrationPhase.PREPARATION
        self.integration_start_time = time.time()

        try:
            logger.info("Starting enterprise feature integration")

            # Initialize core infrastructure
            await self._initialize_core_infrastructure()

            # Register enterprise services
            await self._register_enterprise_services()

            # Start service orchestration
            self.integration_phase = IntegrationPhase.DEPLOYMENT
            await self.service_registry.orchestrate_startup()

            # Validate integration
            self.integration_phase = IntegrationPhase.VALIDATION
            await self._validate_enterprise_integration()

            # Begin monitoring
            self.integration_phase = IntegrationPhase.MONITORING

            self.integration_phase = IntegrationPhase.COMPLETED

            integration_time = time.time() - self.integration_start_time
            logger.info(
                f"Enterprise integration completed in {integration_time:.2f} seconds"
            )

        except Exception as e:
            self.integration_phase = IntegrationPhase.FAILED
            logger.exception("Enterprise integration failed")
            await self.cleanup_enterprise_features()
            raise

    async def cleanup_enterprise_features(self) -> None:
        """Cleanup all enterprise features gracefully."""
        logger.info("Starting enterprise feature cleanup")

        try:
            await self.service_registry.coordinate_shutdown()

            # Cleanup individual components
            if self.security_manager:
                await self.security_manager.cleanup_resources()

            if self.performance_monitor:
                await self.performance_monitor.cleanup()

            logger.info("Enterprise feature cleanup completed")

        except Exception as e:
            logger.exception("Error during enterprise cleanup")

    async def get_integration_status(self) -> dict[str, Any]:
        """Get comprehensive integration status."""
        system_status = await self.service_registry.get_system_status()

        return {
            "integration_phase": self.integration_phase.value,
            "integration_uptime": (
                time.time() - self.integration_start_time
                if self.integration_start_time
                else 0
            ),
            "system_status": system_status,
            "enterprise_features": {
                "security_manager": self.security_manager is not None,
                "performance_monitor": self.performance_monitor is not None,
                "feature_flags": self.feature_flag_manager is not None,
                "blue_green_deployment": self.deployment_manager is not None,
            },
        }

    async def _initialize_core_infrastructure(self) -> None:
        """Initialize core infrastructure components."""
        # Initialize security manager
        self.security_manager = SecurityManager()
        await self.security_manager.initialize_components()

        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        await self.performance_monitor.initialize()

        # Initialize feature flag manager
        self.feature_flag_manager = FeatureFlagManager()

        logger.info("Core infrastructure initialized")

    async def _register_enterprise_services(self) -> None:
        """Register all enterprise services with the registry."""
        # Register enterprise cache service
        from src.services.enterprise.cache import EnterpriseCacheService

        cache_service = EnterpriseCacheService()
        await self.service_registry.register_service(
            cache_service, dependencies=[], criticality="high"
        )

        # Register enterprise search service
        from src.services.enterprise.search import EnterpriseSearchService

        search_service = EnterpriseSearchService()
        await self.service_registry.register_service(
            search_service,
            dependencies=["enterprise_cache_service"],
            criticality="critical",
        )

        logger.info("Enterprise services registered")

    async def _validate_enterprise_integration(self) -> None:
        """Validate enterprise feature integration."""
        # Validate all services are healthy
        system_status = await self.service_registry.get_system_status()

        if system_status["health_summary"]["overall_status"] != "healthy":
            msg = "Enterprise integration validation failed: unhealthy services"
            raise RuntimeError(msg)

        # Validate security framework
        if self.security_manager:
            security_status = await self.security_manager.get_security_status()
            if not security_status.get("components_initialized", False):
                msg = "Security framework validation failed"
                raise RuntimeError(msg)

        logger.info("Enterprise integration validation passed")


# Global integration manager instance
_integration_manager: EnterpriseIntegrationManager | None = None


async def get_integration_manager(
    config: Config = None,
) -> EnterpriseIntegrationManager:
    """Get or create the global enterprise integration manager."""
    global _integration_manager

    if _integration_manager is None:
        if config is None:
            config = Config()
        _integration_manager = EnterpriseIntegrationManager(config)

    return _integration_manager


async def initialize_enterprise_platform(
    config: Config = None,
) -> EnterpriseIntegrationManager:
    """Initialize the complete enterprise platform."""
    integration_manager = await get_integration_manager(config)
    await integration_manager.initialize_enterprise_features()
    return integration_manager


async def cleanup_enterprise_platform() -> None:
    """Cleanup the enterprise platform."""
    global _integration_manager

    if _integration_manager:
        await _integration_manager.cleanup_enterprise_features()
        _integration_manager = None
