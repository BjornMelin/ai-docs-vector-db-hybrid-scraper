"""Enhanced circuit breaker implementation using the circuitbreaker library.

Provides a wrapper around the standard circuitbreaker library with additional features:
- Integration with existing monitoring and metrics
- Service-specific configurations from config
- Fallback mechanisms
- Custom exception handling patterns
- FastAPI middleware support
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

import circuitbreaker
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .circuit_breaker import CircuitBreakerError


logger = logging.getLogger(__name__)

T = TypeVar("T")


class EnhancedCircuitBreakerState(Enum):
    """Enhanced circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class EnhancedCircuitBreakerConfig:
    """Enhanced circuit breaker configuration.

    Uses the standard circuitbreaker library with additional enterprise features.
    """

    # Core circuitbreaker library settings
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    expected_exception: type[Exception] | tuple[type[Exception], ...] | None = None

    # Enhanced features
    enable_metrics: bool = True
    enable_fallback: bool = True
    fallback_function: Callable[..., Any] | None = None

    # Service identification
    service_name: str = "default"

    # Exception handling
    monitored_exceptions: tuple[type[Exception], ...] = field(
        default_factory=lambda: (Exception,)
    )

    # Metrics collection
    collect_detailed_metrics: bool = True

    @classmethod
    def from_service_config(
        cls, service_name: str, config_overrides: dict[str, Any] | None = None
    ) -> "EnhancedCircuitBreakerConfig":
        """Create configuration for a specific service with overrides."""
        # Default service configurations
        service_defaults = {
            "openai": {
                "failure_threshold": 3,
                "recovery_timeout": 30,
                "expected_exception": (Exception,),
            },
            "firecrawl": {
                "failure_threshold": 5,
                "recovery_timeout": 60,
                "expected_exception": (Exception,),
            },
            "qdrant": {
                "failure_threshold": 3,
                "recovery_timeout": 15,
                "expected_exception": (Exception,),
            },
            "redis": {
                "failure_threshold": 2,
                "recovery_timeout": 10,
                "expected_exception": (Exception,),
            },
        }

        # Get service-specific defaults
        defaults = service_defaults.get(service_name, {})

        # Apply overrides
        if config_overrides:
            defaults.update(config_overrides)

        return cls(service_name=service_name, **defaults)

    @classmethod
    def simple_mode(
        cls, service_name: str = "default"
    ) -> "EnhancedCircuitBreakerConfig":
        """Create simple circuit breaker configuration."""
        return cls(
            service_name=service_name,
            failure_threshold=3,
            recovery_timeout=30,
            enable_metrics=False,
            enable_fallback=False,
            collect_detailed_metrics=False,
        )

    @classmethod
    def enterprise_mode(
        cls, service_name: str = "default"
    ) -> "EnhancedCircuitBreakerConfig":
        """Create enterprise circuit breaker configuration with advanced features."""
        return cls(
            service_name=service_name,
            failure_threshold=5,
            recovery_timeout=60,
            enable_metrics=True,
            enable_fallback=True,
            collect_detailed_metrics=True,
        )


@dataclass
class EnhancedCircuitBreakerMetrics:
    """Enhanced circuit breaker metrics tracking."""

    service_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    blocked_requests: int = 0
    circuit_open_count: int = 0
    circuit_close_count: int = 0
    last_failure_time: float = 0
    last_success_time: float = 0
    average_response_time: float = 0
    created_at: float = field(default_factory=time.time)

    @property
    def failure_rate(self) -> float:
        """Calculate current failure rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate current success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        return time.time() - self.created_at


class EnhancedCircuitBreaker:
    """Enhanced circuit breaker using the standard circuitbreaker library.

    Provides additional features on top of the circuitbreaker library:
    - Metrics collection and monitoring integration
    - Service-specific configuration
    - Fallback mechanisms
    - Custom exception handling
    """

    def __init__(self, config: EnhancedCircuitBreakerConfig):
        """Initialize enhanced circuit breaker.

        Args:
            config: Enhanced circuit breaker configuration
        """
        self.config = config
        self.metrics = EnhancedCircuitBreakerMetrics(service_name=config.service_name)
        self._lock = asyncio.Lock()
        self._previous_state = None

        # Create the underlying circuitbreaker
        self._circuit = circuitbreaker.CircuitBreaker(
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
            expected_exception=config.expected_exception or config.monitored_exceptions,
            name=config.service_name,
            fallback_function=config.fallback_function,
        )

        # Track initial state
        self._previous_state = self._circuit.state

        logger.info(
            f"Enhanced circuit breaker initialized for {config.service_name}: "
            f"{config.failure_threshold} threshold, {config.recovery_timeout}s recovery"
        )

    def _check_state_change(self) -> None:
        """Check for state changes and update metrics accordingly."""
        current_state = self._circuit.state
        if current_state != self._previous_state:
            if current_state == "open" and self._previous_state != "open":
                self.metrics.circuit_open_count += 1
                logger.warning(
                    f"Circuit breaker OPEN for {self.config.service_name}: "
                    f"failure_rate={self.metrics.failure_rate:.2%}"
                )
            elif current_state == "closed" and self._previous_state != "closed":
                self.metrics.circuit_close_count += 1
                logger.info(f"Circuit breaker CLOSED for {self.config.service_name}")

            self._previous_state = current_state

    @property
    def state(self) -> EnhancedCircuitBreakerState:
        """Get current circuit state."""
        circuit_state = self._circuit.state
        if circuit_state == "closed":
            return EnhancedCircuitBreakerState.CLOSED
        elif circuit_state == "open":
            return EnhancedCircuitBreakerState.OPEN
        elif circuit_state == "half_open":
            return EnhancedCircuitBreakerState.HALF_OPEN
        else:
            return EnhancedCircuitBreakerState.CLOSED

    async def call(
        self, func: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
            Original exception: If function fails and circuit allows
        """
        start_time = time.time()

        # Check for state changes
        if self.config.enable_metrics:
            self._check_state_change()

        try:
            # Use the circuit breaker as decorator for proper behavior
            if asyncio.iscoroutinefunction(func):
                # Create a wrapped async function
                @self._circuit
                async def wrapped_func():
                    return await func(*args, **kwargs)

                result = await wrapped_func()
            else:
                result = self._circuit.call(func, *args, **kwargs)

            # Record success metrics
            if self.config.enable_metrics:
                await self._record_success(time.time() - start_time)

            return result

        except circuitbreaker.CircuitBreakerError as e:
            # Convert to our custom exception and record metrics
            if self.config.enable_metrics:
                async with self._lock:
                    self.metrics.blocked_requests += 1
                    self._check_state_change()
            raise CircuitBreakerError(str(e)) from e

        except Exception as e:
            # Record failure metrics
            if self.config.enable_metrics:
                await self._record_failure(time.time() - start_time)
                self._check_state_change()
            raise

    async def _record_success(self, execution_time: float) -> None:
        """Record successful execution metrics."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_success_time = time.time()

            # Update average response time
            if self.config.collect_detailed_metrics:
                total_time = (
                    self.metrics.average_response_time
                    * (self.metrics.total_requests - 1)
                    + execution_time
                )
                self.metrics.average_response_time = (
                    total_time / self.metrics.total_requests
                )

    async def _record_failure(self, execution_time: float) -> None:
        """Record failed execution metrics."""
        async with self._lock:
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            self.metrics.last_failure_time = time.time()

            # Update average response time
            if self.config.collect_detailed_metrics:
                total_time = (
                    self.metrics.average_response_time
                    * (self.metrics.total_requests - 1)
                    + execution_time
                )
                self.metrics.average_response_time = (
                    total_time / self.metrics.total_requests
                )

    def get_metrics(self) -> dict[str, Any]:
        """Get circuit breaker metrics.

        Returns:
            Dictionary with current metrics
        """
        return {
            "service_name": self.metrics.service_name,
            "state": self.state.value,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "blocked_requests": self.metrics.blocked_requests,
            "failure_rate": self.metrics.failure_rate,
            "success_rate": self.metrics.success_rate,
            "average_response_time": self.metrics.average_response_time,
            "circuit_open_count": self.metrics.circuit_open_count,
            "circuit_close_count": self.metrics.circuit_close_count,
            "uptime_seconds": self.metrics.uptime_seconds,
            "last_failure_time": self.metrics.last_failure_time,
            "last_success_time": self.metrics.last_success_time,
        }

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._circuit.reset()
        logger.info(f"Circuit breaker reset for {self.config.service_name}")


def enhanced_circuit_breaker(
    config: EnhancedCircuitBreakerConfig | None = None,
    service_name: str = "default",
):
    """Decorator for enhanced circuit breaker functionality.

    Args:
        config: Circuit breaker configuration
        service_name: Name of the service for metrics and logging

    Returns:
        Decorated function with circuit breaker protection
    """
    if config is None:
        config = EnhancedCircuitBreakerConfig.simple_mode(service_name)

    breaker = EnhancedCircuitBreaker(config)

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call(func, *args, **kwargs)

        # Attach circuit breaker for metrics access
        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator


def create_enhanced_circuit_breaker(
    service_name: str = "default",
    mode: str = "simple",
    config_overrides: dict[str, Any] | None = None,
) -> EnhancedCircuitBreaker:
    """Factory function for creating enhanced circuit breakers.

    Args:
        service_name: Name of the service
        mode: "simple" or "enterprise"
        config_overrides: Additional configuration overrides

    Returns:
        Configured EnhancedCircuitBreaker instance
    """
    if mode == "simple":
        config = EnhancedCircuitBreakerConfig.simple_mode(service_name)
    elif mode == "enterprise":
        config = EnhancedCircuitBreakerConfig.enterprise_mode(service_name)
    else:
        config = EnhancedCircuitBreakerConfig.from_service_config(
            service_name, config_overrides
        )

    # Apply any additional overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)

    return EnhancedCircuitBreaker(config)


class EnhancedCircuitBreakerMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for enhanced circuit breaker protection."""

    def __init__(self, app, config: EnhancedCircuitBreakerConfig | None = None):
        """Initialize enhanced circuit breaker middleware.

        Args:
            app: FastAPI application
            config: Circuit breaker configuration
        """
        super().__init__(app)
        self.config = config or EnhancedCircuitBreakerConfig.simple_mode("api")
        self.circuit_breaker = EnhancedCircuitBreaker(self.config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with circuit breaker protection.

        Args:
            request: HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        try:
            return await self.circuit_breaker.call(call_next, request)
        except CircuitBreakerError as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=503, detail=str(e))


# Registry for circuit breakers to enable centralized monitoring
_circuit_breaker_registry: dict[str, EnhancedCircuitBreaker] = {}


def register_circuit_breaker(
    name: str, circuit_breaker: EnhancedCircuitBreaker
) -> None:
    """Register a circuit breaker for monitoring."""
    _circuit_breaker_registry[name] = circuit_breaker


def get_circuit_breaker_registry() -> dict[str, EnhancedCircuitBreaker]:
    """Get all registered circuit breakers."""
    return _circuit_breaker_registry.copy()


def get_all_circuit_breaker_metrics() -> dict[str, dict[str, Any]]:
    """Get metrics for all registered circuit breakers."""
    return {
        name: breaker.get_metrics()
        for name, breaker in _circuit_breaker_registry.items()
    }


# Convenience function for middleware setup
def enhanced_circuit_breaker_middleware(
    app, service_name: str = "api", mode: str = "simple", **kwargs: Any
) -> None:
    """Add enhanced circuit breaker middleware to FastAPI app.

    Args:
        app: FastAPI application
        service_name: Name of the service
        mode: "simple" or "enterprise"
        **kwargs: Additional configuration overrides
    """
    if mode == "simple":
        config = EnhancedCircuitBreakerConfig.simple_mode(service_name)
    elif mode == "enterprise":
        config = EnhancedCircuitBreakerConfig.enterprise_mode(service_name)
    else:
        config = EnhancedCircuitBreakerConfig.from_service_config(service_name, kwargs)

    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    middleware = EnhancedCircuitBreakerMiddleware(app, config)
    app.add_middleware(EnhancedCircuitBreakerMiddleware, config=config)

    # Register for monitoring
    register_circuit_breaker(service_name, middleware.circuit_breaker)
