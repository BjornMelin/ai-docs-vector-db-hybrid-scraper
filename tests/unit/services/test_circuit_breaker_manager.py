"""Circuit-breaker manager configuration tests."""

from src.config import Settings
from src.config.models import CircuitBreakerConfig, Environment
from src.services.circuit_breaker import CircuitBreakerManager


def test_manager_uses_canonical_circuit_breaker_settings() -> None:
    """Factory defaults should come from Settings.circuit_breaker."""
    settings = Settings(
        environment=Environment.TESTING,
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=7.5,
        ),
    )

    manager = CircuitBreakerManager.in_memory(settings)

    assert manager.factory.default_threshold == 2
    assert manager.factory.default_ttl == 7.5
