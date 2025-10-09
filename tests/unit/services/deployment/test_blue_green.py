"""Tests for the blue-green deployment manager."""

from __future__ import annotations

from typing import cast

import pytest

from src.services.deployment.blue_green import (
    BlueGreenConfig,
    BlueGreenDeployment,
    BlueGreenEnvironment,
)
from src.services.health.manager import HealthCheckResult, HealthStatus


@pytest.fixture()
def deployment(mocker: pytest.MockFixture) -> BlueGreenDeployment:
    """Provide a deployment manager with stubbed dependencies."""

    manager = BlueGreenDeployment(qdrant_service=object(), cache_manager=object())
    manager._initialized = True  # noqa: SLF001 - test-only priming
    mocker.patch.object(manager, "_persist_environment_state", mocker.AsyncMock())
    return manager


@pytest.mark.asyncio()
async def test_switch_environments_requires_healthy_target(
    deployment: BlueGreenDeployment,
) -> None:
    """Switching should fail when the target environment is unhealthy."""

    deployment._blue_env.active = True  # noqa: SLF001
    deployment._green_env.health = None  # noqa: SLF001

    success = await deployment.switch_environments()

    assert success is False


@pytest.mark.asyncio()
async def test_switch_environments_succeeds_when_healthy(
    deployment: BlueGreenDeployment,
    mocker: pytest.MockFixture,
) -> None:
    """Switching should succeed when the target environment is healthy."""

    deployment._blue_env.active = True  # noqa: SLF001
    deployment._green_env.health = HealthCheckResult(  # noqa: SLF001
        name="green_deployment",
        status=HealthStatus.HEALTHY,
        message="ok",
        duration_ms=0.0,
    )
    mocker.patch.object(
        deployment,
        "_execute_environment_switch",
        mocker.AsyncMock(return_value=None),
    )

    success = await deployment.switch_environments()

    assert success is True


@pytest.mark.asyncio()
async def test_perform_health_checks_sets_unhealthy_state(
    deployment: BlueGreenDeployment,
    mocker: pytest.MockFixture,
) -> None:
    """Health checks should mark the environment unhealthy after repeated failures."""

    env = BlueGreenEnvironment(name="blue", active=False)
    config = BlueGreenConfig(deployment_id="dep", target_version="1.0.0")

    mocker.patch.object(
        deployment,
        "_perform_single_health_check",
        mocker.AsyncMock(side_effect=[False, False, False]),
    )

    healthy = await deployment._perform_health_checks(env, config)  # noqa: SLF001

    assert healthy is False
    assert env.health is not None
    health = cast(HealthCheckResult, env.health)
    assert getattr(health, "status", None) is HealthStatus.UNHEALTHY


@pytest.mark.asyncio()
async def test_get_deployment_metrics_uses_health_metadata(
    deployment: BlueGreenDeployment,
) -> None:
    """Deployment metrics should include health-derived information."""

    deployment._blue_env.health = HealthCheckResult(  # noqa: SLF001
        name="blue",
        status=HealthStatus.HEALTHY,
        message="ok",
        duration_ms=0.0,
        metadata={"response_time_ms": 42.0, "error_rate": 1.5},
    )
    deployment._blue_env.deployment_id = "dep-blue"  # noqa: SLF001

    metrics = await deployment.get_deployment_metrics()

    assert metrics["blue"].avg_response_time_ms == 42.0
    assert metrics["blue"].error_rate == 1.5
