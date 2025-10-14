"""Unit tests for the deployment manager service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.config.loader import load_settings
from src.config.models import DeploymentConfig, DeploymentStrategy, Environment
from src.services.deployment.manager import (
    DeploymentExecutionError,
    DeploymentManager,
)


def _settings_for(deployment: DeploymentConfig) -> tuple[DeploymentManager, Path]:
    """Return a deployment manager bound to a testing settings instance."""
    settings = load_settings(environment=Environment.TESTING, deployment=deployment)
    manager = DeploymentManager(settings)
    project_root = Path(__file__).resolve().parents[4]
    return manager, project_root


def test_build_plan_defaults_to_release_workflow() -> None:
    """The default deployment plan should surface the release workflow."""
    manager, project_root = _settings_for(DeploymentConfig())

    plan = manager.build_plan()

    assert plan.strategy is DeploymentStrategy.GITHUB_ACTIONS
    assert plan.available
    assert plan.entrypoint == project_root / ".github/workflows/release.yml"
    assert any(command.startswith("git tag") for command in plan.formatted_commands())


def test_build_plan_honours_environment_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment overrides should select the requested strategy."""
    manager, _ = _settings_for(DeploymentConfig())

    monkeypatch.setenv("AI_DOCS_DEPLOYMENT_STRATEGY", "docker_compose")

    plan = manager.build_plan()

    assert plan.strategy is DeploymentStrategy.DOCKER_COMPOSE
    assert "docker compose" in plan.formatted_commands()[0]


def test_disabled_strategy_raises() -> None:
    """Disabled strategies should raise a ValueError when requested."""
    deployment = DeploymentConfig(
        default_strategy=DeploymentStrategy.GITHUB_ACTIONS,
        enabled_strategies={DeploymentStrategy.GITHUB_ACTIONS},
    )
    manager, _ = _settings_for(deployment)

    with pytest.raises(ValueError):
        manager.build_plan(DeploymentStrategy.KUBERNETES)


@pytest.mark.parametrize("strategy", list(DeploymentStrategy))
def test_validate_plan_handles_available_assets(strategy: DeploymentStrategy) -> None:
    """Validation should succeed for supported strategies with valid assets."""
    manager, _ = _settings_for(DeploymentConfig())

    plan = manager.build_plan(strategy)

    manager.validate_plan(plan)


def test_validate_plan_raises_for_missing_compose(tmp_path: Path) -> None:
    """Validation should fail when the docker compose file is missing."""
    manager, _ = _settings_for(DeploymentConfig())
    plan = manager.build_plan(DeploymentStrategy.DOCKER_COMPOSE)
    invalid_plan = replace(plan, entrypoint=tmp_path / "missing-compose.yml")

    with pytest.raises(ValueError):
        manager.validate_plan(invalid_plan)


def test_execute_plan_invokes_runner_for_each_command() -> None:
    """Executing a plan should call the runner for every command."""
    manager, _ = _settings_for(DeploymentConfig())
    plan = manager.build_plan(DeploymentStrategy.GITHUB_ACTIONS)
    executed: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> int:
        executed.append(command)
        return 0

    manager.execute_plan(plan, runner=runner)

    assert executed == list(plan.commands)


def test_execute_plan_raises_on_failed_command() -> None:
    """Non-zero exit codes should raise a DeploymentExecutionError."""
    manager, _ = _settings_for(DeploymentConfig())
    plan = manager.build_plan(DeploymentStrategy.GITHUB_ACTIONS)

    def runner(command: tuple[str, ...]) -> int:
        return 0 if command == plan.commands[0] else 2

    with pytest.raises(DeploymentExecutionError):
        manager.execute_plan(plan, runner=runner)
