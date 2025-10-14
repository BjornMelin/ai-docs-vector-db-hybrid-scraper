"""Tests for the `dev deploy` CLI command."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import scripts.dev as dev
from src.config.models import DeploymentStrategy
from src.services.deployment.manager import DeploymentPlan


@pytest.fixture(autouse=True)
def reload_dev_module() -> None:
    """Reload the dev module to avoid cached parser state between tests."""
    importlib.reload(dev)


def test_deploy_command_outputs_plan(capsys: pytest.CaptureFixture[str]) -> None:
    """The deploy subcommand should print the default deployment plan."""
    exit_code = dev.main(["deploy"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Strategy: github_actions" in output
    assert "release.yml" in output


def test_deploy_command_accepts_override(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy subcommand should honour --strategy overrides."""
    monkeypatch.delenv("AI_DOCS_DEPLOYMENT_STRATEGY", raising=False)

    exit_code = dev.main(["deploy", "--strategy", "docker_compose"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Strategy: docker_compose" in output
    assert "docker compose" in output


def test_deploy_command_apply_executes_plan(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deploy subcommand should validate and execute when --apply is set."""
    sentinel_settings = object()
    plan = DeploymentPlan(
        strategy=DeploymentStrategy.GITHUB_ACTIONS,
        entrypoint=Path(".github/workflows/release.yml"),
        description="test plan",
        commands=(("echo", "ok"),),
        notes=(),
        available=True,
    )
    calls: list[str] = []

    class DummyManager:
        def __init__(self, settings: object) -> None:
            assert settings is sentinel_settings

        def build_plan(self, strategy: str | None) -> DeploymentPlan:
            assert strategy is None
            return plan

        def validate_plan(self, received_plan: DeploymentPlan) -> None:
            assert received_plan is plan
            calls.append("validate")

        def execute_plan(self, received_plan: DeploymentPlan) -> None:
            assert received_plan is plan
            calls.append("execute")

    monkeypatch.setattr("src.config.loader.get_settings", lambda: sentinel_settings)
    monkeypatch.setattr("src.services.deployment.DeploymentManager", DummyManager)

    exit_code = dev.main(["deploy", "--apply"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Deployment commands executed successfully." in output
    assert calls == ["validate", "execute"]
