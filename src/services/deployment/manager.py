"""Deployment strategy resolution, validation, and execution."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run

from opentelemetry.trace import Status, StatusCode
from yaml import YAMLError, safe_load

from src.config.loader import Settings
from src.config.models import DeploymentStrategy
from src.services.observability import get_tracer
from src.services.observability.tracing import log_extra_with_trace


def find_project_root(start: Path, marker: str = "pyproject.toml") -> Path:
    """Return the nearest ancestor directory containing the marker file."""
    current = start.resolve()
    while True:
        if (current / marker).is_file():
            return current
        if current.parent == current:
            msg = f"Could not find {marker} for path {start}"
            raise FileNotFoundError(msg)
        current = current.parent


LOGGER = logging.getLogger(__name__)
_PROJECT_ROOT = find_project_root(Path(__file__))
_ENV_OVERRIDE = "AI_DOCS_DEPLOYMENT_STRATEGY"


Command = tuple[str, ...]
CommandRunner = Callable[[Command], int]


@dataclass(frozen=True, slots=True)
class DeploymentPlan:
    """Concrete deployment plan for the selected strategy."""

    strategy: DeploymentStrategy
    entrypoint: Path
    description: str
    commands: tuple[Command, ...]
    notes: tuple[str, ...]
    available: bool

    def formatted_commands(self) -> tuple[str, ...]:
        """Return shell-friendly command strings."""
        return tuple(" ".join(parts) for parts in self.commands)


class DeploymentExecutionError(RuntimeError):
    """Raised when a deployment command fails to execute."""

    def __init__(self, command: Command, exit_code: int) -> None:
        """Store the failing command and exit code."""
        self.command = command
        self.exit_code = exit_code
        message = f"Command '{' '.join(command)}' exited with status {exit_code}"
        super().__init__(message)


class DeploymentManager:
    """Resolve deployment strategies into actionable plans."""

    def __init__(self, settings: Settings) -> None:
        """Initialise the deployment manager with application settings."""
        self._settings = settings
        self._tracer = get_tracer(__name__)

    def build_plan(
        self, override: DeploymentStrategy | str | None = None
    ) -> DeploymentPlan:
        """Return the deployment plan for the active strategy."""
        with self._tracer.start_as_current_span("deployment.build_plan") as span:
            strategy = self._resolve_strategy(override)
            span.set_attribute("deployment.strategy", strategy.value)
            entrypoint = self._resolve_entrypoint(strategy)
            span.set_attribute("deployment.entrypoint", str(entrypoint))
            commands, notes = self._commands_for(strategy, entrypoint)
            available = entrypoint.exists()
            plan = DeploymentPlan(
                strategy=strategy,
                entrypoint=entrypoint,
                description=self._description_for(strategy),
                commands=commands,
                notes=notes,
                available=available,
            )
            LOGGER.info(
                "Resolved deployment plan",
                extra=log_extra_with_trace(
                    "deployment.build_plan",
                    strategy=strategy.value,
                    entrypoint=str(entrypoint),
                    available=available,
                ),
            )
            return plan

    def validate_plan(self, plan: DeploymentPlan) -> None:
        """Validate that the plan's entrypoint is ready for execution."""
        if plan.strategy is DeploymentStrategy.GITHUB_ACTIONS:
            self._validate_github_actions(plan.entrypoint)
            return
        if plan.strategy is DeploymentStrategy.DOCKER_COMPOSE:
            self._validate_docker_compose(plan.entrypoint)
            return
        if plan.strategy is DeploymentStrategy.KUBERNETES:
            self._validate_kubernetes(plan.entrypoint)
            return
        msg = f"Unsupported deployment strategy: {plan.strategy}"
        raise ValueError(msg)

    def execute_plan(
        self,
        plan: DeploymentPlan,
        *,
        runner: CommandRunner | None = None,
    ) -> None:
        """Execute the plan by invoking each command sequentially."""
        command_runner = runner or self._default_runner
        with self._tracer.start_as_current_span("deployment.execute_plan") as span:
            span.set_attribute("deployment.strategy", plan.strategy.value)
            for command in plan.commands:
                self._validate_command(command)
                command_str = " ".join(command)
                LOGGER.info(
                    "Executing deployment command",
                    extra=log_extra_with_trace(
                        "deployment.execute_plan", command=command_str
                    ),
                )
                try:
                    exit_code = command_runner(command)
                except OSError as exc:  # pragma: no cover - surfaced in failure mode
                    span.record_exception(exc)
                    span.set_status(
                        Status(StatusCode.ERROR, str(exc)),
                    )
                    raise DeploymentExecutionError(command, exc.errno or -1) from exc
                if exit_code != 0:
                    error = DeploymentExecutionError(command, exit_code)
                    span.record_exception(error)
                    span.set_status(Status(StatusCode.ERROR, str(error)))
                    raise error

    @staticmethod
    def _default_runner(command: Command) -> int:
        """Run a deployment command using subprocess."""
        DeploymentManager._validate_command(command)
        completed: CompletedProcess[bytes] = run(  # noqa: S603
            command,
            check=False,
            cwd=_PROJECT_ROOT,
        )
        return completed.returncode

    @staticmethod
    def _validate_command(command: Command) -> None:
        """Ensure command arguments are safe for subprocess execution."""
        if not command:
            msg = "Deployment command must contain at least one argument"
            raise ValueError(msg)
        for index, part in enumerate(command):
            if not isinstance(part, str):
                msg = (
                    "Deployment command arguments must be strings; "
                    f"got {type(part).__name__} at position {index}"
                )
                raise TypeError(msg)
            if part != part.strip():
                msg = "Deployment command arguments must not include surrounding spaces"
                raise ValueError(msg)
            if not part:
                msg = "Deployment command arguments must not be empty"
                raise ValueError(msg)
            if any(char in part for char in ("\n", "\r", "\x00")):
                msg = "Deployment command arguments must not contain control characters"
                raise ValueError(msg)

    def _resolve_strategy(
        self, override: DeploymentStrategy | str | None
    ) -> DeploymentStrategy:
        """Resolve the deployment strategy accounting for overrides."""
        candidate: DeploymentStrategy | None
        if override is None:
            env_override = os.getenv(_ENV_OVERRIDE)
            candidate = self._parse_strategy(env_override) if env_override else None
        else:
            candidate = self._parse_strategy(override)

        strategy = candidate or self._settings.deployment.default_strategy
        if not self._settings.deployment.is_enabled(strategy):
            msg = f"Deployment strategy '{strategy.value}' is disabled"
            raise ValueError(msg)
        return strategy

    @staticmethod
    def _parse_strategy(
        value: DeploymentStrategy | str | None,
    ) -> DeploymentStrategy | None:
        """Convert a raw value into a DeploymentStrategy."""
        if value is None:
            return None
        if isinstance(value, DeploymentStrategy):
            return value
        normalized = value.strip().lower()
        for strategy in DeploymentStrategy:
            if strategy.value == normalized:
                return strategy
        msg = f"Unknown deployment strategy: {value}"
        raise ValueError(msg)

    def _resolve_entrypoint(self, strategy: DeploymentStrategy) -> Path:
        """Return the filesystem entrypoint for the given strategy."""
        config_entry = self._settings.deployment.resolve_entrypoint(strategy)
        return (
            config_entry if config_entry.is_absolute() else _PROJECT_ROOT / config_entry
        )

    def _description_for(self, strategy: DeploymentStrategy) -> str:
        """Return a human-readable description for the strategy."""
        if strategy is DeploymentStrategy.GITHUB_ACTIONS:
            return "Deploy via GitHub Actions release workflow and tagged versions."
        if strategy is DeploymentStrategy.DOCKER_COMPOSE:
            return "Deploy locally using Docker Compose managed services."
        if strategy is DeploymentStrategy.KUBERNETES:
            return "Deploy to a Kubernetes cluster using the provided manifests."
        msg = f"Unsupported deployment strategy: {strategy}"
        raise ValueError(msg)

    def _commands_for(
        self, strategy: DeploymentStrategy, entrypoint: Path
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
        """Return commands and operator notes for the strategy."""
        if strategy is DeploymentStrategy.GITHUB_ACTIONS:
            commands: tuple[tuple[str, ...], ...] = (
                ("git", "tag", "vX.Y.Z"),
                ("git", "push", "origin", "vX.Y.Z"),
                ("gh", "workflow", "run", entrypoint.name),
            )
            notes = (
                "Tagging a release triggers the workflow that publishes artifacts.",
                f"Workflow file located at {entrypoint}.",
                "Ensure registry and signing secrets exist before triggering runs.",
            )
            return commands, notes

        if strategy is DeploymentStrategy.DOCKER_COMPOSE:
            commands = (
                (
                    "docker",
                    "compose",
                    "-f",
                    str(entrypoint),
                    "--profile",
                    "monitoring",
                    "up",
                    "-d",
                ),
                ("python", "scripts/dev.py", "services", "status"),
            )
            notes = (
                "Install Docker Engine and Compose on the deployment host.",
                "Set AI_DOCS__ENVIRONMENT variables for staging or production targets.",
                f"Compose manifest located at {entrypoint}.",
            )
            return commands, notes

        if strategy is DeploymentStrategy.KUBERNETES:
            commands = (
                ("kubectl", "apply", "-f", str(entrypoint)),
                (
                    "kubectl",
                    "rollout",
                    "status",
                    "deployment/ai-docs-vector-db",
                    "--timeout",
                    "2m",
                ),
            )
            notes = (
                "Configure KUBECONFIG or kubectl context before applying manifests.",
                f"Manifests sourced from {entrypoint}.",
                "Adjust HorizontalPodAutoscaler settings before production rollout.",
            )
            return commands, notes

        msg = f"Unsupported deployment strategy: {strategy}"
        raise ValueError(msg)

    @staticmethod
    def _validate_github_actions(entrypoint: Path) -> None:
        """Ensure the GitHub Actions workflow exists and is well-formed."""
        if not entrypoint.is_file():
            msg = f"GitHub Actions workflow not found at {entrypoint}"
            raise ValueError(msg)
        try:
            workflow = safe_load(entrypoint.read_text(encoding="utf-8"))
        except YAMLError as exc:
            msg = f"Invalid GitHub Actions workflow at {entrypoint}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(workflow, dict):
            msg = (
                "Workflow must deserialize into a mapping, "
                f"got {type(workflow).__name__}"
            )
            raise TypeError(msg)
        jobs = workflow.get("jobs")
        if not isinstance(jobs, dict):
            msg = "Workflow requires a jobs section"
            raise TypeError(msg)
        required_jobs = {"prepare", "quality", "build-artifacts"}
        missing = required_jobs.difference(jobs)
        if missing:
            missing_jobs = ", ".join(sorted(missing))
            msg = f"Workflow missing required jobs: {missing_jobs}"
            raise ValueError(msg)

    @staticmethod
    def _validate_docker_compose(entrypoint: Path) -> None:
        """Ensure the docker compose manifest is structurally sound."""
        if not entrypoint.is_file():
            msg = f"docker compose file not found at {entrypoint}"
            raise ValueError(msg)
        try:
            manifest = safe_load(entrypoint.read_text(encoding="utf-8"))
        except YAMLError as exc:
            msg = f"Invalid docker compose manifest at {entrypoint}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(manifest, dict):
            msg = "docker compose manifest must be a mapping"
            raise TypeError(msg)
        services = manifest.get("services")
        if not isinstance(services, dict):
            msg = "docker compose manifest must define services"
            raise TypeError(msg)
        required_services = {"qdrant", "app"}
        missing = required_services.difference(services)
        if missing:
            missing_services = ", ".join(sorted(missing))
            msg = f"docker compose missing required services: {missing_services}"
            raise ValueError(msg)

    @staticmethod
    def _validate_kubernetes(entrypoint: Path) -> None:
        """Ensure Kubernetes manifests exist and reference valid resources."""
        if not entrypoint.is_dir():
            msg = f"Kubernetes manifest directory not found at {entrypoint}"
            raise ValueError(msg)
        kustomization = entrypoint / "kustomization.yaml"
        if not kustomization.is_file():
            msg = f"Kustomization file missing at {kustomization}"
            raise ValueError(msg)
        try:
            config = safe_load(kustomization.read_text(encoding="utf-8"))
        except YAMLError as exc:
            msg = f"Invalid kustomization manifest at {kustomization}: {exc}"
            raise ValueError(msg) from exc
        if not isinstance(config, dict):
            msg = "kustomization manifest must be a mapping"
            raise TypeError(msg)
        resources = config.get("resources")
        if not isinstance(resources, list) or not resources:
            msg = "kustomization manifest must list at least one resource"
            raise ValueError(msg)
        missing_resources = [
            resource for resource in resources if not (entrypoint / resource).exists()
        ]
        if missing_resources:
            missing = ", ".join(sorted(missing_resources))
            msg = f"kustomization references missing resources: {missing}"
            raise ValueError(msg)
