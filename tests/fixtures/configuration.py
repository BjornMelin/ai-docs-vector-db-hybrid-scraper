"""Configuration-centric pytest fixtures for the unit test suite."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import pytest
from dotenv import load_dotenv


_TEST_ENV_PATH = Path(__file__).resolve().parents[1] / ".env.test"
if _TEST_ENV_PATH.exists():  # pragma: no cover - depends on developer machine
    load_dotenv(_TEST_ENV_PATH, override=True)


if TYPE_CHECKING:
    from src.config import Config


def _is_ci_environment() -> bool:
    """Return True when running inside a known CI provider."""
    ci_variables = ("CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE")
    return any(os.getenv(var) for var in ci_variables)


def _parallel_worker_count() -> int:
    """Derive a conservative worker count for local and CI execution."""
    cpu_total = os.cpu_count() or 1
    if _is_ci_environment():
        return max(1, min(4, cpu_total // 2))
    return max(1, min(2, cpu_total))


@pytest.fixture(scope="session")
def app_config() -> dict[str, Any]:
    """Provide a minimal, deterministic application configuration for tests."""
    return {
        "test_mode": True,
        "parallel_workers": _parallel_worker_count(),
        "timeouts": {"default": 30, "browser": 60, "network": 15, "database": 10},
        "test_dirs": [
            "tests/fixtures/cache",
            "tests/fixtures/data",
            "tests/fixtures/logs",
            "logs",
            "cache",
            "data",
        ],
    }


@pytest.fixture(scope="session")
def ci_environment_config(app_config: dict[str, Any]) -> dict[str, Any]:
    """Expose CI detection metadata for infrastructure smoke tests."""
    return {
        "is_ci": _is_ci_environment(),
        "parallel_workers": app_config["parallel_workers"],
        "worker_id": os.getenv("PYTEST_XDIST_WORKER", "master"),
        "platform": {
            "os": sys.platform,
            "is_windows": sys.platform.startswith("win"),
            "is_macos": sys.platform == "darwin",
            "is_linux": sys.platform.startswith("linux"),
        },
    }


class _Checkpoint(TypedDict):
    """Dictionary layout for recorded checkpoints."""

    name: str
    elapsed_seconds: float


class _PerformanceMonitor:
    """Collect coarse execution timings for infrastructure assertions."""

    def __init__(self) -> None:
        self._start: float | None = None
        self._checkpoints: list[_Checkpoint] = []
        self._last_metrics: dict[str, Any] | None = None

    def start(self) -> None:
        """Begin timing."""
        self._checkpoints.clear()
        self._start = time.perf_counter()
        self._last_metrics = None

    def checkpoint(self, name: str) -> _Checkpoint:
        """Record a named checkpoint and return the measurement."""
        if self._start is None:
            msg = "Performance monitor has not been started."
            raise RuntimeError(msg)
        now = time.perf_counter()
        elapsed = now - self._start
        checkpoint: _Checkpoint = {"name": name, "elapsed_seconds": elapsed}
        self._checkpoints.append(checkpoint)
        return checkpoint

    def stop(self) -> dict[str, Any]:
        """Stop timing and return aggregated metrics."""
        if self._start is None:
            msg = "Performance monitor has not been started."
            raise RuntimeError(msg)
        duration = time.perf_counter() - self._start
        metrics: dict[str, Any] = {
            "duration_seconds": duration,
            "checkpoints": list(self._checkpoints),
        }
        self._start = None
        self._last_metrics = metrics
        return metrics

    def assert_under(self, max_seconds: float) -> None:
        """Assert the recorded duration stays below the provided bound."""
        if not self._checkpoints:
            msg = "No checkpoints recorded before assert_under call."
            raise RuntimeError(msg)
        last_duration = self._checkpoints[-1]["elapsed_seconds"]
        if last_duration > max_seconds:
            msg = (
                "Performance budget exceeded: "
                f"{last_duration:.4f}s > {max_seconds:.4f}s"
            )
            raise AssertionError(msg)

    def assert_performance(
        self, max_duration: float | None = None, max_memory_mb: float | None = None
    ) -> None:
        """Assert recorded metrics against optional duration and memory limits."""
        if self._last_metrics is None:
            msg = "Performance monitor has no recorded metrics."
            raise RuntimeError(msg)

        if max_duration is not None:
            duration = float(self._last_metrics["duration_seconds"])
            if duration > max_duration:
                message = (
                    f"Duration {duration:.4f}s exceeded max duration "
                    f"{max_duration:.4f}s"
                )
                raise AssertionError(message)

        if max_memory_mb is not None:
            # Memory tracking is not recorded; this argument is accepted for API parity.
            raise AssertionError("Memory tracking is not supported in this monitor")


@pytest.fixture
def performance_monitor() -> Generator[_PerformanceMonitor, None, None]:
    """Yield a fresh performance monitor for each test."""
    monitor = _PerformanceMonitor()
    yield monitor


@pytest.fixture
def config_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[..., Config]:
    """Build typed Config instances with deterministic directories."""

    def _create_config(**overrides: Any):
        from src.config import Config, Environment  # Local import to avoid cycles

        base_dir = tmp_path_factory.mktemp("config_factory")
        payload: dict[str, Any] = {
            "environment": Environment.TESTING,
            "data_dir": base_dir / "data",
            "cache_dir": base_dir / "cache",
            "logs_dir": base_dir / "logs",
        }
        payload.update(overrides)
        return Config.model_validate(payload)

    return _create_config


__all__ = [
    "app_config",
    "ci_environment_config",
    "performance_monitor",
    "config_factory",
]
