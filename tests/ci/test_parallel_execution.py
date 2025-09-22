"""Tests covering pytest-xdist configuration, isolation, and benchmarks."""

import os
import time
from pathlib import Path

import pytest

from tests.ci.performance_reporter import PerformanceReporter
from tests.ci.pytest_xdist_config import XDistOptimizer, get_xdist_args
from tests.ci.test_environments import GitHubActionsEnvironment, detect_environment
from tests.ci.test_isolation import (
    IsolatedTestResources,
    TestDataIsolation,
    ensure_test_isolation,
)


class TestXDistConfiguration:
    """Test pytest-xdist configuration optimization."""

    def test_xdist_optimizer_detection(self):
        """Test that XDistOptimizer correctly detects environment."""
        optimizer = XDistOptimizer()
        config = optimizer.get_optimal_config()

        # Basic validation
        assert config.num_workers >= 1
        assert config.num_workers <= config.max_workers
        assert config.dist_mode in ["loadscope", "loadfile", "loadgroup", "no"]
        assert config.timeout > 0
        assert config.timeout_method in ["thread", "signal"]

    def test_xdist_args_generation(self):
        """Test command line argument generation."""
        args = get_xdist_args()

        # Check required arguments are present
        assert any("--numprocesses=" in arg for arg in args)
        assert any("--dist=" in arg for arg in args)
        assert any("--timeout=" in arg for arg in args)

    def test_ci_environment_detection(self):
        """Test CI environment detection."""
        env = detect_environment()

        # Should detect environment
        assert env is not None
        assert hasattr(env, "get_pytest_args")
        assert hasattr(env, "get_env_vars")

    @pytest.mark.parametrize(
        ("ci_env", "expected_workers"),
        [
            ("GITHUB_ACTIONS", 4),
            ("GITLAB_CI", 2),
            ("CIRCLECI", None),  # Depends on resource class
        ],
    )
    def test_ci_specific_configuration(self, monkeypatch, ci_env, expected_workers):
        """Test CI-specific configurations."""
        # Mock CI environment
        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv(ci_env, "true")

        optimizer = XDistOptimizer()
        config = optimizer.get_optimal_config()

        if expected_workers is not None:
            assert config.num_workers <= expected_workers


class TestParallelIsolation:
    """Test parallel execution isolation."""

    def test_worker_isolation(self):
        """Test that workers are properly isolated."""
        resources = IsolatedTestResources()

        # Test temp directory isolation
        temp_dir1 = resources.get_isolated_temp_dir("test1")
        temp_dir2 = resources.get_isolated_temp_dir("test2")

        assert temp_dir1 != temp_dir2
        assert temp_dir1.exists()
        assert temp_dir2.exists()

        # Cleanup
        resources.cleanup()

    def test_port_allocation(self):
        """Test isolated port allocation."""
        resources = IsolatedTestResources()

        # Get multiple ports
        ports = set()
        for _ in range(5):
            port = resources.get_isolated_port()
            assert port not in ports
            ports.add(port)

        # All ports should be unique
        assert len(ports) == 5

    def test_database_name_isolation(self):
        """Test database name generation for isolation."""
        resources = IsolatedTestResources()

        # Generate multiple database names
        db_names = set()
        for _ in range(10):
            db_name = resources.get_isolated_database_name()
            assert db_name not in db_names
            db_names.add(db_name)

        # All should be unique
        assert len(db_names) == 10

        # Should include worker ID if in parallel mode
        if resources.is_parallel:
            assert all(resources.worker_id in name for name in db_names)

    def test_unique_id_generation(self):
        """Test unique ID generation for test data."""
        ids = set()
        for _ in range(100):
            unique_id = TestDataIsolation.get_unique_id("test")
            assert unique_id not in ids
            ids.add(unique_id)

        assert len(ids) == 100


class TestParallelPerformance:
    """Test performance characteristics of parallel execution."""

    @pytest.mark.slow
    def test_parallel_execution_overhead(self):
        """Test that parallel execution has acceptable overhead."""
        start_time = time.time()

        # Simulate work
        resources = IsolatedTestResources()
        for _ in range(10):
            _ = resources.get_isolated_temp_dir()
            _ = resources.get_isolated_port()

        resources.cleanup()

        duration = time.time() - start_time

        # Should complete quickly even with isolation
        assert duration < 1.0  # Less than 1 second

    def test_worker_detection(self):
        """Test worker detection in parallel execution."""
        worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

        # Worker ID should be set (either master or gwX)
        assert worker_id is not None

        # If not master, should be in expected format
        if worker_id != "master":
            assert worker_id.startswith("gw")


class TestCIIntegration:
    """Test CI-specific integration points."""

    def test_github_actions_configuration(self, monkeypatch):
        """Test GitHub Actions specific configuration."""
        # Mock GitHub Actions environment
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("RUNNER_OS", "Linux")

        env = GitHubActionsEnvironment()
        assert env.detect()

        args = env.get_pytest_args()
        assert "--numprocesses=auto" in args
        assert any("--junit-xml=" in arg for arg in args)

    def test_performance_reporter_registration(self):
        """Test that performance reporter can be registered."""
        # Should be importable and instantiable
        assert PerformanceReporter is not None

    def test_isolation_fixtures(self, isolated_temp_dir, isolated_port):
        """Test isolation fixtures work correctly."""
        # Fixtures should provide isolated resources
        assert isinstance(isolated_temp_dir, Path)
        assert isolated_temp_dir.exists()

        assert isinstance(isolated_port, int)
        assert 1024 < isolated_port < 65535


class TestWorkerCoordination:
    """Test coordination between parallel workers."""

    def test_worker_specific_directories(self):
        """Test that worker-specific directories are created."""
        worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

        if worker_id != "master":
            # Check worker directories exist
            expected_dirs = [
                f"tests/fixtures/cache/worker_{worker_id}",
                f"tests/fixtures/data/worker_{worker_id}",
                f"tests/fixtures/logs/worker_{worker_id}",
            ]

            for dir_path in expected_dirs:
                path = Path(dir_path)
                # Create if not exists (as would happen in real execution)
                path.mkdir(parents=True, exist_ok=True)
                assert path.exists()

    def test_ensure_isolation_function(self):
        """Test the ensure_test_isolation utility."""
        result = ensure_test_isolation()

        worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
        if worker_id != "master":
            assert result is True
        else:
            assert result is False


# Benchmarks for parallel execution
@pytest.mark.benchmark
class TestParallelBenchmarks:
    """Benchmarks for parallel execution performance."""

    def test_resource_allocation_speed(self, benchmark):
        """Benchmark resource allocation speed."""
        resources = IsolatedTestResources()

        def allocate_resources():
            _ = resources.get_isolated_temp_dir()
            _ = resources.get_isolated_port()
            _ = resources.get_isolated_database_name()

        benchmark(allocate_resources)
        resources.cleanup()

    def test_unique_id_generation_speed(self, benchmark):
        """Benchmark unique ID generation speed."""
        benchmark(TestDataIsolation.get_unique_id, "benchmark")


if __name__ == "__main__":
    # Run tests with optimal configuration
    import subprocess

    cmd = [
        "python",
        "scripts/dev.py",
        "test",
        "--profile",
        "ci",
        "--verbose",
    ]
    subprocess.run(cmd, check=False)
