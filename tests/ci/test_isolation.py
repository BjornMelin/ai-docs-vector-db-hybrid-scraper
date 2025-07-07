"""Test isolation utilities for parallel test execution.

This module provides utilities to ensure proper test isolation
when running tests in parallel with pytest-xdist.
"""

import os
import tempfile
import uuid
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import pytest


class IsolatedTestResources:
    """Manages isolated resources for parallel test execution."""

    def __init__(self, worker_id: str = None):
        self.worker_id = worker_id or os.getenv("PYTEST_XDIST_WORKER", "master")
        self.is_parallel = self.worker_id != "master"
        self._temp_dirs = []
        self._allocated_ports = set()
        self._resource_locks = {}

    def get_isolated_temp_dir(self, prefix: str = "test") -> Path:
        """Get an isolated temporary directory for this test worker."""
        if self.is_parallel:
            # Worker-specific temp directory
            base_dir = Path(tempfile.gettempdir()) / f"pytest_worker_{self.worker_id}"
            base_dir.mkdir(exist_ok=True)
            temp_dir = base_dir / f"{prefix}_{uuid.uuid4().hex[:8]}"
        else:
            # Regular temp directory for non-parallel execution
            temp_dir = Path(tempfile.mkdtemp(prefix=f"{prefix}_"))

        temp_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def get_isolated_port(self, base_port: int = 8000) -> int:
        """Get an isolated port number for this test worker."""
        if not self.is_parallel:
            # For non-parallel execution, use dynamic port allocation
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
                self._allocated_ports.add(port)
                return port

        # For parallel execution, use worker-based port ranges
        worker_num = 0
        if "gw" in self.worker_id:
            try:
                worker_num = int(self.worker_id.replace("gw", ""))
            except ValueError:
                worker_num = hash(self.worker_id) % 100

        # Each worker gets a range of 100 ports
        start_port = base_port + (worker_num * 100)

        for port in range(start_port, start_port + 100):
            if port not in self._allocated_ports:
                # Check if port is actually available
                import socket
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.bind(("127.0.0.1", port))
                        self._allocated_ports.add(port)
                        return port
                except OSError:
                    continue

        raise RuntimeError(f"No free ports available for worker {self.worker_id}")

    def get_isolated_database_name(self, base_name: str = "test_db") -> str:
        """Get an isolated database name for this test worker."""
        if self.is_parallel:
            return f"{base_name}_{self.worker_id}_{uuid.uuid4().hex[:8]}"
        return f"{base_name}_{uuid.uuid4().hex[:8]}"

    def get_isolated_collection_name(self, base_name: str = "test_collection") -> str:
        """Get an isolated vector database collection name."""
        if self.is_parallel:
            return f"{base_name}_{self.worker_id}_{uuid.uuid4().hex[:8]}"
        return f"{base_name}_{uuid.uuid4().hex[:8]}"

    def cleanup(self):
        """Clean up allocated resources."""
        import shutil

        for temp_dir in self._temp_dirs:
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass

        self._temp_dirs.clear()
        self._allocated_ports.clear()
        self._resource_locks.clear()


@pytest.fixture
def isolated_resources(request):
    """Pytest fixture providing isolated test resources."""
    resources = IsolatedTestResources()
    yield resources
    resources.cleanup()


@pytest.fixture
def isolated_temp_dir(isolated_resources):
    """Get an isolated temporary directory."""
    return isolated_resources.get_isolated_temp_dir()


@pytest.fixture
def isolated_port(isolated_resources):
    """Get an isolated port number."""
    return isolated_resources.get_isolated_port()


@pytest.fixture
def isolated_db_name(isolated_resources):
    """Get an isolated database name."""
    return isolated_resources.get_isolated_database_name()


@pytest.fixture
def isolated_collection_name(isolated_resources):
    """Get an isolated vector DB collection name."""
    return isolated_resources.get_isolated_collection_name()


@contextmanager
def isolated_environment_variables(**kwargs) -> Generator[dict[str, str]]:
    """Context manager for isolated environment variables."""
    original_env = {}
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

    # Store original values and set new ones
    for key, value in kwargs.items():
        original_env[key] = os.environ.get(key)
        # Add worker ID to environment variable values for isolation
        if worker_id != "master" and isinstance(value, str):
            value = f"{value}_{worker_id}"
        os.environ[key] = str(value)

    try:
        yield dict(os.environ)
    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class IsolatedAsyncioPolicy:
    """Ensures isolated asyncio event loops for parallel tests."""

    def __init__(self):
        self.worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
        self._original_policy = None

    def __enter__(self):
        import asyncio

        # Store original policy
        self._original_policy = asyncio.get_event_loop_policy()

        # Create new isolated policy
        if os.name == "nt":  # Windows
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        else:
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import asyncio

        # Close any running loops
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            if not loop.is_closed():
                loop.close()
        except RuntimeError:
            pass

        # Restore original policy
        if self._original_policy:
            asyncio.set_event_loop_policy(self._original_policy)


@pytest.fixture
def isolated_asyncio():
    """Fixture providing isolated asyncio environment."""
    with IsolatedAsyncioPolicy():
        yield


class TestDataIsolation:
    """Utilities for test data isolation in parallel execution."""

    @staticmethod
    def get_unique_id(prefix: str = "test") -> str:
        """Generate a unique ID for test data."""
        worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
        timestamp = int(time.time() * 1000000)  # Microsecond precision
        unique_part = uuid.uuid4().hex[:8]

        if worker_id != "master":
            return f"{prefix}_{worker_id}_{timestamp}_{unique_part}"
        return f"{prefix}_{timestamp}_{unique_part}"

    @staticmethod
    def get_isolated_test_data_path(filename: str) -> Path:
        """Get an isolated path for test data files."""
        worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

        if worker_id != "master":
            # Worker-specific data directory
            data_dir = Path("tests/fixtures/data") / f"worker_{worker_id}"
        else:
            data_dir = Path("tests/fixtures/data") / "master"

        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir / filename


# Import time for timestamp generation
import time


# Pytest marker for tests that require isolation
pytest.mark.isolated = pytest.mark.isolated


def requires_isolation(reason: str = "Test requires resource isolation"):
    """Decorator to mark tests that require special isolation."""
    def decorator(func):
        return pytest.mark.isolated(func, reason=reason)
    return decorator


# Hook for pytest-xdist to ensure proper isolation
def pytest_configure_node(node):
    """Configure worker node for proper isolation."""
    worker_id = getattr(node, "workerinput", {}).get("workerid", "master")

    # Set worker-specific environment
    os.environ["PYTEST_XDIST_WORKER"] = worker_id

    # Create worker-specific directories
    worker_dirs = [
        f"tests/fixtures/cache/worker_{worker_id}",
        f"tests/fixtures/data/worker_{worker_id}",
        f"tests/fixtures/logs/worker_{worker_id}",
        f"logs/worker_{worker_id}",
        f"cache/worker_{worker_id}",
    ]

    for dir_path in worker_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


# Utility function for test suites
def ensure_test_isolation():
    """Ensure proper test isolation is set up."""
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

    # Check if we're in parallel execution
    if worker_id != "master":
        # Verify worker-specific directories exist
        required_dirs = [
            f"tests/fixtures/cache/worker_{worker_id}",
            f"tests/fixtures/data/worker_{worker_id}",
            f"tests/fixtures/logs/worker_{worker_id}",
        ]

        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)

        return True

    return False
