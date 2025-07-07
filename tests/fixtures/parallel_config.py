"""Configuration for parallel test execution with pytest-xdist.

This module provides configuration and utilities for optimizing
parallel test execution with proper resource isolation.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


def pytest_configure_node(node):
    """Configure pytest-xdist worker nodes."""
    # Set worker-specific temporary directories
    worker_id = getattr(node, "workerinput", {}).get("workerid", "master")

    if worker_id != "master":
        # Create worker-specific temp directory
        temp_base = Path(tempfile.gettempdir()) / f"pytest_worker_{worker_id}"
        temp_base.mkdir(exist_ok=True)

        # Set environment variables for worker isolation
        os.environ[f"PYTEST_WORKER_TEMP_{worker_id}"] = str(temp_base)
        os.environ["PYTEST_CURRENT_WORKER"] = worker_id


@pytest.fixture(scope="session")
def worker_config():
    """Configuration for the current pytest worker."""
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")

    config = {
        "worker_id": worker_id,
        "is_master": worker_id == "master",
        "temp_dir": None,
        "ports": {},
        "isolation_level": "process" if worker_id != "master" else "thread",
    }

    # Set worker-specific temporary directory
    if worker_id != "master":
        temp_dir = Path(tempfile.gettempdir()) / f"pytest_worker_{worker_id}"
        temp_dir.mkdir(exist_ok=True)
        config["temp_dir"] = temp_dir

        # Assign worker-specific ports to avoid conflicts
        base_port = 6333  # Qdrant default
        worker_num = int(worker_id.replace("gw", "")) if "gw" in worker_id else 0

        config["ports"] = {
            "qdrant": base_port + (worker_num * 10),
            "redis": 6379 + (worker_num * 10),
            "http_mock": 8000 + (worker_num * 10),
        }

    return config


@pytest.fixture
def isolated_worker_env(worker_config):
    """Environment variables isolated per worker."""
    if not worker_config["is_master"]:
        # Set worker-specific environment
        env_vars = {
            "QDRANT_PORT": str(worker_config["ports"]["qdrant"]),
            "REDIS_PORT": str(worker_config["ports"]["redis"]),
            "TEST_HTTP_PORT": str(worker_config["ports"]["http_mock"]),
            "TMPDIR": str(worker_config["temp_dir"]),
            "PYTEST_WORKER_ID": worker_config["worker_id"],
        }

        # Store original values
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            yield env_vars
        finally:
            # Restore original environment
            for key, value in original_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    else:
        yield {}


class ParallelResourceManager:
    """Manages resources for parallel test execution."""

    def __init__(self, worker_config: dict[str, Any]):
        self.worker_config = worker_config
        self._allocated_ports = set()

    def get_free_port(self, base_port: int = 0) -> int:
        """Get a free port for the current worker."""
        import socket

        if self.worker_config["is_master"]:
            # For master process, use any available port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                s.listen(1)
                port = s.getsockname()[1]
                self._allocated_ports.add(port)
                return port
        else:
            # For workers, use worker-specific port range
            worker_num = (
                int(self.worker_config["worker_id"].replace("gw", ""))
                if "gw" in self.worker_config["worker_id"]
                else 0
            )
            start_port = (base_port or 8000) + (worker_num * 100)

            for port in range(start_port, start_port + 100):
                if port not in self._allocated_ports:
                    try:
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.bind(("127.0.0.1", port))
                            self._allocated_ports.add(port)
                            return port
                    except OSError:
                        continue

            raise RuntimeError(
                f"No free ports available for worker {self.worker_config['worker_id']}"
            )

    def create_worker_temp_dir(self, name: str) -> Path:
        """Create a temporary directory specific to this worker."""
        if self.worker_config["temp_dir"]:
            temp_dir = self.worker_config["temp_dir"] / name
        else:
            temp_dir = Path(tempfile.gettempdir()) / f"pytest_master_{name}"

        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir


@pytest.fixture
def parallel_resource_manager(worker_config):
    """Resource manager for parallel test execution."""
    return ParallelResourceManager(worker_config)


@pytest.fixture(scope="session")
def shared_test_state():
    """Shared state across test workers using file-based coordination."""
    state_file = Path(tempfile.gettempdir()) / "pytest_shared_state.json"

    class SharedState:
        def __init__(self, state_file: Path):
            self.state_file = state_file

        def get(self, key: str, default=None):
            """Get a value from shared state."""
            import fcntl
            import json

            if not self.state_file.exists():
                return default

            try:
                with open(self.state_file) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    data = json.load(f)
                    return data.get(key, default)
            except (json.JSONDecodeError, OSError):
                return default

        def set(self, key: str, value):
            """Set a value in shared state."""
            import fcntl
            import json

            # Read existing data
            data = {}
            if self.state_file.exists():
                try:
                    with open(self.state_file) as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                        data = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass

            # Update and write
            data[key] = value

            with open(self.state_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                json.dump(data, f)

    shared_state = SharedState(state_file)

    yield shared_state

    # Cleanup on session end (only master worker)
    worker_id = os.getenv("PYTEST_XDIST_WORKER", "master")
    if worker_id == "master" and state_file.exists():
        try:
            state_file.unlink()
        except OSError:
            pass
