"""Test configuration and environment management utilities.

This module provides utilities for managing test configurations, environment setup,
database initialization, and cleanup operations across different test scenarios.
"""

import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    """Configuration for a test environment."""

    name: str
    database_url: str
    vector_db_url: str
    embedding_api_key: str
    cache_url: str
    temp_dir: str
    log_level: str = "INFO"
    debug_mode: bool = False
    additional_config: dict[str, Any] = None

    def __post_init__(self):
        if self.additional_config is None:
            self.additional_config = {}


class ConfigManager:
    """Manage test configurations and environments."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing test configuration files
        """
        self.config_dir = config_dir or Path(__file__).parent / "configs"
        self.environments = {}
        self.current_environment = None
        self._temp_dirs = []

        # Load default configurations
        self._load_default_configs()

    def _load_default_configs(self):
        """Load default test environment configurations."""
        # Unit test environment
        self.environments["unit"] = EnvironmentConfig(
            name="unit",
            database_url="sqlite:///:memory:",
            vector_db_url="memory://localhost",
            embedding_api_key="test-key",
            cache_url="memory://localhost",
            temp_dir=self._create_temp_dir("unit_tests"),
            debug_mode=True,
        )

        # Integration test environment
        self.environments["integration"] = EnvironmentConfig(
            name="integration",
            database_url="postgresql://test:test@localhost:5432/test_integration",
            vector_db_url="http://localhost:6333",
            embedding_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            cache_url="redis://localhost:6379/1",
            temp_dir=self._create_temp_dir("integration_tests"),
            log_level="DEBUG",
        )

        # End-to-end test environment
        self.environments["e2e"] = EnvironmentConfig(
            name="e2e",
            database_url="postgresql://test:test@localhost:5432/test_e2e",
            vector_db_url="http://localhost:6333",
            embedding_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            cache_url="redis://localhost:6379/2",
            temp_dir=self._create_temp_dir("e2e_tests"),
            additional_config={
                "browser_headless": True,
                "screenshot_on_failure": True,
                "test_timeout": 300,
            },
        )

        # Performance test environment
        self.environments["performance"] = EnvironmentConfig(
            name="performance",
            database_url="postgresql://test:test@localhost:5432/test_performance",
            vector_db_url="http://localhost:6333",
            embedding_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
            cache_url="redis://localhost:6379/3",
            temp_dir=self._create_temp_dir("performance_tests"),
            additional_config={
                "performance_monitoring": True,
                "metrics_collection": True,
                "load_test_duration": 300,
            },
        )

    def _create_temp_dir(self, prefix: str) -> str:
        """Create a temporary directory for tests.

        Args:
            prefix: Prefix for the temporary directory name

        Returns:
            Path to the created temporary directory
        """
        temp_dir = tempfile.mkdtemp(prefix=f"{prefix}_")
        self._temp_dirs.append(temp_dir)
        return temp_dir

    def get_environment(self, name: str) -> EnvironmentConfig:
        """Get test environment configuration.

        Args:
            name: Environment name

        Returns:
            Test environment configuration

        Raises:
            KeyError: If environment not found
        """
        if name not in self.environments:
            msg = f"Test environment '{name}' not found"
            raise KeyError(msg)
        return self.environments[name]

    def set_current_environment(self, name: str):
        """Set the current test environment.

        Args:
            name: Environment name
        """
        self.current_environment = self.get_environment(name)

        # Set environment variables
        self._set_environment_variables(self.current_environment)

    def _set_environment_variables(self, env: EnvironmentConfig):
        """Set environment variables for the test environment.

        Args:
            env: Test environment configuration
        """
        os.environ["TEST_ENVIRONMENT"] = env.name
        os.environ["DATABASE_URL"] = env.database_url
        os.environ["VECTOR_DB_URL"] = env.vector_db_url
        os.environ["EMBEDDING_API_KEY"] = env.embedding_api_key
        os.environ["CACHE_URL"] = env.cache_url
        os.environ["TEMP_DIR"] = env.temp_dir
        os.environ["LOG_LEVEL"] = env.log_level
        os.environ["DEBUG_MODE"] = str(env.debug_mode)

        # Set additional config as environment variables
        for key, value in env.additional_config.items():
            env_key = f"TEST_{key.upper()}"
            os.environ[env_key] = str(value)

    def load_config_file(self, file_path: Path, environment_name: str):
        """Load configuration from a file.

        Args:
            file_path: Path to configuration file
            environment_name: Name for the environment
        """
        if not file_path.exists():
            msg = f"Configuration file not found: {file_path}"
            raise FileNotFoundError(msg)

        if file_path.suffix.lower() == ".yaml" or file_path.suffix.lower() == ".yml":
            with Path(file_path).open() as f:
                config_data = yaml.safe_load(f)
        elif file_path.suffix.lower() == ".json":
            with Path(file_path).open() as f:
                config_data = json.load(f)
        else:
            msg = f"Unsupported configuration file format: {file_path.suffix}"
            raise ValueError(msg)

        # Create environment from config data
        env = EnvironmentConfig(**config_data)
        self.environments[environment_name] = env

    def save_config_file(
        self, environment_name: str, file_path: Path, format: str = "yaml"
    ):
        """Save environment configuration to a file.

        Args:
            environment_name: Name of environment to save
            file_path: Path where to save the configuration
            format: File format ("yaml" or "json")
        """
        env = self.get_environment(environment_name)
        config_data = asdict(env)

        file_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "yaml":
            with Path(file_path).open("w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
        elif format.lower() == "json":
            with Path(file_path).open("w") as f:
                json.dump(config_data, f, indent=2)
        else:
            msg = f"Unsupported format: {format}"
            raise ValueError(msg)

    def cleanup_temp_directories(self):
        """Clean up all temporary directories created during testing."""
        for temp_dir in self._temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        self._temp_dirs.clear()

    def get_all_environments(self) -> dict[str, EnvironmentConfig]:
        """Get all available test environments.

        Returns:
            Dictionary mapping environment names to configurations
        """
        return self.environments.copy()

    @contextmanager
    def temporary_environment(self, **_kwargs):
        """Create a temporary test environment.

        Args:
            **_kwargs: Environment configuration parameters
        """
        # Create temporary environment
        temp_name = f"temp_{len(self.environments)}"
        defaults = {
            "name": temp_name,
            "database_url": "sqlite:///:memory:",
            "vector_db_url": "memory://localhost",
            "embedding_api_key": "test-key",
            "cache_url": "memory://localhost",
            "temp_dir": self._create_temp_dir(temp_name),
        }
        defaults.update(_kwargs)

        temp_env = EnvironmentConfig(**defaults)
        self.environments[temp_name] = temp_env

        # Save current environment
        previous_env = self.current_environment

        try:
            self.set_current_environment(temp_name)
            yield temp_env
        finally:
            # Restore previous environment
            if previous_env:
                self.set_current_environment(previous_env.name)

            # Clean up temporary environment
            if temp_name in self.environments:
                del self.environments[temp_name]


# Module-level configuration manager instance
_config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the test configuration manager instance."""
    return _config_manager


def get_test_environment(name: str | None = None) -> EnvironmentConfig:
    """Get test environment configuration.

    Args:
        name: Environment name (defaults to current environment)

    Returns:
        Test environment configuration
    """
    config_manager = get_config_manager()
    if name is None:
        if config_manager.current_environment is None:
            # Default to unit test environment
            config_manager.set_current_environment("unit")
        return config_manager.current_environment

    return config_manager.get_environment(name)


def setup_test_database(environment_name: str = "unit") -> dict[str, Any]:
    """Setup test database for the specified environment.

    Args:
        environment_name: Name of test environment

    Returns:
        Database connection information
    """
    env = get_test_environment(environment_name)

    # For in-memory databases, return immediately
    if "memory" in env.database_url or ":memory:" in env.database_url:
        return {
            "status": "ready",
            "database_url": env.database_url,
            "type": "in_memory",
        }

    # For real databases, perform setup (mock implementation)
    # In a real implementation, this would create tables, run migrations, etc.
    return {
        "status": "ready",
        "database_url": env.database_url,
        "type": "persistent",
        "tables_created": [
            "documents",
            "embeddings",
            "collections",
            "metadata",
            "search_history",
        ],
        "migrations_run": True,
    }


def cleanup_test_data(environment_name: str = "unit") -> dict[str, Any]:
    """Clean up test data for the specified environment.

    Args:
        environment_name: Name of test environment

    Returns:
        Cleanup results
    """
    env = get_test_environment(environment_name)

    cleanup_results = {
        "environment": environment_name,
        "database_cleaned": False,
        "temp_files_removed": False,
        "cache_cleared": False,
        "errors": [],
    }

    try:
        # Clean up database
        if "memory" in env.database_url or ":memory:" in env.database_url:
            cleanup_results["database_cleaned"] = True
        else:
            # For persistent databases, truncate tables (mock implementation)
            cleanup_results["database_cleaned"] = True
            cleanup_results["tables_truncated"] = [
                "documents",
                "embeddings",
                "collections",
                "metadata",
                "search_history",
            ]

        # Clean up temporary files
        if Path(env.temp_dir).exists():
            temp_files_count = len(list(Path(env.temp_dir).rglob("*")))
            shutil.rmtree(env.temp_dir, ignore_errors=True)
            cleanup_results["temp_files_removed"] = True
            cleanup_results["temp_files_count"] = temp_files_count

        # Clear cache (mock implementation)
        if "memory" in env.cache_url:
            cleanup_results["cache_cleared"] = True
        else:
            # For real cache systems, clear cache
            cleanup_results["cache_cleared"] = True
            cleanup_results["cache_keys_removed"] = 0  # Mock count

    except (ValueError, RuntimeError, OSError) as e:
        cleanup_results["errors"].append(str(e))

    return cleanup_results


def create_test_config(
    name: str, overrides: dict[str, Any] | None = None
) -> EnvironmentConfig:
    """Create a custom test configuration.

    Args:
        name: Configuration name
        overrides: Configuration overrides

    Returns:
        Test environment configuration
    """
    base_config = {
        "name": name,
        "database_url": "sqlite:///:memory:",
        "vector_db_url": "memory://localhost",
        "embedding_api_key": "test-key",
        "cache_url": "memory://localhost",
        "temp_dir": get_config_manager()._create_temp_dir(name),  # noqa: SLF001
        "log_level": "INFO",
        "debug_mode": True,
        "additional_config": {},
    }

    if overrides:
        base_config.update(overrides)

    env = EnvironmentConfig(**base_config)
    get_config_manager().environments[name] = env

    return env


def get_test_data_dir() -> Path:
    """Get the test data directory.

    Returns:
        Path to test data directory
    """
    current_env = get_test_environment()
    test_data_dir = Path(current_env.temp_dir) / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir


def create_test_file(
    filename: str, content: str, subdirectory: str | None = None
) -> Path:
    """Create a test file with specified content.

    Args:
        filename: Name of the file to create
        content: File content
        subdirectory: Optional subdirectory within test data dir

    Returns:
        Path to created file
    """
    test_data_dir = get_test_data_dir()

    if subdirectory:
        file_dir = test_data_dir / subdirectory
        file_dir.mkdir(parents=True, exist_ok=True)
    else:
        file_dir = test_data_dir

    file_path = file_dir / filename
    file_path.write_text(content, encoding="utf-8")

    return file_path


def setup__mock_services(environment_name: str = "unit") -> dict[str, Any]:
    """Setup mock services for testing.

    Args:
        environment_name: Name of test environment

    Returns:
        Mock service configurations
    """
    env = get_test_environment(environment_name)

    _mock_services = {
        "vector_db": {
            "url": env.vector_db_url,
            "collections": ["test_collection"],
            "documents_count": 100,
            "status": "active",
        },
        "embedding_service": {
            "api_key": env.embedding_api_key,
            "model": "text-embedding-ada-002",
            "dimension": 1536,
            "status": "active",
        },
        "cache": {
            "url": env.cache_url,
            "ttl": 3600,
            "max_size": 1000,
            "status": "active",
        },
        "web_scraper": {
            "user_agent": "Test Bot 1.0",
            "timeout": 30,
            "rate_limit": 10,
            "status": "active",
        },
    }

    return _mock_services


def validate_test_environment(environment_name: str) -> dict[str, Any]:
    """Validate a test environment configuration.

    Args:
        environment_name: Name of environment to validate

    Returns:
        Validation results
    """
    try:
        env = get_test_environment(environment_name)
    except KeyError:
        return {
            "valid": False,
            "errors": [f"Environment '{environment_name}' not found"],
        }

    validation_results = {
        "valid": True,
        "environment": environment_name,
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    # Check required fields
    required_fields = [
        "name",
        "database_url",
        "vector_db_url",
        "embedding_api_key",
        "cache_url",
        "temp_dir",
    ]

    for field in required_fields:
        if not hasattr(env, field) or getattr(env, field) is None:
            validation_results["errors"].append(f"Missing required field: {field}")
            validation_results["valid"] = False
        else:
            validation_results["checks"].append(f"✓ {field} is present")

    # Check temp directory
    if hasattr(env, "temp_dir") and env.temp_dir:
        if not Path(env.temp_dir).exists():
            validation_results["warnings"].append(
                f"Temp directory does not exist: {env.temp_dir}"
            )
        else:
            validation_results["checks"].append("✓ Temp directory exists")

    # Check for test API key in production-like environments
    if env.embedding_api_key == "test-key" and environment_name in [
        "integration",
        "e2e",
    ]:
        validation_results["warnings"].append(
            "Using test API key in integration/e2e environment"
        )

    return validation_results


# Cleanup function for pytest fixtures
def cleanup_all_test_environments():
    """Clean up all test environments and temporary resources."""
    config_manager = get_config_manager()

    # Clean up temporary directories
    config_manager.cleanup_temp_directories()

    # Reset environment variables
    test_env_vars = [key for key in os.environ if key.startswith("TEST_")]
    for var in test_env_vars:
        os.environ.pop(var, None)

    # Reset configuration manager state
    config_manager.environments.clear()
    config_manager.current_environment = None
