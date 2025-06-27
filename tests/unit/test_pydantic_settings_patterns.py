"""Tests for pydantic-settings integration patterns.

Tests for the modernized configuration system using pydantic-settings
instead of custom configuration management.
"""

import os  # noqa: PLC0415
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config import Config


class TestPydanticSettingsPatterns:
    """Test modern pydantic-settings patterns."""

    def test_environment_variable_loading(self):
        """Test loading configuration from environment variables.

        Verifies that pydantic-settings properly loads and validates
        configuration from environment variables.
        """

        class TestSettings(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="TEST_",
                case_sensitive=False,
                env_file=".env",
                env_file_encoding="utf-8",
            )

            database_url: str = Field(default="sqlite:///test.db")
            api_key: str = Field(default="default-key")
            cache_size: int = Field(default=1000, ge=1)
            debug_mode: bool = Field(default=False)

        # Test with environment variables
        with patch.dict(
            os.environ,
            {
                "TEST_DATABASE_URL": "postgresql://localhost/mydb",
                "TEST_API_KEY": "secret-key-123",
                "TEST_CACHE_SIZE": "5000",
                "TEST_DEBUG_MODE": "true",
            },
        ):
            settings = TestSettings()

            assert settings.database_url == "postgresql://localhost/mydb"
            assert settings.api_key == "secret-key-123"
            assert settings.cache_size == 5000
            assert settings.debug_mode is True

    def test_dotenv_file_loading(self):
        """Test loading configuration from .env files.

        Verifies that pydantic-settings can load configuration
        from .env files with proper precedence.
        """
        # Create temporary .env file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env.test", delete=False
        ) as f:
            f.write("APP_NAME=test-application\n")
            f.write("PORT=9000\n")
            f.flush()

            try:

                class TestSettings(BaseSettings):
                    model_config = SettingsConfigDict(
                        env_file=f.name,
                        env_file_encoding="utf-8",
                    )

                    app_name: str = Field(default="default-app")
                    port: int = Field(default=8000)

                # Test loading from file
                settings = TestSettings()

                assert settings.app_name == "test-application"
                assert settings.port == 9000
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_configuration_validation(self):
        """Test configuration validation with pydantic.

        Verifies that pydantic properly validates configuration
        values and provides meaningful error messages.
        """

        class ValidatedSettings(BaseSettings):
            model_config = SettingsConfigDict(validate_default=True)

            port: int = Field(ge=1, le=65535, description="Server port")
            timeout: float = Field(gt=0, description="Timeout in seconds")
            workers: int = Field(ge=1, description="Number of workers")

        # Test valid configuration
        settings = ValidatedSettings(port=8080, timeout=30.0, workers=4)
        assert settings.port == 8080
        assert settings.timeout == 30.0
        assert settings.workers == 4

        # Test invalid configurations
        with pytest.raises(ValidationError) as exc_info:
            ValidatedSettings(port=70000, timeout=30.0, workers=4)  # Port too high

        assert "port" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ValidatedSettings(port=8080, timeout=-1.0, workers=4)  # Negative timeout

        assert "timeout" in str(exc_info.value)

    def test_nested_configuration_models(self):
        """Test nested configuration models.

        Verifies that complex nested configurations work
        properly with pydantic-settings.
        """

        class DatabaseSettings(BaseSettings):
            host: str = Field(default="localhost")
            port: int = Field(default=5432)
            name: str = Field(default="mydb")

        class CacheSettings(BaseSettings):
            enabled: bool = Field(default=True)
            ttl: int = Field(default=300)
            max_size: int = Field(default=1000)

        class AppSettings(BaseSettings):
            model_config = SettingsConfigDict(
                env_prefix="APP_",
                env_nested_delimiter="__",
            )

            app_name: str = Field(default="test-app")
            database: DatabaseSettings = Field(default_factory=DatabaseSettings)
            cache: CacheSettings = Field(default_factory=CacheSettings)

        # Test with nested environment variables
        with patch.dict(
            os.environ,
            {
                "APP_APP_NAME": "my-application",
                "APP_DATABASE__HOST": "db.example.com",
                "APP_DATABASE__PORT": "3306",
                "APP_CACHE__TTL": "600",
            },
        ):
            settings = AppSettings()

            assert settings.app_name == "my-application"
            assert settings.database.host == "db.example.com"
            assert settings.database.port == 3306
            assert settings.cache.ttl == 600
            assert settings.cache.enabled is True  # Default value

    def test_configuration_aliases(self):
        """Test configuration aliases and alternative names.

        Verifies that pydantic-settings handles field aliases
        for backward compatibility.
        """

        class AliasedSettings(BaseSettings):
            model_config = SettingsConfigDict(populate_by_name=True)

            database_url: str = Field(
                default="sqlite:///default.db",
                alias="DB_URL",
                description="Database connection URL",
            )
            api_secret: str = Field(
                default="default-secret",
                alias="SECRET_KEY",
                description="API secret key",
            )

        # Test with alias names
        with patch.dict(
            os.environ,
            {
                "DB_URL": "postgresql://localhost/test",
                "SECRET_KEY": "super-secret-key",
            },
        ):
            settings = AliasedSettings()

            assert settings.database_url == "postgresql://localhost/test"
            assert settings.api_secret == "super-secret-key"  # noqa: S105

        # Test with field names
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "mysql://localhost/app",
                "API_SECRET": "another-secret",
            },
        ):
            settings = AliasedSettings()

            assert settings.database_url == "mysql://localhost/app"
            assert settings.api_secret == "another-secret"  # noqa: S105

    def test_configuration_from_multiple_sources(self):
        """Test configuration loading from multiple sources.

        Verifies the precedence order: CLI args > env vars > .env file > defaults
        """

        class MultiSourceSettings(BaseSettings):
            model_config = SettingsConfigDict(
                env_file=".env",
                case_sensitive=False,
            )

            service_name: str = Field(default="default-service")
            port: int = Field(default=8000)
            debug: bool = Field(default=False)

        # Create .env file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write("SERVICE_NAME=env-file-service\n")
            f.write("PORT=9000\n")
            f.write("DEBUG=true\n")
            f.flush()

            try:

                class TestMultiSourceSettings(BaseSettings):
                    model_config = SettingsConfigDict(
                        env_file=f.name,
                        case_sensitive=False,
                    )

                    service_name: str = Field(default="default-service")
                    port: int = Field(default=8000)
                    debug: bool = Field(default=False)

                # Test: env var overrides .env file
                with patch.dict(os.environ, {"PORT": "7000"}):
                    settings = TestMultiSourceSettings()

                    assert settings.service_name == "env-file-service"  # From .env
                    assert settings.port == 7000  # From env var (overrides .env)
                    assert settings.debug is True  # From .env
            finally:
                Path(f.name).unlink(missing_ok=True)

    def test_configuration_serialization(self):
        """Test configuration serialization and export.

        Verifies that configurations can be serialized for
        debugging, logging, or API responses.
        """

        class SerializableSettings(BaseSettings):
            app_name: str = Field(default="test-app")
            version: str = Field(default="1.0.0")
            secret_key: str = Field(
                default="secret", exclude=True
            )  # Exclude from serialization

        settings = SerializableSettings(
            app_name="my-app", version="2.0.0", secret_key="super-secret"
        )

        # Test serialization
        serialized = settings.model_dump()
        assert serialized["app_name"] == "my-app"
        assert serialized["version"] == "2.0.0"
        assert "secret_key" not in serialized  # Should be excluded

        # Test JSON serialization
        json_str = settings.model_dump_json()
        assert "my-app" in json_str
        assert "super-secret" not in json_str  # Should be excluded

    def test_configuration_dynamic_defaults(self):
        """Test dynamic default values in configuration.

        Verifies that dynamic defaults work correctly with
        pydantic-settings.
        """
        from datetime import datetime  # noqa: PLC0415

        class DynamicSettings(BaseSettings):
            created_at: datetime = Field(default_factory=datetime.now)
            instance_id: str = Field(default_factory=lambda: f"instance-{os.getpid()}")

        settings = DynamicSettings()

        assert isinstance(settings.created_at, datetime)
        assert settings.instance_id.startswith("instance-")
        assert str(os.getpid()) in settings.instance_id

    def test_configuration_custom_validation(self):
        """Test custom validation logic in configuration.

        Verifies that custom validators work properly with
        pydantic-settings.
        """
        from pydantic import field_validator  # noqa: PLC0415

        class CustomValidatedSettings(BaseSettings):
            email: str = Field(description="User email address")
            password: str = Field(min_length=8, description="User password")

            @field_validator("email")
            @classmethod
            def validate_email(cls, v):
                if "@" not in v:
                    raise ValueError("Invalid email format")
                return v.lower()

            @field_validator("password")
            @classmethod
            def validate_password(cls, v):
                if not any(c.isupper() for c in v):
                    raise ValueError(
                        "Password must contain at least one uppercase letter"
                    )
                if not any(c.isdigit() for c in v):
                    raise ValueError("Password must contain at least one digit")
                return v

        # Test valid configuration
        settings = CustomValidatedSettings(
            email="USER@EXAMPLE.COM", password="SecurePass123"
        )
        assert settings.email == "user@example.com"  # Lowercased
        assert settings.password == "SecurePass123"  # noqa: S105

        # Test invalid configurations
        with pytest.raises(ValidationError) as exc_info:
            CustomValidatedSettings(email="invalid-email", password="SecurePass123")
        assert "Invalid email format" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CustomValidatedSettings(email="user@example.com", password="weakpass")
        assert "uppercase" in str(exc_info.value)

    def test_configuration_model_rebuild(self):
        """Test configuration model rebuilding for dynamic schemas.

        Verifies that configuration models can be rebuilt
        when schema changes are needed.
        """

        class RebuildableSettings(BaseSettings):
            model_config = SettingsConfigDict(extra="allow")  # Allow extra fields

            basic_field: str = Field(default="basic")

        # Create initial instance
        settings = RebuildableSettings()
        assert settings.basic_field == "basic"

        # Test adding dynamic fields during initialization (simulating plugin-based config)
        new_settings = RebuildableSettings(dynamic_field="custom-value")
        assert new_settings.basic_field == "basic"
        assert new_settings.dynamic_field == "custom-value"


class TestConfigurationIntegrationPatterns:
    """Test integration patterns for configuration systems."""

    @pytest.mark.asyncio
    async def test_async_configuration_loading(self):
        """Test asynchronous configuration loading patterns.

        Verifies that configurations can be loaded asynchronously
        from remote sources or databases.
        """

        class AsyncConfigLoader:
            def __init__(self, base_settings: BaseSettings):
                self.base_settings = base_settings

            async def load_remote_config(self) -> dict:
                """Simulate loading config from remote source."""
                # In real implementation, this would make HTTP requests
                # or database queries
                import asyncio  # noqa: PLC0415

                await asyncio.sleep(0.01)  # Simulate network delay
                return {
                    "remote_setting": "value-from-remote",
                    "cache_size": 5000,
                }

            async def get_merged_config(self) -> dict:
                """Get configuration merged from local and remote sources."""
                local_config = self.base_settings.model_dump()
                remote_config = await self.load_remote_config()

                # Merge configurations (remote overrides local)
                merged = {**local_config, **remote_config}
                return merged

        # Test
        base_settings = Config()
        loader = AsyncConfigLoader(base_settings)

        merged_config = await loader.get_merged_config()

        assert "remote_setting" in merged_config
        assert merged_config["cache_size"] == 5000

    def test_configuration_dependency_injection(self):
        """Test configuration as dependency injection.

        Verifies that configurations can be properly injected
        into services and components.
        """

        class ServiceConfig(BaseSettings):
            service_name: str = Field(default="test-service")
            timeout: float = Field(default=30.0)
            retries: int = Field(default=3)

        class Service:
            def __init__(self, config: ServiceConfig):
                self.config = config
                self.initialized = True

            def get_timeout(self) -> float:
                return self.config.timeout

            def get_service_info(self) -> dict:
                return {
                    "name": self.config.service_name,
                    "timeout": self.config.timeout,
                    "retries": self.config.retries,
                }

        # Test dependency injection
        config = ServiceConfig(service_name="my-service", timeout=60.0)
        service = Service(config)

        assert service.initialized is True
        assert service.get_timeout() == 60.0

        info = service.get_service_info()
        assert info["name"] == "my-service"
        assert info["timeout"] == 60.0
        assert info["retries"] == 3

    def test_configuration_hot_reload_simulation(self):
        """Test hot reload configuration simulation.

        Verifies that configuration changes can be applied
        without restarting the application.
        """

        class HotReloadableConfig:
            def __init__(self, initial_settings: BaseSettings):
                self._settings = initial_settings
                self._reload_callbacks = []

            def add_reload_callback(self, callback):
                """Add callback to be called on config reload."""
                self._reload_callbacks.append(callback)

            def reload(self, new_settings: BaseSettings):
                """Reload configuration with new settings."""
                old_settings = self._settings
                self._settings = new_settings

                # Notify all callbacks of the change
                for callback in self._reload_callbacks:
                    callback(old_settings, new_settings)

            def get(self, key: str, default=None):
                """Get configuration value."""
                return getattr(self._settings, key, default)

        # Test hot reload
        initial_config = Config()
        hot_config = HotReloadableConfig(initial_config)

        # Track reload events
        reload_events = []

        def on_reload(old_settings, new_settings):
            reload_events.append((old_settings, new_settings))

        hot_config.add_reload_callback(on_reload)

        # Simulate configuration change
        new_config = Config()  # Different instance
        hot_config.reload(new_config)

        # Verify reload was triggered
        assert len(reload_events) == 1
        assert reload_events[0][0] == initial_config
        assert reload_events[0][1] == new_config
