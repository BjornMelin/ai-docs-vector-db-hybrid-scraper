"""Async validation tests for configuration system.

Tests async configuration loading and validation patterns.
"""

import asyncio
import json

import pytest

from src.config import (
    Config,
    get_config,
    reset_config
)


class TestAsyncConfigurationLoading:
    """Test async configuration loading patterns."""

    def setup_method(self):
        """Reset config before each test."""
        reset_config()

    @pytest.mark.asyncio
    async def test_async_config_initialization(self):
        """Test async configuration initialization."""

        async def create_config():
            """Simulate async config creation."""
            await asyncio.sleep(0.01)  # Simulate async work
            return Config(debug=True, app_name="Async Test")

        config = await create_config()
        assert config.debug is True
        assert config.app_name == "Async Test"

    @pytest.mark.asyncio
    async def test_concurrent_config_access(self):
        """Test concurrent config access is thread-safe."""

        async def get_config_async():
            """Get config in async context."""
            await asyncio.sleep(0.01)
            return get_config()

        # Create multiple concurrent tasks
        tasks = [get_config_async() for _ in range(10)]
        configs = await asyncio.gather(*tasks)

        # All should return the same instance
        first_config = configs[0]
        for config in configs[1:]:
            assert config is first_config

    @pytest.mark.asyncio
    async def test_async_file_loading_simulation(self):
        """Test simulated async file loading."""
        config_data = {
            "debug": True,
            "environment": "staging",
            "openai": {"api_key": "sk-async-test"},
        }

        async def load_config_async(data):
            """Simulate async config loading from file."""
            await asyncio.sleep(0.01)  # Simulate file I/O
            return Config.model_validate(data)

        config = await load_config_async(config_data)
        assert config.debug is True
        assert config.openai.api_key == "sk-async-test"

    @pytest.mark.asyncio
    async def test_async_config_validation(self):
        """Test async configuration validation."""

        async def validate_config_async(config_data):
            """Async config validation."""
            await asyncio.sleep(0.01)  # Simulate validation work
            try:
                config = Config.model_validate(config_data)
            except (ConnectionError, RuntimeError, ValueError) as e:
                return None, str(e)
            else:
                return config, None

        # Valid config
        valid_config_data = {
            "embedding_provider": "fastembed",
            "chunking": {"chunk_size": 1000, "chunk_overlap": 200},
        }
        config, error = await validate_config_async(valid_config_data)
        assert config is not None
        assert error is None

        # Invalid config
        invalid_config_data = {
            "embedding_provider": "openai",
            "openai": {"api_key": None},  # Missing required API key
        }
        config, error = await validate_config_async(invalid_config_data)
        assert config is None
        assert "OpenAI API key required" in error

    @pytest.mark.asyncio
    async def test_async_nested_config_processing(self):
        """Test async processing of nested configurations."""

        async def process_nested_configs(config):
            """Process nested configurations asynchronously."""
            results = {}

            # Simulate async processing of each nested config
            await asyncio.sleep(0.001)
            results["cache"] = {
                "enabled": config.cache.enable_caching,
                "size": config.cache.local_max_size,
            }

            await asyncio.sleep(0.001)
            results["qdrant"] = {
                "url": config.qdrant.url,
                "batch_size": config.qdrant.batch_size,
            }

            await asyncio.sleep(0.001)
            results["chunking"] = {
                "size": config.chunking.chunk_size,
                "overlap": config.chunking.chunk_overlap,
            }

            return results

        config = Config()
        results = await process_nested_configs(config)

        assert "cache" in results
        assert "qdrant" in results
        assert "chunking" in results
        assert results["cache"]["enabled"] == config.cache.enable_caching
        assert results["qdrant"]["url"] == config.qdrant.url
        assert results["chunking"]["size"] == config.chunking.chunk_size

    @pytest.mark.asyncio
    async def test_async_config_updates(self):
        """Test async configuration updates."""

        async def update_config_async(config, updates):
            """Apply updates to config asynchronously."""
            await asyncio.sleep(0.01)  # Simulate async work

            # Create new config with updates
            config_data = config.model_dump()
            config_data.update(updates)
            return Config.model_validate(config_data)

        original_config = Config(debug=False, app_name="Original")
        updates = {"debug": True, "app_name": "Updated"}

        updated_config = await update_config_async(original_config, updates)

        assert updated_config.debug is True
        assert updated_config.app_name == "Updated"
        # Original should be unchanged
        assert original_config.debug is False
        assert original_config.app_name == "Original"

    @pytest.mark.asyncio
    async def test_async_batch_config_operations(self):
        """Test batch configuration operations asynchronously."""

        async def create_configs_batch(config_specs):
            """Create multiple configs in batch."""

            async def create_single_config(spec):
                await asyncio.sleep(0.001)  # Simulate work
                return Config.model_validate(spec)

            tasks = [create_single_config(spec) for spec in config_specs]
            return await asyncio.gather(*tasks)

        config_specs = [
            {"debug": True, "app_name": "Config 1"},
            {"debug": False, "app_name": "Config 2"},
            {"debug": True, "app_name": "Config 3"},
        ]

        configs = await create_configs_batch(config_specs)

        assert len(configs) == 3
        assert configs[0].debug is True
        assert configs[0].app_name == "Config 1"
        assert configs[1].debug is False
        assert configs[1].app_name == "Config 2"
        assert configs[2].debug is True
        assert configs[2].app_name == "Config 3"

    @pytest.mark.asyncio
    async def test_async_config_serialization(self):
        """Test async configuration serialization patterns."""

        async def serialize_config_async(config, format_type="json"):
            """Serialize config asynchronously."""
            await asyncio.sleep(0.01)  # Simulate serialization work

            data = config.model_dump(mode="json")  # Use JSON mode for serialization

            if format_type == "json":
                return json.dumps(data, indent=2)
            if format_type == "dict":
                return data
            msg = f"Unsupported format: {format_type}"
            raise ValueError(msg)

        config = Config(debug=True, app_name="Serialization Test")

        # Test JSON serialization
        json_data = await serialize_config_async(config, "json")
        assert '"debug": true' in json_data
        assert '"app_name": "Serialization Test"' in json_data

        # Test dict serialization
        dict_data = await serialize_config_async(config, "dict")
        assert dict_data["debug"] is True
        assert dict_data["app_name"] == "Serialization Test"
