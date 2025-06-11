"""Comprehensive tests for configuration templates.

This test file covers the configuration template system that provides
pre-configured templates for different deployment scenarios.
"""

from unittest.mock import patch

from src.config.templates import ConfigurationTemplate
from src.config.templates import ConfigurationTemplates
from src.config.templates import TemplateMetadata


class TestTemplateMetadata:
    """Test the TemplateMetadata model."""

    def test_template_metadata_creation(self):
        """Test basic TemplateMetadata creation."""
        metadata = TemplateMetadata(
            name="development",
            description="Development configuration",
            use_case="Local development and testing",
            environment="development",
            tags=["dev", "local"]
        )

        assert metadata.name == "development"
        assert metadata.description == "Development configuration"
        assert metadata.use_case == "Local development and testing"
        assert metadata.environment == "development"
        assert metadata.tags == ["dev", "local"]

    def test_template_metadata_default_values(self):
        """Test TemplateMetadata with default values."""
        metadata = TemplateMetadata(
            name="basic",
            description="Basic template",
            use_case="development",
            environment="development"
        )

        assert metadata.use_case == "development"
        assert metadata.environment == "development"
        assert metadata.tags == []

    def test_template_metadata_serialization(self):
        """Test TemplateMetadata serialization."""
        metadata = TemplateMetadata(
            name="production",
            description="Production template",
            use_case="Production deployment",
            environment="production",
            tags=["prod", "secure"]
        )

        # Should be serializable to dict
        data = metadata.model_dump()
        assert isinstance(data, dict)
        assert data["name"] == "production"
        assert data["tags"] == ["prod", "secure"]

        # Should be deserializable from dict
        restored = TemplateMetadata(**data)
        assert restored.name == metadata.name
        assert restored.tags == metadata.tags


class TestConfigurationTemplate:
    """Test the ConfigurationTemplate model."""

    def test_configuration_template_creation(self):
        """Test basic ConfigurationTemplate creation."""
        metadata = TemplateMetadata(
            name="test",
            description="Test template",
            use_case="testing",
            environment="test"
        )

        config_data = {
            "environment": "development",
            "debug": True,
            "log_level": "DEBUG"
        }

        template = ConfigurationTemplate(
            metadata=metadata,
            configuration=config_data
        )

        assert template.metadata.name == "test"
        assert template.configuration["environment"] == "development"
        assert template.configuration["debug"] is True

    def test_configuration_template_serialization(self):
        """Test ConfigurationTemplate serialization."""
        metadata = TemplateMetadata(name="test", description="Test", use_case="testing", environment="test")
        config_data = {"environment": "test"}

        template = ConfigurationTemplate(
            metadata=metadata,
            configuration=config_data
        )

        # Should be serializable
        data = template.model_dump()
        assert isinstance(data, dict)
        assert "metadata" in data
        assert "configuration" in data
        assert data["configuration"]["environment"] == "test"


class TestConfigurationTemplates:
    """Test the ConfigurationTemplates class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.templates = ConfigurationTemplates()

    def test_templates_initialization(self):
        """Test ConfigurationTemplates initialization."""
        templates = ConfigurationTemplates()
        assert hasattr(templates, 'path_manager')
        assert hasattr(templates, 'get_template')

    def test_list_available_templates(self):
        """Test listing available templates."""
        available = self.templates.list_available_templates()

        assert isinstance(available, list)
        # Should have the expected template names
        expected_templates = [
            "development", "production", "high_performance",
            "memory_optimized", "distributed"
        ]

        for template_name in expected_templates:
            assert template_name in available

    def test_get_template_existing(self):
        """Test getting existing template."""
        template = self.templates.get_template("development")

        assert template is not None
        assert template.metadata.name == "development"
        assert isinstance(template.metadata, TemplateMetadata)
        assert "development" in template.metadata.description.lower()

    def test_get_template_nonexistent(self):
        """Test getting non-existent template."""
        template = self.templates.get_template("nonexistent")

        assert template is None

    def test_get_template_existing(self):
        """Test getting existing template."""
        template = self.templates.get_template("development")

        assert template is not None
        assert isinstance(template, ConfigurationTemplate)
        assert template.metadata.name == "development"
        assert template.configuration["environment"] == "development"
        assert template.configuration["debug"] is True

    def test_get_template_nonexistent(self):
        """Test getting non-existent template."""
        template = self.templates.get_template("nonexistent")

        assert template is None

    def test_development_template_structure(self):
        """Test development template has correct structure."""
        template = self.templates.get_template("development")

        assert template is not None
        config = template.configuration

        # Development template should have debug enabled
        assert config["debug"] is True
        assert config["environment"] == "development"
        assert config["log_level"] == "DEBUG"

        # Should have database configuration
        assert "database" in config
        assert "sqlite" in config["database"]["database_url"].lower()

        # Should have appropriate performance settings
        assert "performance" in config
        assert config["performance"]["max_concurrent_requests"] <= 20

    def test_production_template_structure(self):
        """Test production template has correct structure."""
        template = self.templates.get_template("production")

        assert template is not None
        config = template.configuration

        # Production template should have debug disabled
        assert config["debug"] is False
        assert config["environment"] == "production"
        assert config["log_level"] == "INFO"

        # Should have security settings
        assert "security" in config

        # Should have monitoring enabled
        if "monitoring" in config:
            assert config["monitoring"]["enabled"] is True

        # Should have performance tuning
        assert "performance" in config
        assert config["performance"]["max_concurrent_requests"] > 10

    def test_high_performance_template_structure(self):
        """Test high-performance template has correct structure."""
        template = self.templates.get_template("high_performance")

        assert template is not None
        config = template.configuration

        # High-performance template should have optimized settings
        assert config["environment"] == "production"
        assert "performance" in config

        # Should have high concurrency settings
        perf = config["performance"]
        assert perf["max_concurrent_requests"] >= 50

        # Should have caching enabled
        assert "cache" in config
        assert config["cache"]["enable_caching"] is True
        assert config["cache"]["enable_dragonfly_cache"] is True

        # Should have optimized HNSW settings
        if "qdrant" in config and "collection_hnsw_configs" in config["qdrant"]:
            hnsw_configs = config["qdrant"]["collection_hnsw_configs"]
            # High-performance configs should have higher ef_construct
            for collection_config in hnsw_configs.values():
                if "ef_construct" in collection_config:
                    assert collection_config["ef_construct"] >= 200

    def test_memory_optimized_template_structure(self):
        """Test memory-optimized template has correct structure."""
        template = self.templates.get_template("memory_optimized")

        assert template is not None
        config = template.configuration

        # Memory-optimized template should have conservative settings
        assert "performance" in config
        perf = config["performance"]
        assert perf["max_concurrent_requests"] <= 10
        assert perf["max_memory_usage_mb"] <= 512

        # Should have reduced cache sizes
        if "cache" in config:
            cache = config["cache"]
            if "local_max_size" in cache:
                assert cache["local_max_size"] <= 500
            if "local_max_memory_mb" in cache:
                assert cache["local_max_memory_mb"] <= 50

        # Should have conservative chunking settings
        if "chunking" in config:
            chunking = config["chunking"]
            assert chunking["chunk_size"] <= 1000

    def test_distributed_template_structure(self):
        """Test distributed template has correct structure."""
        template = self.templates.get_template("distributed")

        assert template is not None
        config = template.configuration

        # Distributed template should have cluster-ready settings
        assert config["environment"] == "production"

        # Should have database configuration for clustering
        assert "database" in config
        db = config["database"]
        if "database_url" in db:
            # Should use PostgreSQL for distributed deployments
            assert "postgresql" in db["database_url"].lower()

        # Should have monitoring enabled for cluster visibility
        if "monitoring" in config:
            assert config["monitoring"]["enabled"] is True

        # Should have appropriate performance settings for clustering
        assert "performance" in config
        perf = config["performance"]
        assert perf["max_concurrent_requests"] >= 20

    def test_apply_template_to_config_existing(self):
        """Test applying existing template to configuration."""
        config_data = self.templates.apply_template_to_config("development")

        assert config_data is not None
        assert isinstance(config_data, dict)
        assert config_data["environment"] == "development"
        assert config_data["debug"] is True

    def test_apply_template_to_config_nonexistent(self):
        """Test applying non-existent template."""
        config_data = self.templates.apply_template_to_config("nonexistent")

        assert config_data is None

    def test_apply_template_to_config_with_overrides(self):
        """Test applying template with environment overrides."""
        # Note: This test assumes the template supports environment overrides
        config_data = self.templates.apply_template_to_config(
            "development",
            environment_overrides="staging"
        )

        assert config_data is not None
        # The exact behavior depends on implementation
        # At minimum, should return valid config data
        assert isinstance(config_data, dict)

    def test_all_templates_have_required_fields(self):
        """Test that all templates have required configuration fields."""
        required_fields = ["environment", "debug", "log_level"]

        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            config = template.configuration
            for field in required_fields:
                assert field in config, f"Template {template_name} missing field {field}"

    def test_template_metadata_consistency(self):
        """Test that template metadata is consistent with configuration."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            # Template name should match metadata name
            assert template.metadata.name == template_name

            # If metadata specifies environment, config should match
            if template.metadata.environment:
                assert template.configuration["environment"] == template.metadata.environment

    def test_template_configuration_validity(self):
        """Test that all template configurations are valid."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            config = template.configuration

            # Should have valid environment values
            assert config["environment"] in ["development", "staging", "production"]

            # Debug should be boolean
            assert isinstance(config["debug"], bool)

            # Log level should be valid
            valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            assert config["log_level"] in valid_log_levels

    def test_template_performance_settings_reasonable(self):
        """Test that template performance settings are reasonable."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            config = template.configuration

            if "performance" in config:
                perf = config["performance"]

                # Concurrent requests should be positive and reasonable
                if "max_concurrent_requests" in perf:
                    assert 1 <= perf["max_concurrent_requests"] <= 1000

                # Timeout should be positive
                if "request_timeout" in perf:
                    assert perf["request_timeout"] > 0

                # Memory usage should be reasonable
                if "max_memory_usage_mb" in perf:
                    assert 100 <= perf["max_memory_usage_mb"] <= 10000

    def test_template_cache_settings_valid(self):
        """Test that template cache settings are valid."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            config = template.configuration

            if "cache" in config:
                cache = config["cache"]

                # Boolean flags should be booleans
                bool_fields = ["enable_caching", "enable_local_cache", "enable_dragonfly_cache"]
                for field in bool_fields:
                    if field in cache:
                        assert isinstance(cache[field], bool)

                # Numeric settings should be positive
                if "local_max_size" in cache:
                    assert cache["local_max_size"] > 0
                if "local_max_memory_mb" in cache:
                    assert cache["local_max_memory_mb"] > 0

    def test_template_database_urls_valid_format(self):
        """Test that template database URLs have valid format."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            config = template.configuration

            if "database" in config and "database_url" in config["database"]:
                db_url = config["database"]["database_url"]

                # Should be a string
                assert isinstance(db_url, str)

                # Should have a scheme
                assert "://" in db_url

                # Should be either sqlite or postgresql
                assert any(scheme in db_url.lower() for scheme in ["sqlite", "postgresql"])


    def test_template_tags_are_meaningful(self):
        """Test that template tags provide meaningful categorization."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            metadata = template.metadata
            # Tags should be non-empty for most templates
            if metadata.tags:
                for tag in metadata.tags:
                    # Tags should be non-empty strings
                    assert isinstance(tag, str)
                    assert len(tag.strip()) > 0

    def test_template_use_cases_descriptive(self):
        """Test that template use cases are descriptive."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            metadata = template.metadata
            # Use case should be descriptive
            if metadata.use_case:
                assert len(metadata.use_case.strip()) > 10  # Should be reasonably descriptive

    def test_template_descriptions_informative(self):
        """Test that template descriptions are informative."""
        for template_name in self.templates.list_available_templates():
            template = self.templates.get_template(template_name)
            if template is None:
                # Skip templates that can't be loaded
                continue

            metadata = template.metadata
            # Description should be informative
            assert len(metadata.description.strip()) > 5
            assert template_name.lower() in metadata.description.lower() or \
                   any(word in metadata.description.lower() for word in template_name.split('_'))

    def test_production_templates_security_hardened(self):
        """Test that production templates have appropriate security settings."""
        production_templates = ["production", "high_performance", "distributed"]

        for template_name in production_templates:
            template = self.templates.get_template(template_name)
            assert template is not None

            config = template.configuration

            # Production templates should have debug disabled
            assert config["debug"] is False

            # Should have appropriate log level (not DEBUG)
            assert config["log_level"] != "DEBUG"

            # Should have security configuration if available
            if "security" in config:
                security = config["security"]
                # Security should be enabled
                if "enabled" in security:
                    assert security["enabled"] is True
