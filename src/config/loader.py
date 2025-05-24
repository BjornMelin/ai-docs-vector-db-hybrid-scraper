"""Configuration loader utilities for managing configuration sources.

This module provides utilities for loading configuration from various sources
including environment variables, files, and documentation sites.
"""

import json
import os
from pathlib import Path
from typing import Any

from .models import DocumentationSite
from .models import UnifiedConfig


class ConfigLoader:
    """Utility class for loading and managing configuration."""

    @staticmethod
    def load_documentation_sites(config_path: Path | str) -> list[DocumentationSite]:
        """Load documentation sites from JSON configuration file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(
                f"Documentation sites config not found: {config_path}"
            )

        with open(config_path) as f:
            data = json.load(f)

        sites = []
        for site_data in data.get("sites", []):
            # Convert URL string to HttpUrl
            site = DocumentationSite(**site_data)
            sites.append(site)

        return sites

    @staticmethod
    def merge_env_config(base_config: dict[str, Any]) -> dict[str, Any]:
        """Merge environment variables into base configuration.

        Supports nested configuration using double underscore delimiter.
        Example: AI_DOCS__QDRANT__URL -> config.qdrant.url
        """
        env_prefix = "AI_DOCS__"

        for key, value in os.environ.items():
            if not key.startswith(env_prefix):
                continue

            # Remove prefix and split by delimiter
            config_path = key[len(env_prefix) :].lower().split("__")

            # Navigate to the target location in config
            current = base_config
            for part in config_path[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]

            # Set the value
            final_key = config_path[-1]

            # Try to parse as JSON for complex types
            try:
                parsed_value = json.loads(value)
                current[final_key] = parsed_value
            except json.JSONDecodeError:
                # Handle boolean strings
                if value.lower() in ["true", "false"]:
                    current[final_key] = value.lower() == "true"
                # Handle numeric strings
                elif value.isdigit():
                    current[final_key] = int(value)
                elif value.replace(".", "").isdigit():
                    current[final_key] = float(value)
                else:
                    current[final_key] = value

        return base_config

    @staticmethod
    def load_config(
        config_file: Path | str | None = None,
        env_file: Path | str | None = None,
        include_env: bool = True,
        documentation_sites_file: Path | str | None = None,
    ) -> UnifiedConfig:
        """Load configuration from multiple sources with priority.

        Priority order (highest to lowest):
        1. Environment variables
        2. Config file (if provided)
        3. .env file (if provided or found)
        4. Default values

        Args:
            config_file: Path to configuration file (JSON, YAML, or TOML)
            env_file: Path to .env file (defaults to .env in current directory)
            include_env: Whether to include environment variables
            documentation_sites_file: Path to documentation sites JSON file

        Returns:
            Loaded configuration instance
        """
        # Load base configuration
        config_data = {}

        # Load from config file if provided
        if config_file:
            config = UnifiedConfig.load_from_file(config_file)
            config_data = config.model_dump()

        # Merge environment variables if enabled
        if include_env:
            config_data = ConfigLoader.merge_env_config(config_data)

        # Create configuration instance
        config = UnifiedConfig(**config_data)

        # Load documentation sites if file provided
        if documentation_sites_file:
            sites = ConfigLoader.load_documentation_sites(documentation_sites_file)
            config.documentation_sites = sites

        return config

    @staticmethod
    def create_example_config(output_path: Path | str, format: str = "json") -> None:
        """Create an example configuration file with all options."""
        # Create example configuration with some sample values
        example_config = UnifiedConfig(
            environment="development",
            debug=True,
            log_level="DEBUG",
            embedding_provider="openai",
            crawl_provider="crawl4ai",
        )

        # Add example documentation sites
        example_config.documentation_sites = [
            DocumentationSite(
                name="Example Documentation",
                url="https://docs.example.com",
                max_pages=100,
                priority="high",
                description="Example documentation site",
            ),
            DocumentationSite(
                name="Another Example",
                url="https://docs.another-example.com",
                max_pages=50,
                priority="medium",
                description="Another example site",
                exclude_patterns=["*/api/*", "*/internal/*"],
            ),
        ]

        # Save to file
        example_config.save_to_file(output_path, format=format)

    @staticmethod
    def create_env_template(output_path: Path | str) -> None:
        """Create a .env.example template file."""
        template = """# AI Documentation Vector DB Configuration
# Copy this file to .env and fill in your values

# Environment
AI_DOCS__ENVIRONMENT=development
AI_DOCS__DEBUG=false
AI_DOCS__LOG_LEVEL=INFO

# Provider Selection
AI_DOCS__EMBEDDING_PROVIDER=fastembed
AI_DOCS__CRAWL_PROVIDER=crawl4ai

# OpenAI Configuration
AI_DOCS__OPENAI__API_KEY=sk-your-openai-api-key
AI_DOCS__OPENAI__MODEL=text-embedding-3-small
AI_DOCS__OPENAI__DIMENSIONS=1536

# Firecrawl Configuration
AI_DOCS__FIRECRAWL__API_KEY=your-firecrawl-api-key

# Qdrant Configuration
AI_DOCS__QDRANT__URL=http://localhost:6333
AI_DOCS__QDRANT__API_KEY=
AI_DOCS__QDRANT__COLLECTION_NAME=documents

# Cache Configuration
AI_DOCS__CACHE__ENABLE_CACHING=true
AI_DOCS__CACHE__REDIS_URL=redis://localhost:6379
AI_DOCS__CACHE__TTL_EMBEDDINGS=86400
AI_DOCS__CACHE__TTL_CRAWL=3600
AI_DOCS__CACHE__TTL_QUERIES=7200

# Performance Settings
AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=10
AI_DOCS__PERFORMANCE__MAX_MEMORY_USAGE_MB=1000

# Security Settings
AI_DOCS__SECURITY__REQUIRE_API_KEYS=true
AI_DOCS__SECURITY__ENABLE_RATE_LIMITING=true
AI_DOCS__SECURITY__RATE_LIMIT_REQUESTS=100
"""

        output_path = Path(output_path)
        output_path.write_text(template.strip())

    @staticmethod
    def validate_config(config: UnifiedConfig) -> tuple[bool, list[str]]:
        """Validate configuration completeness and correctness.

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = config.validate_completeness()

        # Additional validation
        if config.environment == "production":
            if config.debug:
                issues.append("Debug mode should be disabled in production")
            if config.log_level == "DEBUG":
                issues.append("Log level should not be DEBUG in production")
            if not config.security.require_api_keys:
                issues.append("API keys should be required in production")

        # Check for test/example values
        if config.openai.api_key and "your-" in config.openai.api_key:
            issues.append("OpenAI API key appears to be a placeholder")
        if config.firecrawl.api_key and "your-" in config.firecrawl.api_key:
            issues.append("Firecrawl API key appears to be a placeholder")

        return len(issues) == 0, issues
