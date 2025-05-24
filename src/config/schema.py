"""Configuration schema generation and export utilities.

This module provides tools to generate JSON Schema, OpenAPI Schema,
and other schema formats from the Pydantic configuration models.
"""

import json
from pathlib import Path
from typing import Any

from pydantic.json_schema import GenerateJsonSchema
from pydantic.json_schema import JsonSchemaValue

from .models import UnifiedConfig


class ConfigJsonSchema(GenerateJsonSchema):
    """Custom JSON Schema generator for configuration."""

    def generate_field_schema(
        self, schema_or_field: Any, mode: str = "validation", **kwargs: Any
    ) -> JsonSchemaValue:
        """Generate field schema with enhanced descriptions."""
        json_schema = super().generate_field_schema(
            schema_or_field, mode=mode, **kwargs
        )

        # Add custom properties for better documentation
        if isinstance(json_schema, dict):
            # Add examples for specific fields
            field_name = kwargs.get("field_name", "")
            if field_name == "openai_api_key":
                json_schema["examples"] = ["sk-your-openai-api-key-here"]
            elif field_name == "qdrant_url":
                json_schema["examples"] = [
                    "http://localhost:6333",
                    "https://qdrant.example.com",
                ]
            elif field_name == "redis_url":
                json_schema["examples"] = [
                    "redis://localhost:6379",
                    "redis://:password@redis.example.com:6380/0",
                ]

        return json_schema


class ConfigSchemaGenerator:
    """Generate various schema formats for the configuration."""

    @staticmethod
    def generate_json_schema(
        include_defaults: bool = True,
        include_examples: bool = True,
        mode: str = "validation",
    ) -> dict[str, Any]:
        """Generate JSON Schema for the configuration.

        Args:
            include_defaults: Include default values in schema
            include_examples: Include example values
            mode: Schema mode ('validation' or 'serialization')

        Returns:
            JSON Schema dictionary
        """
        schema = UnifiedConfig.model_json_schema(
            schema_generator=ConfigJsonSchema,
            mode=mode,
            by_alias=False,
            ref_template="#/definitions/{model}",
        )

        # Add schema metadata
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["title"] = "AI Documentation Vector DB Configuration"
        schema["description"] = (
            "Complete configuration schema for the AI Documentation Vector DB system"
        )

        # Add custom schema properties
        if include_examples:
            schema["examples"] = [
                {
                    "environment": "production",
                    "debug": False,
                    "embedding_provider": "openai",
                    "openai": {
                        "api_key": "sk-your-openai-api-key-here",
                        "model": "text-embedding-3-small",
                    },
                }
            ]

        return schema

    @staticmethod
    def generate_typescript_types(schema: dict[str, Any] | None = None) -> str:  # noqa: PLR0915
        """Generate TypeScript type definitions from schema.

        Args:
            schema: JSON Schema (generated if not provided)

        Returns:
            TypeScript type definitions as string
        """
        if schema is None:
            schema = ConfigSchemaGenerator.generate_json_schema()

        # Simple TypeScript generation (could use json-schema-to-typescript for complex cases)
        lines = []
        lines.append(
            "// Auto-generated TypeScript types for AI Documentation Vector DB Configuration"
        )
        lines.append("")

        # Generate enums
        lines.append("export enum Environment {")
        lines.append('  DEVELOPMENT = "development",')
        lines.append('  TESTING = "testing",')
        lines.append('  PRODUCTION = "production"')
        lines.append("}")
        lines.append("")

        lines.append("export enum LogLevel {")
        lines.append('  DEBUG = "DEBUG",')
        lines.append('  INFO = "INFO",')
        lines.append('  WARNING = "WARNING",')
        lines.append('  ERROR = "ERROR",')
        lines.append('  CRITICAL = "CRITICAL"')
        lines.append("}")
        lines.append("")

        lines.append("export enum EmbeddingProvider {")
        lines.append('  OPENAI = "openai",')
        lines.append('  FASTEMBED = "fastembed"')
        lines.append("}")
        lines.append("")

        lines.append("export enum CrawlProvider {")
        lines.append('  CRAWL4AI = "crawl4ai",')
        lines.append('  FIRECRAWL = "firecrawl"')
        lines.append("}")
        lines.append("")

        # Generate main interface
        lines.append("export interface UnifiedConfig {")
        lines.append("  environment?: Environment;")
        lines.append("  debug?: boolean;")
        lines.append("  log_level?: LogLevel;")
        lines.append("  app_name?: string;")
        lines.append("  version?: string;")
        lines.append("  embedding_provider?: EmbeddingProvider;")
        lines.append("  crawl_provider?: CrawlProvider;")
        lines.append("  cache?: CacheConfig;")
        lines.append("  qdrant?: QdrantConfig;")
        lines.append("  openai?: OpenAIConfig;")
        lines.append("  fastembed?: FastEmbedConfig;")
        lines.append("  firecrawl?: FirecrawlConfig;")
        lines.append("  crawl4ai?: Crawl4AIConfig;")
        lines.append("  chunking?: ChunkingConfig;")
        lines.append("  performance?: PerformanceConfig;")
        lines.append("  security?: SecurityConfig;")
        lines.append("  documentation_sites?: DocumentationSite[];")
        lines.append("  data_dir?: string;")
        lines.append("  cache_dir?: string;")
        lines.append("  logs_dir?: string;")
        lines.append("}")
        lines.append("")

        # Generate sub-interfaces (simplified)
        lines.append("export interface CacheConfig {")
        lines.append("  enable_caching?: boolean;")
        lines.append("  enable_local_cache?: boolean;")
        lines.append("  enable_redis_cache?: boolean;")
        lines.append("  redis_url?: string;")
        lines.append("  ttl_embeddings?: number;")
        lines.append("  ttl_crawl?: number;")
        lines.append("  ttl_queries?: number;")
        lines.append("  local_max_size?: number;")
        lines.append("  local_max_memory_mb?: number;")
        lines.append("}")
        lines.append("")

        lines.append("export interface QdrantConfig {")
        lines.append("  url?: string;")
        lines.append("  api_key?: string | null;")
        lines.append("  timeout?: number;")
        lines.append("  prefer_grpc?: boolean;")
        lines.append("  collection_name?: string;")
        lines.append("  batch_size?: number;")
        lines.append("  max_retries?: number;")
        lines.append("}")
        lines.append("")

        lines.append("export interface OpenAIConfig {")
        lines.append("  api_key?: string | null;")
        lines.append("  model?: string;")
        lines.append("  dimensions?: number;")
        lines.append("  batch_size?: number;")
        lines.append("}")

        return "\n".join(lines)

    @staticmethod
    def generate_markdown_docs(schema: dict[str, Any] | None = None) -> str:  # noqa: PLR0915
        """Generate Markdown documentation from schema.

        Args:
            schema: JSON Schema (generated if not provided)

        Returns:
            Markdown documentation as string
        """
        if schema is None:
            schema = ConfigSchemaGenerator.generate_json_schema()

        lines = []
        lines.append("# AI Documentation Vector DB Configuration Schema")
        lines.append("")
        lines.append(
            "This document describes the complete configuration schema for the AI Documentation Vector DB system."
        )
        lines.append("")
        lines.append("## Configuration Structure")
        lines.append("")

        # Generate table of contents
        lines.append("### Table of Contents")
        lines.append("")
        lines.append("- [Root Configuration](#root-configuration)")
        lines.append("- [Cache Configuration](#cache-configuration)")
        lines.append("- [Qdrant Configuration](#qdrant-configuration)")
        lines.append("- [OpenAI Configuration](#openai-configuration)")
        lines.append("- [FastEmbed Configuration](#fastembed-configuration)")
        lines.append("- [Firecrawl Configuration](#firecrawl-configuration)")
        lines.append("- [Crawl4AI Configuration](#crawl4ai-configuration)")
        lines.append("- [Chunking Configuration](#chunking-configuration)")
        lines.append("- [Performance Configuration](#performance-configuration)")
        lines.append("- [Security Configuration](#security-configuration)")
        lines.append(
            "- [Documentation Site Configuration](#documentation-site-configuration)"
        )
        lines.append("")

        # Root configuration
        lines.append("## Root Configuration")
        lines.append("")
        lines.append("The root configuration object contains the following properties:")
        lines.append("")
        lines.append("| Property | Type | Default | Description |")
        lines.append("|----------|------|---------|-------------|")
        lines.append(
            "| `environment` | `string` | `development` | Application environment (development, testing, production) |"
        )
        lines.append("| `debug` | `boolean` | `false` | Enable debug mode |")
        lines.append(
            "| `log_level` | `string` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |"
        )
        lines.append(
            "| `app_name` | `string` | `AI Documentation Vector DB` | Application name |"
        )
        lines.append("| `version` | `string` | `0.1.0` | Application version |")
        lines.append(
            "| `embedding_provider` | `string` | `fastembed` | Embedding provider (openai, fastembed) |"
        )
        lines.append(
            "| `crawl_provider` | `string` | `crawl4ai` | Crawl provider (crawl4ai, firecrawl) |"
        )
        lines.append("")

        # Add more sections...
        lines.append("## Cache Configuration")
        lines.append("")
        lines.append("The `cache` object configures the caching system:")
        lines.append("")
        lines.append("| Property | Type | Default | Description |")
        lines.append("|----------|------|---------|-------------|")
        lines.append(
            "| `enable_caching` | `boolean` | `true` | Enable caching system |"
        )
        lines.append(
            "| `enable_local_cache` | `boolean` | `true` | Enable local in-memory cache |"
        )
        lines.append(
            "| `enable_redis_cache` | `boolean` | `true` | Enable Redis cache |"
        )
        lines.append(
            "| `redis_url` | `string` | `redis://localhost:6379` | Redis connection URL |"
        )
        lines.append(
            "| `ttl_embeddings` | `integer` | `86400` | Embeddings cache TTL in seconds (24 hours) |"
        )
        lines.append(
            "| `ttl_crawl` | `integer` | `3600` | Crawl cache TTL in seconds (1 hour) |"
        )
        lines.append(
            "| `ttl_queries` | `integer` | `7200` | Query cache TTL in seconds (2 hours) |"
        )
        lines.append("")

        # Environment variables section
        lines.append("## Environment Variables")
        lines.append("")
        lines.append(
            "All configuration values can be set via environment variables using the `AI_DOCS__` prefix and double underscore (`__`) for nested values:"
        )
        lines.append("")
        lines.append("```bash")
        lines.append("# Set environment")
        lines.append("export AI_DOCS__ENVIRONMENT=production")
        lines.append("")
        lines.append("# Set nested values")
        lines.append("export AI_DOCS__OPENAI__API_KEY=sk-your-api-key")
        lines.append("export AI_DOCS__CACHE__REDIS_URL=redis://redis.example.com:6379")
        lines.append("```")
        lines.append("")

        return "\n".join(lines)

    @staticmethod
    def save_schema(
        output_dir: Path | str, formats: list[str] | None = None
    ) -> dict[str, Path]:
        """Save schema in multiple formats.

        Args:
            output_dir: Directory to save schema files
            formats: List of formats to generate (default: all)

        Returns:
            Dictionary mapping format names to output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["json", "typescript", "markdown"]

        saved_files = {}

        # Generate base schema
        schema = ConfigSchemaGenerator.generate_json_schema()

        # Save JSON Schema
        if "json" in formats:
            json_path = output_dir / "config-schema.json"
            with open(json_path, "w") as f:
                json.dump(schema, f, indent=2)
            saved_files["json"] = json_path

        # Save TypeScript types
        if "typescript" in formats:
            ts_path = output_dir / "config-types.ts"
            ts_content = ConfigSchemaGenerator.generate_typescript_types(schema)
            ts_path.write_text(ts_content)
            saved_files["typescript"] = ts_path

        # Save Markdown documentation
        if "markdown" in formats:
            md_path = output_dir / "config-schema.md"
            md_content = ConfigSchemaGenerator.generate_markdown_docs(schema)
            md_path.write_text(md_content)
            saved_files["markdown"] = md_path

        return saved_files
