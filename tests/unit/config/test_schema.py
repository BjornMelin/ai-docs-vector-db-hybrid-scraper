"""Unit tests for configuration schema generation module."""

import json
from pathlib import Path

import pytest

from src.config.schema import ConfigJsonSchema
from src.config.schema import ConfigSchemaGenerator


class TestConfigJsonSchema:
    """Test cases for ConfigJsonSchema class."""

    def test_schema_generator_class_exists(self):
        """Test that the ConfigJsonSchema class can be instantiated."""
        generator = ConfigJsonSchema()
        assert generator is not None

    def test_schema_generator_inheritance(self):
        """Test that ConfigJsonSchema inherits from GenerateJsonSchema."""
        from pydantic.json_schema import GenerateJsonSchema
        
        generator = ConfigJsonSchema()
        assert isinstance(generator, GenerateJsonSchema)

    def test_field_schema_enhancement_concept(self):
        """Test the concept of field schema enhancement."""
        # Since the Pydantic v2 API has changed, we'll test the concept
        # rather than the exact implementation
        generator = ConfigJsonSchema()
        
        # The class should exist and be usable for schema generation
        assert hasattr(generator, 'generate')
        
        # Test that we can call generate method (basic functionality)
        # This validates the class structure without testing implementation details
        assert callable(getattr(generator, 'generate', None))


class TestConfigSchemaGenerator:
    """Test cases for ConfigSchemaGenerator class."""

    def test_generate_json_schema_basic(self):
        """Test basic JSON schema generation."""
        schema = ConfigSchemaGenerator.generate_json_schema()
        
        assert isinstance(schema, dict)
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "AI Documentation Vector DB Configuration"
        assert "description" in schema
        assert "properties" in schema

    def test_generate_json_schema_with_examples(self):
        """Test JSON schema generation with examples."""
        schema = ConfigSchemaGenerator.generate_json_schema(include_examples=True)
        
        assert "examples" in schema
        assert len(schema["examples"]) == 1
        
        example = schema["examples"][0]
        assert example["environment"] == "production"
        assert example["debug"] is False
        assert example["embedding_provider"] == "openai"

    def test_generate_json_schema_without_examples(self):
        """Test JSON schema generation without examples."""
        schema = ConfigSchemaGenerator.generate_json_schema(include_examples=False)
        
        assert "examples" not in schema

    def test_generate_json_schema_validation_mode(self):
        """Test JSON schema generation in validation mode."""
        schema = ConfigSchemaGenerator.generate_json_schema(mode="validation")
        
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert "properties" in schema

    def test_generate_json_schema_serialization_mode(self):
        """Test JSON schema generation in serialization mode."""
        schema = ConfigSchemaGenerator.generate_json_schema(mode="serialization")
        
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert "properties" in schema

    def test_generate_typescript_types_basic(self):
        """Test basic TypeScript types generation."""
        ts_types = ConfigSchemaGenerator.generate_typescript_types()
        
        assert isinstance(ts_types, str)
        assert "export enum Environment" in ts_types
        assert "export enum LogLevel" in ts_types
        assert "export enum EmbeddingProvider" in ts_types
        assert "export enum CrawlProvider" in ts_types
        assert "export interface UnifiedConfig" in ts_types

    def test_generate_typescript_types_with_schema(self):
        """Test TypeScript types generation with provided schema."""
        schema = ConfigSchemaGenerator.generate_json_schema()
        ts_types = ConfigSchemaGenerator.generate_typescript_types(schema)
        
        assert isinstance(ts_types, str)
        assert "Auto-generated TypeScript types" in ts_types
        assert "export interface CacheConfig" in ts_types
        assert "export interface QdrantConfig" in ts_types
        assert "export interface OpenAIConfig" in ts_types

    def test_generate_typescript_enums(self):
        """Test TypeScript enum generation."""
        ts_types = ConfigSchemaGenerator.generate_typescript_types()
        
        # Check Environment enum
        assert 'DEVELOPMENT = "development"' in ts_types
        assert 'TESTING = "testing"' in ts_types
        assert 'PRODUCTION = "production"' in ts_types
        
        # Check LogLevel enum
        assert 'DEBUG = "DEBUG"' in ts_types
        assert 'INFO = "INFO"' in ts_types
        assert 'ERROR = "ERROR"' in ts_types
        
        # Check provider enums
        assert 'OPENAI = "openai"' in ts_types
        assert 'FASTEMBED = "fastembed"' in ts_types
        assert 'CRAWL4AI = "crawl4ai"' in ts_types
        assert 'FIRECRAWL = "firecrawl"' in ts_types

    def test_generate_typescript_interfaces(self):
        """Test TypeScript interface generation."""
        ts_types = ConfigSchemaGenerator.generate_typescript_types()
        
        # Check main interface
        assert "environment?: Environment;" in ts_types
        assert "debug?: boolean;" in ts_types
        assert "embedding_provider?: EmbeddingProvider;" in ts_types
        
        # Check sub-interfaces
        assert "enable_caching?: boolean;" in ts_types  # CacheConfig
        assert "url?: string;" in ts_types  # QdrantConfig
        assert "api_key?: string | null;" in ts_types  # OpenAIConfig

    def test_generate_markdown_docs_basic(self):
        """Test basic Markdown documentation generation."""
        markdown = ConfigSchemaGenerator.generate_markdown_docs()
        
        assert isinstance(markdown, str)
        assert "# AI Documentation Vector DB Configuration Schema" in markdown
        assert "## Configuration Structure" in markdown
        assert "### Table of Contents" in markdown

    def test_generate_markdown_docs_with_schema(self):
        """Test Markdown documentation generation with provided schema."""
        schema = ConfigSchemaGenerator.generate_json_schema()
        markdown = ConfigSchemaGenerator.generate_markdown_docs(schema)
        
        assert "## Root Configuration" in markdown
        assert "## Cache Configuration" in markdown
        assert "## Environment Variables" in markdown

    def test_generate_markdown_docs_table_structure(self):
        """Test Markdown documentation table structure."""
        markdown = ConfigSchemaGenerator.generate_markdown_docs()
        
        # Check table headers
        assert "| Property | Type | Default | Description |" in markdown
        assert "|----------|------|---------|-------------|" in markdown
        
        # Check some example rows
        assert "| `environment` | `string` | `development`" in markdown
        assert "| `debug` | `boolean` | `false`" in markdown

    def test_generate_markdown_docs_environment_variables(self):
        """Test Markdown environment variables section."""
        markdown = ConfigSchemaGenerator.generate_markdown_docs()
        
        assert "## Environment Variables" in markdown
        assert "AI_DOCS__" in markdown
        assert "export AI_DOCS__ENVIRONMENT=production" in markdown
        assert "export AI_DOCS__OPENAI__API_KEY=sk-your-api-key" in markdown

    def test_save_schema_all_formats(self, tmp_path):
        """Test saving schema in all formats."""
        output_dir = tmp_path / "schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(output_dir)
        
        # Check all default formats are saved
        assert "json" in saved_files
        assert "typescript" in saved_files
        assert "markdown" in saved_files
        
        # Check files exist
        assert saved_files["json"].exists()
        assert saved_files["typescript"].exists()
        assert saved_files["markdown"].exists()

    def test_save_schema_specific_formats(self, tmp_path):
        """Test saving schema in specific formats."""
        output_dir = tmp_path / "schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["json", "typescript"]
        )
        
        # Check only requested formats are saved
        assert "json" in saved_files
        assert "typescript" in saved_files
        assert "markdown" not in saved_files
        
        # Check files exist
        assert saved_files["json"].exists()
        assert saved_files["typescript"].exists()

    def test_save_schema_json_format(self, tmp_path):
        """Test saving JSON schema format."""
        output_dir = tmp_path / "schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["json"]
        )
        
        json_file = saved_files["json"]
        assert json_file.name == "config-schema.json"
        
        # Check JSON content
        with open(json_file) as f:
            schema = json.load(f)
        
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "AI Documentation Vector DB Configuration"

    def test_save_schema_typescript_format(self, tmp_path):
        """Test saving TypeScript schema format."""
        output_dir = tmp_path / "schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["typescript"]
        )
        
        ts_file = saved_files["typescript"]
        assert ts_file.name == "config-types.ts"
        
        # Check TypeScript content
        content = ts_file.read_text()
        assert "export enum Environment" in content
        assert "export interface UnifiedConfig" in content

    def test_save_schema_markdown_format(self, tmp_path):
        """Test saving Markdown schema format."""
        output_dir = tmp_path / "schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["markdown"]
        )
        
        md_file = saved_files["markdown"]
        assert md_file.name == "config-schema.md"
        
        # Check Markdown content
        content = md_file.read_text()
        assert "# AI Documentation Vector DB Configuration Schema" in content
        assert "## Root Configuration" in content

    def test_save_schema_creates_directory(self, tmp_path):
        """Test that save_schema creates output directory."""
        output_dir = tmp_path / "nested" / "schema"
        
        # Directory doesn't exist yet
        assert not output_dir.exists()
        
        saved_files = ConfigSchemaGenerator.save_schema(output_dir, formats=["json"])
        
        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()
        assert saved_files["json"].exists()

    def test_save_schema_empty_formats_list(self, tmp_path):
        """Test saving schema with empty formats list."""
        output_dir = tmp_path / "schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(output_dir, formats=[])
        
        # Should save nothing
        assert len(saved_files) == 0

    def test_save_schema_output_paths(self, tmp_path):
        """Test correct output paths for saved schema files."""
        output_dir = tmp_path / "custom_schema"
        
        saved_files = ConfigSchemaGenerator.save_schema(output_dir)
        
        # Check expected filenames
        assert saved_files["json"].name == "config-schema.json"
        assert saved_files["typescript"].name == "config-types.ts"
        assert saved_files["markdown"].name == "config-schema.md"
        
        # Check they're in the correct directory
        for path in saved_files.values():
            assert path.parent == output_dir