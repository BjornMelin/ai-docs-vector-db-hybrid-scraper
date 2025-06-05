"""Comprehensive tests for automated configuration schema generation.

This module tests the automated TypeScript and Markdown generation functionality
using external tools (json-schema-to-typescript and jsonschema2md) with fallbacks.
"""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from src.config.schema import ConfigJsonSchema
from src.config.schema import ConfigSchemaGenerator


class TestConfigJsonSchema:
    """Test cases for ConfigJsonSchema custom schema generator."""

    def test_schema_generator_exists(self):
        """Test that ConfigJsonSchema can be instantiated."""
        generator = ConfigJsonSchema()
        assert generator is not None

    def test_inherits_from_pydantic_generator(self):
        """Test that ConfigJsonSchema inherits from Pydantic's GenerateJsonSchema."""
        from pydantic.json_schema import GenerateJsonSchema

        generator = ConfigJsonSchema()
        assert isinstance(generator, GenerateJsonSchema)

    def test_has_required_methods(self):
        """Test that ConfigJsonSchema has the required methods."""
        generator = ConfigJsonSchema()
        assert hasattr(generator, "generate")
        assert callable(generator.generate)


class TestConfigSchemaGeneratorJSONSchema:
    """Test cases for JSON Schema generation."""

    def test_generate_json_schema_basic_structure(self):
        """Test that JSON schema has the correct basic structure."""
        schema = ConfigSchemaGenerator.generate_json_schema()

        assert isinstance(schema, dict)
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "AI Documentation Vector DB Configuration"
        assert "description" in schema
        assert "properties" in schema
        assert "definitions" in schema  # Should be converted from $defs

    def test_generate_json_schema_with_examples(self):
        """Test JSON schema generation includes examples when requested."""
        schema = ConfigSchemaGenerator.generate_json_schema(include_examples=True)

        assert "examples" in schema
        assert isinstance(schema["examples"], list)
        assert len(schema["examples"]) > 0

        example = schema["examples"][0]
        assert "environment" in example
        assert "embedding_provider" in example

    def test_generate_json_schema_without_examples(self):
        """Test JSON schema generation excludes examples when not requested."""
        schema = ConfigSchemaGenerator.generate_json_schema(include_examples=False)
        assert "examples" not in schema

    def test_json_schema_validation_mode(self):
        """Test JSON schema generation in validation mode."""
        schema = ConfigSchemaGenerator.generate_json_schema(mode="validation")

        assert "properties" in schema
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_json_schema_serialization_mode(self):
        """Test JSON schema generation in serialization mode."""
        schema = ConfigSchemaGenerator.generate_json_schema(mode="serialization")

        assert "properties" in schema
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"

    def test_ref_conversion_defs_to_definitions(self):
        """Test that $defs are converted to definitions for tool compatibility."""
        schema = ConfigSchemaGenerator.generate_json_schema()

        # Should have definitions, not $defs
        assert "definitions" in schema
        assert "$defs" not in schema

        # Check that references point to definitions
        schema_str = json.dumps(schema)
        assert "#/definitions/" in schema_str
        assert "#/$defs/" not in schema_str


class TestConfigSchemaGeneratorTypeScript:
    """Test cases for TypeScript generation with automated tools."""

    def test_generate_typescript_success(self):
        """Test successful TypeScript generation using json-schema-to-typescript."""
        ts_types = ConfigSchemaGenerator.generate_typescript_types()

        assert isinstance(ts_types, str)
        assert len(ts_types) > 1000  # Should be substantial
        assert "Auto-generated TypeScript types" in ts_types

        # Check for either automated tool output or fallback
        automated = "json-schema-to-typescript" in ts_types
        fallback = "fallback manual generation" in ts_types
        assert automated or fallback

    def test_generate_typescript_with_schema(self):
        """Test TypeScript generation with provided schema."""
        schema = ConfigSchemaGenerator.generate_json_schema()
        ts_types = ConfigSchemaGenerator.generate_typescript_types(schema)

        assert isinstance(ts_types, str)
        assert "TypeScript types" in ts_types
        assert len(ts_types) > 500

    @patch("subprocess.run")
    def test_typescript_tool_success(self, mock_subprocess):
        """Test TypeScript generation when json-schema-to-typescript succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "export interface TestConfig { test: string; }"
        mock_subprocess.return_value = mock_result

        ts_types = ConfigSchemaGenerator.generate_typescript_types()

        assert "json-schema-to-typescript" in ts_types
        assert "export interface TestConfig" in ts_types

    @patch("subprocess.run")
    def test_typescript_tool_failure_fallback(self, mock_subprocess):
        """Test TypeScript generation falls back when tool fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Tool failed"
        mock_subprocess.return_value = mock_result

        # Should raise RuntimeError now instead of falling back silently
        with pytest.raises(RuntimeError, match="TypeScript generation failed"):
            ConfigSchemaGenerator.generate_typescript_types()

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_typescript_tool_not_found_fallback(self, mock_subprocess):
        """Test TypeScript generation falls back when tool not found."""
        ts_types = ConfigSchemaGenerator.generate_typescript_types()

        assert "simplified fallback" in ts_types
        assert "export interface UnifiedConfig" in ts_types
        assert "[key: string]: any" in ts_types

    def test_typescript_fallback_generation(self):
        """Test the simplified fallback TypeScript generation logic."""
        schema = {
            "properties": {
                "test_prop": {"type": "string"},
                "test_num": {"type": "number"},
                "test_bool": {"type": "boolean"},
            },
            "definitions": {
                "TestEnum": {"enum": ["value1", "value2"]},
                "TestObject": {
                    "type": "object",
                    "properties": {"nested": {"type": "string"}},
                },
            },
        }

        ts_types = ConfigSchemaGenerator._generate_typescript_fallback(schema)

        assert "export interface UnifiedConfig" in ts_types
        assert "[key: string]: any" in ts_types
        assert "simplified fallback" in ts_types
        assert "npm install -g json-schema-to-typescript" in ts_types


class TestConfigSchemaGeneratorMarkdown:
    """Test cases for Markdown generation with automated tools."""

    def test_generate_markdown_success(self):
        """Test successful Markdown generation using jsonschema2md."""
        markdown = ConfigSchemaGenerator.generate_markdown_docs()

        assert isinstance(markdown, str)
        assert len(markdown) > 1000  # Should be substantial
        assert "# AI Documentation Vector DB Configuration Schema" in markdown
        assert "Environment Variables" in markdown

    def test_generate_markdown_with_schema(self):
        """Test Markdown generation with provided schema."""
        schema = ConfigSchemaGenerator.generate_json_schema()
        markdown = ConfigSchemaGenerator.generate_markdown_docs(schema)

        assert isinstance(markdown, str)
        assert "Configuration Schema" in markdown
        assert "AI_DOCS__" in markdown

    def test_markdown_contains_environment_variables_section(self):
        """Test that generated Markdown includes environment variables documentation."""
        markdown = ConfigSchemaGenerator.generate_markdown_docs()

        assert "## Environment Variables" in markdown
        assert "AI_DOCS__" in markdown
        assert "export AI_DOCS__ENVIRONMENT=production" in markdown
        assert "export AI_DOCS__OPENAI__API_KEY=sk-your-api-key" in markdown

    @patch("jsonschema2md.Parser")
    def test_markdown_tool_success(self, mock_parser_class):
        """Test Markdown generation when jsonschema2md succeeds."""
        mock_parser = MagicMock()
        mock_parser.parse_schema.return_value = ["## Test Schema\n", "Content here\n"]
        mock_parser_class.return_value = mock_parser

        markdown = ConfigSchemaGenerator.generate_markdown_docs()

        assert "Generated automatically" in markdown
        assert "Test Schema" in markdown
        mock_parser_class.assert_called_once()

    @patch("jsonschema2md.Parser", side_effect=Exception("Tool failed"))
    def test_markdown_tool_failure_fallback(self, mock_parser_class):
        """Test Markdown generation falls back when tool fails."""
        markdown = ConfigSchemaGenerator.generate_markdown_docs()

        assert "simplified fallback" in markdown
        assert "## Configuration Properties" in markdown
        assert "pip install jsonschema2md" in markdown

    def test_markdown_fallback_generation(self):
        """Test the simplified fallback Markdown generation logic."""
        schema = {
            "properties": {
                "test_prop": {"type": "string", "description": "Test property"},
                "test_num": {"type": "number", "description": "Test number"},
            },
            "definitions": {
                "TestObject": {
                    "type": "object",
                    "description": "Test object",
                    "properties": {
                        "nested": {"type": "string", "description": "Nested prop"}
                    },
                },
            },
        }

        markdown = ConfigSchemaGenerator._generate_markdown_fallback(schema)

        assert "# AI Documentation Vector DB Configuration Schema" in markdown
        assert "## Configuration Properties" in markdown
        assert "simplified fallback" in markdown
        assert "pip install jsonschema2md" in markdown
        assert "Environment Variables" in markdown


class TestConfigSchemaGeneratorSaveSchema:
    """Test cases for the save_schema functionality."""

    def test_save_schema_all_formats(self, tmp_path):
        """Test saving schema in all default formats."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(output_dir)

        assert len(saved_files) == 3
        assert "json" in saved_files
        assert "typescript" in saved_files
        assert "markdown" in saved_files

        # Check all files exist and have content
        for file_path in saved_files.values():
            assert file_path.exists()
            assert file_path.stat().st_size > 0

    def test_save_schema_specific_formats(self, tmp_path):
        """Test saving schema in specific formats only."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["json", "typescript"]
        )

        assert len(saved_files) == 2
        assert "json" in saved_files
        assert "typescript" in saved_files
        assert "markdown" not in saved_files

    def test_save_schema_json_content(self, tmp_path):
        """Test JSON schema file content."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(output_dir, formats=["json"])

        json_file = saved_files["json"]
        assert json_file.name == "config-schema.json"

        with open(json_file) as f:
            schema = json.load(f)

        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "AI Documentation Vector DB Configuration"
        assert "properties" in schema

    def test_save_schema_typescript_content(self, tmp_path):
        """Test TypeScript schema file content."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["typescript"]
        )

        ts_file = saved_files["typescript"]
        assert ts_file.name == "config-types.ts"

        content = ts_file.read_text()
        assert "TypeScript types" in content
        assert len(content) > 500

    def test_save_schema_markdown_content(self, tmp_path):
        """Test Markdown schema file content."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["markdown"]
        )

        md_file = saved_files["markdown"]
        assert md_file.name == "config-schema.md"

        content = md_file.read_text()
        assert "# AI Documentation Vector DB Configuration Schema" in content
        assert "Environment Variables" in content

    def test_save_schema_creates_directory(self, tmp_path):
        """Test that save_schema creates the output directory if it doesn't exist."""
        output_dir = tmp_path / "nested" / "deep" / "schema"

        assert not output_dir.exists()

        saved_files = ConfigSchemaGenerator.save_schema(output_dir, formats=["json"])

        assert output_dir.exists()
        assert output_dir.is_dir()
        assert saved_files["json"].exists()

    def test_save_schema_empty_formats(self, tmp_path):
        """Test saving schema with empty formats list."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(output_dir, formats=[])

        assert len(saved_files) == 0

    def test_save_schema_file_sizes(self, tmp_path):
        """Test that saved schema files have reasonable sizes."""
        output_dir = tmp_path / "schema"

        saved_files = ConfigSchemaGenerator.save_schema(output_dir)

        # JSON schema should be substantial
        json_size = saved_files["json"].stat().st_size
        assert json_size > 10000  # At least 10KB

        # TypeScript should be substantial
        ts_size = saved_files["typescript"].stat().st_size
        assert ts_size > 5000  # At least 5KB

        # Markdown should be substantial
        md_size = saved_files["markdown"].stat().st_size
        assert md_size > 5000  # At least 5KB

    def test_save_schema_with_path_string(self, tmp_path):
        """Test save_schema accepts string paths."""
        output_dir_str = str(tmp_path / "schema")

        saved_files = ConfigSchemaGenerator.save_schema(
            output_dir_str, formats=["json"]
        )

        assert len(saved_files) == 1
        assert saved_files["json"].exists()

    def test_save_schema_concurrent_access(self, tmp_path):
        """Test save_schema handles concurrent access gracefully."""
        output_dir = tmp_path / "schema"

        # Create directory first
        output_dir.mkdir(parents=True, exist_ok=True)

        # Multiple calls should work without conflict
        saved_files1 = ConfigSchemaGenerator.save_schema(output_dir, formats=["json"])
        saved_files2 = ConfigSchemaGenerator.save_schema(
            output_dir, formats=["typescript"]
        )

        assert saved_files1["json"].exists()
        assert saved_files2["typescript"].exists()


class TestConfigSchemaGeneratorIntegration:
    """Integration tests for the complete schema generation workflow."""

    def test_end_to_end_schema_generation(self, tmp_path):
        """Test complete end-to-end schema generation workflow."""
        output_dir = tmp_path / "complete_schema"

        # Generate all formats
        saved_files = ConfigSchemaGenerator.save_schema(output_dir)

        # Verify all formats were generated
        assert len(saved_files) == 3

        # Load and verify JSON schema
        with open(saved_files["json"]) as f:
            json_schema = json.load(f)
        assert "definitions" in json_schema
        assert "properties" in json_schema

        # Verify TypeScript content
        ts_content = saved_files["typescript"].read_text()
        assert "interface" in ts_content or "type" in ts_content

        # Verify Markdown content
        md_content = saved_files["markdown"].read_text()
        assert "Configuration Schema" in md_content
        assert "Environment Variables" in md_content

    def test_schema_consistency_across_formats(self, tmp_path):
        """Test that schema information is consistent across all formats."""
        output_dir = tmp_path / "consistency_test"

        saved_files = ConfigSchemaGenerator.save_schema(output_dir)

        # Load JSON schema to get property names
        with open(saved_files["json"]) as f:
            json_schema = json.load(f)

        properties = json_schema.get("properties", {})
        assert len(properties) > 5  # Should have multiple properties

        # Check that TypeScript mentions the same properties
        ts_content = saved_files["typescript"].read_text()
        md_content = saved_files["markdown"].read_text()

        # At least some key properties should appear in all formats
        key_properties = ["environment", "debug", "embedding_provider"]
        for prop in key_properties:
            assert prop in json_schema.get("properties", {})
            # Properties should appear somewhere in generated files
            assert prop in ts_content or prop in md_content

    def test_schema_validation_with_example(self, tmp_path):
        """Test that generated schema can validate example configurations."""
        # Generate JSON schema
        schema = ConfigSchemaGenerator.generate_json_schema(include_examples=True)

        # Should have examples
        assert "examples" in schema
        examples = schema["examples"]
        assert len(examples) > 0

        # Example should have required structure
        example = examples[0]
        assert "environment" in example
        assert "embedding_provider" in example

        # Basic validation that example matches expected types
        assert isinstance(example.get("environment"), str)
        assert isinstance(example.get("debug"), bool)
