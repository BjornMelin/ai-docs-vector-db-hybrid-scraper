"""Configuration schema generation and export utilities.

This module provides tools to generate JSON Schema, OpenAPI Schema,
and other schema formats from the Pydantic configuration models.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import jsonschema2md
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
                json_schema["examples"] = [
                    "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                ]
                json_schema["description"] = (
                    json_schema.get("description", "")
                    + " WARNING: Never expose real API keys in examples or templates!"
                )
            elif field_name == "qdrant_url":
                json_schema["examples"] = [
                    "http://localhost:6333",
                    "https://qdrant.example.com",
                ]
            elif field_name == "redis_url":
                json_schema["examples"] = [
                    "redis://localhost:6379",
                    "redis://:CHANGEME_REDIS_PASSWORD@redis.example.com:6380/0",
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

        # Convert $defs to definitions for compatibility with json-schema-to-typescript
        if "$defs" in schema:
            schema["definitions"] = schema.pop("$defs")

        # Update all $ref pointers to use definitions instead of $defs
        def convert_refs(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if (
                        key == "$ref"
                        and isinstance(value, str)
                        and value.startswith("#/$defs/")
                    ):
                        obj[key] = value.replace("#/$defs/", "#/definitions/")
                    else:
                        convert_refs(value)
            elif isinstance(obj, list):
                for item in obj:
                    convert_refs(item)

        convert_refs(schema)

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
                        "api_key": "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                        "model": "text-embedding-3-small",
                    },
                }
            ]

        return schema

    @staticmethod
    def generate_typescript_types(schema: dict[str, Any] | None = None) -> str:
        """Generate TypeScript type definitions from JSON Schema using json-schema-to-typescript.

        Args:
            schema: JSON Schema (generated if not provided)

        Returns:
            TypeScript type definitions as string

        Raises:
            RuntimeError: If TypeScript generation fails
        """
        if schema is None:
            schema = ConfigSchemaGenerator.generate_json_schema()

        try:
            # Try using npx to run json-schema-to-typescript
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as temp_file:
                json.dump(schema, temp_file, indent=2)
                temp_file.flush()

                # Use npx to run json-schema-to-typescript
                result = subprocess.run(
                    [
                        "npx",
                        "--package=json-schema-to-typescript",
                        "json2ts",
                        temp_file.name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )

                # Clean up temp file
                Path(temp_file.name).unlink()

                if result.returncode == 0:
                    # Add header comment to generated TypeScript
                    header = (
                        "// Auto-generated TypeScript types for AI Documentation Vector DB Configuration\n"
                        "// Generated using json-schema-to-typescript\n\n"
                    )
                    return header + result.stdout
                else:
                    raise RuntimeError(f"TypeScript generation failed: {result.stderr}")

        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Fallback to manual generation if json-schema-to-typescript is not available
            return ConfigSchemaGenerator._generate_typescript_fallback(schema)

    @staticmethod
    def _generate_typescript_fallback(schema: dict[str, Any]) -> str:
        """Simple fallback TypeScript generation when automated tools fail.

        Args:
            schema: JSON Schema

        Returns:
            Basic TypeScript type definitions as string
        """
        return """// Auto-generated TypeScript types for AI Documentation Vector DB Configuration
// Generated using simplified fallback (automated tools unavailable)

export interface UnifiedConfig {
  [key: string]: any;
}

// Note: Install 'json-schema-to-typescript' globally for detailed type generation:
// npm install -g json-schema-to-typescript
"""

    @staticmethod
    def generate_markdown_docs(schema: dict[str, Any] | None = None) -> str:
        """Generate Markdown documentation from JSON Schema using jsonschema2md.

        Args:
            schema: JSON Schema (generated if not provided)

        Returns:
            Markdown documentation as string
        """
        if schema is None:
            schema = ConfigSchemaGenerator.generate_json_schema()

        try:
            # Use jsonschema2md to generate comprehensive documentation
            parser = jsonschema2md.Parser(
                examples_as_yaml=False,
                show_examples="all",
                header_level=1,
            )

            # Generate markdown lines from the schema
            md_lines = parser.parse_schema(schema)
            generated_content = "".join(md_lines)

            # Add custom header and additional sections
            header_content = [
                "# AI Documentation Vector DB Configuration Schema",
                "",
                "This document describes the complete configuration schema for the AI Documentation Vector DB system.",
                "Generated automatically from the Pydantic configuration models.",
                "",
                "## Environment Variables",
                "",
                "All configuration values can be set via environment variables using the `AI_DOCS__` prefix and double underscore (`__`) for nested values:",
                "",
                "```bash",
                "# Set environment",
                "export AI_DOCS__ENVIRONMENT=production",
                "",
                "# Set nested values",
                "export AI_DOCS__OPENAI__API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                "export AI_DOCS__CACHE__REDIS_URL=redis://redis.example.com:6379",
                "```",
                "",
                "## Configuration Schema",
                "",
            ]

            return "\n".join(header_content) + generated_content

        except Exception:
            # Fallback to manual generation if jsonschema2md fails
            return ConfigSchemaGenerator._generate_markdown_fallback(schema)

    @staticmethod
    def _generate_markdown_fallback(schema: dict[str, Any]) -> str:
        """Simple fallback Markdown generation when automated tools fail.

        Args:
            schema: JSON Schema

        Returns:
            Basic Markdown documentation as string
        """
        return """# AI Documentation Vector DB Configuration Schema

This document describes the complete configuration schema for the AI Documentation Vector DB system.
Generated using simplified fallback (automated tools unavailable).

## Configuration Properties

The configuration schema includes comprehensive settings for all system components.

## Environment Variables

All configuration values can be set via environment variables using the `AI_DOCS__` prefix:

```bash
# Set environment
export AI_DOCS__ENVIRONMENT=production

# Set nested values
export AI_DOCS__OPENAI__API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export AI_DOCS__CACHE__REDIS_URL=redis://redis.example.com:6379
```

Note: Install 'jsonschema2md' for detailed documentation generation:
```bash
pip install jsonschema2md
```
"""

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
