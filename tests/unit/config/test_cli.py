"""Unit tests for configuration CLI module."""

import json
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

from click.testing import CliRunner
from src.config.cli import cli


class TestConfigCLI:
    """Test cases for configuration CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert (
            "AI Documentation Vector DB configuration management tool" in result.output
        )

    def test_create_example_json(self):
        """Test creating example configuration in JSON format."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ["create-example", "--format", "json", "--output", "test.json"]
            )

            assert result.exit_code == 0
            assert "Created example config at: test.json" in result.output
            assert Path("test.json").exists()

            # Verify JSON content
            with open("test.json") as f:
                config = json.load(f)
            assert config["environment"] == "development"
            assert config["debug"] is True

    def test_create_example_yaml(self):
        """Test creating example configuration in YAML format."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ["create-example", "--format", "yaml", "--output", "test.yaml"]
            )

            assert result.exit_code == 0
            assert "Created example config at: test.yaml" in result.output
            assert Path("test.yaml").exists()

    def test_create_example_default_output(self):
        """Test creating example configuration with default output."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["create-example"])

            assert result.exit_code == 0
            assert Path("config.example.json").exists()

    def test_create_env_template(self):
        """Test creating .env template."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(
                cli, ["create-env-template", "--output", "test.env"]
            )

            assert result.exit_code == 0
            assert "Created .env template at: test.env" in result.output
            assert Path("test.env").exists()

            # Verify template content
            content = Path("test.env").read_text()
            assert "AI_DOCS__ENVIRONMENT=development" in content
            assert "AI_DOCS__OPENAI__API_KEY=" in content

    def test_create_env_template_default_output(self):
        """Test creating .env template with default output."""
        with self.runner.isolated_filesystem():
            result = self.runner.invoke(cli, ["create-env-template"])

            assert result.exit_code == 0
            assert Path(".env.example").exists()

    @patch("src.config.cli.ConfigLoader.load_config")
    @patch("src.config.cli.ConfigLoader.validate_config")
    def test_validate_success(self, mock_validate, mock_load):
        """Test successful configuration validation."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_validate.return_value = (True, [])

        result = self.runner.invoke(cli, ["validate"])

        assert result.exit_code == 0
        assert "Configuration is valid!" in result.output
        mock_load.assert_called_once()
        mock_validate.assert_called_once_with(mock_config)

    @patch("src.config.cli.ConfigLoader.load_config")
    @patch("src.config.cli.ConfigLoader.validate_config")
    def test_validate_with_issues(self, mock_validate, mock_load):
        """Test configuration validation with issues."""
        mock_config = MagicMock()
        mock_load.return_value = mock_config
        mock_validate.return_value = (False, ["Issue 1", "Issue 2"])

        result = self.runner.invoke(cli, ["validate"])

        assert result.exit_code == 0
        assert "Configuration has issues:" in result.output
        assert "Issue 1" in result.output
        assert "Issue 2" in result.output

    @patch("src.config.cli.ConfigLoader.load_config")
    def test_validate_with_config_file(self, mock_load):
        """Test validation with specific config file."""
        with self.runner.isolated_filesystem():
            # Create test config file
            config_data = {"environment": "testing"}
            with open("test.json", "w") as f:
                json.dump(config_data, f)

            mock_config = MagicMock()
            mock_load.return_value = mock_config

            self.runner.invoke(cli, ["validate", "--config-file", "test.json"])

            mock_load.assert_called_once_with(
                config_file="test.json", env_file=None, include_env=True
            )

    @patch("src.config.cli.ConfigLoader.load_config")
    @patch("src.config.cli.ConfigLoader.validate_config")
    def test_validate_show_config(self, mock_validate, mock_load):
        """Test validation with show config option."""
        mock_config = MagicMock()
        mock_config.model_dump_json.return_value = '{"test": "config"}'
        mock_load.return_value = mock_config
        mock_validate.return_value = (True, [])

        result = self.runner.invoke(cli, ["validate", "--show-config"])

        assert result.exit_code == 0
        assert "Loaded Configuration:" in result.output

    @patch("src.config.cli.ConfigLoader.load_config")
    def test_validate_loading_error(self, mock_load):
        """Test validation with configuration loading error."""
        mock_load.side_effect = ValueError("Invalid config")

        result = self.runner.invoke(cli, ["validate"])

        assert result.exit_code == 1
        assert "Error loading configuration:" in result.output

    def test_convert_json_to_yaml(self):
        """Test converting configuration from JSON to YAML."""
        with self.runner.isolated_filesystem():
            # Create source JSON file
            config_data = {"environment": "testing", "debug": True}
            with open("source.json", "w") as f:
                json.dump(config_data, f)

            with patch("src.config.models.UnifiedConfig.load_from_file") as mock_load:
                mock_config = MagicMock()
                mock_load.return_value = mock_config

                result = self.runner.invoke(
                    cli,
                    [
                        "convert",
                        "source.json",
                        "target.yaml",
                        "--to-format",
                        "yaml",
                    ],
                )

                assert result.exit_code == 0
                assert (
                    "Converted source.json (json) → target.yaml (yaml)" in result.output
                )
                mock_load.assert_called_once()
                mock_config.save_to_file.assert_called_once_with(
                    Path("target.yaml"), format="yaml"
                )

    def test_convert_auto_detect_format(self):
        """Test converting with auto-detected input format."""
        with self.runner.isolated_filesystem():
            # Create source YAML file
            with open("source.yml", "w") as f:
                f.write("environment: testing\n")

            with patch("src.config.models.UnifiedConfig.load_from_file") as mock_load:
                mock_config = MagicMock()
                mock_load.return_value = mock_config

                result = self.runner.invoke(
                    cli,
                    ["convert", "source.yml", "target.json", "--to-format", "json"],
                )

                assert result.exit_code == 0
                assert (
                    "Converted source.yml (yaml) → target.json (json)" in result.output
                )

    def test_convert_unknown_format(self):
        """Test converting with unknown input format."""
        with self.runner.isolated_filesystem():
            with open("source.xml", "w") as f:
                f.write("<config></config>")

            result = self.runner.invoke(
                cli, ["convert", "source.xml", "target.json", "--to-format", "json"]
            )

            assert result.exit_code == 1
            assert "Could not auto-detect input format" in result.output

    def test_convert_env_format(self):
        """Test converting from .env format."""
        with self.runner.isolated_filesystem():
            with open(".env.test", "w") as f:
                f.write("AI_DOCS__DEBUG=true\n")

            with patch("src.config.cli.UnifiedConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config

                result = self.runner.invoke(
                    cli,
                    [
                        "convert",
                        ".env.test",
                        "config.json",
                        "--from-format",
                        "env",
                        "--to-format",
                        "json",
                    ],
                )

                assert result.exit_code == 0

    def test_convert_loading_error(self):
        """Test converting with file loading error."""
        with self.runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("invalid json")

            result = self.runner.invoke(
                cli, ["convert", "invalid.json", "target.yaml", "--to-format", "yaml"]
            )

            assert result.exit_code == 1
            assert "Error converting configuration:" in result.output

    @patch("src.config.cli.ConfigLoader.load_config")
    def test_show_providers_openai(self, mock_load):
        """Test showing providers with OpenAI configuration."""
        mock_config = MagicMock()
        mock_config.embedding_provider = "openai"
        mock_config.crawl_provider = "firecrawl"
        mock_config.get_active_providers.return_value = {
            "embedding": MagicMock(
                model="text-embedding-3-small", dimensions=1536, api_key="sk-test"
            ),
            "crawl": MagicMock(api_url="https://api.firecrawl.dev", api_key="fc-test"),
        }
        mock_load.return_value = mock_config

        result = self.runner.invoke(cli, ["show-providers"])

        assert result.exit_code == 0
        assert "Active Provider Configuration" in result.output
        assert "OpenAI" in result.output
        assert "Firecrawl" in result.output

    @patch("src.config.cli.ConfigLoader.load_config")
    def test_show_providers_fastembed(self, mock_load):
        """Test showing providers with FastEmbed configuration."""
        mock_config = MagicMock()
        mock_config.embedding_provider = "fastembed"
        mock_config.crawl_provider = "crawl4ai"
        mock_config.get_active_providers.return_value = {
            "embedding": MagicMock(model="BAAI/bge-small-en-v1.5", max_length=512),
            "crawl": MagicMock(
                browser_type="chromium", headless=True, max_concurrent_crawls=3
            ),
        }
        mock_load.return_value = mock_config

        result = self.runner.invoke(cli, ["show-providers"])

        assert result.exit_code == 0
        assert "FastEmbed" in result.output
        assert "Crawl4AI" in result.output

    @patch("src.config.cli.ConfigLoader.load_config")
    def test_show_providers_loading_error(self, mock_load):
        """Test show providers with configuration loading error."""
        mock_load.side_effect = ValueError("Invalid config")

        result = self.runner.invoke(cli, ["show-providers"])

        assert result.exit_code == 1
        assert "Error loading configuration:" in result.output

    @patch("src.config.cli.ConfigLoader.load_documentation_sites")
    @patch("src.config.models.UnifiedConfig.load_from_file")
    def test_migrate_sites(self, mock_load_config, mock_load_sites):
        """Test migrating documentation sites."""
        with self.runner.isolated_filesystem():
            # Create sites file
            with open("sites.json", "w") as f:
                json.dump(
                    {"sites": [{"name": "Test", "url": "https://example.com"}]}, f
                )

            # Create config file
            with open("config.json", "w") as f:
                json.dump({"environment": "development"}, f)

            mock_sites = [MagicMock(name="Test Site")]
            mock_load_sites.return_value = mock_sites
            mock_config = MagicMock()
            mock_load_config.return_value = mock_config

            result = self.runner.invoke(
                cli,
                [
                    "migrate-sites",
                    "--source",
                    "sites.json",
                    "--config-file",
                    "config.json",
                ],
            )

            assert result.exit_code == 0
            assert "Migrated 1 documentation sites" in result.output
            mock_config.save_to_file.assert_called_once()

    @patch("src.config.cli.ConfigLoader.load_documentation_sites")
    def test_migrate_sites_no_config(self, mock_load_sites):
        """Test migrating sites without existing config file."""
        with self.runner.isolated_filesystem():
            with open("sites.json", "w") as f:
                json.dump({"sites": []}, f)

            mock_load_sites.return_value = []

            with patch("src.config.cli.UnifiedConfig") as mock_config_class:
                mock_config = MagicMock()
                mock_config_class.return_value = mock_config

                result = self.runner.invoke(
                    cli, ["migrate-sites", "--source", "sites.json"]
                )

                assert result.exit_code == 0
                assert "Migrated 0 documentation sites" in result.output
                mock_config.save_to_file.assert_called_once()

    @patch("src.config.cli.ConfigLoader.load_documentation_sites")
    def test_migrate_sites_error(self, mock_load_sites):
        """Test migrate sites with loading error."""
        mock_load_sites.side_effect = FileNotFoundError("Sites file not found")

        result = self.runner.invoke(cli, ["migrate-sites", "--source", "missing.json"])

        # Click returns exit code 2 for file not found, which is expected
        assert result.exit_code in [1, 2]  # Accept both application and CLI errors
        # Check if it's either our error message or click's file error
        assert (
            "Error migrating sites:" in result.output or "missing.json" in result.output
        )

    @patch("qdrant_client.QdrantClient")
    @patch("src.config.cli.ConfigLoader.load_config")
    def test_check_connections_success(self, mock_load, mock_client_class):
        """Test successful connection checking."""
        mock_config = MagicMock()
        mock_config.qdrant.url = "http://localhost:6333"
        mock_config.qdrant.api_key = None
        mock_config.cache.enable_dragonfly_cache = False
        mock_config.embedding_provider = "fastembed"
        mock_config.crawl_provider = "crawl4ai"
        mock_load.return_value = mock_config

        mock_client = MagicMock()
        mock_collections = MagicMock()
        mock_collections.collections = [MagicMock(), MagicMock()]
        mock_client.get_collections.return_value = mock_collections
        mock_client_class.return_value = mock_client

        result = self.runner.invoke(cli, ["check-connections"])

        assert result.exit_code == 0
        assert "Checking Service Connections" in result.output
        assert "Qdrant connected (2 collections)" in result.output

    @patch("qdrant_client.QdrantClient")
    @patch("src.config.cli.ConfigLoader.load_config")
    def test_check_connections_qdrant_failure(self, mock_load, mock_client_class):
        """Test connection checking with Qdrant failure."""
        mock_config = MagicMock()
        mock_config.qdrant.url = "http://localhost:6333"
        mock_config.qdrant.api_key = None
        mock_config.cache.enable_dragonfly_cache = False
        mock_config.embedding_provider = "fastembed"
        mock_config.crawl_provider = "crawl4ai"
        mock_load.return_value = mock_config

        mock_client_class.side_effect = Exception("Connection refused")

        result = self.runner.invoke(cli, ["check-connections"])

        assert result.exit_code == 0
        assert "Qdrant connection failed: Connection refused" in result.output

    @patch("redis.from_url")
    @patch("qdrant_client.QdrantClient")
    @patch("src.config.cli.ConfigLoader.load_config")
    def test_check_connections_redis_enabled(self, mock_load, mock_qdrant, mock_redis):
        """Test connection checking with Redis enabled."""
        mock_config = MagicMock()
        mock_config.qdrant.url = "http://localhost:6333"
        mock_config.cache.enable_dragonfly_cache = True
        mock_config.cache.dragonfly_url = "redis://localhost:6379"
        mock_config.embedding_provider = "fastembed"
        mock_config.crawl_provider = "crawl4ai"
        mock_load.return_value = mock_config

        # Mock Qdrant success
        mock_qdrant_client = MagicMock()
        mock_qdrant.return_value = mock_qdrant_client

        # Mock Redis success
        mock_redis_client = MagicMock()
        mock_redis.return_value = mock_redis_client

        result = self.runner.invoke(cli, ["check-connections"])

        assert result.exit_code == 0
        assert "Checking Redis" in result.output
        assert "Redis connected" in result.output

    @patch("src.config.cli.ConfigSchemaGenerator.save_schema")
    def test_generate_schema_all_formats(self, mock_save):
        """Test generating schema in all formats."""
        mock_save.return_value = {
            "json": Path("schema/config-schema.json"),
            "typescript": Path("schema/config-types.ts"),
            "markdown": Path("schema/config-schema.md"),
        }

        result = self.runner.invoke(cli, ["generate-schema", "--output-dir", "schema"])

        assert result.exit_code == 0
        assert "Generated configuration schema files:" in result.output
        assert "json: schema/config-schema.json" in result.output
        mock_save.assert_called_once_with("schema", None)

    @patch("src.config.cli.ConfigSchemaGenerator.save_schema")
    def test_generate_schema_specific_formats(self, mock_save):
        """Test generating schema in specific formats."""
        mock_save.return_value = {
            "json": Path("schema/config-schema.json"),
            "typescript": Path("schema/config-types.ts"),
        }

        result = self.runner.invoke(
            cli,
            [
                "generate-schema",
                "--output-dir",
                "schema",
                "--format",
                "json",
                "--format",
                "typescript",
            ],
        )

        assert result.exit_code == 0
        mock_save.assert_called_once_with("schema", ["json", "typescript"])

    @patch("src.config.cli.ConfigSchemaGenerator.save_schema")
    def test_generate_schema_error(self, mock_save):
        """Test generate schema with error."""
        mock_save.side_effect = OSError("Permission denied")

        result = self.runner.invoke(cli, ["generate-schema"])

        assert result.exit_code == 1
        assert "Error generating schema:" in result.output

    @patch("src.config.cli.ConfigSchemaGenerator.generate_json_schema")
    def test_show_schema(self, mock_generate):
        """Test showing schema in terminal."""
        mock_generate.return_value = {"type": "object", "properties": {}}

        result = self.runner.invoke(cli, ["show-schema"])

        assert result.exit_code == 0
        mock_generate.assert_called_once()

    @patch("src.config.cli.ConfigSchemaGenerator.generate_json_schema")
    def test_show_schema_error(self, mock_generate):
        """Test show schema with error."""
        mock_generate.side_effect = ValueError("Schema generation failed")

        result = self.runner.invoke(cli, ["show-schema"])

        assert result.exit_code == 1
        assert "Error displaying schema:" in result.output

    def test_migrate_config_success(self):
        """Test successful configuration migration."""
        with self.runner.isolated_filesystem():
            # Create config file
            config_data = {"version": "0.1.0", "environment": "development"}
            with open("config.json", "w") as f:
                json.dump(config_data, f)

            with patch("src.config.cli.ConfigMigrator") as mock_migrator:
                mock_migrator.detect_config_version.return_value = "0.1.0"
                mock_migrator.migrate_between_versions.return_value = {
                    "version": "0.2.0",
                    "environment": "development",
                }
                mock_migrator.create_migration_report.return_value = (
                    "Migration successful"
                )
                mock_migrator.auto_migrate.return_value = (True, "Migration completed")

                result = self.runner.invoke(
                    cli, ["migrate", "config.json", "--target-version", "0.2.0"]
                )

                assert result.exit_code == 0
                assert "Current version: 0.1.0" in result.output
                assert "Target version: 0.2.0" in result.output

    def test_migrate_config_dry_run(self):
        """Test configuration migration dry run."""
        with self.runner.isolated_filesystem():
            config_data = {"version": "0.1.0", "environment": "development"}
            with open("config.json", "w") as f:
                json.dump(config_data, f)

            with patch("src.config.cli.ConfigMigrator") as mock_migrator:
                mock_migrator.detect_config_version.return_value = "0.1.0"
                mock_migrator.migrate_between_versions.return_value = {
                    "version": "0.2.0",
                    "environment": "development",
                }
                mock_migrator.create_migration_report.return_value = "Migration report"

                result = self.runner.invoke(
                    cli, ["migrate", "config.json", "--dry-run"]
                )

                assert result.exit_code == 0
                assert "Dry run complete - no changes made" in result.output

    def test_migrate_config_same_version(self):
        """Test migration when already at target version."""
        with self.runner.isolated_filesystem():
            config_data = {"version": "0.2.0", "environment": "development"}
            with open("config.json", "w") as f:
                json.dump(config_data, f)

            with patch("src.config.cli.ConfigMigrator") as mock_migrator:
                mock_migrator.detect_config_version.return_value = "0.2.0"

                result = self.runner.invoke(
                    cli, ["migrate", "config.json", "--target-version", "0.2.0"]
                )

                assert result.exit_code == 0
                assert "Configuration already at target version" in result.output

    def test_migrate_config_unknown_version(self):
        """Test migration with unknown version."""
        with self.runner.isolated_filesystem():
            config_data = {"unknown": "config"}
            with open("config.json", "w") as f:
                json.dump(config_data, f)

            with patch("src.config.cli.ConfigMigrator") as mock_migrator:
                mock_migrator.detect_config_version.return_value = None

                result = self.runner.invoke(cli, ["migrate", "config.json"])

                assert result.exit_code == 1
                assert "Could not detect configuration version" in result.output

    def test_migrate_config_unsupported_format(self):
        """Test migration with unsupported file format."""
        with self.runner.isolated_filesystem():
            with open("config.xml", "w") as f:
                f.write("<config></config>")

            result = self.runner.invoke(cli, ["migrate", "config.xml"])

            assert result.exit_code == 1
            assert "Unsupported file format" in result.output

    @patch("src.config.cli.ConfigMigrator")
    def test_show_migration_path(self, mock_migrator):
        """Test showing migration paths."""
        mock_migrator.VERSIONS = {
            "0.1.0": "Initial version",
            "0.2.0": "Cache improvements",
            "0.3.0": "Security features",
        }

        result = self.runner.invoke(cli, ["show-migration-path"])

        assert result.exit_code == 0
        assert "Configuration Version History" in result.output

    @patch("src.config.cli.ConfigMigrator")
    def test_show_migration_path_specific(self, mock_migrator):
        """Test showing specific migration path."""
        mock_migrator.VERSIONS = {"0.1.0": "Initial", "0.2.0": "Updated"}

        result = self.runner.invoke(
            cli,
            [
                "show-migration-path",
                "--from-version",
                "0.1.0",
                "--to-version",
                "0.2.0",
            ],
        )

        assert result.exit_code == 0
        assert "Migration path from 0.1.0 to 0.2.0:" in result.output

    @patch("src.config.cli.ConfigMigrator")
    def test_show_migration_path_legacy(self, mock_migrator):
        """Test showing legacy migration path."""
        mock_migrator.VERSIONS = {"0.3.0": "Current"}

        result = self.runner.invoke(
            cli,
            [
                "show-migration-path",
                "--from-version",
                "legacy",
                "--to-version",
                "0.3.0",
            ],
        )

        assert result.exit_code == 0
        assert "Convert legacy format to unified configuration" in result.output
