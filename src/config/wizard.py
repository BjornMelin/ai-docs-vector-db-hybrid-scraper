"""Interactive configuration wizard for guided setup and management.

This module provides an interactive wizard that guides users through configuration
setup, template selection, and environment-specific customization.
"""

import json
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .backup_restore import ConfigBackupManager
from .enhanced_validators import ConfigurationValidator
from .migrations import ConfigMigrationManager
from .migrations import create_default_migrations
from .models import UnifiedConfig
from .templates import ConfigurationTemplates
from .utils import ConfigPathManager


class ConfigurationWizard:
    """Interactive configuration wizard with rich CLI interface."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize configuration wizard.

        Args:
            base_dir: Base directory for configuration management
        """
        self.console = Console()
        self.path_manager = ConfigPathManager(base_dir or Path("config"))
        self.templates = ConfigurationTemplates()
        self.backup_manager = ConfigBackupManager(base_dir)
        self.migration_manager = ConfigMigrationManager(base_dir)

        # Initialize with default migrations
        create_default_migrations(self.migration_manager)

        self.path_manager.ensure_directories()

    def run_setup_wizard(self, config_path: Path | None = None) -> Path:
        """Run the complete configuration setup wizard.

        Args:
            config_path: Target path for configuration file

        Returns:
            Path to created configuration file
        """
        self.console.print(
            Panel.fit(
                "[bold blue]AI Documentation Vector DB[/bold blue]\n"
                "[dim]Advanced Configuration Setup Wizard[/dim]",
                title="🧙‍♂️ Configuration Wizard",
                border_style="blue",
            )
        )

        # Step 1: Choose setup mode
        setup_mode = self._choose_setup_mode()

        if setup_mode == "template":
            return self._template_based_setup(config_path)
        elif setup_mode == "interactive":
            return self._interactive_setup(config_path)
        elif setup_mode == "migrate":
            return self._migration_setup(config_path)
        else:  # import
            return self._import_setup(config_path)

    def run_validation_wizard(self, config_path: Path) -> bool:
        """Run configuration validation with suggestions.

        Args:
            config_path: Path to configuration file to validate

        Returns:
            True if configuration is valid or fixed
        """
        self.console.print(
            Panel.fit(
                "[bold yellow]Configuration Validation[/bold yellow]\n"
                f"[dim]Analyzing: {config_path}[/dim]",
                title="🔍 Validation Wizard",
                border_style="yellow",
            )
        )

        # Load configuration
        try:
            config = UnifiedConfig.load_from_file(config_path)
        except Exception as e:
            self.console.print(f"[red]❌ Failed to load configuration: {e}[/red]")
            return False

        # Run validation
        validator = ConfigurationValidator(str(config.environment))
        report = validator.validate_configuration(config)

        # Display validation results
        self._display_validation_report(report)

        if report.is_valid:
            self.console.print("[green]✅ Configuration is valid![/green]")
            return True

        # Offer to fix issues
        if report.errors:
            fix_issues = questionary.confirm(
                "Would you like to attempt automatic fixes for some issues?"
            ).ask()

            if fix_issues:
                return self._attempt_fixes(config_path, report)

        return False

    def run_backup_wizard(self, config_path: Path) -> str | None:
        """Run backup creation wizard.

        Args:
            config_path: Path to configuration file to backup

        Returns:
            Backup ID if successful
        """
        self.console.print(
            Panel.fit(
                "[bold green]Configuration Backup[/bold green]\n"
                f"[dim]Creating backup for: {config_path}[/dim]",
                title="💾 Backup Wizard",
                border_style="green",
            )
        )

        # Get backup options
        description = questionary.text(
            "Enter a description for this backup (optional):", default=""
        ).ask()

        tags_input = questionary.text(
            "Enter tags separated by commas (optional):", default=""
        ).ask()

        tags = [tag.strip() for tag in tags_input.split(",") if tag.strip()]

        compress = questionary.confirm(
            "Compress the backup to save space?", default=True
        ).ask()

        try:
            backup_id = self.backup_manager.create_backup(
                config_path,
                description=description or None,
                tags=tags,
                compress=compress,
            )

            self.console.print(f"[green]✅ Backup created: {backup_id}[/green]")
            return backup_id

        except Exception as e:
            self.console.print(f"[red]❌ Backup failed: {e}[/red]")
            return None

    def run_restore_wizard(self) -> bool:
        """Run backup restore wizard.

        Returns:
            True if restore was successful
        """
        self.console.print(
            Panel.fit(
                "[bold cyan]Configuration Restore[/bold cyan]\n"
                "[dim]Restore from backup[/dim]",
                title="🔄 Restore Wizard",
                border_style="cyan",
            )
        )

        # List available backups
        backups = self.backup_manager.list_backups(limit=20)

        if not backups:
            self.console.print("[yellow]No backups found.[/yellow]")
            return False

        # Display backup table
        table = Table(title="Available Backups")
        table.add_column("ID", style="cyan")
        table.add_column("Config", style="green")
        table.add_column("Created", style="yellow")
        table.add_column("Environment", style="blue")
        table.add_column("Description", style="dim")

        backup_choices = []
        for backup in backups:
            table.add_row(
                backup.backup_id[:12] + "...",
                backup.config_name,
                backup.created_at[:19],
                backup.environment or "unknown",
                backup.description or "No description",
            )
            backup_choices.append(
                (backup.backup_id, f"{backup.backup_id[:12]}... - {backup.config_name}")
            )

        self.console.print(table)

        # Choose backup to restore
        selected_backup = questionary.select(
            "Select backup to restore:",
            choices=[choice[1] for choice in backup_choices],
        ).ask()

        if not selected_backup:
            return False

        # Find selected backup ID
        backup_id = None
        for bid, label in backup_choices:
            if label == selected_backup:
                backup_id = bid
                break

        if not backup_id:
            return False

        # Get restore options
        target_path = questionary.path(
            "Enter target path for restored configuration:",
            default="config/restored_config.json",
        ).ask()

        create_backup = questionary.confirm(
            "Create backup of current configuration before restore?", default=True
        ).ask()

        # Perform restore
        try:
            result = self.backup_manager.restore_backup(
                backup_id, Path(target_path), create_pre_restore_backup=create_backup
            )

            if result.success:
                self.console.print(
                    f"[green]✅ Configuration restored to: {result.config_path}[/green]"
                )
                if result.pre_restore_backup:
                    self.console.print(
                        f"[dim]Pre-restore backup: {result.pre_restore_backup}[/dim]"
                    )
                return True
            else:
                self.console.print("[red]❌ Restore failed[/red]")
                for error in result.warnings:
                    self.console.print(f"[yellow]⚠️  {error}[/yellow]")
                return False

        except Exception as e:
            self.console.print(f"[red]❌ Restore failed: {e}[/red]")
            return False

    def _choose_setup_mode(self) -> str:
        """Choose configuration setup mode."""
        return questionary.select(
            "How would you like to set up your configuration?",
            choices=[
                ("Start with a template", "template"),
                ("Interactive step-by-step setup", "interactive"),
                ("Migrate existing configuration", "migrate"),
                ("Import from file", "import"),
            ],
        ).ask()

    def _template_based_setup(self, config_path: Path | None) -> Path:
        """Template-based configuration setup."""
        # Choose template
        available_templates = self.templates.list_available_templates()

        template_descriptions = {
            "development": "🛠️  Development - Debug logging, local database, fast iteration",
            "production": "🚀 Production - Security hardening, performance optimization",
            "high_performance": "⚡ High Performance - Maximum throughput and concurrency",
            "memory_optimized": "💾 Memory Optimized - Resource-constrained environments",
            "distributed": "🌐 Distributed - Multi-node cluster deployment",
        }

        template_choices = []
        for template_name in available_templates:
            description = template_descriptions.get(
                template_name, f"📄 {template_name}"
            )
            template_choices.append((description, template_name))

        selected_template = questionary.select(
            "Choose a configuration template:",
            choices=[choice[0] for choice in template_choices],
        ).ask()

        # Find selected template name
        template_name = None
        for desc, name in template_choices:
            if desc == selected_template:
                template_name = name
                break

        if not template_name:
            raise ValueError("No template selected")

        # Get customization options
        customize = questionary.confirm(
            "Would you like to customize the template?", default=False
        ).ask()

        # Apply template
        config_data = self.templates.apply_template_to_config(template_name)
        if not config_data:
            raise ValueError(f"Failed to load template: {template_name}")

        # Customize if requested
        if customize:
            config_data = self._customize_template(config_data)

        # Determine output path
        if not config_path:
            default_name = f"{template_name}_config.json"
            config_path = Path(
                questionary.path(
                    "Enter path for configuration file:", default=default_name
                ).ask()
            )

        # Create configuration
        config = UnifiedConfig(**config_data)
        config.save_to_file(config_path)

        self.console.print(f"[green]✅ Configuration created: {config_path}[/green]")
        self.console.print(f"[dim]Template: {template_name}[/dim]")

        return config_path

    def _interactive_setup(self, config_path: Path | None) -> Path:
        """Interactive step-by-step configuration setup."""
        self.console.print("[bold]Interactive Configuration Setup[/bold]")

        config_data = {}

        # Basic settings
        config_data["environment"] = questionary.select(
            "Select environment:", choices=["development", "staging", "production"]
        ).ask()

        config_data["debug"] = questionary.confirm(
            "Enable debug mode?", default=config_data["environment"] == "development"
        ).ask()

        config_data["log_level"] = questionary.select(
            "Select log level:", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
        ).ask()

        # Provider selection
        config_data["embedding_provider"] = questionary.select(
            "Select embedding provider:", choices=["fastembed", "openai"]
        ).ask()

        config_data["crawl_provider"] = questionary.select(
            "Select crawling provider:", choices=["crawl4ai", "firecrawl"]
        ).ask()

        # API keys
        if config_data["embedding_provider"] == "openai":
            api_key = questionary.password("Enter OpenAI API key:").ask()
            config_data["openai"] = {"api_key": api_key}

        if config_data["crawl_provider"] == "firecrawl":
            api_key = questionary.password("Enter Firecrawl API key:").ask()
            config_data["firecrawl"] = {"api_key": api_key}

        # Database configuration
        use_postgres = questionary.confirm(
            "Use PostgreSQL database? (No = SQLite)",
            default=config_data["environment"] == "production",
        ).ask()

        if use_postgres:
            db_url = questionary.text(
                "Enter PostgreSQL connection URL:",
                default="postgresql+asyncpg://user:CHANGEME_DB_PASSWORD@localhost:5432/aidocs",
            ).ask()
            config_data["database"] = {"database_url": db_url}

        # Caching
        enable_redis = questionary.confirm(
            "Enable Redis/DragonflyDB caching?",
            default=config_data["environment"] != "development",
        ).ask()

        if enable_redis:
            redis_url = questionary.text(
                "Enter Redis URL:", default="redis://localhost:6379"
            ).ask()
            config_data["cache"] = {
                "enable_dragonfly_cache": True,
                "dragonfly_url": redis_url,
            }

        # Determine output path
        if not config_path:
            env = config_data["environment"]
            default_name = f"{env}_config.json"
            config_path = Path(
                questionary.path(
                    "Enter path for configuration file:", default=default_name
                ).ask()
            )

        # Create configuration
        config = UnifiedConfig(**config_data)
        config.save_to_file(config_path)

        self.console.print(f"[green]✅ Configuration created: {config_path}[/green]")

        return config_path

    def _migration_setup(self, config_path: Path | None) -> Path:
        """Migration-based configuration setup."""
        # Find existing configuration
        if not config_path:
            config_path = Path(
                questionary.path(
                    "Enter path to existing configuration:",
                    validate=lambda x: Path(x).exists() or "File does not exist",
                ).ask()
            )

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Check current version
        current_version = self.migration_manager.get_current_version(config_path)
        self.console.print(f"[dim]Current version: {current_version}[/dim]")

        # Get target version
        target_version = questionary.text(
            "Enter target version:", default="2.0.0"
        ).ask()

        # Create migration plan
        plan = self.migration_manager.create_migration_plan(
            current_version, target_version
        )

        if not plan:
            self.console.print("[red]❌ No migration path found[/red]")
            return config_path

        # Display migration plan
        self.console.print(
            f"[yellow]Migration plan: {current_version} → {target_version}[/yellow]"
        )
        for migration_id in plan.migrations:
            self.console.print(f"  • {migration_id}")

        self.console.print(f"[dim]Estimated duration: {plan.estimated_duration}[/dim]")

        if plan.requires_downtime:
            self.console.print("[yellow]⚠️  This migration requires downtime[/yellow]")

        # Confirm migration
        proceed = questionary.confirm("Proceed with migration?").ask()

        if not proceed:
            return config_path

        # Execute migration
        results = self.migration_manager.apply_migration_plan(plan, config_path)

        # Display results
        for result in results:
            if result.success:
                self.console.print(f"[green]✅ {result.migration_id}[/green]")
                for change in result.changes_made:
                    self.console.print(f"    {change}")
            else:
                self.console.print(f"[red]❌ {result.migration_id}[/red]")
                for error in result.errors:
                    self.console.print(f"    {error}")

        return config_path

    def _import_setup(self, config_path: Path | None) -> Path:
        """Import configuration from external file."""
        import_path = Path(
            questionary.path(
                "Enter path to configuration file to import:",
                validate=lambda x: Path(x).exists() or "File does not exist",
            ).ask()
        )

        # Determine output path
        if not config_path:
            config_path = Path(
                questionary.path(
                    "Enter path for new configuration file:",
                    default="imported_config.json",
                ).ask()
            )

        # Load and validate imported configuration
        try:
            imported_config = UnifiedConfig.load_from_file(import_path)

            # Validate imported configuration
            validator = ConfigurationValidator()
            report = validator.validate_configuration(imported_config)

            if not report.is_valid:
                self.console.print(
                    "[yellow]⚠️  Imported configuration has validation issues:[/yellow]"
                )
                self._display_validation_report(report)

                proceed = questionary.confirm("Import anyway?").ask()
                if not proceed:
                    return config_path

            # Save imported configuration
            imported_config.save_to_file(config_path)

            self.console.print(
                f"[green]✅ Configuration imported: {config_path}[/green]"
            )

        except Exception as e:
            self.console.print(f"[red]❌ Import failed: {e}[/red]")
            return config_path

        return config_path

    def _customize_template(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Customize template configuration."""
        self.console.print("[bold]Template Customization[/bold]")

        # Environment override
        new_env = questionary.select(
            f"Change environment from '{config_data.get('environment', 'unknown')}'?",
            choices=["Keep current", "development", "staging", "production"],
        ).ask()

        if new_env != "Keep current":
            config_data["environment"] = new_env

        # Debug mode
        if "debug" in config_data:
            new_debug = questionary.confirm(
                f"Debug mode currently: {config_data['debug']}. Change?",
                default=config_data["debug"],
            ).ask()
            config_data["debug"] = new_debug

        # API keys
        if questionary.confirm("Configure API keys?").ask():
            if config_data.get("embedding_provider") == "openai":
                api_key = questionary.password(
                    "Enter OpenAI API key (leave empty to skip):"
                ).ask()
                if api_key:
                    if "openai" not in config_data:
                        config_data["openai"] = {}
                    config_data["openai"]["api_key"] = api_key

            if config_data.get("crawl_provider") == "firecrawl":
                api_key = questionary.password(
                    "Enter Firecrawl API key (leave empty to skip):"
                ).ask()
                if api_key:
                    if "firecrawl" not in config_data:
                        config_data["firecrawl"] = {}
                    config_data["firecrawl"]["api_key"] = api_key

        return config_data

    def _display_validation_report(self, report) -> None:
        """Display validation report with rich formatting."""
        if report.errors:
            self.console.print("[red]❌ Errors:[/red]")
            for issue in report.errors:
                self.console.print(f"  {issue}")

        if report.warnings:
            self.console.print("[yellow]⚠️  Warnings:[/yellow]")
            for issue in report.warnings:
                self.console.print(f"  {issue}")

        if report.info:
            self.console.print("[blue]i  Information:[/blue]")
            for issue in report.info:
                self.console.print(f"  {issue}")

    def _attempt_fixes(self, config_path: Path, report) -> bool:
        """Attempt to automatically fix validation issues."""
        self.console.print("[yellow]Attempting automatic fixes...[/yellow]")

        # Create backup before fixes
        backup_id = self.backup_manager.create_backup(
            config_path,
            description="Pre-fix backup",
            tags=["automatic", "validation-fix"],
        )

        self.console.print(f"[dim]Backup created: {backup_id}[/dim]")

        # Load configuration
        with open(config_path) as f:
            config_data = json.load(f)

        fixes_applied = []

        # Apply fixes for common issues
        for issue in report.errors:
            if issue.field_path == "debug" and "production" in issue.message:
                config_data["debug"] = False
                fixes_applied.append("Set debug=false for production")

            elif "api_key" in issue.field_path and "required" in issue.message:
                # Prompt for API key
                provider = issue.field_path.split(".")[0]
                api_key = questionary.password(f"Enter {provider} API key:").ask()
                if api_key:
                    if provider not in config_data:
                        config_data[provider] = {}
                    config_data[provider]["api_key"] = api_key
                    fixes_applied.append(f"Set {provider} API key")

        if fixes_applied:
            # Save fixed configuration
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)

            self.console.print("[green]✅ Fixes applied:[/green]")
            for fix in fixes_applied:
                self.console.print(f"  • {fix}")

            # Re-validate
            config = UnifiedConfig.load_from_file(config_path)
            validator = ConfigurationValidator(str(config.environment))
            new_report = validator.validate_configuration(config)

            if new_report.is_valid:
                self.console.print("[green]✅ Configuration is now valid![/green]")
                return True
            else:
                self.console.print("[yellow]⚠️  Some issues remain[/yellow]")
                self._display_validation_report(new_report)
                return False
        else:
            self.console.print("[yellow]No automatic fixes available[/yellow]")
            return False
