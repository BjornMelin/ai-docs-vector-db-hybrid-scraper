"""CLI demonstration of progressive configuration system.

This module provides an interactive CLI interface that showcases the sophisticated
configuration system capabilities for portfolio demonstration.

Features:
- Interactive configuration discovery
- Progressive complexity demonstration
- Auto-detection showcase
- Validation and error handling demo
- Configuration export and templates
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .builders import ConfigBuilderFactory, ProgressiveConfigurationGuide
from .discovery import (
    ConfigurationOptimizer,
    IntelligentValidator,
    discover_optimal_configuration,
    get_system_recommendations,
    validate_configuration_intelligently,
)
from .examples import ConfigurationShowcase, run_configuration_showcase


console = Console()


@click.group()
def config_cli():
    """AI Documentation Vector DB - Progressive Configuration System."""
    pass


@config_cli.command()
@click.option(
    "--persona",
    type=click.Choice(["development", "production", "research", "enterprise"]),
    help="Configuration persona (auto-detected if not specified)",
)
@click.option("--output", "-o", type=Path, help="Output configuration file path")
@click.option(
    "--format", "output_format", type=click.Choice(["json", "yaml"]), default="json"
)
@click.option("--no-auto-detect", is_flag=True, help="Disable service auto-detection")
def create(
    persona: str | None,
    output: Path | None,
    output_format: str,
    no_auto_detect: bool,
):
    """Create configuration with guided setup."""

    async def _create_config():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Step 1: Environment analysis
            task = progress.add_task("Analyzing environment...", total=None)
            system_env, recommendations = await get_system_recommendations()

            # Step 2: Persona selection
            progress.update(task, description="Determining optimal persona...")
            if not persona:
                optimal_persona = recommendations.recommended_persona
                console.print(
                    f"🎯 Auto-detected optimal persona: [bold green]{optimal_persona}[/bold green]"
                )
                console.print(f"   Confidence: {recommendations.confidence_score:.1%}")
                for reason in recommendations.reasoning:
                    console.print(f"   • {reason}")
            else:
                optimal_persona = persona
                console.print(
                    f"🎭 Using specified persona: [bold blue]{optimal_persona}[/bold blue]"
                )

            # Step 3: Configuration building
            progress.update(task, description="Building configuration...")
            guide = ProgressiveConfigurationGuide(
                optimal_persona, auto_detect=not no_auto_detect
            )
            discovery = await guide.start_guided_setup()

            progress.remove_task(task)

        # Display discovery results
        _display_discovery_results(discovery, system_env, recommendations)

        # Interactive feature selection
        features = guide.get_progressive_features()
        feature_level = _interactive_feature_selection(features)

        # Build final configuration
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Building final configuration...", total=None)
            config = await guide.build_configuration(feature_level)

        # Display configuration summary
        _display_config_summary(config, feature_level)

        # Save configuration if requested
        if output:
            _save_configuration(config, output, output_format)

        return config

    asyncio.run(_create_config())


@config_cli.command()
@click.argument("config_file", type=Path)
@click.option("--persona", help="Expected persona for context-aware validation")
def validate(config_file: Path, persona: str | None):
    """Validate configuration with intelligent suggestions."""

    async def _validate_config():
        if not config_file.exists():
            console.print(f"❌ Configuration file not found: {config_file}")
            return

        # Load configuration
        with open(config_file) as f:
            if config_file.suffix == ".json":
                config_data = json.load(f)
            else:
                console.print(
                    "❌ Only JSON configuration files supported for validation"
                )
                return

        # Perform intelligent validation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Validating configuration...", total=None)
            report = await validate_configuration_intelligently(config_data, persona)

        # Display validation results
        _display_validation_report(report)

    asyncio.run(_validate_config())


@config_cli.command()
def discover():
    """Discover optimal configuration based on environment analysis."""

    async def _discover():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Discovering environment...", total=None)
            recommendations = await discover_optimal_configuration()

        # Display recommendations
        console.print("\n🔍 Configuration Discovery Results")
        console.print("=" * 50)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Aspect", style="cyan")
        table.add_column("Recommendation", style="green")
        table.add_column("Confidence", style="yellow")

        table.add_row(
            "Persona",
            recommendations.recommended_persona,
            f"{recommendations.confidence_score:.1%}",
        )

        for provider_type, provider in recommendations.suggested_providers.items():
            table.add_row(f"{provider_type.title()} Provider", provider, "High")

        console.print(table)

        # Display reasoning
        if recommendations.reasoning:
            console.print("\n💡 Reasoning:")
            for reason in recommendations.reasoning:
                console.print(f"  • {reason}")

        # Display performance recommendations
        if recommendations.performance_recommendations:
            console.print("\n⚡ Performance Recommendations:")
            for rec in recommendations.performance_recommendations:
                console.print(f"  • {rec}")

        # Display cost estimates
        if recommendations.estimated_costs:
            console.print("\n💰 Estimated Costs:")
            for service, cost in recommendations.estimated_costs.items():
                console.print(f"  • {service.title()}: {cost}")

    asyncio.run(_discover())


@config_cli.command()
def showcase():
    """Run the complete configuration system showcase."""

    async def _showcase():
        console.print(
            Panel.fit(
                "[bold blue]AI Documentation Vector DB[/bold blue]\n"
                "[bold green]Progressive Configuration System Showcase[/bold green]\n\n"
                "Demonstrating sophisticated configuration patterns,\n"
                "auto-detection, and enterprise-grade features.",
                title="Configuration Showcase",
                border_style="blue",
            )
        )

        try:
            discovery, config = await run_configuration_showcase()

            console.print("\n✨ Showcase completed successfully!")
            console.print(
                f"Final configuration: {discovery.persona} persona with {discovery.configuration_complexity} complexity"
            )

        except Exception as e:
            console.print(f"\n❌ Showcase failed: {e}")
            return 1

    return asyncio.run(_showcase())


@config_cli.command()
@click.option(
    "--persona",
    type=click.Choice(["development", "production", "research", "enterprise"]),
)
@click.option("--output-dir", type=Path, default=Path("config_examples"))
def examples(persona: str | None, output_dir: Path):
    """Generate configuration examples for different personas."""

    output_dir.mkdir(exist_ok=True)

    personas = [persona] if persona else ConfigBuilderFactory.get_available_personas()

    for p in personas:
        builder = ConfigBuilderFactory.create_builder(p, auto_detect=False)
        config = builder.build()

        # Save example configuration
        example_file = output_dir / f"{p}_example.json"
        with open(example_file, "w") as f:
            json.dump(config.model_dump(), f, indent=2, default=str)

        console.print(f"✓ Generated {p} example: {example_file}")

    # Generate usage examples
    usage_file = output_dir / "usage_examples.py"
    with open(usage_file, "w") as f:
        f.write(
            '"""Configuration usage examples.\n\nGenerated examples for different configuration personas.\n"""\n\n'
        )

        from .examples import USAGE_EXAMPLES

        for name, example in USAGE_EXAMPLES.items():
            f.write(f"# {name.replace('_', ' ').title()}\n")
            f.write(f"{example}\n\n")

    console.print(f"✓ Generated usage examples: {usage_file}")
    console.print(f"\n📁 All examples saved to: {output_dir}")


def _display_discovery_results(discovery, system_env, recommendations):
    """Display configuration discovery results."""
    console.print("\n🔍 Configuration Discovery Complete")
    console.print("=" * 50)

    # System information
    console.print(f"💻 System: {system_env.platform} {system_env.architecture}")
    console.print(
        f"🧠 Resources: {system_env.memory_gb}GB RAM, {system_env.cpu_count} cores"
    )
    console.print(f"🐍 Python: {system_env.python_version}")

    if system_env.is_docker:
        console.print("🐳 Docker environment detected")
    if system_env.is_ci:
        console.print("🔄 CI environment detected")

    # Discovery results
    console.print(f"\n🎭 Persona: [bold blue]{discovery.persona}[/bold blue]")
    console.print(f"📊 Complexity: {discovery.configuration_complexity}")
    console.print(f"⏱️  Setup time: {discovery.estimated_setup_time}")

    if discovery.auto_detected_services:
        console.print(
            f"🔌 Auto-detected: {', '.join(discovery.auto_detected_services)}"
        )


def _interactive_feature_selection(features: Dict[str, Any]) -> str:
    """Interactive feature level selection."""
    console.print("\n🎚️  Select Configuration Complexity Level")
    console.print("-" * 40)

    levels = ["essential", "intermediate", "advanced"]
    descriptions = {
        "essential": "Basic features for getting started quickly",
        "intermediate": "Balanced features for most use cases",
        "advanced": "Full enterprise features and capabilities",
    }

    for i, level in enumerate(levels, 1):
        feature_list = features.get(level, [])
        console.print(f"{i}. [bold]{level.title()}[/bold] - {descriptions[level]}")
        console.print(f"   Features: {len(feature_list)} available")
        for feature in feature_list[:2]:
            console.print(f"   • {feature}")
        if len(feature_list) > 2:
            console.print(f"   • ... and {len(feature_list) - 2} more")
        console.print()

    while True:
        try:
            choice = click.prompt("Select level (1-3)", type=int)
            if 1 <= choice <= 3:
                return levels[choice - 1]
            else:
                console.print("Please enter 1, 2, or 3")
        except click.Abort:
            return "essential"  # Default


def _display_config_summary(config, feature_level: str):
    """Display configuration summary."""
    console.print(f"\n✅ Configuration Built - {feature_level.title()} Level")
    console.print("=" * 50)

    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Component", style="cyan")
    summary_table.add_column("Configuration", style="green")

    summary_table.add_row("Environment", config.environment.value)
    summary_table.add_row("Embedding Provider", config.embedding_provider.value)
    summary_table.add_row("Crawl Provider", config.crawl_provider.value)
    summary_table.add_row("Debug Mode", "Enabled" if config.debug else "Disabled")
    summary_table.add_row(
        "Caching", "Enabled" if config.cache.enable_caching else "Disabled"
    )
    summary_table.add_row(
        "Monitoring", "Enabled" if config.monitoring.enabled else "Disabled"
    )
    summary_table.add_row(
        "Observability", "Enabled" if config.observability.enabled else "Disabled"
    )

    console.print(summary_table)


def _display_validation_report(report):
    """Display configuration validation report."""
    console.print("\n📋 Configuration Validation Report")
    console.print("=" * 50)

    # Overall status
    status = "✅ Valid" if report.is_valid else "❌ Invalid"
    console.print(f"Status: {status}")

    # Scores
    scores_table = Table(show_header=True, header_style="bold magenta")
    scores_table.add_column("Aspect", style="cyan")
    scores_table.add_column("Score", style="green")
    scores_table.add_column("Rating", style="yellow")

    def get_rating(score):
        if score >= 80:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 40:
            return "Fair"
        else:
            return "Needs Improvement"

    scores_table.add_row(
        "Security", f"{report.security_score}/100", get_rating(report.security_score)
    )
    scores_table.add_row(
        "Performance",
        f"{report.performance_score}/100",
        get_rating(report.performance_score),
    )
    scores_table.add_row(
        "Maintainability",
        f"{report.maintainability_score}/100",
        get_rating(report.maintainability_score),
    )

    console.print(scores_table)

    # Errors
    if report.errors:
        console.print(f"\n❌ Errors ({report.error_count}):")
        for error in report.errors:
            console.print(f"  • {error['field']}: {error['message']}")
            for suggestion in error.get("suggestions", []):
                console.print(f"    💡 {suggestion}")

    # Warnings
    if report.warnings:
        console.print(f"\n⚠️  Warnings ({report.warning_count}):")
        for warning in report.warnings:
            console.print(f"  • {warning['field']}: {warning['message']}")
            console.print(f"    💡 {warning['recommendation']}")

    # Suggestions
    if report.suggestions:
        console.print(f"\n💡 Suggestions ({report.suggestions_count}):")
        for suggestion in report.suggestions:
            console.print(f"  • {suggestion['field']}: {suggestion['message']}")
            console.print(f"    ➡️  {suggestion['recommendation']}")


def _save_configuration(config, output: Path, output_format: str):
    """Save configuration to file."""
    if output_format == "json":
        with open(output, "w") as f:
            json.dump(config.model_dump(), f, indent=2, default=str)
    elif output_format == "yaml":
        import yaml

        with open(output, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)

    console.print(f"\n💾 Configuration saved to: {output}")


if __name__ == "__main__":
    config_cli()
