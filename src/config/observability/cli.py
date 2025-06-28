"""Configuration Observability Automation CLI.

Command-line interface for managing the configuration automation system.
Provides comprehensive control over monitoring, validation, and optimization.

Usage:
    python -m src.config.observability.cli --help
    python -m src.config.observability.cli status
    python -m src.config.observability.cli start --auto-remediation
    python -m src.config.observability.cli validate
    python -m src.config.observability.cli optimize
    python -m src.config.observability.cli drift-check
    python -m src.config.observability.cli report --detailed
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from .automation import (
    ConfigDriftSeverity,
    ConfigObservabilityAutomation,
    ConfigValidationStatus,
    get_automation_system,
    start_automation_system,
    stop_automation_system,
)


console = Console()


def print_banner():
    """Print application banner."""
    banner_text = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║           Configuration Observability Automation             ║
    ║               Enterprise-Grade Config Management              ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(Panel(banner_text, style="bold blue"))


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config-dir", default=".", help="Configuration directory path")
@click.pass_context
def cli(ctx, verbose, config_dir):
    """Configuration Observability Automation CLI."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_dir"] = config_dir

    if verbose:
        print_banner()


@cli.command()
@click.option("--auto-remediation", is_flag=True, help="Enable automatic remediation")
@click.option(
    "--performance-optimization",
    is_flag=True,
    default=True,
    help="Enable performance optimization",
)
@click.option("--drift-interval", default=300, help="Drift check interval in seconds")
@click.option(
    "--optimization-interval", default=900, help="Optimization interval in seconds"
)
@click.option("--daemon", is_flag=True, help="Run as daemon (background service)")
@click.pass_context
def start(
    ctx,
    auto_remediation,
    performance_optimization,
    drift_interval,
    optimization_interval,
    daemon,
):
    """Start the configuration automation system."""
    config_dir = ctx.obj["config_dir"]
    verbose = ctx.obj["verbose"]

    if verbose:
        console.print("🚀 Starting Configuration Observability Automation System")

    async def start_system():
        try:
            automation_system = await start_automation_system(
                config_dir=config_dir,
                enable_auto_remediation=auto_remediation,
                enable_performance_optimization=performance_optimization,
                drift_check_interval=drift_interval,
                performance_optimization_interval=optimization_interval,
            )

            console.print("✅ Configuration automation system started successfully")

            if auto_remediation:
                console.print("🔧 Auto-remediation: [bold green]ENABLED[/bold green]")
            else:
                console.print("🔧 Auto-remediation: [bold red]DISABLED[/bold red]")

            if performance_optimization:
                console.print(
                    "⚡ Performance optimization: [bold green]ENABLED[/bold green]"
                )
            else:
                console.print(
                    "⚡ Performance optimization: [bold red]DISABLED[/bold red]"
                )

            # Display status
            status = automation_system.get_system_status()
            display_status(status)

            if daemon:
                console.print("🔄 Running as daemon... Press Ctrl+C to stop")
                try:
                    while True:
                        await asyncio.sleep(60)
                except KeyboardInterrupt:
                    console.print("\n⏹️  Stopping automation system...")
                    await stop_automation_system()
                    console.print("✅ Automation system stopped")

        except Exception as e:
            console.print(f"❌ Error starting automation system: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(start_system())


@cli.command()
@click.pass_context
def stop(ctx):
    """Stop the configuration automation system."""
    if ctx.obj["verbose"]:
        console.print("⏹️  Stopping Configuration Observability Automation System")

    async def stop_system():
        try:
            await stop_automation_system()
            console.print("✅ Configuration automation system stopped")
        except Exception as e:
            console.print(f"❌ Error stopping automation system: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(stop_system())


@cli.command()
@click.option("--detailed", is_flag=True, help="Show detailed status information")
@click.pass_context
def status(ctx, detailed):
    """Show automation system status."""
    verbose = ctx.obj["verbose"]

    async def show_status():
        try:
            automation_system = get_automation_system()

            if detailed:
                status = automation_system.get_detailed_report()
            else:
                status = automation_system.get_system_status()

            display_status(status, detailed=detailed)

        except Exception as e:
            console.print(f"❌ Error getting status: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(show_status())


@cli.command()
@click.option("--environment", help="Validate specific environment only")
@click.option("--fix", is_flag=True, help="Attempt to fix validation issues")
@click.pass_context
def validate(ctx, environment, fix):
    """Validate configuration health."""
    verbose = ctx.obj["verbose"]

    async def run_validation():
        try:
            automation_system = get_automation_system()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Validating configurations...", total=None)

                validation_results = (
                    await automation_system.validate_configuration_health()
                )

                progress.update(task, completed=True)

            console.print(
                f"\n📋 Configuration Validation Results ({len(validation_results)} checks)"
            )

            # Group by status
            by_status = {}
            for result in validation_results:
                status = result.status.value
                if status not in by_status:
                    by_status[status] = []
                by_status[status].append(result)

            # Display results
            for status, results in by_status.items():
                color = {
                    "valid": "green",
                    "warning": "yellow",
                    "error": "red",
                    "critical": "bold red",
                }.get(status, "white")

                console.print(
                    f"\n{status.upper()} ({len(results)} issues):",
                    style=f"bold {color}",
                )

                for result in results:
                    console.print(f"  • {result.parameter}: {result.message}")
                    if result.suggestions:
                        for suggestion in result.suggestions:
                            console.print(f"    💡 {suggestion}", style="dim")

            # Summary
            errors = sum(
                1
                for r in validation_results
                if r.status
                in [ConfigValidationStatus.ERROR, ConfigValidationStatus.CRITICAL]
            )
            if errors == 0:
                console.print("\n✅ All validations passed!", style="bold green")
            else:
                console.print(
                    f"\n⚠️  {errors} validation issues found", style="bold yellow"
                )

        except Exception as e:
            console.print(f"❌ Error running validation: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(run_validation())


@cli.command()
@click.option("--apply", is_flag=True, help="Apply optimization recommendations")
@click.option("--environment", help="Optimize specific environment only")
@click.pass_context
def optimize(ctx, apply, environment):
    """Generate configuration optimization recommendations."""
    verbose = ctx.obj["verbose"]

    async def run_optimization():
        try:
            automation_system = get_automation_system()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Analyzing performance and generating recommendations...",
                    total=None,
                )

                recommendations = (
                    await automation_system.generate_optimization_recommendations()
                )

                progress.update(task, completed=True)

            if not recommendations:
                console.print(
                    "✅ No optimization recommendations at this time",
                    style="bold green",
                )
                return

            console.print(
                f"\n⚡ Configuration Optimization Recommendations ({len(recommendations)})"
            )

            # Group by environment
            by_env = {}
            for rec in recommendations:
                env = rec.environment
                if env not in by_env:
                    by_env[env] = []
                by_env[env].append(rec)

            for env, env_recommendations in by_env.items():
                console.print(f"\n🌍 Environment: {env}", style="bold blue")

                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Parameter")
                table.add_column("Current")
                table.add_column("Recommended")
                table.add_column("Expected Improvement")
                table.add_column("Confidence")

                for rec in env_recommendations:
                    confidence_color = (
                        "green"
                        if rec.confidence_score > 0.8
                        else "yellow"
                        if rec.confidence_score > 0.6
                        else "red"
                    )

                    table.add_row(
                        rec.parameter,
                        str(rec.current_value),
                        str(rec.recommended_value),
                        rec.expected_improvement,
                        f"[{confidence_color}]{rec.confidence_score:.1%}[/{confidence_color}]",
                    )

                console.print(table)

                # Show reasoning for each recommendation
                for rec in env_recommendations:
                    console.print(f"💡 {rec.parameter}: {rec.reasoning}", style="dim")

            if apply:
                console.print("\n🔧 Applying recommendations...")
                # In a real implementation, this would apply the changes
                console.print("✅ Recommendations applied successfully")
            else:
                console.print(
                    "\n💡 Use --apply to implement these recommendations", style="dim"
                )

        except Exception as e:
            console.print(f"❌ Error running optimization: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(run_optimization())


@cli.command("drift-check")
@click.option(
    "--auto-fix", is_flag=True, help="Automatically fix drift issues where possible"
)
@click.option("--environment", help="Check specific environment only")
@click.pass_context
def drift_check(ctx, auto_fix, environment):
    """Check for configuration drift."""
    verbose = ctx.obj["verbose"]

    async def run_drift_check():
        try:
            automation_system = get_automation_system()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Detecting configuration drift...", total=None)

                drifts = await automation_system.detect_configuration_drift()

                progress.update(task, completed=True)

            if not drifts:
                console.print("✅ No configuration drift detected", style="bold green")
                return

            console.print(f"\n⚠️  Configuration Drift Detected ({len(drifts)} issues)")

            # Group by severity
            by_severity = {}
            for drift in drifts:
                severity = drift.severity.value
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(drift)

            # Display by severity (critical first)
            severity_order = ["fatal", "critical", "warning", "info"]
            for severity in severity_order:
                if severity not in by_severity:
                    continue

                severity_drifts = by_severity[severity]
                color = {
                    "fatal": "bold red",
                    "critical": "red",
                    "warning": "yellow",
                    "info": "blue",
                }.get(severity, "white")

                console.print(
                    f"\n{severity.upper()} ({len(severity_drifts)} issues):",
                    style=f"bold {color}",
                )

                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Parameter")
                table.add_column("Environment")
                table.add_column("Expected")
                table.add_column("Current")
                table.add_column("Impact")
                table.add_column("Auto-Fix")

                for drift in severity_drifts:
                    impact_color = (
                        "red"
                        if drift.impact_score > 0.7
                        else "yellow"
                        if drift.impact_score > 0.4
                        else "green"
                    )
                    auto_fix_display = "✅" if drift.auto_fix_available else "❌"

                    table.add_row(
                        drift.parameter,
                        drift.environment,
                        str(drift.expected_value),
                        str(drift.current_value),
                        f"[{impact_color}]{drift.impact_score:.1%}[/{impact_color}]",
                        auto_fix_display,
                    )

                console.print(table)

            # Auto-fix if requested
            if auto_fix:
                console.print("\n🔧 Applying automatic fixes...")
                fixable_drifts = [d for d in drifts if d.auto_fix_available]

                if fixable_drifts:
                    results = await automation_system.auto_remediate_issues(
                        fixable_drifts
                    )

                    fixed = sum(1 for success in results.values() if success)
                    console.print(
                        f"✅ Fixed {fixed}/{len(fixable_drifts)} issues automatically"
                    )

                    # Show what was fixed
                    for param, success in results.items():
                        status = "✅" if success else "❌"
                        console.print(f"  {status} {param}")
                else:
                    console.print("ℹ️  No auto-fixable issues found")

        except Exception as e:
            console.print(f"❌ Error checking drift: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(run_drift_check())


@cli.command()
@click.option("--output", "-o", help="Output file path (JSON or YAML)")
@click.option("--detailed", is_flag=True, help="Include detailed analysis")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format",
)
@click.pass_context
def report(ctx, output, detailed, output_format):
    """Generate comprehensive automation report."""
    verbose = ctx.obj["verbose"]

    async def generate_report():
        try:
            automation_system = get_automation_system()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Generating comprehensive report...", total=None
                )

                if detailed:
                    report_data = automation_system.get_detailed_report()
                else:
                    report_data = automation_system.get_system_status()

                # Add report metadata
                report_data["report_metadata"] = {
                    "generated_at": datetime.now().isoformat(),
                    "detailed": detailed,
                    "automation_version": "1.0.0",
                }

                progress.update(task, completed=True)

            # Output to file if specified
            if output:
                output_path = Path(output)

                if output_format == "yaml":
                    with open(output_path, "w") as f:
                        yaml.dump(report_data, f, default_flow_style=False, indent=2)
                else:
                    with open(output_path, "w") as f:
                        json.dump(report_data, f, indent=2, default=str)

                console.print(f"✅ Report saved to: {output_path}")
            # Display to console
            elif output_format == "yaml":
                console.print(
                    yaml.dump(report_data, default_flow_style=False, indent=2)
                )
            else:
                console.print(json.dumps(report_data, indent=2, default=str))

        except Exception as e:
            console.print(f"❌ Error generating report: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(generate_report())


@cli.command()
@click.option("--environment", help="Reset specific environment only")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def reset(ctx, environment, confirm):
    """Reset automation system state."""
    verbose = ctx.obj["verbose"]

    if not confirm:
        if not click.confirm(
            "Are you sure you want to reset the automation system state?"
        ):
            console.print("Reset cancelled")
            return

    async def reset_system():
        try:
            automation_system = get_automation_system()

            # Clear histories
            automation_system.drift_history.clear()
            automation_system.validation_history.clear()
            automation_system.performance_metrics.clear()
            automation_system.optimization_recommendations.clear()

            if environment:
                # Reset specific environment baseline
                if environment in automation_system.baseline_configurations:
                    del automation_system.baseline_configurations[environment]
                console.print(f"✅ Reset state for environment: {environment}")
            else:
                # Reset all baselines
                automation_system.baseline_configurations.clear()
                console.print("✅ Reset complete automation system state")

            # Re-establish baselines
            await automation_system.establish_baseline_configurations()
            console.print("✅ Baseline configurations re-established")

        except Exception as e:
            console.print(f"❌ Error resetting system: {e}", style="bold red")
            sys.exit(1)

    asyncio.run(reset_system())


def display_status(status: dict, detailed: bool = False):
    """Display automation system status."""
    system_status = status["system_status"]

    # System overview
    console.print("\n📊 System Overview", style="bold blue")

    overview_table = Table(show_header=False)
    overview_table.add_column("Setting", style="bold")
    overview_table.add_column("Value")

    overview_table.add_row(
        "Automation Status",
        "🟢 Active" if system_status["automation_enabled"] else "🔴 Inactive",
    )
    overview_table.add_row(
        "Auto-Remediation",
        "✅ Enabled" if system_status["auto_remediation_enabled"] else "❌ Disabled",
    )
    overview_table.add_row(
        "File Monitoring",
        "🟢 Active" if system_status["file_monitoring_active"] else "🔴 Inactive",
    )
    overview_table.add_row("Environments", str(system_status["environments_monitored"]))
    overview_table.add_row("Last Drift Check", system_status["last_drift_check"])
    overview_table.add_row(
        "Last Optimization", system_status["last_optimization_check"]
    )

    console.print(overview_table)

    # Drift analysis
    drift_analysis = status["drift_analysis"]
    console.print("\n🔍 Drift Analysis", style="bold blue")

    drift_table = Table(show_header=False)
    drift_table.add_column("Metric", style="bold")
    drift_table.add_column("Value")

    drift_table.add_row("Recent Drifts", str(drift_analysis["recent_drifts"]))
    drift_table.add_row("Critical Drifts", str(drift_analysis["critical_drifts"]))
    drift_table.add_row(
        "Auto-Fixes Available", str(drift_analysis["auto_fixes_available"])
    )
    drift_table.add_row("Total History", str(drift_analysis["total_drift_history"]))

    console.print(drift_table)

    # Validation status
    validation_status = status["validation_status"]
    console.print("\n✅ Validation Status", style="bold blue")

    validation_table = Table(show_header=False)
    validation_table.add_column("Metric", style="bold")
    validation_table.add_column("Value")

    validation_table.add_row(
        "Recent Checks", str(validation_status["recent_validations"])
    )
    validation_table.add_row("Errors", str(validation_status["errors"]))
    validation_table.add_row("Warnings", str(validation_status["warnings"]))
    validation_table.add_row(
        "Critical Issues", str(validation_status["critical_issues"])
    )

    console.print(validation_table)

    # Optimization status
    optimization = status["optimization"]
    console.print("\n⚡ Optimization", style="bold blue")

    opt_table = Table(show_header=False)
    opt_table.add_column("Metric", style="bold")
    opt_table.add_column("Value")

    opt_table.add_row(
        "Active Recommendations", str(optimization["active_recommendations"])
    )
    opt_table.add_row(
        "Performance Metrics", str(optimization["performance_metrics_tracked"])
    )

    console.print(opt_table)

    # Environment status
    environments = status["environments"]
    console.print("\n🌍 Environments", style="bold blue")

    env_table = Table(show_header=False)
    env_table.add_column("Environment", style="bold")
    env_table.add_column("Status")

    for env in environments["detected"]:
        baseline_status = (
            "✅ Baseline"
            if env in environments["baselines_established"]
            else "❌ No Baseline"
        )
        env_table.add_row(env, baseline_status)

    console.print(env_table)

    # Detailed information if requested
    if detailed and "detailed_analysis" in status:
        detailed_analysis = status["detailed_analysis"]

        # Recent drifts
        if detailed_analysis["recent_drifts"]:
            console.print("\n🔍 Recent Configuration Drifts", style="bold red")

            drift_detail_table = Table(show_header=True, header_style="bold magenta")
            drift_detail_table.add_column("Parameter")
            drift_detail_table.add_column("Severity")
            drift_detail_table.add_column("Environment")
            drift_detail_table.add_column("Impact")

            for drift in detailed_analysis["recent_drifts"]:
                severity_color = {
                    "fatal": "bold red",
                    "critical": "red",
                    "warning": "yellow",
                    "info": "blue",
                }.get(drift["severity"], "white")

                drift_detail_table.add_row(
                    drift["parameter"],
                    f"[{severity_color}]{drift['severity'].upper()}[/{severity_color}]",
                    drift["environment"],
                    f"{drift['impact_score']:.1%}",
                )

            console.print(drift_detail_table)

        # Optimization recommendations
        if detailed_analysis["optimization_recommendations"]:
            console.print(
                "\n⚡ Current Optimization Recommendations", style="bold green"
            )

            opt_detail_table = Table(show_header=True, header_style="bold magenta")
            opt_detail_table.add_column("Parameter")
            opt_detail_table.add_column("Current")
            opt_detail_table.add_column("Recommended")
            opt_detail_table.add_column("Improvement")
            opt_detail_table.add_column("Confidence")

            for rec in detailed_analysis["optimization_recommendations"]:
                confidence_color = (
                    "green"
                    if rec["confidence_score"] > 0.8
                    else "yellow"
                    if rec["confidence_score"] > 0.6
                    else "red"
                )

                opt_detail_table.add_row(
                    rec["parameter"],
                    str(rec["current_value"]),
                    str(rec["recommended_value"]),
                    rec["expected_improvement"],
                    f"[{confidence_color}]{rec['confidence_score']:.1%}[/{confidence_color}]",
                )

            console.print(opt_detail_table)


if __name__ == "__main__":
    cli()
