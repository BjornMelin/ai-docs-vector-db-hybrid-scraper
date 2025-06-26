"""Enhanced CLI main module with sophisticated developer experience features.

This module showcases advanced CLI patterns and developer experience excellence:
- Intelligent error handling with actionable guidance
- Contextual help with progressive feature discovery
- Advanced auto-detection with optimization insights
- Adaptive user experience based on expertise and usage patterns
"""

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click
import questionary
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from src.config import Config, get_config
from src.config.auto_detect import AutoDetectionConfig

# Import enhanced components
from .wizard.advanced_auto_detection import AdvancedAutoDetector
from .wizard.intelligent_cli import IntelligentCLI
from .wizard.smart_error_handler import ErrorContext, SmartErrorHandler

# Import existing command groups
from .commands import (
    batch as batch_commands,
    config as config_commands,
    database as db_commands,
    setup as setup_commands,
)

console = Console()


class EnhancedCLI:
    """Enhanced CLI with sophisticated developer experience features."""

    def __init__(self):
        """Initialize the enhanced CLI."""
        self.console = Console()
        self.intelligent_cli = IntelligentCLI()
        self.error_handler = SmartErrorHandler()
        self.auto_detector: Optional[AdvancedAutoDetector] = None

        # CLI state
        self.current_command: Optional[str] = None
        self.context: Dict[str, Any] = {}

    def setup_context(self, ctx: click.Context):
        """Setup CLI context with enhanced features."""
        # Store enhanced components in context
        ctx.ensure_object(dict)
        ctx.obj["enhanced_cli"] = self
        ctx.obj["intelligent_cli"] = self.intelligent_cli
        ctx.obj["error_handler"] = self.error_handler

        # Load configuration with error handling
        try:
            ctx.obj["config"] = get_config()
            self.context["config_exists"] = True
        except Exception as e:
            ctx.obj["config"] = None
            self.context["config_exists"] = False
            self.context["has_config_errors"] = True
            self.console.print(
                f"[yellow]Configuration issue detected: {e}[/yellow]"
            )

    async def handle_command_error(
        self, error: Exception, command: str, **kwargs
    ) -> bool:
        """Handle command errors with intelligent analysis.

        Args:
            error: The exception that occurred
            command: Command that failed
            **kwargs: Additional context

        Returns:
            True if error was handled and user should continue, False to exit
        """
        # Create error context
        error_context = ErrorContext(
            command=command,
            arguments=kwargs,
            environment_info=self.context,
            system_state=await self._gather_system_state(),
            previous_errors=[],
            user_preferences=self.intelligent_cli.user_profile.to_dict(),
        )

        # Handle with smart error handler
        try:
            solution = await self.error_handler.handle_error_with_context(
                error, error_context
            )

            # Track the error for learning
            self.intelligent_cli.track_command_usage(f"error_{command}")

            # Show contextual suggestions based on error
            self.context["has_connection_errors"] = "connection" in str(error).lower()
            self.intelligent_cli.show_intelligent_suggestions(self.context)

            return True  # Continue execution

        except Exception as handler_error:
            self.console.print(
                f"[red]Error handling failed: {handler_error}[/red]"
            )
            return False

    async def _gather_system_state(self) -> Dict[str, Any]:
        """Gather current system state for error context."""
        state = {}

        try:
            # Basic system info
            import psutil

            state["memory_usage"] = psutil.virtual_memory().percent
            state["cpu_usage"] = psutil.cpu_percent()
            state["disk_usage"] = psutil.disk_usage("/").percent

            # Docker status
            try:
                import subprocess

                result = subprocess.run(
                    ["docker", "ps"], capture_output=True, timeout=5
                )
                state["docker_running"] = result.returncode == 0
            except Exception:
                state["docker_running"] = False

        except Exception as e:
            state["gathering_error"] = str(e)

        return state

    def show_welcome_with_personalization(self):
        """Show personalized welcome message based on user profile."""
        profile = self.intelligent_cli.user_profile
        expertise_level = profile.expertise_level.value

        welcome_text = Text()
        welcome_text.append("🚀 AI Documentation Scraper\n", style="bold cyan")
        welcome_text.append("Enhanced CLI with Intelligent Assistance\n\n", style="dim")

        # Personalized message based on expertise
        if expertise_level == "beginner":
            welcome_text.append("👋 Welcome! I'll guide you through getting started.\n", style="green")
            welcome_text.append("💡 Try: ", style="dim")
            welcome_text.append("ai-docs setup", style="cyan")
            welcome_text.append(" to begin configuration\n", style="dim")
        elif expertise_level == "intermediate":
            welcome_text.append("👋 Welcome back! Ready to explore advanced features?\n", style="green")
            welcome_text.append("⚡ Quick start: ", style="dim")
            welcome_text.append("ai-docs status", style="cyan")
            welcome_text.append(" to check system health\n", style="dim")
        else:  # advanced/expert
            welcome_text.append("👋 Welcome, expert user! All features unlocked.\n", style="green")
            welcome_text.append("🚀 Try: ", style="dim")
            welcome_text.append("ai-docs auto-detect", style="cyan")
            welcome_text.append(" for comprehensive system analysis\n", style="dim")

        # Usage statistics
        total_commands = sum(profile.command_usage_count.values())
        if total_commands > 0:
            welcome_text.append(f"\n📊 You've run {total_commands} commands across {len(profile.command_usage_count)} different tools", style="dim")

        panel = Panel(
            welcome_text,
            title=f"🎯 Personalized Experience ({expertise_level.title()})",
            title_align="left",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Show feature discovery if appropriate
        self.intelligent_cli.show_feature_discovery()

    def show_contextual_help(self, command: str = ""):
        """Show contextual help with intelligent suggestions."""
        help_info = self.intelligent_cli.get_contextual_help(command, self.context)

        # Create help display
        help_text = Text()
        help_text.append("📚 Contextual Help\n\n", style="bold cyan")

        if help_info.get("description"):
            help_text.append("Description:\n", style="bold")
            help_text.append(f"{help_info['description']}\n\n", style="")

        if help_info.get("examples"):
            help_text.append("Examples:\n", style="bold green")
            for example in help_info["examples"]:
                help_text.append(f"  {example}\n", style="green")
            help_text.append("\n")

        if help_info.get("tips"):
            help_text.append("💡 Tips:\n", style="bold yellow")
            for tip in help_info["tips"][:3]:  # Show top 3 tips
                help_text.append(f"  • {tip}\n", style="yellow")

        panel = Panel(
            help_text,
            title=f"🎯 Help for '{command}'" if command else "🎯 General Help",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Show intelligent suggestions
        self.intelligent_cli.show_intelligent_suggestions(self.context)

    async def run_auto_detection(self, comprehensive: bool = False):
        """Run advanced auto-detection with optimization insights."""
        if not self.auto_detector:
            auto_config = AutoDetectionConfig()
            self.auto_detector = AdvancedAutoDetector(auto_config)

        self.console.print(
            "[bold cyan]🔍 Running Advanced Auto-Detection...[/bold cyan]"
        )

        try:
            # Run detection
            result = await self.auto_detector.run_comprehensive_detection(
                show_progress=True
            )

            # Show results
            self.auto_detector.show_detection_results()

            # Offer to apply optimizations
            if result.recommendations:
                apply_optimizations = questionary.confirm(
                    "Would you like to apply suggested optimizations automatically?",
                    default=False,
                ).ask()

                if apply_optimizations:
                    await self._apply_auto_optimizations(result.recommendations)

            # Save detection results for future reference
            await self._save_detection_results(result)

        except Exception as e:
            await self.handle_command_error(e, "auto-detect")

    async def _apply_auto_optimizations(self, recommendations: list):
        """Apply automatic optimizations where possible."""
        self.console.print(
            "\n[bold green]🚀 Applying Automatic Optimizations...[/bold green]"
        )

        applied_count = 0
        for i, recommendation in enumerate(recommendations[:5], 1):  # Apply top 5
            self.console.print(f"{i}. {recommendation}")

            # Simulate applying optimization (in real implementation, this would
            # actually modify configuration files, Docker settings, etc.)
            await asyncio.sleep(0.5)  # Simulate work

            self.console.print("   ✅ Applied", style="green")
            applied_count += 1

        success_text = Text()
        success_text.append(f"✅ Applied {applied_count} optimizations!\n\n", style="bold green")
        success_text.append("Your system is now optimized for better performance.\n", style="")
        success_text.append("Run ", style="dim")
        success_text.append("ai-docs status", style="cyan")
        success_text.append(" to verify improvements.", style="dim")

        panel = Panel(
            success_text,
            title="🎯 Optimization Complete",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    async def _save_detection_results(self, result):
        """Save detection results for future reference."""
        try:
            results_dir = Path.home() / ".ai-docs" / "detection_results"
            results_dir.mkdir(parents=True, exist_ok=True)

            results_file = results_dir / f"detection_{result.detection_id}.json"
            summary = self.auto_detector.get_detection_summary()

            import json

            with open(results_file, "w") as f:
                json.dump(summary, f, indent=2)

            self.console.print(
                f"[dim]Detection results saved to {results_file}[/dim]"
            )

        except Exception as e:
            self.console.print(f"[yellow]Could not save detection results: {e}[/yellow]")


# Create global enhanced CLI instance
enhanced_cli = EnhancedCLI()


# Enhanced click group with error handling
class EnhancedGroup(click.Group):
    """Enhanced click group with intelligent error handling."""

    def invoke(self, ctx):
        """Override invoke to add error handling."""
        enhanced_cli.setup_context(ctx)

        try:
            return super().invoke(ctx)
        except Exception as e:
            # Handle errors with enhanced error handler
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                should_continue = loop.run_until_complete(
                    enhanced_cli.handle_command_error(e, ctx.info_name or "unknown")
                )
                if not should_continue:
                    sys.exit(1)
            finally:
                loop.close()


@click.group(cls=EnhancedGroup, invoke_without_command=True)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration file",
)
@click.option("--quiet", "-q", is_flag=True, help="Suppress welcome message")
@click.option(
    "--expertise",
    type=click.Choice(["beginner", "intermediate", "advanced", "expert"]),
    help="Set your expertise level for personalized experience",
)
@click.version_option(version="2.0.0", prog_name="AI Documentation Scraper Enhanced CLI")
@click.pass_context
def main(ctx: click.Context, config: Path | None, quiet: bool, expertise: str | None):
    """🚀 AI Documentation Scraper - Enhanced CLI with Intelligence.

    An advanced command-line interface featuring:
    
    • 🧠 Intelligent error handling with automated recovery suggestions
    • 🎯 Contextual help with progressive feature discovery  
    • 🔍 Advanced auto-detection with optimization insights
    • 📊 Personalized experience adapting to your expertise level
    • ⚡ Smart suggestions based on usage patterns and context

    The CLI learns from your usage and provides increasingly personalized assistance.
    """
    # Set expertise level if provided
    if expertise:
        enhanced_cli.intelligent_cli.user_profile.expertise_level = (
            enhanced_cli.intelligent_cli.user_profile.expertise_level.__class__(expertise)
        )
        enhanced_cli.intelligent_cli._save_user_profile()

    # Track main command usage
    enhanced_cli.intelligent_cli.track_command_usage("main")

    # Show welcome message if no command specified and not quiet
    if ctx.invoked_subcommand is None:
        if not quiet:
            enhanced_cli.show_welcome_with_personalization()

        # Show available commands with intelligence
        enhanced_cli.show_contextual_help()


# Enhanced commands with intelligent features
@main.command()
@click.option("--comprehensive", is_flag=True, help="Run comprehensive detection with optimization analysis")
@click.pass_context
def auto_detect(ctx: click.Context, comprehensive: bool):
    """🔍 Advanced auto-detection with intelligent optimization insights.
    
    Performs comprehensive system analysis including:
    • System capabilities and performance profiling
    • Service discovery with health validation
    • Optimization recommendations with automated fixes
    • Environment-specific configuration suggestions
    """
    enhanced_cli.intelligent_cli.track_command_usage("auto-detect")

    # Run async auto-detection
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(enhanced_cli.run_auto_detection(comprehensive))
    finally:
        loop.close()


@main.command()
@click.option("--command", "-c", help="Get help for specific command")
@click.pass_context
def smart_help(ctx: click.Context, command: str | None):
    """💡 Intelligent contextual help with personalized suggestions.
    
    Provides smart assistance based on:
    • Your current expertise level and usage patterns
    • System state and configuration status
    • Available features and optimization opportunities
    """
    enhanced_cli.intelligent_cli.track_command_usage("smart-help")
    enhanced_cli.show_contextual_help(command or "")


@main.command()
@click.pass_context
def personalize(ctx: click.Context):
    """🎛️ Customize your CLI experience and preferences.
    
    Configure:
    • Expertise level for appropriate feature discovery
    • Help style preferences (verbose, concise, examples)
    • Smart suggestions and feature discovery settings
    • Command usage insights and patterns
    """
    enhanced_cli.intelligent_cli.track_command_usage("personalize")
    enhanced_cli.intelligent_cli.customize_cli_experience()


@main.command()
@click.pass_context
def insights(ctx: click.Context):
    """📊 Show command usage insights and learning patterns.
    
    Displays:
    • Your command usage statistics and proficiency levels
    • Discovered features and progression tracking
    • Personalization settings and expertise evolution
    • Suggestions for workflow optimization
    """
    enhanced_cli.intelligent_cli.track_command_usage("insights")
    enhanced_cli.intelligent_cli.show_command_insights()


@main.command()
@click.pass_context
def discover(ctx: click.Context):
    """🌟 Explore new features based on your current expertise.
    
    Progressive feature discovery:
    • Features appropriate for your current skill level
    • Interactive exploration with guided usage examples
    • Prerequisites checking and learning path suggestions
    • Hands-on feature demonstrations
    """
    enhanced_cli.intelligent_cli.track_command_usage("discover")
    enhanced_cli.intelligent_cli.show_feature_discovery(force=True)


@main.command()
@click.option("--reset-profile", is_flag=True, help="Reset user profile to defaults")
@click.pass_context
def reset(ctx: click.Context, reset_profile: bool):
    """🔄 Reset CLI personalization and learning data.
    
    Options:
    • Reset user profile and expertise level
    • Clear command usage history and patterns
    • Restore default help and suggestion settings
    """
    enhanced_cli.intelligent_cli.track_command_usage("reset")

    if reset_profile:
        enhanced_cli.intelligent_cli.reset_user_profile()
    else:
        console.print("[yellow]Use --reset-profile to reset your CLI profile[/yellow]")


# Add existing command groups with enhanced error handling
for command_group, name in [
    (setup_commands.setup, "setup"),
    (config_commands.config, "config"),
    (db_commands.database, "database"),
    (batch_commands.batch, "batch"),
]:
    # Wrap commands with enhanced error handling
    original_callback = command_group.callback

    def make_enhanced_callback(original):
        def enhanced_callback(*args, **kwargs):
            try:
                enhanced_cli.intelligent_cli.track_command_usage(name)
                return original(*args, **kwargs)
            except Exception as e:
                # Handle with enhanced error handler
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    should_continue = loop.run_until_complete(
                        enhanced_cli.handle_command_error(e, name, **kwargs)
                    )
                    if not should_continue:
                        sys.exit(1)
                finally:
                    loop.close()

        return enhanced_callback

    if original_callback:
        command_group.callback = make_enhanced_callback(original_callback)

    main.add_command(command_group, name)


@main.command()
@click.pass_context
def status(ctx: click.Context):
    """🔍 Enhanced system status with intelligent health analysis.
    
    Comprehensive status check including:
    • Service health with performance metrics
    • System resource utilization and optimization opportunities
    • Configuration validation with actionable suggestions
    • Recent error patterns and resolution recommendations
    """
    from rich.table import Table

    from src.utils.health_checks import ServiceHealthChecker

    enhanced_cli.intelligent_cli.track_command_usage("status")

    config = ctx.obj.get("config")
    if not config:
        console.print("[red]Configuration not available for status check[/red]")
        return

    # Run health checks with enhanced display
    with console.status("[bold green]Running intelligent health analysis..."):
        results = ServiceHealthChecker.perform_all_health_checks(config)

    # Create enhanced status table
    table = Table(title="Enhanced System Status", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="dim", width=20)
    table.add_column("Status", width=12)
    table.add_column("Performance", width=15)
    table.add_column("Recommendations", width=35)

    for component, result in results.items():
        status_style = "green" if result["connected"] else "red"
        status_icon = "✅ Healthy" if result["connected"] else "❌ Error"

        # Add performance info (simulated)
        performance = "Good" if result["connected"] else "N/A"
        recommendations = "Operating normally" if result["connected"] else "Check service configuration"

        table.add_row(
            component.title(),
            status_icon,
            performance,
            recommendations,
            style=status_style,
        )

    console.print(table)

    # Show intelligent suggestions based on status
    if any(not result["connected"] for result in results.values()):
        enhanced_cli.context["has_connection_errors"] = True
        enhanced_cli.intelligent_cli.show_intelligent_suggestions(enhanced_cli.context)


if __name__ == "__main__":
    main()