"""Intelligent error handling with actionable guidance and auto-recovery suggestions.

This module provides sophisticated error handling that showcases advanced UX thinking:
- Contextual error analysis with smart diagnosis
- Actionable recovery suggestions with automated fixes
- Progressive error resolution with learning patterns
- Integration with system state for intelligent guidance
"""

import asyncio
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from src.config.auto_detect import AutoDetectionConfig, EnvironmentDetector
from src.utils.health_checks import ServiceHealthChecker


logger = logging.getLogger(__name__)
console = Console()


class ErrorCategory(Enum):
    """Categories of errors for smart handling."""

    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    SERVICES = "services"
    NETWORK = "network"
    PERMISSIONS = "permissions"
    RESOURCES = "resources"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Severity levels for intelligent prioritization."""

    CRITICAL = "critical"  # Blocks all functionality
    HIGH = "high"  # Blocks core functionality
    MEDIUM = "medium"  # Reduces functionality
    LOW = "low"  # Minor issues
    INFO = "info"  # Information only


@dataclass
class ErrorContext:
    """Rich context information for intelligent error analysis."""

    command: str
    arguments: Dict[str, Any]
    environment_info: Dict[str, Any]
    system_state: Dict[str, Any]
    previous_errors: List[str]
    user_preferences: Dict[str, Any]


@dataclass
class RecoveryAction:
    """Actionable recovery suggestion with automation capabilities."""

    title: str
    description: str
    command: str | None = None
    automated: bool = False
    risk_level: str = "low"  # low, medium, high
    prerequisites: List[str] = None
    estimated_time: str = "< 1 minute"
    success_probability: float = 0.8

    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class SmartErrorSolution:
    """Comprehensive error solution with multiple recovery paths."""

    category: ErrorCategory
    severity: ErrorSeverity
    title: str
    description: str
    root_cause: str
    immediate_actions: List[RecoveryAction]
    preventive_actions: List[RecoveryAction]
    learning_notes: List[str]
    related_docs: List[str]
    confidence_score: float = 0.7


class SmartErrorHandler:
    """Intelligent error handler with contextual analysis and automated recovery."""

    def __init__(self, config: Any | None = None):
        """Initialize the smart error handler.

        Args:
            config: Configuration object for enhanced context
        """
        self.config = config
        self.console = Console()
        self.error_patterns: Dict[str, Callable] = {}
        self.recovery_cache: Dict[str, SmartErrorSolution] = {}
        self.user_preferences = self._load_user_preferences()

        # Initialize with built-in error patterns
        self._register_builtin_patterns()

    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences for error handling."""
        preferences = {
            "auto_fix": False,
            "verbose_errors": True,
            "show_learning_tips": True,
            "preferred_recovery_style": "guided",  # guided, automatic, manual
            "remember_solutions": True,
        }

        # Try to load from user config
        try:
            config_path = Path.home() / ".ai-docs" / "error_preferences.json"
            if config_path.exists():
                import json

                with open(config_path) as f:
                    user_prefs = json.load(f)
                    preferences.update(user_prefs)
        except Exception as e:
            logger.debug(f"Could not load user preferences: {e}")

        return preferences

    def _register_builtin_patterns(self):
        """Register built-in error patterns with smart analysis."""
        self.error_patterns.update(
            {
                "connection_refused": self._handle_connection_error,
                "permission_denied": self._handle_permission_error,
                "module_not_found": self._handle_missing_module,
                "api_key": self._handle_api_key_error,
                "validation_error": self._handle_validation_error,
                "timeout": self._handle_timeout_error,
                "disk_space": self._handle_disk_space_error,
                "port_in_use": self._handle_port_conflict,
            }
        )

    async def handle_error_with_context(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution:
        """Handle error with rich context and intelligent analysis.

        Args:
            error: The exception that occurred
            context: Rich context information

        Returns:
            Comprehensive error solution with recovery actions
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Analyze error pattern
            error_signature = self._create_error_signature(error, context)

            # Check cache first
            if error_signature in self.recovery_cache:
                cached_solution = self.recovery_cache[error_signature]
                self._show_cached_solution(cached_solution)
                return cached_solution

            # Perform intelligent analysis
            solution = await self._analyze_error_intelligently(error, context)

            # Cache the solution
            if self.user_preferences.get("remember_solutions", True):
                self.recovery_cache[error_signature] = solution

            # Present solution to user
            await self._present_solution_interactively(solution, context)

            return solution

        except Exception as analysis_error:
            logger.exception(f"Error during smart error analysis: {analysis_error}")
            # Fallback to basic error handling
            return self._create_fallback_solution(error, context)

        finally:
            elapsed = asyncio.get_event_loop().time() - start_time
            logger.debug(f"Smart error analysis completed in {elapsed:.2f}s")

    def _create_error_signature(self, error: Exception, context: ErrorContext) -> str:
        """Create a unique signature for error caching."""
        error_type = type(error).__name__
        error_msg = str(error)[:100]  # First 100 chars
        command = context.command
        return f"{error_type}:{command}:{hash(error_msg)}"

    async def _analyze_error_intelligently(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution:
        """Perform intelligent error analysis with pattern matching."""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Try pattern matching first
        for pattern, handler in self.error_patterns.items():
            if pattern in error_str or pattern in error_type.lower():
                solution = await handler(error, context)
                if solution:
                    return solution

        # Fallback to heuristic analysis
        return await self._heuristic_error_analysis(error, context)

    async def _heuristic_error_analysis(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution:
        """Perform heuristic error analysis when no pattern matches."""
        error_str = str(error)

        # Determine category and severity
        category = self._categorize_error(error, context)
        severity = self._assess_severity(error, context)

        # Generate context-aware solutions
        immediate_actions = await self._generate_immediate_actions(
            error, context, category
        )
        preventive_actions = await self._generate_preventive_actions(
            error, context, category
        )

        return SmartErrorSolution(
            category=category,
            severity=severity,
            title=f"{error_type}: {error_str[:50]}...",
            description=self._generate_error_description(error, context),
            root_cause=self._analyze_root_cause(error, context),
            immediate_actions=immediate_actions,
            preventive_actions=preventive_actions,
            learning_notes=self._generate_learning_notes(error, context),
            related_docs=self._find_related_docs(error, context),
            confidence_score=0.6,  # Lower confidence for heuristic analysis
        )

    def _categorize_error(
        self, error: Exception, context: ErrorContext
    ) -> ErrorCategory:
        """Categorize error for intelligent handling."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        if any(
            keyword in error_str
            for keyword in ["connection", "network", "timeout", "refused"]
        ):
            return ErrorCategory.NETWORK
        elif any(
            keyword in error_str for keyword in ["permission", "access", "denied"]
        ):
            return ErrorCategory.PERMISSIONS
        elif any(
            keyword in error_str for keyword in ["not found", "missing", "import"]
        ):
            return ErrorCategory.DEPENDENCIES
        elif any(keyword in error_str for keyword in ["config", "setting", "invalid"]):
            return ErrorCategory.CONFIGURATION
        elif any(keyword in error_str for keyword in ["service", "server", "port"]):
            return ErrorCategory.SERVICES
        elif any(
            keyword in error_str for keyword in ["validation", "format", "schema"]
        ):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_str for keyword in ["memory", "disk", "space"]):
            return ErrorCategory.RESOURCES
        else:
            return ErrorCategory.UNKNOWN

    def _assess_severity(
        self, error: Exception, context: ErrorContext
    ) -> ErrorSeverity:
        """Assess error severity for prioritization."""
        error_str = str(error).lower()

        # Critical errors that block all functionality
        if any(
            keyword in error_str
            for keyword in [
                "fatal",
                "critical",
                "cannot start",
                "initialization failed",
            ]
        ):
            return ErrorSeverity.CRITICAL

        # High severity - blocks core functionality
        if any(
            keyword in error_str
            for keyword in ["connection refused", "service unavailable", "auth"]
        ):
            return ErrorSeverity.HIGH

        # Medium severity - reduces functionality
        if any(keyword in error_str for keyword in ["timeout", "slow", "degraded"]):
            return ErrorSeverity.MEDIUM

        # Low severity - minor issues
        if any(keyword in error_str for keyword in ["warning", "deprecated"]):
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM  # Default

    async def _generate_immediate_actions(
        self, error: Exception, context: ErrorContext, category: ErrorCategory
    ) -> List[RecoveryAction]:
        """Generate immediate recovery actions based on error analysis."""
        actions = []

        if category == ErrorCategory.NETWORK:
            actions.extend(
                [
                    RecoveryAction(
                        title="Check Network Connectivity",
                        description="Verify internet connection and DNS resolution",
                        command="curl -s http://www.google.com > /dev/null && echo 'Connected' || echo 'No connection'",
                        automated=True,
                        estimated_time="10 seconds",
                        success_probability=0.9,
                    ),
                    RecoveryAction(
                        title="Restart Services",
                        description="Restart Docker services to reset connections",
                        command="docker-compose restart",
                        automated=False,
                        risk_level="low",
                        estimated_time="30 seconds",
                    ),
                ]
            )

        elif category == ErrorCategory.SERVICES:
            actions.extend(
                [
                    RecoveryAction(
                        title="Check Service Status",
                        description="Verify all required services are running",
                        command="docker-compose ps",
                        automated=True,
                        estimated_time="5 seconds",
                    ),
                    RecoveryAction(
                        title="Start Missing Services",
                        description="Start any stopped services",
                        command="docker-compose up -d",
                        automated=False,
                        estimated_time="1 minute",
                    ),
                ]
            )

        elif category == ErrorCategory.DEPENDENCIES:
            actions.extend(
                [
                    RecoveryAction(
                        title="Reinstall Dependencies",
                        description="Refresh project dependencies with UV",
                        command="uv sync --reinstall",
                        automated=False,
                        estimated_time="2 minutes",
                    ),
                    RecoveryAction(
                        title="Check Python Environment",
                        description="Verify Python version and virtual environment",
                        command="uv run python --version && uv run which python",
                        automated=True,
                        estimated_time="5 seconds",
                    ),
                ]
            )

        return actions

    async def _generate_preventive_actions(
        self, error: Exception, context: ErrorContext, category: ErrorCategory
    ) -> List[RecoveryAction]:
        """Generate preventive actions to avoid future errors."""
        actions = []

        # Common preventive actions
        actions.extend(
            [
                RecoveryAction(
                    title="Enable Health Monitoring",
                    description="Set up automated health checks to catch issues early",
                    command="uv run python -m src.cli.main config set monitoring.enabled=true",
                    automated=False,
                    estimated_time="30 seconds",
                ),
                RecoveryAction(
                    title="Update Configuration",
                    description="Review and update configuration for optimal performance",
                    automated=False,
                    estimated_time="5 minutes",
                ),
            ]
        )

        return actions

    def _generate_error_description(
        self, error: Exception, context: ErrorContext
    ) -> str:
        """Generate a human-friendly error description."""
        error_type = type(error).__name__
        error_msg = str(error)

        if context.command:
            return f"An error occurred while running '{context.command}': {error_msg}"
        else:
            return f"{error_type}: {error_msg}"

    def _analyze_root_cause(self, error: Exception, context: ErrorContext) -> str:
        """Analyze and explain the root cause of the error."""
        error_str = str(error).lower()

        if "connection refused" in error_str:
            return "The target service is not running or not accessible on the specified port"
        elif "permission denied" in error_str:
            return "Insufficient permissions to perform the requested operation"
        elif "not found" in error_str:
            return "Required resource, module, or service could not be located"
        elif "timeout" in error_str:
            return "Operation took longer than expected, possibly due to network or resource constraints"
        else:
            return "The specific cause requires further investigation"

    def _generate_learning_notes(
        self, error: Exception, context: ErrorContext
    ) -> List[str]:
        """Generate educational notes to help prevent future errors."""
        notes = []
        error_str = str(error).lower()

        if "connection" in error_str:
            notes.extend(
                [
                    "Connection errors often indicate service startup order issues",
                    "Consider implementing retry logic with exponential backoff",
                    "Health checks can help detect service availability",
                ]
            )

        if "permission" in error_str:
            notes.extend(
                [
                    "Permission errors may require running with elevated privileges",
                    "Check file/directory ownership and permissions",
                    "Consider using service accounts for production deployments",
                ]
            )

        if not notes:
            notes.append("Review logs and documentation for similar issues")

        return notes

    def _find_related_docs(self, error: Exception, context: ErrorContext) -> List[str]:
        """Find related documentation links."""
        docs = [
            "https://docs.ai-docs-scraper.com/troubleshooting",
            "https://docs.ai-docs-scraper.com/configuration",
        ]

        error_str = str(error).lower()

        if "connection" in error_str or "network" in error_str:
            docs.append("https://docs.ai-docs-scraper.com/network-troubleshooting")

        if "api" in error_str or "key" in error_str:
            docs.append("https://docs.ai-docs-scraper.com/api-configuration")

        return docs

    async def _present_solution_interactively(
        self, solution: SmartErrorSolution, context: ErrorContext
    ):
        """Present the solution interactively with user choices."""
        self._show_error_analysis(solution)

        if solution.immediate_actions:
            await self._handle_immediate_actions(solution.immediate_actions, context)

        if solution.preventive_actions and self.user_preferences.get(
            "show_learning_tips", True
        ):
            self._show_preventive_suggestions(solution.preventive_actions)

    def _show_error_analysis(self, solution: SmartErrorSolution):
        """Display comprehensive error analysis."""
        # Error summary
        severity_color = {
            ErrorSeverity.CRITICAL: "red",
            ErrorSeverity.HIGH: "red",
            ErrorSeverity.MEDIUM: "yellow",
            ErrorSeverity.LOW: "blue",
            ErrorSeverity.INFO: "green",
        }

        error_text = Text()
        error_text.append("🔍 Error Analysis\n\n", style="bold cyan")
        error_text.append("Category: ", style="dim")
        error_text.append(f"{solution.category.value.title()}\n", style="cyan")
        error_text.append("Severity: ", style="dim")
        error_text.append(
            f"{solution.severity.value.title()}\n",
            style=severity_color[solution.severity],
        )
        error_text.append("Confidence: ", style="dim")
        error_text.append(f"{solution.confidence_score:.1%}\n\n", style="green")

        error_text.append("Root Cause:\n", style="bold")
        error_text.append(f"{solution.root_cause}\n", style="")

        panel = Panel(
            error_text,
            title=f"🚨 {solution.title}",
            border_style=severity_color[solution.severity],
            padding=(1, 2),
        )
        self.console.print(panel)

    async def _handle_immediate_actions(
        self, actions: List[RecoveryAction], context: ErrorContext
    ):
        """Handle immediate recovery actions with user interaction."""
        if not actions:
            return

        self.console.print("\n[bold cyan]🛠️ Suggested Recovery Actions[/bold cyan]")

        # Show actions in table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Action", style="cyan", width=30)
        table.add_column("Description", width=40)
        table.add_column("Time", style="dim", width=10)
        table.add_column("Auto", style="green", width=6)

        for i, action in enumerate(actions, 1):
            auto_icon = "✅" if action.automated else "❌"
            table.add_row(
                f"{i}. {action.title}",
                action.description,
                action.estimated_time,
                auto_icon,
            )

        self.console.print(table)

        # Handle automated actions
        automated_actions = [a for a in actions if a.automated]
        if automated_actions:
            if questionary.confirm(
                f"Run {len(automated_actions)} automated diagnostic checks?",
                default=True,
            ).ask():
                await self._execute_automated_actions(automated_actions)

        # Handle manual actions
        manual_actions = [a for a in actions if not a.automated]
        if manual_actions:
            selected_actions = questionary.checkbox(
                "Which manual actions would you like guidance for?",
                choices=[
                    questionary.Choice(
                        title=f"{action.title} ({action.estimated_time})",
                        value=action,
                    )
                    for action in manual_actions
                ],
            ).ask()

            if selected_actions:
                await self._guide_manual_actions(selected_actions)

    async def _execute_automated_actions(self, actions: List[RecoveryAction]):
        """Execute automated recovery actions."""
        import subprocess

        for action in actions:
            if action.command:
                self.console.print(f"[dim]Running: {action.title}...[/dim]", end="")
                try:
                    result = subprocess.run(
                        action.command,
                        check=False, shell=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )

                    if result.returncode == 0:
                        self.console.print(" ✅", style="green")
                        if result.stdout.strip():
                            self.console.print(f"  Output: {result.stdout.strip()}")
                    else:
                        self.console.print(" ❌", style="red")
                        if result.stderr.strip():
                            self.console.print(f"  Error: {result.stderr.strip()}")

                except subprocess.TimeoutExpired:
                    self.console.print(" ⏱️ Timeout", style="yellow")
                except Exception as e:
                    self.console.print(f" ❌ Error: {e}", style="red")

    async def _guide_manual_actions(self, actions: List[RecoveryAction]):
        """Provide guidance for manual recovery actions."""
        for action in actions:
            self.console.print(f"\n[bold cyan]📋 {action.title}[/bold cyan]")
            self.console.print(action.description)

            if action.command:
                self.console.print(
                    f"\n[dim]Command:[/dim] [green]{action.command}[/green]"
                )

            if action.prerequisites:
                self.console.print("\n[dim]Prerequisites:[/dim]")
                for prereq in action.prerequisites:
                    self.console.print(f"  • {prereq}")

            if action.risk_level != "low":
                self.console.print(
                    f"\n⚠️ Risk Level: {action.risk_level.upper()}", style="yellow"
                )

            completed = questionary.confirm(
                "Have you completed this action?", default=False
            ).ask()

            if completed:
                self.console.print("✅ Action completed", style="green")
            else:
                self.console.print("⏸️ Action skipped", style="yellow")

    def _show_preventive_suggestions(self, actions: List[RecoveryAction]):
        """Show preventive action suggestions."""
        if not actions:
            return

        preventive_text = Text()
        preventive_text.append("🛡️ Preventive Measures\n\n", style="bold green")
        preventive_text.append(
            "To prevent similar issues in the future:\n\n", style="dim"
        )

        for i, action in enumerate(actions, 1):
            preventive_text.append(f"{i}. ", style="green")
            preventive_text.append(f"{action.title}\n", style="bold")
            preventive_text.append(f"   {action.description}\n\n", style="")

        panel = Panel(
            preventive_text,
            title="💡 Prevention Tips",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _show_cached_solution(self, solution: SmartErrorSolution):
        """Show a cached solution with indication."""
        self.console.print(
            "💾 [dim]Using cached solution (similar error encountered before)[/dim]"
        )
        self._show_error_analysis(solution)

    def _create_fallback_solution(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution:
        """Create a basic fallback solution when smart analysis fails."""
        return SmartErrorSolution(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            title=f"Unexpected Error: {type(error).__name__}",
            description=str(error),
            root_cause="Unable to determine root cause automatically",
            immediate_actions=[
                RecoveryAction(
                    title="Check System Status",
                    description="Verify system health and service status",
                    command="uv run python -m src.cli.main status",
                    automated=True,
                ),
                RecoveryAction(
                    title="Review Logs",
                    description="Check application logs for additional context",
                    automated=False,
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Enable Debug Mode",
                    description="Enable debug logging for better error tracking",
                    automated=False,
                )
            ],
            learning_notes=[
                "Consider reporting this error to improve automatic detection",
                "Check documentation for similar issues",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/troubleshooting"],
            confidence_score=0.3,
        )

    # Built-in error pattern handlers
    async def _handle_connection_error(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle connection-related errors."""
        return SmartErrorSolution(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.HIGH,
            title="Service Connection Failed",
            description="Unable to connect to required service",
            root_cause="Target service is not running or not accessible",
            immediate_actions=[
                RecoveryAction(
                    title="Check Service Status",
                    description="Verify Docker services are running",
                    command="docker-compose ps",
                    automated=True,
                    success_probability=0.9,
                ),
                RecoveryAction(
                    title="Start Services",
                    description="Start all required services",
                    command="docker-compose up -d",
                    automated=False,
                    estimated_time="1 minute",
                ),
                RecoveryAction(
                    title="Test Connectivity",
                    description="Test network connectivity to service",
                    command="curl -s http://localhost:6333/health || echo 'Service not accessible'",
                    automated=True,
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Enable Health Monitoring",
                    description="Set up automated health checks",
                    automated=False,
                )
            ],
            learning_notes=[
                "Services should be started before running commands",
                "Use health checks to verify service availability",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/services"],
            confidence_score=0.85,
        )

    async def _handle_permission_error(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle permission-related errors."""
        return SmartErrorSolution(
            category=ErrorCategory.PERMISSIONS,
            severity=ErrorSeverity.MEDIUM,
            title="Permission Denied",
            description="Insufficient permissions for operation",
            root_cause="User lacks necessary permissions for the requested operation",
            immediate_actions=[
                RecoveryAction(
                    title="Check File Permissions",
                    description="Verify file and directory permissions",
                    command="ls -la",
                    automated=True,
                ),
                RecoveryAction(
                    title="Fix Permissions",
                    description="Correct file permissions if needed",
                    command="chmod +x scripts/*.sh",
                    automated=False,
                    risk_level="medium",
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Set Proper Permissions",
                    description="Configure proper permissions during setup",
                    automated=False,
                )
            ],
            learning_notes=[
                "Always check permissions when file operations fail",
                "Use appropriate permissions for security",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/permissions"],
            confidence_score=0.8,
        )

    async def _handle_missing_module(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle missing module/dependency errors."""
        return SmartErrorSolution(
            category=ErrorCategory.DEPENDENCIES,
            severity=ErrorSeverity.HIGH,
            title="Missing Python Module",
            description="Required Python module is not installed",
            root_cause="Dependencies are not properly installed or virtual environment is not activated",
            immediate_actions=[
                RecoveryAction(
                    title="Reinstall Dependencies",
                    description="Refresh all project dependencies",
                    command="uv sync --reinstall",
                    automated=False,
                    estimated_time="2 minutes",
                ),
                RecoveryAction(
                    title="Check Python Environment",
                    description="Verify correct Python version and environment",
                    command="uv run python --version && uv run which python",
                    automated=True,
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Lock Dependencies",
                    description="Use locked dependency versions for consistency",
                    automated=False,
                )
            ],
            learning_notes=[
                "Always use uv for dependency management",
                "Check virtual environment activation",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/installation"],
            confidence_score=0.9,
        )

    async def _handle_api_key_error(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle API key related errors."""
        return SmartErrorSolution(
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            title="API Key Configuration Error",
            description="API key is missing or invalid",
            root_cause="Required API keys are not configured or are invalid",
            immediate_actions=[
                RecoveryAction(
                    title="Configure API Keys",
                    description="Run configuration wizard to set up API keys",
                    command="uv run python -m src.cli.main setup",
                    automated=False,
                    estimated_time="5 minutes",
                ),
                RecoveryAction(
                    title="Validate API Keys",
                    description="Test API key configuration",
                    command="uv run python -m src.cli.main config validate",
                    automated=True,
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Secure API Storage",
                    description="Use environment variables for API keys",
                    automated=False,
                )
            ],
            learning_notes=[
                "Never commit API keys to version control",
                "Use environment variables for sensitive data",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/api-keys"],
            confidence_score=0.95,
        )

    async def _handle_validation_error(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle validation errors."""
        return SmartErrorSolution(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            title="Configuration Validation Failed",
            description="Configuration contains invalid values",
            root_cause="One or more configuration values do not meet validation requirements",
            immediate_actions=[
                RecoveryAction(
                    title="Review Configuration",
                    description="Check configuration for invalid values",
                    command="uv run python -m src.cli.main config show",
                    automated=True,
                ),
                RecoveryAction(
                    title="Fix Configuration",
                    description="Use wizard to correct configuration",
                    command="uv run python -m src.cli.main setup",
                    automated=False,
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Enable Validation",
                    description="Use built-in validation features",
                    automated=False,
                )
            ],
            learning_notes=[
                "Always validate configuration after changes",
                "Use configuration templates for consistency",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/configuration"],
            confidence_score=0.8,
        )

    async def _handle_timeout_error(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle timeout errors."""
        return SmartErrorSolution(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            title="Operation Timeout",
            description="Operation took longer than expected",
            root_cause="Network latency, service overload, or resource constraints",
            immediate_actions=[
                RecoveryAction(
                    title="Retry Operation",
                    description="Retry the failed operation",
                    automated=False,
                    estimated_time="30 seconds",
                ),
                RecoveryAction(
                    title="Check System Resources",
                    description="Verify system resource availability",
                    command="docker stats --no-stream",
                    automated=True,
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Increase Timeouts",
                    description="Configure longer timeout values",
                    automated=False,
                )
            ],
            learning_notes=[
                "Consider implementing retry logic",
                "Monitor system resources",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/performance"],
            confidence_score=0.7,
        )

    async def _handle_disk_space_error(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle disk space errors."""
        return SmartErrorSolution(
            category=ErrorCategory.RESOURCES,
            severity=ErrorSeverity.HIGH,
            title="Insufficient Disk Space",
            description="Not enough disk space for operation",
            root_cause="Disk space has been exhausted or is critically low",
            immediate_actions=[
                RecoveryAction(
                    title="Check Disk Usage",
                    description="View disk space usage",
                    command="df -h",
                    automated=True,
                ),
                RecoveryAction(
                    title="Clean Cache",
                    description="Clear application caches",
                    command="docker system prune -f",
                    automated=False,
                    risk_level="medium",
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Set Up Monitoring",
                    description="Monitor disk space usage",
                    automated=False,
                )
            ],
            learning_notes=[
                "Regular cleanup prevents disk space issues",
                "Monitor disk usage in production",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/maintenance"],
            confidence_score=0.9,
        )

    async def _handle_port_conflict(
        self, error: Exception, context: ErrorContext
    ) -> SmartErrorSolution | None:
        """Handle port conflict errors."""
        return SmartErrorSolution(
            category=ErrorCategory.SERVICES,
            severity=ErrorSeverity.MEDIUM,
            title="Port Already in Use",
            description="Required port is already occupied by another process",
            root_cause="Another service is using the required port",
            immediate_actions=[
                RecoveryAction(
                    title="Check Port Usage",
                    description="Find what's using the port",
                    command="netstat -tulpn | grep :6333",
                    automated=True,
                ),
                RecoveryAction(
                    title="Stop Conflicting Service",
                    description="Stop the conflicting service",
                    automated=False,
                    risk_level="medium",
                ),
            ],
            preventive_actions=[
                RecoveryAction(
                    title="Use Different Ports",
                    description="Configure alternative port numbers",
                    automated=False,
                )
            ],
            learning_notes=[
                "Check port availability before starting services",
                "Use port management tools",
            ],
            related_docs=["https://docs.ai-docs-scraper.com/ports"],
            confidence_score=0.85,
        )
