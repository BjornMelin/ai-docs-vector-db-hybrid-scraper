"""Intelligent CLI with contextual help, progressive feature discovery, and smart suggestions.

This module provides sophisticated CLI patterns that showcase advanced UX thinking:
- Context-aware help with smart suggestions
- Progressive feature discovery with learning patterns
- Intelligent command completion and validation
- Adaptive user experience based on expertise level
"""

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import click
import questionary
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from .smart_error_handler import SmartErrorHandler


console = Console()


class UserExpertiseLevel(Enum):
    """User expertise levels for adaptive experience."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class FeatureCategory(Enum):
    """Categories for progressive feature discovery."""

    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"


@dataclass
class UserProfile:
    """User profile for personalized CLI experience."""

    expertise_level: UserExpertiseLevel = UserExpertiseLevel.BEGINNER
    discovered_features: Set[str] = field(default_factory=set)
    command_usage_count: Dict[str, int] = field(default_factory=dict)
    preferred_help_style: str = "verbose"  # verbose, concise, examples
    enable_suggestions: bool = True
    enable_feature_discovery: bool = True
    last_updated: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "expertise_level": self.expertise_level.value,
            "discovered_features": list(self.discovered_features),
            "command_usage_count": self.command_usage_count,
            "preferred_help_style": self.preferred_help_style,
            "enable_suggestions": self.enable_suggestions,
            "enable_feature_discovery": self.enable_feature_discovery,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        return cls(
            expertise_level=UserExpertiseLevel(data.get("expertise_level", "beginner")),
            discovered_features=set(data.get("discovered_features", [])),
            command_usage_count=data.get("command_usage_count", {}),
            preferred_help_style=data.get("preferred_help_style", "verbose"),
            enable_suggestions=data.get("enable_suggestions", True),
            enable_feature_discovery=data.get("enable_feature_discovery", True),
            last_updated=data.get("last_updated", ""),
        )


@dataclass
class CommandFeature:
    """Represents a CLI feature for progressive discovery."""

    name: str
    category: FeatureCategory
    description: str
    example: str
    prerequisites: List[str] = field(default_factory=list)
    related_features: List[str] = field(default_factory=list)
    documentation_url: str = ""
    complexity_score: int = 1  # 1-10 scale


@dataclass
class ContextualSuggestion:
    """Contextual suggestion for intelligent assistance."""

    title: str
    description: str
    command: str
    confidence: float  # 0.0-1.0
    trigger_context: str
    category: str = "general"
    examples: List[str] = field(default_factory=list)


class IntelligentCLI:
    """Intelligent CLI with contextual help and progressive feature discovery."""

    def __init__(self, config_dir: Path | None = None):
        """Initialize the intelligent CLI.

        Args:
            config_dir: Directory for storing user profiles and preferences
        """
        self.console = Console()
        self.config_dir = config_dir or Path.home() / ".ai-docs"
        self.config_dir.mkdir(exist_ok=True)

        self.user_profile = self._load_user_profile()
        self.error_handler = SmartErrorHandler()
        self.command_features = self._initialize_command_features()
        self.contextual_suggestions = self._initialize_contextual_suggestions()

        # Track command context for smart suggestions
        self.current_context: Dict[str, Any] = {}
        self.command_history: List[str] = []

    def _load_user_profile(self) -> UserProfile:
        """Load user profile from disk."""
        profile_path = self.config_dir / "user_profile.json"

        if profile_path.exists():
            try:
                with open(profile_path) as f:
                    data = json.load(f)
                return UserProfile.from_dict(data)
            except Exception as e:
                console.print(f"[yellow]Could not load user profile: {e}[/yellow]")

        return UserProfile()

    def _save_user_profile(self):
        """Save user profile to disk."""
        profile_path = self.config_dir / "user_profile.json"

        try:
            with open(profile_path, "w") as f:
                json.dump(self.user_profile.to_dict(), f, indent=2)
        except Exception as e:
            console.print(f"[yellow]Could not save user profile: {e}[/yellow]")

    def _initialize_command_features(self) -> Dict[str, CommandFeature]:
        """Initialize the command feature registry."""
        features = {
            "setup_wizard": CommandFeature(
                name="Interactive Setup Wizard",
                category=FeatureCategory.BASIC,
                description="Template-driven configuration wizard with real-time validation",
                example="ai-docs setup --profile personal",
                related_features=["config_validation", "profile_management"],
                documentation_url="https://docs.ai-docs-scraper.com/setup",
            ),
            "config_validation": CommandFeature(
                name="Configuration Validation",
                category=FeatureCategory.BASIC,
                description="Real-time configuration validation with helpful error messages",
                example="ai-docs config validate",
                related_features=["setup_wizard"],
                complexity_score=2,
            ),
            "profile_management": CommandFeature(
                name="Profile Management",
                category=FeatureCategory.INTERMEDIATE,
                description="Multiple configuration profiles for different environments",
                example="ai-docs setup --profile production",
                prerequisites=["setup_wizard"],
                related_features=["config_export"],
                complexity_score=3,
            ),
            "batch_operations": CommandFeature(
                name="Batch Operations",
                category=FeatureCategory.ADVANCED,
                description="Process multiple documents or collections in parallel",
                example="ai-docs batch process --pattern '**/*.md' --collection docs",
                prerequisites=["basic_setup"],
                complexity_score=5,
            ),
            "advanced_search": CommandFeature(
                name="Advanced Search Features",
                category=FeatureCategory.ADVANCED,
                description="Semantic search with clustering, ranking, and federation",
                example="ai-docs search --query 'AI documentation' --cluster --rank",
                prerequisites=["basic_search"],
                complexity_score=7,
            ),
            "monitoring_setup": CommandFeature(
                name="Monitoring & Observability",
                category=FeatureCategory.ADVANCED,
                description="Comprehensive monitoring with Prometheus and Grafana",
                example="ai-docs monitor setup --enable-alerts",
                prerequisites=["production_config"],
                complexity_score=8,
            ),
            "rag_integration": CommandFeature(
                name="RAG Answer Generation",
                category=FeatureCategory.EXPERIMENTAL,
                description="AI-powered answer generation from search results",
                example="ai-docs rag generate --query 'How to setup?' --model gpt-4",
                prerequisites=["advanced_search", "api_keys"],
                complexity_score=9,
            ),
        }

        return features

    def _initialize_contextual_suggestions(self) -> List[ContextualSuggestion]:
        """Initialize contextual suggestions based on common scenarios."""
        return [
            ContextualSuggestion(
                title="First-time Setup",
                description="It looks like this is your first time! Try the setup wizard.",
                command="ai-docs setup",
                confidence=0.9,
                trigger_context="no_config_found",
                category="onboarding",
                examples=["ai-docs setup --profile personal"],
            ),
            ContextualSuggestion(
                title="Configuration Issues",
                description="Configuration errors detected. Run validation to diagnose.",
                command="ai-docs config validate",
                confidence=0.8,
                trigger_context="config_error",
                category="troubleshooting",
                examples=["ai-docs config show", "ai-docs config export"],
            ),
            ContextualSuggestion(
                title="Service Connection Problems",
                description="Services appear to be down. Check status and start if needed.",
                command="ai-docs status",
                confidence=0.85,
                trigger_context="connection_error",
                category="troubleshooting",
                examples=["docker-compose up -d", "ai-docs database health"],
            ),
            ContextualSuggestion(
                title="Ready for Advanced Features",
                description="You've mastered the basics! Explore advanced search features.",
                command="ai-docs help advanced",
                confidence=0.7,
                trigger_context="intermediate_user",
                category="progression",
                examples=["ai-docs search --help", "ai-docs batch --help"],
            ),
            ContextualSuggestion(
                title="Performance Optimization",
                description="Enable monitoring to track performance and optimize usage.",
                command="ai-docs monitor setup",
                confidence=0.6,
                trigger_context="heavy_usage",
                category="optimization",
                examples=["ai-docs monitor status", "ai-docs cache optimize"],
            ),
        ]

    def track_command_usage(self, command: str):
        """Track command usage for adaptive experience."""
        self.command_history.append(command)
        if len(self.command_history) > 100:
            self.command_history = self.command_history[-100:]

        # Update usage count
        self.user_profile.command_usage_count[command] = (
            self.user_profile.command_usage_count.get(command, 0) + 1
        )

        # Update expertise level based on usage patterns
        self._update_expertise_level()

        # Save profile
        self._save_user_profile()

    def _update_expertise_level(self):
        """Update user expertise level based on usage patterns."""
        total_commands = sum(self.user_profile.command_usage_count.values())
        unique_commands = len(self.user_profile.command_usage_count)
        advanced_commands = sum(
            count
            for cmd, count in self.user_profile.command_usage_count.items()
            if any(
                advanced in cmd for advanced in ["batch", "advanced", "monitor", "rag"]
            )
        )

        if total_commands >= 100 and advanced_commands >= 10:
            self.user_profile.expertise_level = UserExpertiseLevel.EXPERT
        elif total_commands >= 50 and unique_commands >= 10:
            self.user_profile.expertise_level = UserExpertiseLevel.ADVANCED
        elif total_commands >= 20 and unique_commands >= 5:
            self.user_profile.expertise_level = UserExpertiseLevel.INTERMEDIATE

    def get_contextual_help(
        self, command: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Get contextual help for a command based on user profile and context.

        Args:
            command: Command name
            context: Current context information

        Returns:
            Contextual help information
        """
        context = context or {}
        help_info = {
            "command": command,
            "expertise_level": self.user_profile.expertise_level.value,
            "suggestions": [],
            "related_features": [],
            "examples": [],
            "tips": [],
        }

        # Get command-specific help
        if command in self.command_features:
            feature = self.command_features[command]
            help_info["description"] = feature.description
            help_info["category"] = feature.category.value
            help_info["examples"] = [feature.example]
            help_info["related_features"] = [
                self.command_features[name]
                for name in feature.related_features
                if name in self.command_features
            ]

        # Add contextual suggestions
        suggestions = self._get_contextual_suggestions(command, context)
        help_info["suggestions"] = suggestions

        # Add expertise-appropriate tips
        tips = self._get_expertise_tips(command)
        help_info["tips"] = tips

        return help_info

    def _get_contextual_suggestions(
        self, command: str, context: Dict[str, Any]
    ) -> List[ContextualSuggestion]:
        """Get contextual suggestions based on command and context."""
        suggestions = []

        for suggestion in self.contextual_suggestions:
            # Check if suggestion is relevant to current context
            if self._is_suggestion_relevant(suggestion, command, context):
                suggestions.append(suggestion)

        # Sort by confidence
        suggestions.sort(key=lambda s: s.confidence, reverse=True)
        return suggestions[:3]  # Return top 3 suggestions

    def _is_suggestion_relevant(
        self, suggestion: ContextualSuggestion, command: str, context: Dict[str, Any]
    ) -> bool:
        """Check if a suggestion is relevant to the current context."""
        # Check trigger context
        if suggestion.trigger_context == "no_config_found":
            return not context.get("config_exists", True)
        elif suggestion.trigger_context == "config_error":
            return context.get("has_config_errors", False)
        elif suggestion.trigger_context == "connection_error":
            return context.get("has_connection_errors", False)
        elif suggestion.trigger_context == "intermediate_user":
            return self.user_profile.expertise_level in [
                UserExpertiseLevel.INTERMEDIATE,
                UserExpertiseLevel.ADVANCED,
            ]
        elif suggestion.trigger_context == "heavy_usage":
            total_usage = sum(self.user_profile.command_usage_count.values())
            return total_usage >= 30

        return False

    def _get_expertise_tips(self, command: str) -> List[str]:
        """Get expertise-appropriate tips for a command."""
        tips = []

        if self.user_profile.expertise_level == UserExpertiseLevel.BEGINNER:
            tips.extend(
                [
                    "💡 Use --help with any command for detailed information",
                    "🎯 Try the setup wizard if you haven't configured yet: ai-docs setup",
                    "📚 Check documentation at docs.ai-docs-scraper.com",
                ]
            )
        elif self.user_profile.expertise_level == UserExpertiseLevel.INTERMEDIATE:
            tips.extend(
                [
                    "⚡ Use tab completion for faster command entry",
                    "🔧 Explore advanced options with --help",
                    "📊 Consider enabling monitoring for insights",
                ]
            )
        elif self.user_profile.expertise_level == UserExpertiseLevel.ADVANCED:
            tips.extend(
                [
                    "🚀 Explore batch operations for efficiency",
                    "🎛️ Customize configuration profiles",
                    "🔍 Try advanced search features",
                ]
            )
        else:  # Expert
            tips.extend(
                [
                    "⚡ Use shell aliases for frequent commands",
                    "🔬 Experiment with RAG integration",
                    "🛠️ Contribute to feature development",
                ]
            )

        return tips

    def show_feature_discovery(self, force: bool = False):
        """Show progressive feature discovery based on user progress."""
        if not self.user_profile.enable_feature_discovery and not force:
            return

        # Find undiscovered features appropriate for user level
        available_features = self._get_available_features()

        if not available_features:
            return

        self.console.print(
            "\n[bold cyan]🌟 Feature Discovery: New capabilities available![/bold cyan]"
        )

        for feature in available_features[:3]:  # Show top 3
            self._show_feature_highlight(feature)

        if questionary.confirm(
            "Would you like to learn more about these features?", default=False
        ).ask():
            self._interactive_feature_exploration(available_features)

    def _get_available_features(self) -> List[CommandFeature]:
        """Get features available for discovery based on user progress."""
        available = []

        for feature in self.command_features.values():
            # Skip if already discovered
            if feature.name in self.user_profile.discovered_features:
                continue

            # Check prerequisites
            if not self._check_prerequisites(feature):
                continue

            # Check if appropriate for current expertise level
            if not self._is_feature_appropriate(feature):
                continue

            available.append(feature)

        # Sort by complexity and relevance
        available.sort(key=lambda f: (f.complexity_score, f.category.value))
        return available

    def _check_prerequisites(self, feature: CommandFeature) -> bool:
        """Check if user has met feature prerequisites."""
        for prereq in feature.prerequisites:
            if prereq not in self.user_profile.discovered_features:
                # Check by command usage as alternative
                if not any(
                    prereq in cmd for cmd in self.user_profile.command_usage_count
                ):
                    return False
        return True

    def _is_feature_appropriate(self, feature: CommandFeature) -> bool:
        """Check if feature is appropriate for current expertise level."""
        level_map = {
            UserExpertiseLevel.BEGINNER: [FeatureCategory.BASIC],
            UserExpertiseLevel.INTERMEDIATE: [
                FeatureCategory.BASIC,
                FeatureCategory.INTERMEDIATE,
            ],
            UserExpertiseLevel.ADVANCED: [
                FeatureCategory.BASIC,
                FeatureCategory.INTERMEDIATE,
                FeatureCategory.ADVANCED,
            ],
            UserExpertiseLevel.EXPERT: list(FeatureCategory),
        }

        return feature.category in level_map[self.user_profile.expertise_level]

    def _show_feature_highlight(self, feature: CommandFeature):
        """Show a highlighted feature for discovery."""
        feature_text = Text()
        feature_text.append(f"✨ {feature.name}\n", style="bold cyan")
        feature_text.append(f"{feature.description}\n\n", style="")
        feature_text.append("Example: ", style="dim")
        feature_text.append(f"{feature.example}\n", style="green")

        if feature.complexity_score > 5:
            feature_text.append("⚠️ Advanced feature", style="yellow")

        panel = Panel(
            feature_text,
            title=f"🆕 {feature.category.value.title()} Feature",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def _interactive_feature_exploration(self, features: List[CommandFeature]):
        """Interactive exploration of new features."""
        selected_features = questionary.checkbox(
            "Which features would you like to explore?",
            choices=[
                questionary.Choice(
                    title=f"{feature.name} ({feature.category.value})", value=feature
                )
                for feature in features
            ],
        ).ask()

        for feature in selected_features or []:
            self._explore_feature_interactively(feature)
            self.user_profile.discovered_features.add(feature.name)

        self._save_user_profile()

    def _explore_feature_interactively(self, feature: CommandFeature):
        """Interactively explore a specific feature."""
        self.console.print(f"\n[bold cyan]🔍 Exploring: {feature.name}[/bold cyan]")

        # Show detailed information
        info_text = Text()
        info_text.append(f"Category: {feature.category.value.title()}\n", style="dim")
        info_text.append(f"Complexity: {feature.complexity_score}/10\n", style="dim")
        info_text.append(f"\n{feature.description}\n\n", style="")
        info_text.append("Example Usage:\n", style="bold")
        info_text.append(f"{feature.example}\n\n", style="green")

        if feature.related_features:
            info_text.append("Related Features:\n", style="bold")
            for related in feature.related_features:
                info_text.append(f"• {related}\n", style="cyan")

        if feature.documentation_url:
            info_text.append(
                f"\nDocumentation: {feature.documentation_url}", style="blue"
            )

        panel = Panel(
            info_text,
            title=f"📖 {feature.name}",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

        # Offer to try the feature
        if questionary.confirm(
            f"Would you like to try {feature.name} now?", default=False
        ).ask():
            self._guide_feature_usage(feature)

    def _guide_feature_usage(self, feature: CommandFeature):
        """Guide user through feature usage."""
        self.console.print(f"\n[bold green]🚀 Let's try {feature.name}![/bold green]")

        # Show step-by-step guidance
        steps = self._get_feature_steps(feature)

        for i, step in enumerate(steps, 1):
            self.console.print(f"\n[bold cyan]Step {i}:[/bold cyan] {step['title']}")
            self.console.print(step["description"])

            if "command" in step:
                self.console.print(
                    f"\n[dim]Command:[/dim] [green]{step['command']}[/green]"
                )

            if not questionary.confirm("Ready for next step?", default=True).ask():
                break

        self.console.print(
            "\n[bold green]✅ Feature exploration complete![/bold green]"
        )

    def _get_feature_steps(self, feature: CommandFeature) -> List[Dict[str, str]]:
        """Get step-by-step guidance for a feature."""
        # This would be customized per feature
        if "setup" in feature.name.lower():
            return [
                {
                    "title": "Choose Profile",
                    "description": "Select a configuration profile that matches your needs",
                    "command": "ai-docs setup --profile personal",
                },
                {
                    "title": "Configure API Keys",
                    "description": "Set up your API keys for external services",
                },
                {
                    "title": "Validate Configuration",
                    "description": "Test your configuration",
                    "command": "ai-docs config validate",
                },
            ]
        elif "batch" in feature.name.lower():
            return [
                {
                    "title": "Prepare Documents",
                    "description": "Organize documents you want to process",
                },
                {
                    "title": "Run Batch Operation",
                    "description": "Process multiple documents at once",
                    "command": "ai-docs batch process --pattern '**/*.md'",
                },
                {
                    "title": "Monitor Progress",
                    "description": "Watch the batch processing progress",
                },
            ]
        else:
            return [
                {
                    "title": "Learn Command Options",
                    "description": f"Explore available options for {feature.name}",
                    "command": f"{feature.example.split()[0]} --help",
                },
                {
                    "title": "Try Basic Usage",
                    "description": "Start with basic usage",
                    "command": feature.example,
                },
            ]

    def show_intelligent_suggestions(self, context: Dict[str, Any] = None):
        """Show intelligent suggestions based on current context."""
        if not self.user_profile.enable_suggestions:
            return

        context = context or {}
        suggestions = self._get_contextual_suggestions("", context)

        if not suggestions:
            return

        self.console.print("\n[bold cyan]💡 Smart Suggestions[/bold cyan]")

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Suggestion", style="cyan", width=30)
        table.add_column("Description", width=40)
        table.add_column("Confidence", style="green", width=10)

        for suggestion in suggestions:
            confidence_percent = f"{suggestion.confidence:.0%}"
            table.add_row(suggestion.title, suggestion.description, confidence_percent)

        self.console.print(table)

        # Offer to execute suggestion
        if len(suggestions) == 1:
            suggestion = suggestions[0]
            if questionary.confirm(
                f"Execute suggested command: {suggestion.command}?", default=False
            ).ask():
                self._execute_suggested_command(suggestion)
        else:
            selected = questionary.select(
                "Which suggestion would you like to try?",
                choices=[
                    questionary.Choice(title=f"{s.title} - {s.command}", value=s)
                    for s in suggestions
                ]
                + [questionary.Choice(title="None", value=None)],
            ).ask()

            if selected:
                self._execute_suggested_command(selected)

    def _execute_suggested_command(self, suggestion: ContextualSuggestion):
        """Execute a suggested command with guidance."""
        self.console.print(
            f"\n[bold green]🚀 Executing: {suggestion.command}[/bold green]"
        )

        # Show additional context if available
        if suggestion.examples:
            self.console.print("\n[dim]Related examples:[/dim]")
            for example in suggestion.examples:
                self.console.print(f"  [green]{example}[/green]")

        # This would integrate with actual command execution
        self.console.print(f"\n[dim]Would execute: {suggestion.command}[/dim]")

    def customize_cli_experience(self):
        """Allow user to customize their CLI experience."""
        self.console.print("\n[bold cyan]🎛️ Customize Your CLI Experience[/bold cyan]")

        # Expertise level
        current_level = self.user_profile.expertise_level.value
        expertise_level = questionary.select(
            "What's your expertise level with AI documentation tools?",
            choices=[
                questionary.Choice(
                    title="Beginner (current)"
                    if current_level == "beginner"
                    else "Beginner",
                    value="beginner",
                ),
                questionary.Choice(
                    title="Intermediate (current)"
                    if current_level == "intermediate"
                    else "Intermediate",
                    value="intermediate",
                ),
                questionary.Choice(
                    title="Advanced (current)"
                    if current_level == "advanced"
                    else "Advanced",
                    value="advanced",
                ),
                questionary.Choice(
                    title="Expert (current)"
                    if current_level == "expert"
                    else "Expert",
                    value="expert",
                ),
            ],
            default=current_level,
        ).ask()

        if expertise_level:
            self.user_profile.expertise_level = UserExpertiseLevel(expertise_level)

        # Help style
        help_style = questionary.select(
            "Preferred help style?",
            choices=[
                "verbose - Detailed explanations and examples",
                "concise - Brief, to-the-point information",
                "examples - Focus on practical examples",
            ],
            default=f"{self.user_profile.preferred_help_style} - {self._get_help_style_description()}",
        ).ask()

        if help_style:
            self.user_profile.preferred_help_style = help_style.split(" - ")[0]

        # Feature preferences
        self.user_profile.enable_suggestions = questionary.confirm(
            "Enable smart suggestions?", default=self.user_profile.enable_suggestions
        ).ask()

        self.user_profile.enable_feature_discovery = questionary.confirm(
            "Enable progressive feature discovery?",
            default=self.user_profile.enable_feature_discovery,
        ).ask()

        # Save preferences
        self._save_user_profile()

        self.console.print("\n[bold green]✅ Preferences saved![/bold green]")
        self._show_personalization_summary()

    def _get_help_style_description(self) -> str:
        """Get description for current help style."""
        descriptions = {
            "verbose": "Detailed explanations and examples",
            "concise": "Brief, to-the-point information",
            "examples": "Focus on practical examples",
        }
        return descriptions.get(self.user_profile.preferred_help_style, "")

    def _show_personalization_summary(self):
        """Show summary of current personalization settings."""
        summary_text = Text()
        summary_text.append(
            "🎯 Your Personalized CLI Experience\n\n", style="bold cyan"
        )

        summary_text.append("Expertise Level: ", style="dim")
        summary_text.append(
            f"{self.user_profile.expertise_level.value.title()}\n", style="cyan"
        )

        summary_text.append("Help Style: ", style="dim")
        summary_text.append(
            f"{self.user_profile.preferred_help_style.title()}\n", style="cyan"
        )

        summary_text.append("Smart Suggestions: ", style="dim")
        summary_text.append(
            f"{'Enabled' if self.user_profile.enable_suggestions else 'Disabled'}\n",
            style="green" if self.user_profile.enable_suggestions else "red",
        )

        summary_text.append("Feature Discovery: ", style="dim")
        summary_text.append(
            f"{'Enabled' if self.user_profile.enable_feature_discovery else 'Disabled'}\n",
            style="green" if self.user_profile.enable_feature_discovery else "red",
        )

        summary_text.append(
            f"\nCommands Used: {len(self.user_profile.command_usage_count)}\n",
            style="dim",
        )
        summary_text.append(
            f"Features Discovered: {len(self.user_profile.discovered_features)}",
            style="dim",
        )

        panel = Panel(
            summary_text,
            title="🎛️ CLI Personalization",
            border_style="cyan",
            padding=(1, 2),
        )
        self.console.print(panel)

    def show_command_insights(self):
        """Show insights about command usage patterns."""
        if not self.user_profile.command_usage_count:
            self.console.print("[yellow]No command usage data available yet.[/yellow]")
            return

        self.console.print("\n[bold cyan]📊 Your Command Usage Insights[/bold cyan]")

        # Most used commands
        sorted_commands = sorted(
            self.user_profile.command_usage_count.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        usage_table = Table(show_header=True, header_style="bold cyan")
        usage_table.add_column("Command", style="cyan")
        usage_table.add_column("Usage Count", style="green")
        usage_table.add_column("Proficiency", style="yellow")

        for command, count in sorted_commands[:10]:
            proficiency = (
                "Expert"
                if count >= 20
                else "Advanced"
                if count >= 10
                else "Intermediate"
                if count >= 5
                else "Beginner"
            )
            usage_table.add_row(command, str(count), proficiency)

        self.console.print(usage_table)

        # Usage summary
        total_commands = sum(self.user_profile.command_usage_count.values())
        unique_commands = len(self.user_profile.command_usage_count)

        summary_text = Text()
        summary_text.append(f"Total Commands: {total_commands}\n", style="green")
        summary_text.append(f"Unique Commands: {unique_commands}\n", style="cyan")
        summary_text.append(
            f"Expertise Level: {self.user_profile.expertise_level.value.title()}\n",
            style="yellow",
        )

        panel = Panel(
            summary_text,
            title="📈 Usage Summary",
            border_style="green",
            padding=(1, 2),
        )
        self.console.print(panel)

    def reset_user_profile(self):
        """Reset user profile to defaults."""
        if questionary.confirm(
            "Are you sure you want to reset your CLI profile? This will clear all personalization.",
            default=False,
        ).ask():
            self.user_profile = UserProfile()
            self._save_user_profile()
            self.console.print(
                "[bold green]✅ Profile reset successfully![/bold green]"
            )
        else:
            self.console.print("[yellow]Profile reset cancelled.[/yellow]")
