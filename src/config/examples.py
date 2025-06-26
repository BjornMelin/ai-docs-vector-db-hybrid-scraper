"""Configuration examples showcasing sophisticated capabilities.

This module provides comprehensive examples demonstrating the advanced features
and capabilities of our configuration system, including:

- Progressive complexity examples
- Portfolio showcase scenarios
- Advanced Pydantic v2 validation patterns
- Auto-detection and intelligent defaults
- Enterprise-grade configuration patterns
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List

from .builders import (
    ConfigBuilderFactory,
    ConfigValidationError,
    ProgressiveConfigurationGuide,
    guided_config_setup,
    quick_config,
    validate_configuration_with_help,
)
from .core import Config


logger = logging.getLogger(__name__)


class ConfigurationShowcase:
    """Showcase advanced configuration capabilities for portfolio demonstration."""

    @staticmethod
    async def demonstrate_progressive_complexity():
        """Demonstrate progressive complexity and guided discovery."""
        print("🔧 Configuration System Showcase - Progressive Complexity")
        print("=" * 60)

        # 1. Auto-discovery of optimal persona
        print("\n1. Auto-Discovery of Optimal Configuration Persona")
        print("-" * 50)

        optimal_persona = await ConfigBuilderFactory.discover_optimal_persona()
        print(f"✓ Auto-detected optimal persona: {optimal_persona}")

        # 2. Quick start for immediate productivity
        print("\n2. Quick Start Configuration (< 30 seconds)")
        print("-" * 50)

        quick_config_result = await quick_config(persona="development")
        print(
            f"✓ Quick config created with {len(quick_config_result.model_fields)} configuration sections"
        )
        print(f"✓ Auto-detection: {quick_config_result.auto_detection.enabled}")
        print(f"✓ Environment: {quick_config_result.environment.value}")

        # 3. Progressive guided setup
        print("\n3. Progressive Guided Setup with Discovery")
        print("-" * 50)

        guide = await guided_config_setup()
        discovery = await guide.start_guided_setup()

        print(f"✓ Persona: {discovery.persona}")
        print(f"✓ Complexity: {discovery.configuration_complexity}")
        print(f"✓ Setup time: {discovery.estimated_setup_time}")
        print(
            f"✓ Auto-detected services: {', '.join(discovery.auto_detected_services) or 'None'}"
        )
        print(f"✓ Available features: {len(discovery.available_features)}")

        # 4. Feature-level progression
        print("\n4. Progressive Feature Disclosure")
        print("-" * 50)

        features = guide.get_progressive_features()
        for level, feature_list in features.items():
            print(f"  {level.title()}: {len(feature_list)} features")
            for feature in feature_list[:2]:  # Show first 2
                print(f"    • {feature}")
            if len(feature_list) > 2:
                print(f"    • ... and {len(feature_list) - 2} more")

        # 5. Advanced validation with helpful errors
        print("\n5. Intelligent Validation with Helpful Guidance")
        print("-" * 50)

        invalid_config = {
            "embedding_provider": "invalid_provider",
            "openai": {"api_key": "invalid"},
        }
        errors = validate_configuration_with_help(invalid_config)
        if errors:
            print("✓ Validation errors with helpful suggestions:")
            for error in errors[:2]:  # Show first 2
                print(f"  • {error}")

        return discovery

    @staticmethod
    async def demonstrate_persona_configurations():
        """Demonstrate different persona configurations."""
        print("\n🎭 Persona-Based Configuration Showcase")
        print("=" * 60)

        personas = ConfigBuilderFactory.get_available_personas()

        for persona in personas:
            print(f"\n{persona.title()} Persona Configuration")
            print("-" * 40)

            builder = ConfigBuilderFactory.create_builder(persona, auto_detect=False)
            discovery = await builder.discover_configuration()

            print(f"Features: {len(discovery.available_features)}")
            print(f"Complexity: {discovery.configuration_complexity}")
            print(f"Setup time: {discovery.estimated_setup_time}")

            # Show key features
            for feature in discovery.available_features[:3]:
                print(f"  • {feature}")
            if len(discovery.available_features) > 3:
                print(f"  • ... and {len(discovery.available_features) - 3} more")

    @staticmethod
    async def demonstrate_advanced_features():
        """Demonstrate advanced configuration features."""
        print("\n🚀 Advanced Configuration Features")
        print("=" * 60)

        # Enterprise configuration with all features
        builder = ConfigBuilderFactory.create_builder("enterprise")
        config = builder.build()

        features = []

        # Circuit breaker sophistication
        if config.circuit_breaker.use_enhanced_circuit_breaker:
            features.append("✓ Enhanced circuit breakers with adaptive timeouts")

        # Observability stack
        if config.observability.enabled:
            instrumentation = []
            if config.observability.instrument_fastapi:
                instrumentation.append("FastAPI")
            if config.observability.instrument_httpx:
                instrumentation.append("HTTP clients")
            if config.observability.instrument_redis:
                instrumentation.append("Redis")
            features.append(
                f"✓ OpenTelemetry instrumentation: {', '.join(instrumentation)}"
            )

        # Deployment features
        if config.deployment.enable_feature_flags:
            deployment_features = []
            if config.deployment.enable_ab_testing:
                deployment_features.append("A/B testing")
            if config.deployment.enable_blue_green:
                deployment_features.append("Blue-green")
            if config.deployment.enable_canary:
                deployment_features.append("Canary")
            features.append(
                f"✓ Deployment strategies: {', '.join(deployment_features)}"
            )

        # Configuration drift detection
        if config.drift_detection.enabled:
            features.append("✓ Configuration drift detection with auto-remediation")

        print("Advanced Enterprise Features:")
        for feature in features:
            print(f"  {feature}")

        return config

    @staticmethod
    def demonstrate_pydantic_patterns():
        """Demonstrate advanced Pydantic v2 validation patterns."""
        print("\n📋 Advanced Pydantic v2 Validation Patterns")
        print("=" * 60)

        patterns = [
            "✓ Field validators with helpful error messages",
            "✓ Model validators for cross-field validation",
            "✓ Computed fields for derived properties",
            "✓ Settings with environment variable mapping",
            "✓ Nested model validation with error aggregation",
            "✓ Custom validation error types with suggestions",
            "✓ SecretStr integration for sensitive data",
            "✓ Dynamic defaults with factory functions",
        ]

        print("Demonstrated Pydantic v2 Patterns:")
        for pattern in patterns:
            print(f"  {pattern}")

        # Show example validation error
        try:
            Config(openai={"api_key": "invalid-key"}, embedding_provider="openai")
        except Exception as e:
            print("\nExample validation error handling:")
            print(f"  Error: {str(e)[:100]}...")


async def run_configuration_showcase():
    """Run the complete configuration showcase."""
    try:
        showcase = ConfigurationShowcase()

        # Progressive complexity demonstration
        discovery = await showcase.demonstrate_progressive_complexity()

        # Persona configurations
        await showcase.demonstrate_persona_configurations()

        # Advanced features
        config = await showcase.demonstrate_advanced_features()

        # Pydantic patterns
        showcase.demonstrate_pydantic_patterns()

        print("\n✅ Configuration System Showcase Complete")
        print("=" * 60)
        print("Portfolio Highlights:")
        print("  • Progressive complexity with guided discovery")
        print("  • 4 persona-based configuration builders")
        print("  • 20+ sophisticated configuration models")
        print("  • Auto-detection and intelligent defaults")
        print("  • Enterprise-grade validation and error handling")
        print("  • Advanced Pydantic v2 patterns and best practices")

        return discovery, config

    except Exception as e:
        logger.exception("Configuration showcase failed")
        print(f"\n❌ Showcase failed: {e}")
        raise


# Example usage patterns for documentation
USAGE_EXAMPLES = {
    "quick_start": """
# Quick start - one line configuration
config = await quick_config("development")
""",
    "guided_setup": """
# Guided setup with discovery
guide = await guided_config_setup()
discovery = await guide.start_guided_setup()

# Progressive feature enablement
config = await guide.build_configuration("essential")  # Start simple
config = await guide.build_configuration("intermediate")  # Add features
config = await guide.build_configuration("advanced")  # Full power
""",
    "persona_builder": """
# Persona-specific builders
dev_builder = ConfigBuilderFactory.create_builder("development")
prod_builder = ConfigBuilderFactory.create_builder("production")
research_builder = ConfigBuilderFactory.create_builder("research")
enterprise_builder = ConfigBuilderFactory.create_builder("enterprise")

# Build with overrides
config = dev_builder.build(
    openai={"api_key": "sk-..."},
    qdrant={"url": "http://localhost:6333"}
)
""",
    "auto_detection": """
# Auto-detection and intelligent defaults
builder = ConfigBuilderFactory.create_builder("production", auto_detect=True)
discovery = await builder.discover_configuration()

print(f"Auto-detected services: {discovery.auto_detected_services}")
print(f"Recommended providers: {discovery.recommended_providers}")

config = builder.build()  # Applies auto-detected settings
""",
    "validation_with_help": """
# Validation with helpful error messages
config_data = {"embedding_provider": "invalid", "openai": {"api_key": "bad"}}
errors = validate_configuration_with_help(config_data)

for error in errors:
    print(f"Validation error: {error}")
    # Output includes helpful suggestions for fixing issues
""",
    "enterprise_features": """
# Enterprise configuration with all features
builder = ConfigBuilderFactory.create_builder("enterprise")
config = builder.build()

# Access sophisticated features
assert config.circuit_breaker.use_enhanced_circuit_breaker
assert config.observability.track_ai_operations
assert config.deployment.enable_feature_flags
assert config.drift_detection.enabled
""",
}


# Configuration complexity examples
COMPLEXITY_EXAMPLES = {
    "simple": {
        "description": "Minimal configuration for getting started",
        "setup_time": "< 5 minutes",
        "features": ["Local embeddings", "Basic caching", "Debug logging"],
        "example": """
config = await quick_config("development")
# Ready to use with sensible defaults
""",
    },
    "moderate": {
        "description": "Production-ready with monitoring and security",
        "setup_time": "10-15 minutes",
        "features": ["OpenAI embeddings", "Redis caching", "Monitoring", "Security"],
        "example": """
builder = ConfigBuilderFactory.create_builder("production")
config = builder.build(openai={"api_key": "sk-..."})
# Production-ready with monitoring and security
""",
    },
    "advanced": {
        "description": "Enterprise-grade with full observability and resilience",
        "setup_time": "20-30 minutes",
        "features": [
            "Circuit breakers",
            "Observability",
            "Feature flags",
            "A/B testing",
            "Drift detection",
            "Auto-remediation",
        ],
        "example": """
builder = ConfigBuilderFactory.create_builder("enterprise")
config = builder.build()
# Full enterprise features with sophisticated patterns
""",
    },
}


if __name__ == "__main__":
    # Run showcase if executed directly
    asyncio.run(run_configuration_showcase())
