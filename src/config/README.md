# Progressive Configuration System

A sophisticated configuration system demonstrating progressive complexity patterns and advanced Pydantic v2 validation.

## 🌟 Portfolio Showcase Features

### Progressive Complexity & Guided Discovery
- **Persona-based builders** for different use cases (Development, Production, Research, Enterprise)
- **Progressive disclosure** of advanced features with guided setup
- **Auto-detection** and intelligent service discovery
- **Smart defaults** based on environment analysis

### Advanced Pydantic v2 Patterns
- **20+ configuration models** with sophisticated validation
- **Custom field validators** with helpful error messages
- **Model validators** for cross-field validation logic
- **Computed fields** for derived properties
- **SecretStr integration** for sensitive data handling

### Enterprise-Grade Features
- **Circuit breakers** with adaptive timeout and bulkhead isolation
- **Configuration drift detection** with auto-remediation
- **Feature flags** and deployment strategy management
- **Observability** with OpenTelemetry instrumentation
- **Intelligent validation** with context-aware suggestions

## 🚀 Quick Start

### Simple Configuration (< 30 seconds)
```python
from src.config import quick_config

# One-line configuration with smart defaults
config = await quick_config("development")
```

### Guided Setup with Discovery
```python
from src.config import guided_config_setup

# Progressive guided setup
guide = await guided_config_setup()
discovery = await guide.start_guided_setup()

# Start with essential features
config = await guide.build_configuration("essential")

# Upgrade to intermediate when ready
config = await guide.build_configuration("intermediate")

# Full enterprise features
config = await guide.build_configuration("advanced")
```

### Advanced Builder Pattern
```python
from src.config import ConfigBuilderFactory

# Create persona-specific builder
builder = ConfigBuilderFactory.create_builder("enterprise")

# Perform discovery
discovery = await builder.discover_configuration()
print(f"Auto-detected services: {discovery.auto_detected_services}")

# Build with overrides
config = builder.build(
    openai={"api_key": "sk-..."},
    qdrant={"url": "http://custom-qdrant:6333"}
)
```

## 🎭 Configuration Personas

### Development Persona
**Target**: Local development and debugging
```python
builder = ConfigBuilderFactory.create_builder("development")
```
- Local embeddings (FastEmbed)
- Browser automation (Crawl4AI)  
- Debug logging enabled
- Relaxed security settings
- Local caching only

### Production Persona
**Target**: Production deployments
```python
builder = ConfigBuilderFactory.create_builder("production")
```
- OpenAI embeddings
- Redis distributed caching
- Circuit breakers enabled
- Monitoring & observability
- Security & rate limiting

### Research Persona
**Target**: Research and experimentation
```python
builder = ConfigBuilderFactory.create_builder("research")
```
- OpenAI embeddings with full precision
- HyDE query expansion
- RAG generation enabled
- Extended caching for experiments
- Cost tracking enabled

### Enterprise Persona
**Target**: Enterprise deployments
```python
builder = ConfigBuilderFactory.create_builder("enterprise")
```
- All advanced features enabled
- Feature flags & A/B testing
- Configuration drift detection
- Advanced observability
- Auto-remediation capabilities

## 🔍 Intelligent Discovery & Validation

### Auto-Detection
```python
from src.config import discover_optimal_configuration

# Discover optimal configuration based on environment
recommendations = await discover_optimal_configuration()

print(f"Recommended persona: {recommendations.recommended_persona}")
print(f"Confidence: {recommendations.confidence_score:.1%}")
print(f"Auto-detected services: {recommendations.auto_detected_services}")
```

### Intelligent Validation
```python
from src.config import validate_configuration_intelligently

# Validate with context-aware suggestions
config_data = {"embedding_provider": "openai", "openai": {"api_key": "invalid"}}
report = await validate_configuration_intelligently(config_data, "production")

print(f"Security score: {report.security_score}/100")
print(f"Performance score: {report.performance_score}/100")

# Get helpful error messages with suggestions
for error in report.errors:
    print(f"Error: {error['message']}")
    for suggestion in error['suggestions']:
        print(f"  💡 {suggestion}")
```

### System Analysis
```python
from src.config import get_system_recommendations

# Get comprehensive system analysis
system_env, recommendations = await get_system_recommendations()

print(f"System: {system_env.platform} {system_env.architecture}")
print(f"Resources: {system_env.memory_gb}GB RAM, {system_env.cpu_count} cores")
print(f"Recommended providers: {recommendations.suggested_providers}")
```

## 📊 Progressive Feature Levels

### Essential (< 5 minutes setup)
- Basic configuration with local providers
- Minimal monitoring
- Simple validation

### Intermediate (10-15 minutes setup)
- Cloud providers integration
- Advanced caching
- Security features
- Monitoring & metrics

### Advanced (20-30 minutes setup)
- Full enterprise features
- Circuit breakers & resilience
- Observability stack
- Feature flags & deployment
- Configuration drift detection

## 🛠️ Advanced Validation Patterns

### Field Validators with Suggestions
```python
@field_validator("api_key")
@classmethod
def validate_api_key(cls, v: str | None) -> str | None:
    if v and not v.startswith("sk-"):
        raise ValueError(
            "OpenAI API key must start with 'sk-'\n"
            "💡 Get API key from https://platform.openai.com/api-keys\n"
            "💡 Set via environment: AI_DOCS_OPENAI__API_KEY=sk-..."
        )
    return v
```

### Cross-Field Model Validation
```python
@model_validator(mode="after")
def validate_provider_keys(self) -> "Config":
    if (
        self.embedding_provider == EmbeddingProvider.OPENAI
        and not self.openai.api_key
    ):
        raise ConfigValidationError(
            "OpenAI API key required when using OpenAI embedding provider",
            field_path="openai.api_key",
            suggestions=[
                "Set OPENAI_API_KEY environment variable",
                "Or switch to 'fastembed' provider for local embeddings"
            ]
        )
    return self
```

### Context-Aware Validation
```python
validator = IntelligentValidator()
report = await validator.validate_with_intelligence(config_data, persona="production")

# Gets suggestions based on:
# - System environment (memory, CPU, platform)
# - Selected persona requirements
# - Auto-detected services
# - Security best practices
```

## 🏗️ Enterprise Architecture Patterns

### Circuit Breaker Configuration
```python
circuit_breaker = {
    "use_enhanced_circuit_breaker": True,
    "enable_detailed_metrics": True,
    "enable_fallback_mechanisms": True,
    "enable_adaptive_timeout": True,
    "service_overrides": {
        "openai": {"failure_threshold": 3, "recovery_timeout": 30.0},
        "qdrant": {"failure_threshold": 2, "recovery_timeout": 15.0}
    }
}
```

### Observability Stack
```python
observability = {
    "enabled": True,
    "track_ai_operations": True,
    "track_costs": True,
    "instrument_fastapi": True,
    "instrument_httpx": True,
    "instrument_redis": True,
    "otlp_endpoint": "http://jaeger:4317"
}
```

### Configuration Drift Detection
```python
drift_detection = {
    "enabled": True,
    "snapshot_interval_minutes": 10,
    "alert_on_severity": ["high", "critical"],
    "enable_auto_remediation": True,
    "auto_remediation_severity_threshold": "high"
}
```

## 🔧 CLI Interface

Interactive CLI for configuration management:

```bash
# Quick configuration creation
python -m src.config.cli_demo create --persona development

# Guided setup with discovery
python -m src.config.cli_demo create --output config.json

# Validate existing configuration
python -m src.config.cli_demo validate config.json --persona production

# Discover optimal configuration
python -m src.config.cli_demo discover

# Run complete showcase
python -m src.config.cli_demo showcase

# Generate examples
python -m src.config.cli_demo examples --output-dir examples/
```

## 📋 Configuration Templates

Pre-built templates for different scenarios:

- `development.json` - Local development
- `production.json` - Production deployment
- `enterprise-showcase.json` - Full enterprise features
- `research.json` - Research & experimentation
- `minimal.json` - Minimal configuration
- `testing.json` - Test environments

## 🎯 Portfolio Highlights

### Technical Sophistication
- **20+ Pydantic models** with advanced validation patterns
- **Progressive complexity** with guided discovery
- **Auto-detection** and intelligent service discovery
- **Context-aware validation** with helpful suggestions

### Enterprise-Grade Features
- **Circuit breakers** with adaptive patterns
- **Configuration drift detection** and auto-remediation
- **Feature flags** and deployment strategies
- **Full observability** with OpenTelemetry

### Clean API Design
- **Simple interfaces** hiding complex validation logic
- **Progressive disclosure** of advanced features
- **Persona-based builders** for different use cases
- **Intelligent defaults** based on environment analysis

### Advanced Patterns
- **Custom validation errors** with suggestions
- **Cross-field validation** with business logic
- **Computed fields** for derived properties
- **Settings with environment mapping**

## 🚀 Next Steps

1. **Quick Start**: Use `quick_config()` for immediate productivity
2. **Guided Setup**: Try `guided_config_setup()` for discovery
3. **Advanced Features**: Explore enterprise builders
4. **Validation**: Use intelligent validation for configuration review
5. **CLI Demo**: Run the showcase to see all capabilities

This configuration system demonstrates sophisticated software architecture patterns while maintaining simplicity for common use cases - a perfect showcase of progressive complexity in action.