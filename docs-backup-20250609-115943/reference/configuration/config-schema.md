# Unified Configuration System

> **Status**: Current  
> **Last Updated**: 2025-06-09  
> **Purpose**: Config Schema reference documentation  
> **Audience**: Developers needing technical details

The AI Documentation Vector DB now uses a comprehensive unified configuration system built with Pydantic v2 and pydantic-settings. This system consolidates all application settings into a single, well-structured configuration model with support for multiple sources, validation, and type safety.

## Table of Contents

- [Overview](#overview)
- [Configuration Structure](#configuration-structure)
- [Configuration Sources](#configuration-sources)
- [Usage Guide](#usage-guide)
- [Configuration Management CLI](#configuration-management-cli)
- [Templates](#templates)
- [Migration](#migration)
- [API Reference](#api-reference)
- [Best Practices](#best-practices)

## Overview

### Key Features

- **Single Source of Truth**: All configuration in one `UnifiedConfig` class
- **Type Safety**: Full Pydantic v2 validation and type checking
- **Multiple Sources**: Support for files, environment variables, and code
- **Validation**: Comprehensive validation with helpful error messages
- **Schema Generation**: Auto-generate JSON Schema, TypeScript types, and docs
- **Migration Tools**: Automatic migration between configuration versions
- **Templates**: Pre-configured templates for common use cases
- **CLI Management**: Rich CLI for configuration management
- **Comprehensive Testing**: 45+ tests covering all configuration aspects
- **Security Validation**: Built-in security checks and validation rules

### Architecture

```plaintext
┌─────────────────────────────────────────────────────┐
│                  UnifiedConfig                      │
│  ┌─────────────────────────────────────────────┐   │
│  │ Environment Settings (env, debug, log_level) │   │
│  ├─────────────────────────────────────────────┤   │
│  │ Provider Selection (embedding, crawl)        │   │
│  ├─────────────────────────────────────────────┤   │
│  │ Component Configurations:                    │   │
│  │ - CacheConfig                               │   │
│  │ - QdrantConfig                              │   │
│  │ - OpenAIConfig                              │   │
│  │ - FastEmbedConfig                           │   │
│  │ - FirecrawlConfig                           │   │
│  │ - Crawl4AIConfig                            │   │
│  │ - ChunkingConfig                            │   │
│  │ - PerformanceConfig                         │   │
│  │ - SecurityConfig                            │   │
│  ├─────────────────────────────────────────────┤   │
│  │ Documentation Sites List                     │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Configuration Structure

### Root Configuration

```python
class UnifiedConfig(BaseSettings):
    # Environment settings
    environment: Environment  # development, testing, production
    debug: bool
    log_level: LogLevel  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    # Application metadata
    app_name: str
    version: str
    
    # Provider selection
    embedding_provider: EmbeddingProvider  # openai, fastembed
    crawl_provider: CrawlProvider  # crawl4ai, firecrawl
    
    # Component configurations
    cache: CacheConfig
    qdrant: QdrantConfig
    openai: OpenAIConfig
    fastembed: FastEmbedConfig
    firecrawl: FirecrawlConfig
    crawl4ai: Crawl4AIConfig
    chunking: ChunkingConfig
    performance: PerformanceConfig
    security: SecurityConfig
    
    # Documentation sites
    documentation_sites: list[DocumentationSite]
    
    # File paths
    data_dir: Path
    cache_dir: Path
    logs_dir: Path
```

### Component Configurations

Each component has its own configuration class with specific settings:

#### CacheConfig

- Enable/disable caching layers
- Redis connection settings
- TTL configurations
- Memory limits

#### QdrantConfig

- Vector database connection
- Collection settings
- Performance tuning
- Quantization options

#### Provider Configs

- API keys and authentication
- Model selection
- Rate limiting
- Cost tracking

## Configuration Sources

The system loads configuration from multiple sources in priority order:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (JSON, YAML, TOML)
3. **`.env` Files**
4. **Default Values** (lowest priority)

### Environment Variables

All settings can be overridden using environment variables with the `AI_DOCS__` prefix:

```bash
# Simple values
export AI_DOCS__ENVIRONMENT=production
export AI_DOCS__DEBUG=false

# Nested values (use double underscore)
export AI_DOCS__OPENAI__API_KEY=sk-your-api-key
export AI_DOCS__CACHE__REDIS_URL=redis://localhost:6379

# Arrays (as JSON)
export AI_DOCS__SECURITY__ALLOWED_DOMAINS='["example.com", "docs.example.com"]'
```

### Configuration Files

Support for multiple formats:

```json
// config.json
{
  "environment": "production",
  "embedding_provider": "openai",
  "openai": {
    "api_key": "sk-your-api-key",
    "model": "text-embedding-3-small"
  }
}
```

```yaml
# config.yaml
environment: production
embedding_provider: openai
openai:
  api_key: sk-your-api-key
  model: text-embedding-3-small
```

## Usage Guide

### Basic Usage

```python
from src.config import UnifiedConfig, get_config

# Get the global configuration instance
config = get_config()

# Access settings
print(config.environment)
print(config.openai.api_key)
print(config.cache.enable_caching)

# Get active provider configurations
providers = config.get_active_providers()
embedding_config = providers["embedding"]
crawl_config = providers["crawl"]
```

### Loading Configuration

```python
from src.config_loader import ConfigLoader

# Load from multiple sources
config = ConfigLoader.load_config(
    config_file="config.json",
    env_file=".env",
    include_env=True,
    documentation_sites_file="config/documentation-sites.json"
)

# Validate configuration
is_valid, issues = ConfigLoader.validate_config(config)
if not is_valid:
    for issue in issues:
        print(f"Issue: {issue}")
```

### Custom Configuration

```python
from src.config import UnifiedConfig, set_config

# Create custom configuration
custom_config = UnifiedConfig(
    environment="testing",
    debug=True,
    embedding_provider="fastembed",
    cache={"enable_redis_cache": False}
)

# Set as global configuration
set_config(custom_config)
```

## Configuration Management CLI

The system includes a comprehensive CLI for managing configurations:

### Available Commands

```bash
# Create example configuration
python -m src.manage_config create-example -o config.json

# Create .env template
python -m src.manage_config create-env-template

# Validate configuration
python -m src.manage_config validate -c config.json

# Convert between formats
python -m src.manage_config convert config.json config.yaml --to-format yaml

# Show active providers
python -m src.manage_config show-providers

# Migrate documentation sites
python -m src.manage_config migrate-sites

# Check service connections
python -m src.manage_config check-connections

# Generate schema
python -m src.manage_config generate-schema -o schema/

# Show schema in terminal
python -m src.manage_config show-schema

# Migrate configuration
python -m src.manage_config migrate config.json --target-version 0.3.0

# Show migration paths
python -m src.manage_config show-migration-path
```

### Examples

#### Validate Configuration

```bash
$ python -m src.manage_config validate -c config.json --show-config
✓ Configuration is valid!

Loaded Configuration:
{
  "environment": "production",
  "debug": false,
  ...
}
```

#### Check Connections

```bash
$ python -m src.manage_config check-connections
Checking Service Connections...

🔍 Checking Qdrant...
✓ Qdrant connected (3 collections)

🔍 Checking Redis...
✓ Redis connected

🔍 Checking OpenAI...
✓ OpenAI connected
```

## Templates

Pre-configured templates are available in `config/templates/`:

### Available Templates

1. **production.json** - Production deployment with high performance
2. **development.json** - Local development with debug features
3. **local-only.json** - Privacy-focused without cloud services
4. **testing.json** - Optimized for test execution
5. **minimal.json** - Minimal configuration with defaults

### Using Templates

```bash
# Copy a template
cp config/templates/production.json config.json

# Set sensitive values via environment
export AI_DOCS__OPENAI__API_KEY=sk-your-api-key

# Validate
python -m src.manage_config validate -c config.json
```

## Migration

### Automatic Migration

The system can automatically migrate configurations between versions:

```bash
# Migrate to latest version
python -m src.manage_config migrate config.json

# Dry run to see changes
python -m src.manage_config migrate config.json --dry-run

# Migrate without backup
python -m src.manage_config migrate config.json --no-backup
```

### Manual Migration

For existing projects:

```bash
# Run migration script
python scripts/migrate_config.py

# This will:
# 1. Convert documentation-sites.json to unified format
# 2. Migrate environment variables
# 3. Create config.json and .env.example
# 4. Validate the configuration
```

## API Reference

### Core Classes

#### UnifiedConfig

Main configuration class with all settings.

```python
config = UnifiedConfig(
    environment="production",
    embedding_provider="openai",
    openai={"api_key": "sk-..."}
)
```

#### ConfigLoader

Utilities for loading configurations from various sources.

```python
config = ConfigLoader.load_config(
    config_file="config.json",
    include_env=True
)
```

#### ConfigValidator

Comprehensive validation utilities.

```python
# Validate configuration
is_valid, issues = ConfigValidator.validate_config(config)

# Check environment variables
env_results = ConfigValidator.check_env_vars()

# Validate connections
connections = ConfigValidator.validate_config_connections(config)

# Generate validation report
report = ConfigValidator.generate_validation_report(config)
```

#### ConfigSchemaGenerator

Generate schema in various formats.

```python
# Generate JSON Schema
schema = ConfigSchemaGenerator.generate_json_schema()

# Generate TypeScript types
ts_types = ConfigSchemaGenerator.generate_typescript_types()

# Generate Markdown documentation
docs = ConfigSchemaGenerator.generate_markdown_docs()

# Save all formats
ConfigSchemaGenerator.save_schema("schema/")
```

#### ConfigMigrator

Handle configuration version migrations.

```python
# Detect version
version = ConfigMigrator.detect_config_version(config_data)

# Migrate between versions
migrated = ConfigMigrator.migrate_between_versions(
    config_data, "0.1.0", "0.3.0"
)

# Auto-migrate file
success, message = ConfigMigrator.auto_migrate("config.json")
```

### Helper Functions

```python
from src.config import get_config, set_config, reset_config

# Get global instance
config = get_config()

# Set custom instance
set_config(custom_config)

# Reset to defaults
reset_config()
```

## Best Practices

### 1. Environment-Specific Configuration

Use different configurations for each environment:

```bash
# Development
cp config/templates/development.json config.dev.json
export CONFIG_FILE=config.dev.json

# Production
cp config/templates/production.json config.prod.json
export CONFIG_FILE=config.prod.json
```

### 2. Secrets Management

Never commit secrets to version control:

```bash
# .gitignore
.env
config.json
*.secret

# Use environment variables
export AI_DOCS__OPENAI__API_KEY=${OPENAI_API_KEY}
export AI_DOCS__FIRECRAWL__API_KEY=${FIRECRAWL_API_KEY}
```

### 3. Validation

Always validate configuration before use:

```python
# In application startup
config = get_config()
issues = config.validate_completeness()
if issues:
    logger.error(f"Configuration issues: {issues}")
    sys.exit(1)
```

### 4. Type Safety

Use type hints with configuration:

```python
from src.config import UnifiedConfig

def process_documents(config: UnifiedConfig) -> None:
    if config.embedding_provider == "openai":
        # TypeScript knows config.openai is available
        client = OpenAI(api_key=config.openai.api_key)
```

### 5. Configuration as Code

Track configuration changes in version control:

```bash
# Track templates and schemas
git add config/templates/
git add schema/

# Document configuration changes
git commit -m "feat(config): add new caching options"
```

### 6. Testing Configuration

Test configuration changes thoroughly:

```bash
# Run configuration-specific tests
uv run pytest tests/unit/config/ -v

# Test specific configuration scenarios
uv run pytest tests/unit/config/test_unified_config.py::test_environment_loading
uv run pytest tests/unit/config/test_enums.py::test_all_enums_are_string_enums

# Validate configuration with comprehensive checks
python -m src.manage_config validate --comprehensive
```

### 7. Monitoring

Monitor configuration usage:

```python
# Log configuration on startup
logger.info(f"Starting with environment: {config.environment}")
logger.info(f"Embedding provider: {config.embedding_provider}")
logger.info(f"Cache enabled: {config.cache.enable_caching}")

# Track configuration validation
report = ConfigValidator.generate_validation_report(config)
logger.debug(f"Configuration report:\n{report}")
```

## Troubleshooting

### Common Issues

1. **Missing API Keys**

   ```bash
   Error: OpenAI API key required when using OpenAI embedding provider
   Solution: Set AI_DOCS__OPENAI__API_KEY environment variable
   ```

2. **Invalid Environment Variables**

   ```bash
   Error: AI_DOCS__DEBUG: Invalid boolean value 'yes'
   Solution: Use true/false, 1/0, on/off for boolean values
   ```

3. **Connection Failures**

   ```bash
   Error: Qdrant connection failed: Connection refused
   Solution: Ensure Qdrant is running on the configured URL
   ```

4. **Migration Errors**

   ```bash
   Error: Could not detect configuration version
   Solution: Add "version": "0.3.0" to your configuration file
   ```

### Debug Mode

Enable debug mode for detailed configuration information:

```bash
export AI_DOCS__DEBUG=true
export AI_DOCS__LOG_LEVEL=DEBUG
```

### Support

For configuration issues:

1. Run validation: `python -m src.manage_config validate`
2. Check connections: `python -m src.manage_config check-connections`
3. Review generated report in validation output
4. Check environment variables with validation tools
