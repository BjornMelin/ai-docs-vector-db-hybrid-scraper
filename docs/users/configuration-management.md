---
title: Configuration Management
audience: users
status: active
owner: product-education
last_reviewed: 2025-03-13
---

## Configuration Management Guide

> **Status**: Active  
> **Last Updated**: 2025-01-11  
> **Purpose**: Complete guide for configuration setup, backup, and management  
> **Audience**: End users setting up and managing system configurations

The AI Documentation Vector DB system includes a comprehensive configuration management system with interactive wizards, templates, backup/restore capabilities, and migration tools to make configuration setup and maintenance simple and reliable.

## 🚀 Quick Start

### Interactive Configuration Wizard

The easiest way to get started is with the interactive configuration wizard:

```bash
# Launch the configuration setup wizard
uv run python -m src.cli.main config wizard

# The wizard will guide you through:
# 1. Choose setup mode (template, interactive, migration, or import)
# 2. Select environment and providers
# 3. Configure API keys and services
# 4. Validate and save configuration
```

### Quick Template Setup

For immediate setup, use one of the pre-configured templates:

```bash
# Development environment (recommended for local testing)
uv run python -m src.cli.main config template apply development -o config.json

# Production environment (security hardened)
uv run python -m src.cli.main config template apply production -o config.json

# Validate your configuration
uv run python -m src.cli.main config validate config.json --health-check
```

### Import an existing configuration

Already have a JSON or YAML file? Load it directly:

```bash
# Validate without saving to disk
uv run python -m src.cli.main config load config/production.json --validate-only

# Load into the current CLI session (subsequent commands use it)
uv run python -m src.cli.main config load config/staging.yaml
```

## 🧙‍♂️ Configuration Wizard

### Setup Modes

The configuration wizard supports four different setup modes:

#### 1. Template-Based Setup

Start with an optimized template and customize as needed:

```bash
uv run python -m src.cli.main config wizard
# Select: "Start with a template"
# Choose from: development, production, high_performance, memory_optimized, distributed
# Customize settings interactively
```

**Available Templates:**

- **🛠️ Development**: Debug logging, local database, fast iteration
- **🚀 Production**: Security hardening, performance optimization
- **⚡ High Performance**: Maximum throughput and concurrency
- **💾 Memory Optimized**: Resource-constrained environments
- **🌐 Distributed**: Multi-node cluster deployment

#### 2. Interactive Step-by-Step Setup

Build your configuration from scratch with guided prompts:

```bash
# Follow prompts for:
# - Environment selection (development/staging/production)
# - Provider choices (OpenAI, FastEmbed, Crawl4AI, Firecrawl)
# - Database configuration (PostgreSQL/SQLite)
# - Caching setup (Redis/DragonflyDB)
# - Security settings
```

#### 3. Migration Setup

Upgrade existing configurations to newer versions:

```bash
# Automatically detect current version and upgrade
# - Shows migration plan with estimated time
# - Creates backup before migration
# - Applies schema changes step by step
```

#### 4. Import Setup

Import configuration from external files:

```bash
# Import from JSON, YAML, or TOML files
# - Validates imported configuration
# - Shows any compatibility issues
# - Converts to current schema format
```

### Wizard Features

The configuration wizard includes:

- **Rich Visual Interface**: Beautiful progress bars and formatting
- **Input Validation**: Real-time validation of settings
- **Smart Defaults**: Environment-appropriate default values
- **API Key Management**: Secure handling of sensitive data
- **Conflict Detection**: Identifies potential configuration issues
- **Backup Creation**: Automatic backup before changes

## 🎨 Configuration Templates

### Template Overview

Templates provide optimized configurations for specific use cases:

| Template             | Use Case                      | Key Features                                  |
| -------------------- | ----------------------------- | --------------------------------------------- |
| **development**      | Local development and testing | Debug logging, local services, fast iteration |
| **production**       | Production deployment         | Security hardening, performance optimization  |
| **high_performance** | High-traffic applications     | Maximum throughput, aggressive caching        |
| **memory_optimized** | Resource-constrained systems  | Minimal memory usage, conservative settings   |
| **distributed**      | Multi-node deployments        | Cluster configuration, load balancing         |

### Using Templates

#### List Available Templates

```bash
uv run python -m src.cli.main config template list
```

Output:

```
Available Configuration Templates
┌─────────────────┬──────────────────────────────────┬─────────────────────────────────┐
│ Template        │ Description                      │ Use Case                        │
├─────────────────┼──────────────────────────────────┼─────────────────────────────────┤
│ development     │ Development with debugging       │ Local development and testing   │
│ production      │ Production with security         │ Production deployment           │
│ high_performance│ Maximum throughput optimization  │ High-traffic applications       │
│ memory_optimized│ Resource-constrained environments│ Memory-limited deployments     │
│ distributed     │ Multi-node cluster deployment   │ Large-scale distributed systems │
└─────────────────┴──────────────────────────────────┴─────────────────────────────────┘
```

#### Apply a Template

```bash
# Apply template to create configuration
uv run python -m src.cli.main config template apply production -o config.json

# Apply with environment-specific overrides
uv run python -m src.cli.main config template apply production \
  --environment-override staging \
  -o staging-config.json
```

#### Template Customization

Templates can be customized during application:

```bash
# The wizard will ask:
# - Change environment settings?
# - Configure API keys?
# - Modify provider settings?
# - Adjust performance parameters?
```

### Template Configuration Examples

#### Development Template

```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "embedding_provider": "fastembed",
  "crawl_provider": "crawl4ai",
  "database": {
    "database_url": "sqlite+aiosqlite:///data/dev.db",
    "echo_queries": true,
    "pool_size": 5
  },
  "cache": {
    "enable_caching": true,
    "enable_local_cache": true,
    "enable_dragonfly_cache": false,
    "cache_ttl_seconds": {
      "embeddings": 300,
      "crawl": 300,
      "search": 300
    }
  },
  "security": {
    "api_key_required": false,
    "enable_rate_limiting": false
  }
}
```

#### Production Template

```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "embedding_provider": "openai",
  "crawl_provider": "crawl4ai",
  "database": {
    "database_url": "postgresql+asyncpg://user:password@localhost:5432/aidocs_prod",
    "pool_size": 20,
    "max_overflow": 10,
    "adaptive_pool_sizing": true,
    "enable_query_monitoring": true
  },
  "cache": {
    "enable_caching": true,
    "enable_dragonfly_cache": true,
    "dragonfly_url": "redis://dragonfly:6379",
    "cache_ttl_seconds": {
      "embeddings": 86400,
      "crawl": 3600,
      "search": 7200
    }
  },
  "security": {
    "api_key_required": true,
    "api_keys": ["prod-key-1"],
    "enable_rate_limiting": true,
    "default_rate_limit": 100,
    "rate_limit_window": 60
  },
  "monitoring": {
    "enabled": true,
    "include_system_metrics": true,
    "enable_performance_monitoring": true
  }
}
```

## 💾 Backup and Restore System

## File Integrity Monitoring Prerequisites

The `/config/file-watch/enable` endpoint streams file change events from
osquery. Before enabling it:

- Install and run `osqueryd` with the `file_events` table enabled.
- Ensure the results log (default: `/var/log/osquery/osqueryd.results.log`) is
  readable by the API process. Override the path with the
  `OSQUERY_RESULTS_LOG` environment variable when needed.
- Configure include/exclude glob patterns in your osquery configuration; the
  API validates paths but does not mutate osquery settings.
- Confirm the agent is producing events before enabling the watcher; the API
  will return `503` if the provider fails to become ready.

### Creating Backups

The system provides Git-like versioning for configuration backups:

```bash
# Create a backup with description
uv run python -m src.cli.main config backup create config.json \
  --description "Before production deployment" \
  --tags "production,deployment" \
  --compress

# Quick backup
uv run python -m src.cli.main config backup create config.json
```

### Listing Backups

```bash
# List all backups
uv run python -m src.cli.main config backup list

# Filter by configuration name
uv run python -m src.cli.main config backup list --config-name production

# Filter by environment
uv run python -m src.cli.main config backup list --environment production

# Filter by tags
uv run python -m src.cli.main config backup list --tags "deployment,staging"

# Limit results
uv run python -m src.cli.main config backup list --limit 10
```

Example output:

```
Configuration Backups
┌──────────────┬────────────┬─────────────────────┬─────────────┬──────┬─────────────────────────────────┐
│ ID           │ Config     │ Created             │ Environment │ Size │ Description                     │
├──────────────┼────────────┼─────────────────────┼─────────────┼──────┼─────────────────────────────────┤
│ 20250111...  │ production │ 2025-01-11 10:30:15│ production  │ 2.3MB│ Before production deployment    │
│ 20250110...  │ staging    │ 2025-01-10 15:45:32│ staging     │ 1.8MB│ Weekly backup                   │
│ 20250110...  │ development│ 2025-01-10 09:12:07│ development │ 1.2MB│ Feature testing backup          │
└──────────────┴────────────┴─────────────────────┴─────────────┴──────┴─────────────────────────────────┘
```

### Restoring from Backup

```bash
# Restore a specific backup
uv run python -m src.cli.main config backup restore 20250111_abcd1234 \
  --target restored-config.json

# Force restore (override conflicts)
uv run python -m src.cli.main config backup restore 20250111_abcd1234 \
  --target config.json \
  --force

# Restore without creating pre-restore backup
uv run python -m src.cli.main config backup restore 20250111_abcd1234 \
  --no-pre-backup
```

### Backup Features

- **Automatic Compression**: Reduces backup size by 60-80%
- **Metadata Tracking**: Environment, template source, creation time
- **Conflict Detection**: Identifies incompatible changes
- **Incremental Backups**: Only store changes between versions
- **Search and Filtering**: Find backups by tags, environment, or date
- **Pre-restore Backups**: Automatic backup before restore operations

## 🔄 Configuration Migration System

### Migration Overview

The migration system handles configuration schema upgrades safely:

- **Version Tracking**: Tracks current configuration version
- **Migration Plans**: Shows required steps before execution
- **Rollback Support**: Undo migrations when needed
- **Dry Run Mode**: Preview changes without applying them
- **Automatic Backups**: Creates backups before migrations

### Creating Migration Plans

```bash
# Create migration plan
uv run python -m src.cli.main config migrate plan config.json 2.0.0
```

Example output:

```
📋 Migration Plan

From: 1.0.0 → To: 2.0.0
Estimated Duration: ~4 minutes
✅ No Downtime Required

Migration Steps:
  1. 1.0.0_to_1.1.0 - Add enhanced validation metadata
  2. 1.1.0_to_1.2.0 - Update cache configuration structure
  3. 1.2.0_to_2.0.0 - Major version upgrade with new features

Rollback Plan Available:
  1. 1.2.0_to_2.0.0
  2. 1.1.0_to_1.2.0
  3. 1.0.0_to_1.1.0
```

### Applying Migrations

```bash
# Apply migration plan
uv run python -m src.cli.main config migrate apply config.json 2.0.0

# Dry run (preview changes only)
uv run python -m src.cli.main config migrate apply config.json 2.0.0 --dry-run

# Force migration (skip warnings)
uv run python -m src.cli.main config migrate apply config.json 2.0.0 --force
```

### Rolling Back Migrations

```bash
# Rollback specific migration
uv run python -m src.cli.main config migrate rollback config.json 1.1.0_to_1.2.0

# Dry run rollback
uv run python -m src.cli.main config migrate rollback config.json 1.1.0_to_1.2.0 --dry-run
```

### Migration Status

```bash
# Check migration status
uv run python -m src.cli.main config migrate status config.json
```

Example output:

```
📊 Migration Status

Current Version: 1.2.0
Available Migrations: 5
Applied Migrations: 3

Available Migrations
┌─────────────────┬─────────┬──────┬─────────────────────────────────────┬─────────┐
│ Migration ID    │ From    │ To   │ Description                         │ Applied │
├─────────────────┼─────────┼──────┼─────────────────────────────────────┼─────────┤
│ 1.0.0_to_1.1.0  │ 1.0.0   │ 1.1.0│ Add enhanced validation metadata    │ ✅      │
│ 1.1.0_to_1.2.0  │ 1.1.0   │ 1.2.0│ Update cache configuration structure│ ✅      │
│ 1.2.0_to_2.0.0  │ 1.2.0   │ 2.0.0│ Major version upgrade               │ ❌      │
└─────────────────┴─────────┴──────┴─────────────────────────────────────┴─────────┘
```

## ✅ Configuration Validation

### Basic Validation

```bash
# Validate configuration file
uv run python -m src.cli.main config validate config.json

# Validate with health checks
uv run python -m src.cli.main config validate config.json --health-check
```

### Validation Features

The enhanced validation system provides:

- **Syntax Validation**: JSON/YAML structure and type checking
- **Business Rules**: Environment-specific validation rules
- **Service Connectivity**: Tests connections to external services
- **Performance Checks**: Validates performance-related settings
- **Security Validation**: Checks security configuration
- **Automatic Fixes**: Suggests fixes for common issues

### Validation Output

```
✅ Configuration is valid!

Configuration Summary
┌───────────────┬─────────────┬─────────────────────────────────┐
│ Component     │ Status      │ Details                         │
├───────────────┼─────────────┼─────────────────────────────────┤
│ Qdrant        │ Configured  │ http://localhost:6333           │
│ OpenAI        │ Configured  │ API key provided                │
│ FastEmbed     │ Enabled     │ BAAI/bge-small-en-v1.5          │
│ Redis Cache   │ Enabled     │ redis://localhost:6379          │
└───────────────┴─────────────┴─────────────────────────────────┘

Health Check Results:
┌─────────┬────────────┬─────────────────────────────────┐
│ Service │ Status     │ Details                         │
├─────────┼────────────┼─────────────────────────────────┤
│ Qdrant  │ ✅ Healthy │ Connected                       │
│ Redis   │ ✅ Healthy │ Connected                       │
│ OpenAI  │ ✅ Healthy │ Connected                       │
└─────────┴────────────┴─────────────────────────────────┘
```

### Validation with Automatic Fixes

The system can automatically fix common configuration issues:

```bash
# Run validation wizard with auto-fix
uv run python -m src.cli.main config wizard --config-path config.json
# Select validation mode and follow prompts for fixes
```

## 🔧 Advanced Configuration Management

### Configuration Formats

The system supports multiple configuration formats:

```bash
# JSON (default)
uv run python -m src.cli.main config template apply development -o config.json

# YAML
uv run python -m src.cli.main config convert config.json config.yaml

# TOML
uv run python -m src.cli.main config convert config.json config.toml
```

### Environment Variables

Override any configuration setting with environment variables:

```bash
# Use AI_DOCS__ prefix for nested configuration
export AI_DOCS__ENVIRONMENT=production
export AI_DOCS__DEBUG=false
export AI_DOCS__OPENAI__API_KEY=sk-your-key
export AI_DOCS__CACHE__REDIS_URL=redis://localhost:6379

# Arrays as JSON
export AI_DOCS__SECURITY__ALLOWED_DOMAINS='["example.com", "docs.example.com"]'
# Restrict CORS origins for production deployments
export AI_DOCS__SECURITY__CORS_ALLOWED_ORIGINS='["https://docs.example.com"]'
# Or override via a simple comma-separated allow list
export CORS_ALLOWED_ORIGINS=https://docs.example.com,https://app.example.com
```

### Configuration Display

```bash
# Show configuration overview
uv run python -m src.cli.main config show

# Show specific section
uv run python -m src.cli.main config show --section openai

# Export as JSON with syntax highlighting
uv run python -m src.cli.main config show --format json

# Export as YAML
uv run python -m src.cli.main config show --format yaml
```

### Configuration Conversion

```bash
# Convert between formats
uv run python -m src.cli.main config convert config.json config.yaml --format yaml
uv run python -m src.cli.main config convert config.yaml config.toml --format toml

# Auto-detect format from extension
uv run python -m src.cli.main config convert config.json config.yaml
```

## 🛡️ Best Practices

### 1. Environment-Specific Configurations

Use different configurations for each environment:

```bash
# Development
uv run python -m src.cli.main config template apply development -o config.dev.json

# Staging
uv run python -m src.cli.main config template apply production -o config.staging.json
# Customize for staging environment

# Production
uv run python -m src.cli.main config template apply production -o config.prod.json
```

### 2. Regular Backups

Create backups before major changes:

```bash
# Before deployment
uv run python -m src.cli.main config backup create config.json \
  --description "Pre-deployment backup" \
  --tags "deployment,$(date +%Y-%m-%d)"

# Weekly backups
uv run python -m src.cli.main config backup create config.json \
  --description "Weekly backup" \
  --tags "weekly,automated"
```

### 3. Version Control Integration

Track configuration templates in version control:

```bash
# Store template configurations
git add config/templates/
git commit -m "Add configuration templates"

# Store environment-specific settings as templates
git add config/environments/
git commit -m "Add environment configurations"
```

### 4. Security Management

Protect sensitive configuration data:

```bash
# Use environment variables for secrets
export AI_DOCS__OPENAI__API_KEY="${OPENAI_API_KEY}"
export AI_DOCS__ANTHROPIC__API_KEY="${ANTHROPIC_API_KEY}"

# Never commit actual API keys
echo "config.json" >> .gitignore
echo ".env" >> .gitignore
```

### 5. Validation Before Deployment

Always validate configurations before deployment:

```bash
# Complete validation with health checks
uv run python -m src.cli.main config validate config.json --health-check

# Test migration before applying
uv run python -m src.cli.main config migrate apply config.json 2.0.0 --dry-run
```

## 🆘 Troubleshooting

### Common Issues

#### Missing API Keys

```
Error: OpenAI API key required when using OpenAI embedding provider
Solution: Set AI_DOCS__OPENAI__API_KEY environment variable or use wizard to configure
```

#### Service Connection Failures

```
Error: Qdrant connection failed: Connection refused
Solution: Ensure Qdrant is running on configured URL or update configuration
```

#### Migration Conflicts

```
Error: Migration conflict detected - environment mismatch
Solution: Use --force flag or resolve conflicts manually
```

#### Invalid Configuration Format

```
Error: Invalid JSON format in configuration file
Solution: Use 'config convert' command to fix format or validate with wizard
```

### Debug Mode

Enable debug output for detailed troubleshooting:

```bash
# Enable debug logging
export AI_DOCS__DEBUG=true
export AI_DOCS__LOG_LEVEL=DEBUG

# Run commands with verbose output
uv run python -m src.cli.main config validate config.json --health-check
```

### Getting Help

```bash
# Get help for any command
uv run python -m src.cli.main config --help
uv run python -m src.cli.main config wizard --help
uv run python -m src.cli.main config backup --help
uv run python -m src.cli.main config migrate --help

# Run configuration diagnostics
uv run python -m src.cli.main config validate config.json --comprehensive
```

## 📚 Related Resources

- **[Setup & Configuration Guide](../developers/setup-and-configuration.md)**: Complete API documentation
- **[Operations Guide](../operators/operations.md)**: Production deployment and operations patterns
- **[Security Guide](../operators/security.md)**: Security configuration best practices
- **[Troubleshooting Guide](./troubleshooting.md)**: Common issues and solutions

---

_🛠️ The advanced configuration management system makes setup and maintenance simple while providing enterprise-grade features for backup, migration, and validation._
