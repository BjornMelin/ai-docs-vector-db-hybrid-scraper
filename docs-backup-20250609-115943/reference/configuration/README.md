# Configuration Reference

> **Purpose**: Complete configuration options and settings  
> **Audience**: System administrators and developers

## Configuration Documentation

### Core Configuration
- [**Config Schema**](../reference/configuration/config-schema.md) - Complete configuration reference with all options

## Configuration Structure

The system uses a hierarchical configuration structure:

```yaml
# Core system settings
system:
  environment: "production"
  debug: false
  
# Vector database configuration
vector_db:
  provider: "qdrant"
  host: "localhost"
  port: 6333
  
# Embedding configuration
embeddings:
  provider: "fastembed"
  model: "BAAI/bge-small-en-v1.5"
  
# Browser automation settings
browser:
  headless: true
  timeout: 30000
  rate_limits:
    lightweight: 100
    medium: 50
```

## Configuration Sources

Configuration is loaded in this priority order:
1. **Environment variables** (highest priority)
2. **Configuration files** (`config.yaml`, `config.json`)
3. **Default values** (lowest priority)

## Environment Variables

Key environment variables:
- `QDRANT_HOST` - Vector database host
- `OPENAI_API_KEY` - OpenAI API key for embeddings
- `LOG_LEVEL` - Logging verbosity
- `PORT` - Server port (default: 8000)

## Configuration Validation

All configuration is validated using Pydantic models:
- **Type checking** - Ensures correct data types
- **Range validation** - Validates numeric ranges
- **Required fields** - Enforces mandatory settings
- **Default values** - Provides sensible defaults

## Related Documentation

- 🛠️ [How-to Guides](../../how-to-guides/) - Configuration examples
- 📋 [API Reference](../api/) - API configuration
- 🚀 [Getting Started](../../getting-started/) - Initial setup