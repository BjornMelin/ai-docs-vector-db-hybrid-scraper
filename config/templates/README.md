# Configuration Templates

This directory contains pre-configured templates for common use cases. Each template is optimized for specific scenarios and can be used as a starting point for your configuration.

## Available Templates

### 🚀 production.json

**Use Case:** Production deployment with high performance and reliability

- Uses OpenAI embeddings for best quality
- Redis caching enabled for performance
- Vector quantization enabled for storage efficiency
- Security features enabled (API keys, rate limiting)
- Optimized batch sizes and concurrency settings

### 🛠️ development.json

**Use Case:** Local development and testing

- Debug mode enabled with verbose logging
- Uses local FastEmbed for faster iteration
- Redis caching disabled (local cache only)
- Smaller batch sizes for easier debugging
- Browser not headless for visual debugging

### 🏠 local-only.json

**Use Case:** Privacy-conscious deployment without cloud services

- All processing done locally with FastEmbed
- No external API dependencies
- Optimized for single-machine deployment
- Models cached locally
- Restricted to localhost connections only

### 🧪 testing.json

**Use Case:** Automated testing and CI/CD pipelines

- Minimal resource usage
- Caching disabled for test isolation
- Small batch sizes for fast execution
- Low timeouts for quick failure detection
- Single concurrent operations for predictability

### 📄 minimal.json

**Use Case:** Quick start with minimal configuration

- Relies on sensible defaults
- Only essential settings specified
- Good starting point for customization
- Easiest to understand and modify

## How to Use

1. **Copy a template:**

   ```bash
   cp config/templates/production.json config.json
   ```

2. **Set environment variables for sensitive data:**

   ```bash
   export AI_DOCS__OPENAI__API_KEY=sk-REPLACE-WITH-YOUR-OPENAI-API-KEY
   export AI_DOCS__FIRECRAWL__API_KEY=fc-REPLACE-WITH-YOUR-FIRECRAWL-KEY
   ```

3. **Customize as needed:**
   - Edit the copied `config.json` file
   - Override specific values via environment variables
   - Use the config management CLI for validation

4. **Validate your configuration:**

   ```bash
   python -m src.manage_config validate --config-file config.json
   ```

## Template Selection Guide

Choose your template based on:

| Scenario | Recommended Template | Key Features |
|----------|---------------------|--------------|
| Production deployment | `production.json` | High performance, security, monitoring |
| Local development | `development.json` | Debug features, visual feedback |
| Privacy requirements | `local-only.json` | No cloud dependencies |
| CI/CD pipelines | `testing.json` | Fast, isolated, predictable |
| Getting started | `minimal.json` | Simple, easy to understand |

## Customization Tips

1. **API Keys:** Never store API keys in config files. Use environment variables:

   ```bash
   AI_DOCS__OPENAI__API_KEY=sk-REPLACE-WITH-YOUR-OPENAI-API-KEY
   AI_DOCS__FIRECRAWL__API_KEY=fc-REPLACE-WITH-YOUR-FIRECRAWL-KEY
   ```

2. **Performance Tuning:**
   - Adjust `batch_size` based on your memory
   - Increase `max_concurrent_requests` for better throughput
   - Tune cache TTLs based on update frequency

3. **Cost Optimization:**
   - Use `fastembed` for free local embeddings
   - Enable vector quantization for storage savings
   - Set budget limits for API providers

4. **Security Hardening:**
   - Always enable `require_api_keys` in production
   - Configure `allowed_domains` for crawling restrictions
   - Use rate limiting to prevent abuse

## Environment-Specific Overrides

You can override any setting using environment variables:

```bash
# Override environment
export AI_DOCS__ENVIRONMENT=staging

# Override nested settings
export AI_DOCS__CACHE__REDIS_URL=redis://prod-redis:6379
export AI_DOCS__QDRANT__URL=http://qdrant-cluster:6333

# Override arrays (as JSON)
export AI_DOCS__SECURITY__ALLOWED_DOMAINS='["example.com", "docs.example.com"]'
```

## Migrating from Old Configuration

If you have existing configuration files:

1. Run the migration script:

   ```bash
   python scripts/migrate_config.py
   ```

2. Review the generated `config.json`

3. Choose the most appropriate template and merge your settings

## Need Help?

- Run `python -m src.manage_config --help` for CLI options
- Check `docs/UNIFIED_CONFIG.md` for detailed documentation
- View the schema with `python -m src.manage_config show-schema`
