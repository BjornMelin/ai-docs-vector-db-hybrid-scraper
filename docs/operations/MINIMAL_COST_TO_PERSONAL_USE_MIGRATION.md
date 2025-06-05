# Migration Guide: minimal-cost to personal-use Configuration

This guide helps users migrate from the deprecated `minimal-cost` configuration to the new `personal-use` configuration.

## Overview

The `minimal-cost` configuration template has been renamed to `personal-use` to better reflect its intended use case: individual developers and personal projects. The functionality remains the same, but the naming is now more descriptive.

## Migration Steps

### 1. Update Configuration Files

If you're using `config/templates/minimal-cost.json`:

```bash
# If you have a custom config based on minimal-cost
cp config.json config.json.backup

# Copy the new personal-use template
cp config/templates/personal-use.json config.json

# Apply any custom settings from your backup
```

### 2. Update Docker Compose Commands

Replace any references to `minimal-cost` in your Docker commands:

**Old:**
```bash
docker-compose -f docker-compose.minimal-cost.yml --profile minimal-cost up -d
```

**New:**
```bash
docker-compose -f docker-compose.personal-use.yml --profile personal-use up -d
```

### 3. Update Environment Variables

If you have scripts or CI/CD pipelines that reference the configuration:

**Old:**
```bash
export CONFIG_TEMPLATE=minimal-cost
```

**New:**
```bash
export CONFIG_TEMPLATE=personal-use
```

### 4. Update Documentation References

Update any internal documentation or README files that reference `minimal-cost` configuration.

## What Changed?

- **Name Only**: The configuration values remain identical
- **File Locations**:
  - `config/templates/minimal-cost.json` → `config/templates/personal-use.json`
  - `docker-compose.minimal-cost.yml` → `docker-compose.personal-use.yml`
- **Docker Profile**: `minimal-cost` → `personal-use`

## Benefits of the New Name

1. **Clarity**: "personal-use" clearly indicates the target audience
2. **Consistency**: Aligns with other template naming conventions
3. **Discoverability**: Easier for new users to understand the purpose

## No Action Required If...

- You're using a custom `config.json` not based on the template
- You're using other templates (production, development, testing, etc.)
- You've already migrated to `personal-use`

## Need Help?

If you encounter any issues during migration:

1. Check the [Configuration Templates README](../config/templates/README.md)
2. Validate your configuration:
   ```bash
   python -m src.manage_config validate --config-file config.json
   ```
3. Review the [Troubleshooting Guide](./TROUBLESHOOTING.md)

## Timeline

- The `minimal-cost` configuration name was deprecated in v1.0
- Full removal of `minimal-cost` references is planned for v2.0
- The `personal-use` configuration is the recommended replacement

---

Last updated: January 2025