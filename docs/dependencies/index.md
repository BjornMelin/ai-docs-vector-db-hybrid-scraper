# Dependency Management Documentation

> **Version**: 0.1.0  
> **Last Updated**: July 8, 2025

## Overview

This section contains comprehensive documentation about dependency management, updates, and migration guides for the AI Docs Vector DB Hybrid Scraper project.

## Documentation Structure

### 📋 [Dependency Changes Summary](./DEPENDENCY_CHANGES_SUMMARY.md)
Quick reference guide with all recent dependency updates, new additions, and breaking changes in a concise format.

### 📚 [Dependency Upgrade Guide](./DEPENDENCY_UPGRADE_GUIDE.md)
Comprehensive documentation of all dependency changes including:
- Detailed update history
- Breaking changes and migration notes
- Performance improvements
- Security updates
- Python version compatibility

### ✅ [Migration Checklist](./MIGRATION_CHECKLIST.md)
Step-by-step checklist for developers to follow when updating their local environment after dependency changes.

## Quick Links

### Recent Updates (June-July 2025)
- **faker**: 36.1.0 → 37.4.0
- **mutmut**: 2.5.1 → 3.3.0
- **pyarrow**: 18.1.0 → 20.0.0
- **cachetools**: 5.3.0 → 6.1.0
- **prometheus-client**: 0.21.1 → 0.22.1

### New Features
- **Rate Limiting**: `slowapi` integration
- **Circuit Breakers**: `purgatory-circuitbreaker`
- **Async Caching**: `aiocache` with Redis
- **Advanced Embeddings**: `FlagEmbedding`
- **AI Validation**: `pydantic-ai`

### Key Commands
```bash
# Update dependencies
uv pip sync

# Run quality checks
task quality

# Run full test suite
task test-full

# Check for breaking changes
python scripts/check_breaking_changes.py
```

## Dependency Management Strategy

### 1. Version Pinning Policy
- **Production**: Exact versions for critical dependencies
- **Development**: Minor version flexibility with `~=`
- **Optional**: Broader ranges for flexibility

### 2. Update Schedule
- **Security**: Immediate via Dependabot
- **Minor**: Weekly review and testing
- **Major**: Quarterly planning with migration guides

### 3. Testing Requirements
- All updates must pass full test suite
- Performance benchmarks must not regress >5%
- Security scan must pass
- Coverage must remain ≥80%

## Support

For dependency-related issues:
1. Check the troubleshooting section in guides
2. Run automated migration scripts
3. Open an issue with specific error details

---

[← Back to Documentation](../index.md)