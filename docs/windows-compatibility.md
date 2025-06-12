# Windows Compatibility Guide

This document outlines known Windows compatibility issues and their solutions for the AI Docs Vector DB Hybrid Scraper project.

## Current Status

✅ **Working**: Core Python functionality, FastAPI services, basic CLI tools  
⚠️ **Partial**: Native dependencies (tree-sitter, hiredis), some build tools  
❌ **Broken**: None currently identified  

## Known Issues & Solutions

### 1. Tree-sitter Native Dependencies

**Issue**: `tree-sitter` packages may fail to build on Windows due to missing C compiler or build tools.

**Solution**:
- Ensure Microsoft C++ Build Tools are installed
- Use pre-built wheels when available
- Fall back to pure Python alternatives if needed

**Workaround in CI**: The CI pipeline includes fallback installation methods for Windows.

### 2. Redis Hiredis Extension

**Issue**: `redis[hiredis]` may fail to compile on Windows.

**Solution**:
- Use `redis` without hiredis extension as fallback
- Performance impact is minimal for most use cases

**Configuration**: 
```python
# In src/config/redis_config.py
REDIS_CONNECTION_CLASS = "redis.Connection"  # instead of hiredis.Connection
```

### 3. Directory Creation Commands

**Issue**: Unix-style directory creation commands may fail on Windows.

**Solution**: Use cross-platform commands in CI workflows:
```bash
# Instead of: mkdir -p tests/fixtures/{cache,data,logs}
mkdir -p tests/fixtures/cache tests/fixtures/data tests/fixtures/logs
```

### 4. Path Separators

**Issue**: Hard-coded forward slashes in file paths.

**Solution**: Use `pathlib.Path` or `os.path.join()` for cross-platform compatibility:
```python
from pathlib import Path

# Good
config_path = Path("config") / "settings.yml"

# Bad  
config_path = "config/settings.yml"
```

## Testing on Windows

### Local Development

1. **Install Prerequisites**:
   ```cmd
   # Install uv
   pip install uv
   
   # Install Microsoft C++ Build Tools (for native deps)
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

2. **Install Dependencies**:
   ```cmd
   uv sync --dev
   ```

3. **Run Tests**:
   ```cmd
   uv run pytest tests/unit
   ```

### CI Pipeline

The GitHub Actions CI pipeline includes Windows testing with:
- Multiple Python versions (3.12, 3.13)
- Fallback installation methods for problematic dependencies
- Cross-platform shell commands

## Future Improvements

1. **Package Pre-built Wheels**: Consider creating pre-built wheels for problematic native dependencies
2. **Docker Support**: Provide Windows container support for consistent environments
3. **Documentation**: Expand Windows-specific setup instructions

## Reporting Issues

If you encounter Windows-specific issues:

1. Check this document for known issues
2. Test with the latest dependencies: `uv sync --upgrade`
3. Report new issues with:
   - Windows version
   - Python version
   - Full error message
   - Steps to reproduce

## Dependencies Status

| Package | Windows Support | Notes |
|---------|----------------|-------|
| `crawl4ai` | ✅ Good | Works with playwright |
| `tree-sitter` | ⚠️ Partial | May need build tools |
| `redis[hiredis]` | ⚠️ Partial | Fallback available |
| `qdrant-client` | ✅ Good | Pure Python |
| `fastapi` | ✅ Good | Full support |
| `uvicorn` | ✅ Good | Full support |
| `playwright` | ✅ Good | Auto-installs browsers |

Last Updated: 2025-06-12