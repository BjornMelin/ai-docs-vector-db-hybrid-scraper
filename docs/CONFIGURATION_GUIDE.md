# Configuration Best Practices Guide

This guide documents the modern Python configuration standards implemented in the AI Docs Vector DB Hybrid Scraper project.

## Overview

All project configurations follow 2025 Python best practices, centralizing settings in `pyproject.toml` and using modern tools like `uv`, `ruff`, and optimized testing configurations.

## Key Configuration Files

### 1. `pyproject.toml` - Central Configuration Hub

Our `pyproject.toml` serves as the single source of truth for:

- **Project Metadata**: Name, version, description, authors
- **Dependencies**: Both runtime and development dependencies
- **Tool Configurations**: Ruff, Black, Pytest, Coverage, Mypy
- **Build System**: Using Hatchling for modern Python packaging
- **Performance Settings**: UV optimizations, bytecode compilation

#### Key Optimizations Applied:

```toml
[tool.uv]
compile-bytecode = true      # Faster imports
link-mode = "copy"          # Better for Docker
resolution = "highest"      # Prefer latest versions

[tool.ruff]
fix = true                  # Auto-fix issues
cache-dir = ".ruff_cache"   # Enable caching
respect-gitignore = true    # Skip ignored files

[tool.pytest.ini_options]
import-mode = "importlib"   # Modern import handling
considerextrafiles = true   # Re-run failed tests first
```

### 2. Test Configuration

#### `pytest.ini` Features:
- Platform-specific test markers
- Performance optimizations (parallel execution, caching)
- Comprehensive marker system for test categorization
- Modern import modes and failure handling

#### `pytest-platform.ini` Features:
- OS-specific timeout settings
- Platform-aware test selection
- Performance threshold adjustments per platform

### 3. Development Environment

#### `.editorconfig` - Universal Editor Settings
Ensures consistent coding standards across all editors:
- Python: 4 spaces, max line 88
- YAML/JSON: 2 spaces
- Markdown: Preserve trailing spaces

#### `.vscode/settings.json` - VS Code Optimizations
- Ruff integration for linting/formatting
- Python path configuration
- Performance exclusions for large directories
- Task automation shortcuts

### 4. CI/CD Configuration

#### GitHub Actions Optimizations:
- Matrix testing across Python 3.11-3.12
- Platform-specific test configurations
- Parallel job execution
- Smart caching strategies
- Change detection for targeted testing

Environment variables for performance:
```yaml
UV_CACHE_COMPRESSION: 1
UV_COMPILE_BYTECODE: 1
PYTEST_XDIST_AUTO_NUM_WORKERS: 4
```

### 5. Code Quality Tools

#### Pre-commit Hooks:
- Ruff for fast linting and formatting
- Security checks with Bandit and Gitleaks
- Type checking with Mypy
- Dependency vulnerability scanning
- Markdown and YAML linting

#### `.yamllint.yml` Configuration:
- Line length limits
- Indentation rules
- Quote consistency
- Comment formatting

### 6. Environment Variables

#### `.env.example` Structure:
- Grouped by functionality
- Extensive inline documentation
- Security-first defaults
- Performance tuning options

Key sections:
- Core API Keys
- Service URLs
- Performance Settings
- Security Configuration
- Feature Flags

## Modern Python Standards Implemented

### 1. Dependency Management
- All dependencies in `pyproject.toml`
- Optional dependency groups for modular installation
- Version constraints following SemVer
- UV for fast, deterministic installs

### 2. Testing Best Practices
- Pytest with comprehensive plugins
- Coverage tracking with branch coverage
- Performance benchmarking
- Property-based testing with Hypothesis
- Mutation testing support

### 3. Security Configuration
- Bandit for security scanning
- Secret detection in pre-commit
- Dependency vulnerability checks
- Rate limiting and input validation

### 4. Performance Optimizations
- Bytecode compilation
- Parallel test execution
- Smart caching strategies
- Optimized import modes
- Resource limits

## Configuration Validation

Run the validation script to ensure configuration consistency:

```bash
uv run python scripts/validate_config.py
```

This checks:
- Dependency consistency
- Tool configuration completeness
- CI/CD alignment
- Environment template completeness

## Migration from Legacy Configurations

### From `setup.py` to `pyproject.toml`:
- All metadata moved to `[project]` table
- Build configuration in `[build-system]`
- Tool configs in `[tool.*]` sections

### From `requirements.txt` to `pyproject.toml`:
- Runtime deps in `dependencies`
- Dev deps in `optional-dependencies`
- Use `uv sync` instead of `pip install`

### From Multiple Config Files:
- `.flake8` → `[tool.ruff]`
- `setup.cfg` → `pyproject.toml`
- `tox.ini` → GitHub Actions + `pyproject.toml`
- `.coveragerc` → `[tool.coverage]`

## Best Practices Summary

1. **Single Source of Truth**: Use `pyproject.toml` for all Python configurations
2. **Modern Tools**: Prefer `uv` over `pip`, `ruff` over `flake8`/`black`
3. **Performance First**: Enable caching, parallel execution, bytecode compilation
4. **Security by Default**: Pre-commit hooks, dependency scanning, secret detection
5. **Platform Awareness**: Test and configure for multiple platforms
6. **Continuous Validation**: Regular config checks in CI/CD

## Maintenance Guidelines

### Regular Updates:
1. Run `uv sync` to update lock file
2. Check for tool updates: `uv tool list --outdated`
3. Review and update pre-commit hooks: `pre-commit autoupdate`
4. Validate configurations: `uv run python scripts/validate_config.py`

### Adding New Tools:
1. Add tool configuration to `pyproject.toml`
2. Update relevant CI/CD workflows
3. Add to pre-commit hooks if applicable
4. Document in this guide

### Performance Monitoring:
1. Track test execution times
2. Monitor CI/CD pipeline duration
3. Profile import times with bytecode compilation
4. Benchmark with different configuration options

## Troubleshooting

### Common Issues:

1. **Import Errors**: Ensure `PYTHONPATH` includes `src/`
2. **Test Discovery**: Check `testpaths` and marker configuration
3. **Linting Conflicts**: Run `ruff check --diff` to preview changes
4. **CI Failures**: Verify Python versions match local environment

### Debug Commands:

```bash
# Verify configuration
uv run python -m pytest --collect-only

# Check ruff configuration
uv run ruff check --show-settings

# Validate pyproject.toml
uv run python -m pyproject_validator

# Test environment setup
uv run python -c "import sys; print(sys.path)"
```

## Future Enhancements

Planned configuration improvements:
1. Add `pdm` backend support option
2. Implement `hatch` environment management
3. Add `nox` for complex test scenarios
4. Integrate `pants` for monorepo support
5. Add `pixi` for conda compatibility

---

Last Updated: 2025-01-23
Version: 1.0.0