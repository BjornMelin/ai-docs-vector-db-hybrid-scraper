# Configuration Best Practices Guide

> **Status**: Active  
> **Last Updated**: 2025-06-23  
> **Purpose**: Modern Python configuration standards and tooling best practices  
> **Audience**: Developers implementing configuration standards  
> **Prerequisites**: Python 3.11+, modern development tools (3.13+ recommended)

This guide documents the 2025 Python configuration standards implemented in the AI Docs Vector DB Hybrid Scraper project, focusing on modern tooling, performance optimizations, and development best practices.

## 🎯 Quick Start

### Essential Configuration Files

```bash
# Core configuration files in project root
.editorconfig          # Universal editor settings
.yamllint.yml         # YAML linting rules
pyproject.toml        # Central Python configuration hub
pytest.ini            # Test configuration
```

### Key Tools and Standards

- **Package Management**: `uv` for fast, deterministic installs
- **Linting & Formatting**: `ruff` for comprehensive code quality
- **Testing**: `pytest` with modern optimizations
- **Build System**: `hatchling` for packaging
- **Configuration**: Centralized in `pyproject.toml`

## 🏗️ Configuration Architecture

### Central Configuration Hub: `pyproject.toml`

Our `pyproject.toml` serves as the single source of truth for:

- **Project Metadata**: Name, version, description, authors
- **Dependencies**: Both runtime and development dependencies  
- **Tool Configurations**: Ruff, Pytest, Coverage, Mypy settings
- **Build System**: Modern Python packaging with Hatchling
- **Performance Settings**: UV optimizations and bytecode compilation

#### Core Optimizations

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

### Development Environment Standards

#### `.editorconfig` - Universal Editor Settings
Ensures consistent coding standards across all editors:
- **Python**: 4 spaces, max line 88, double quotes
- **YAML/JSON**: 2 spaces indentation
- **Markdown**: Preserve trailing spaces, 80 char limit
- **TypeScript/JavaScript**: 2 spaces, single quotes

#### `.yamllint.yml` - YAML Quality Standards
- Line length limits (120 chars)
- Consistent indentation (2 spaces)
- Quote consistency and comment formatting
- Comprehensive ignore patterns for build artifacts

## 🧪 Testing Configuration

### Test Organization and Markers

```ini
# Platform-specific markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    benchmark: marks tests as benchmarks
    windows: marks tests as Windows-specific
    linux: marks tests as Linux-specific
    macos: marks tests as macOS-specific
```

### Performance Optimizations

- **Parallel Execution**: `pytest-xdist` for multi-core testing
- **Smart Caching**: Test result caching with `--lf` (last failed)
- **Import Optimization**: Modern `importlib` mode
- **Platform-Aware**: OS-specific timeout and performance settings

### Coverage Configuration

```toml
[tool.coverage.run]
branch = true
source = ["src"]
omit = [
    "tests/*",
    "scripts/*",
    "docs/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError"
]
```

## 🔧 Development Tools Integration

### VS Code Configuration

Located in `.vscode/settings.json`:
- **Ruff Integration**: Primary linter and formatter
- **Python Path**: Automatic `src/` path configuration  
- **Performance**: Exclusions for large directories
- **Task Automation**: Quick access to common commands

### Pre-commit Hooks

Comprehensive quality gates:
- **Ruff**: Fast linting and formatting
- **Security**: Bandit security scanning, secret detection
- **Type Checking**: Mypy static analysis
- **Dependencies**: Vulnerability scanning
- **Documentation**: Markdown and YAML linting

## 🚀 CI/CD Configuration

### GitHub Actions Optimizations

```yaml
strategy:
  matrix:
    python-version: ["3.11", "3.12", "3.13"]
    os: [ubuntu-latest, windows-latest, macos-latest]

env:
  UV_CACHE_COMPRESSION: 1
  UV_COMPILE_BYTECODE: 1
  PYTEST_XDIST_AUTO_NUM_WORKERS: 4
```

### Performance Features

- **Matrix Testing**: Multiple Python versions and platforms
- **Smart Caching**: UV dependency caching
- **Parallel Execution**: Optimized worker allocation
- **Change Detection**: Targeted testing based on file changes

## 📋 Modern Python Standards

### 1. Dependency Management

```toml
[project]
dependencies = [
    "fastapi>=0.104.0",
    "pydantic>=2.5.0",
    "uvicorn[standard]>=0.24.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.1.0",
    "mypy>=1.7.0"
]
```

**Key Principles**:
- All dependencies in `pyproject.toml`
- Optional dependency groups for modular installation
- SemVer constraints with minimum versions
- Use `uv sync` for deterministic installs

### 2. Code Quality Standards

```toml
[tool.ruff]
line-length = 88
target-version = "py311"
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings  
    "F",    # pyflakes
    "I",    # isort
    "B",    # flake8-bugbear
    "C4",   # flake8-comprehensions
    "UP",   # pyupgrade
]
```

### 3. Security Configuration

- **Bandit**: Security vulnerability scanning
- **Secret Detection**: Pre-commit hooks for credential detection
- **Dependency Scanning**: Regular vulnerability checks
- **Input Validation**: Pydantic models with strict validation

## 🔄 Migration Guidelines

### From Legacy Configurations

| Legacy File | Modern Equivalent | Migration Action |
|-------------|------------------|------------------|
| `setup.py` | `pyproject.toml` | Move metadata to `[project]` table |
| `requirements.txt` | `pyproject.toml` | Convert to `dependencies` array |
| `.flake8` | `[tool.ruff]` | Migrate rules to Ruff config |
| `setup.cfg` | `pyproject.toml` | Consolidate tool configurations |
| `tox.ini` | GitHub Actions | Replace with modern CI/CD |

### Step-by-Step Migration

1. **Create `pyproject.toml`** with project metadata
2. **Migrate dependencies** from requirements files
3. **Configure tools** (Ruff, Pytest, Coverage)
4. **Update CI/CD** to use `uv` and modern practices
5. **Remove legacy files** after validation

## 🛠️ Maintenance and Validation

### Regular Maintenance Tasks

```bash
# Update dependencies
uv sync --upgrade

# Check for tool updates
uv tool list --outdated

# Update pre-commit hooks
pre-commit autoupdate

# Validate configuration
uv run python scripts/validate_config.py
```

### Configuration Validation Script

Creates automated checks for:
- Dependency consistency across environments
- Tool configuration completeness
- CI/CD workflow alignment
- Environment template accuracy

### Performance Monitoring

Track key metrics:
- Test execution times across platforms
- CI/CD pipeline duration optimization
- Import performance with bytecode compilation
- Build time improvements with caching

## 🐛 Troubleshooting

### Common Configuration Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Import Errors | Module not found | Ensure `PYTHONPATH` includes `src/` |
| Test Discovery | No tests collected | Check `testpaths` in `pyproject.toml` |
| Linting Conflicts | Contradictory rules | Run `ruff check --diff` to preview |
| CI Failures | Version mismatches | Verify Python versions match locally |

### Debug Commands

```bash
# Verify configuration parsing
uv run python -m pytest --collect-only

# Check tool settings
uv run ruff check --show-settings

# Validate project metadata
uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb')))"

# Test environment setup
uv run python -c "import sys; print('\\n'.join(sys.path))"
```

## 🔮 Future Enhancements

### Planned Improvements

1. **Alternative Backends**: Add `pdm` and `poetry` compatibility
2. **Environment Management**: Integrate `hatch` environments
3. **Complex Testing**: Add `nox` for matrix testing scenarios
4. **Monorepo Support**: Evaluate `pants` build system
5. **Conda Compatibility**: Add `pixi` for scientific computing

### Emerging Standards

- **Strict Type Checking**: Gradual adoption of `strict = true`
- **Performance Profiling**: Automated performance regression detection
- **Security Automation**: Enhanced dependency vulnerability scanning
- **Documentation Generation**: Automated API docs from docstrings

## 📚 Additional Resources

### Documentation Links

- [UV Documentation](https://docs.astral.sh/uv/) - Modern Python package management
- [Ruff Documentation](https://docs.astral.sh/ruff/) - Fast Python linter and formatter
- [Pytest Documentation](https://docs.pytest.org/) - Testing framework best practices
- [Pydantic V2 Guide](https://docs.pydantic.dev/latest/) - Data validation and settings

### Internal References

- [Configuration Guide](./configuration.md) - Runtime system configuration
- [Architecture Documentation](./architecture.md) - System design overview
- [Contributing Guidelines](./contributing.md) - Development workflow standards

---

**Last Updated**: 2025-06-23  
**Version**: 2.0.0  
**Next Review**: 2025-09-23