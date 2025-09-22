# Developer Experience Guide

## Quick Start

### 1. Initial Setup
```bash
# Install dependencies
uv sync

# Setup development environment
task setup-complete

# Start services
task services-start

# Start development server
task dev-simple
```

### 2. Daily Development Workflow
```bash
# Fast unit tests (< 60s)
task test

# Code quality checks
task quality

# Documentation preview
task docs-serve
```

## Unified Task Interface

### Core Development
- `task dev` - Start development server (all interfaces)
- `task dev-simple` - Simple mode development server
- `task dev-enterprise` - Enterprise mode development server

### Testing & Quality (Optimized Feedback Loops)
- `task test` - Fast test suite (unit + fast integration)
- `task test-unit` - Unit tests only (< 30s)
- `task test-integration` - Integration tests only
- `task test-full` - Comprehensive test suite
- `task coverage` - Test coverage report

### Code Quality
- `task lint` - Lint code with ruff
- `task format` - Format code with ruff
- `task typecheck` - Type check with mypy
- `task quality` - All quality checks (format + lint + typecheck)

### Documentation
- `task docs-serve` - Serve documentation locally
- `task docs-build` - Build documentation
- `task docs-add-status` - Update status indicators
- `task docs-update-links` - Update documentation links

### Services & Infrastructure
- `task services-start` - Start Qdrant + Redis
- `task services-stop` - Stop all services
- `task validate-config` - Validate configuration

### Performance & Monitoring
- `task benchmark` - Run performance benchmarks
- `task profile` - Performance profiling

## Fast Test Feedback Loops

### Test Profiles
- **Unit** (`task test-unit`): < 30s, unit tests only
- **Fast** (`task test`): < 60s, unit + fast integration tests  
- **Integration** (`task test-integration`): < 300s, integration tests only
- **Full** (`task test-full`): Complete test suite

### Test Optimization Features
- Parallel execution (auto-detected CPU cores)
- Failed-first execution for faster debugging
- Smart test selection based on markers
- Coverage integration without performance impact

## Configuration Management

### Environment Variables (AI_DOCS__ prefix)
```bash
# Core configuration
AI_DOCS__MODE=simple                    # simple, enterprise
AI_DOCS__DEBUG=false                   # Debug mode

# Services
AI_DOCS__QDRANT_URL=http://localhost:6333
AI_DOCS__REDIS_URL=redis://localhost:6379

# Performance
AI_DOCS__MAX_CONCURRENT_CRAWLS=10
AI_DOCS__CACHE_TTL=3600
```

### Configuration Files
- `.env.local` - Local development overrides
- `.env.simplified` - Minimal configuration template
- `pyproject.toml` - All tool configuration

## IDE Integration

### VS Code Setup
Configuration automatically provided in `.vscode/`:
- Python interpreter path
- Test runner configuration
- Formatting and linting settings
- Task runner integration
- Environment variables for terminals

### Available VS Code Tasks
- **Dev Simple**: Start development server (simple mode)
- **Dev Enterprise**: Start development server (enterprise mode)
- **Test Fast**: Run fast test suite
- **Test Unit**: Run unit tests only
- **Quality**: Run code quality checks
- **Services Start**: Start local services
- **Docs Serve**: Serve documentation
- **Validate Config**: Validate configuration

## Pre-commit Hooks

Automated quality checks on every commit:
- **Code Formatting**: Ruff format
- **Code Linting**: Ruff check with auto-fix
- **Type Checking**: MyPy validation
- **Security Scanning**: Bandit + Gitleaks
- **Documentation Updates**: Auto-update status indicators
- **Configuration Validation**: Environment and setup validation

## Performance Targets

### Test Execution Times
- Unit tests: < 30 seconds
- Fast profile: < 60 seconds
- Integration tests: < 300 seconds
- Full test suite: < 600 seconds

### Development Server
- Cold start: < 10 seconds
- Hot reload: < 2 seconds

### Code Quality Checks
- Format + lint: < 15 seconds
- Type checking: < 30 seconds

## Troubleshooting

### Common Issues

#### Slow Tests
```bash
# Run only fastest tests
task test-unit

# Check test durations
task test --verbose
```

#### Configuration Issues
```bash
# Validate current configuration
task validate-config

# Reset to defaults
cp .env.simplified .env.local
```

#### Service Connection Issues
```bash
# Start services
task services-start

# Check service health
curl localhost:6333/health  # Qdrant
redis-cli ping              # Redis
```

#### Pre-commit Hook Failures
```bash
# Skip hooks for emergency commits
git commit --no-verify

# Fix formatting issues
task quality

# Update pre-commit hooks
pre-commit autoupdate
```

## Advanced Usage

### Custom Test Execution
```bash
# Run specific test with custom parallel workers
python scripts/run_fast_tests.py --profile=fast --parallel=8 --verbose

# Run tests with coverage
python scripts/run_fast_tests.py --profile=unit --coverage
```

### Documentation Automation
```bash
# Update documentation manually
python scripts/docs_automation.py

# Build documentation only
python scripts/docs_automation.py --build-only

# Skip link validation
python scripts/docs_automation.py --skip-validation
```

### Configuration Validation
```bash
# Standard validation
python scripts/validate_config.py

# Strict mode (warnings as errors)
python scripts/validate_config.py --strict
```

## Best Practices

### Development Workflow
1. **Start with tests**: `task test-unit` for rapid feedback
2. **Code quality first**: `task quality` before commits
3. **Service validation**: `task validate-config` for environment issues
4. **Documentation updates**: Automated via pre-commit hooks

### Performance Optimization
1. **Use appropriate test profiles**: Don't run full test suite for unit changes
2. **Leverage parallel execution**: Test runner auto-detects optimal workers
3. **Monitor performance targets**: Built-in timing validation
4. **Cache optimization**: Services start once, persist across development

### Configuration Management
1. **Use AI_DOCS__ prefix**: Consistent environment variable naming
2. **Mode-specific defaults**: Simple vs enterprise configurations
3. **Local overrides**: `.env.local` for personal customization
4. **Validation automation**: Pre-commit hooks catch issues early

## Migration from Scripts

### Old Script → New Task Mapping
- `./scripts/run_fast_tests.py` → `task test`
- `./scripts/start-services.sh` → `python scripts/dev.py services start`
- `./scripts/benchmark_query_api.py` → `task benchmark`
- `ruff check . --fix && ruff format .` → `task quality`
- `mkdocs serve` → `task docs-serve`
- Custom combinations → Unified `task` interface

### Benefits of Unified Interface
- **Discoverability**: `task --list` shows all available commands
- **Consistency**: Same interface across all operations
- **Performance**: Optimized execution paths
- **Documentation**: Self-documenting task names
- **IDE Integration**: Built-in VS Code task support