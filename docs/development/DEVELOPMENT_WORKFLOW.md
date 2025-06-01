# Development Workflow Guide

This document outlines the development workflow, coding standards, and best practices for contributing to the vector knowledge base system.

## Table of Contents

- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Git Workflow](#git-workflow)
- [Testing Workflow](#testing-workflow)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)
- [Quality Gates](#quality-gates)

## Development Setup

### Prerequisites

- **Python 3.13+** for latest performance improvements
- **uv package manager** - 10-100x faster than pip
- **Docker Desktop** - for Qdrant and DragonflyDB
- **Git** with proper configuration
- **VS Code** (recommended) with Python extension

### Initial Setup

```bash
# Clone repository
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Install dependencies with uv
uv sync

# Start services
./scripts/start-services.sh

# Verify setup
uv run pytest tests/unit/ --tb=short -q
```

### Development Environment

```bash
# Activate virtual environment
source .venv/bin/activate

# Install pre-commit hooks
pre-commit install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Coding Standards

### Python Code Style

We follow Python community standards with specific configurations:

```bash
# Linting and formatting (required before commit)
ruff check . --fix
ruff format .

# Type checking
mypy src/

# Import sorting
ruff check --select I --fix .
```

### Code Organization

```python
# File structure standards
src/
├── config/          # Configuration models and loading
├── models/          # Pydantic data models
├── services/        # Business logic services
├── mcp/            # MCP protocol implementation
├── core/           # Core utilities and patterns
└── utils/          # Helper utilities

# Import organization
from __future__ import annotations  # For forward references

# Standard library imports
import asyncio
import logging
from pathlib import Path

# Third-party imports
import pytest
from pydantic import BaseModel

# Local imports
from src.config import get_config
from src.models.api_contracts import SearchRequest
```

### Naming Conventions

- **Files/Modules**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`

### Documentation Standards

```python
def search_documents(
    query: str,
    collection_name: str = "documents",
    limit: int = 10
) -> list[SearchResult]:
    """Search documents using vector similarity.
    
    Args:
        query: Search query string
        collection_name: Target collection name
        limit: Maximum number of results
        
    Returns:
        List of search results sorted by relevance
        
    Raises:
        ValidationError: If query is empty or invalid
        ServiceError: If vector database is unavailable
    """
```

## Git Workflow

### Branch Naming

```bash
# Feature branches
git checkout -b feature/add-hybrid-search
git checkout -b feature/improve-chunking-algorithm

# Bug fixes
git checkout -b fix/memory-leak-embedding-cache
git checkout -b fix/validation-error-handling

# Documentation
git checkout -b docs/update-api-reference
git checkout -b docs/add-testing-guide

# Refactoring
git checkout -b refactor/service-layer-architecture
git checkout -b refactor/unify-configuration-system
```

### Commit Messages

Follow conventional commits format:

```bash
# Types: feat, fix, docs, test, refactor, perf, ci, chore

# Features
git commit -m "feat(search): add hybrid vector search with reranking"
git commit -m "feat(models): implement Pydantic v2 validation for API contracts"

# Fixes
git commit -m "fix(embedding): resolve memory leak in batch processing"
git commit -m "fix(config): handle missing environment variables gracefully"

# Tests
git commit -m "test(models): add comprehensive validation tests for SearchRequest"
git commit -m "test(security): add URL validation edge case tests"

# Documentation
git commit -m "docs(api): update API reference with new models"
git commit -m "docs(architecture): document service layer patterns"

# Breaking changes
git commit -m "feat(config)!: migrate to unified configuration system

BREAKING CHANGE: Configuration structure has changed. 
Run migration script to update existing configs."
```

### Pull Request Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes with proper commits
git add .
git commit -m "feat(module): implement new feature"

# 3. Push and create PR
git push origin feature/new-feature
# Create PR via GitHub interface

# 4. Address review feedback
git add .
git commit -m "fix(review): address code review feedback"
git push origin feature/new-feature

# 5. Squash merge into main
# PR gets squashed and merged via GitHub
```

## Testing Workflow

### Test-Driven Development (TDD)

```bash
# 1. Write failing test
uv run pytest tests/unit/test_new_feature.py::test_feature_success -v
# Test fails (expected)

# 2. Write minimal code to pass test
# Implement just enough to make test pass

# 3. Run test again
uv run pytest tests/unit/test_new_feature.py::test_feature_success -v
# Test passes

# 4. Refactor and add more tests
# Add edge cases, error scenarios, etc.

# 5. Run full test suite
uv run pytest tests/unit/ --tb=short
```

### Test Categories

```bash
# Unit tests (primary focus)
uv run pytest tests/unit/ -v

# Model validation tests
uv run pytest tests/unit/models/ -v

# Configuration tests
uv run pytest tests/unit/config/ -v

# Service integration tests
uv run pytest tests/unit/services/ -v

# Security tests
uv run pytest tests/unit/test_security.py -v
```

### Coverage Requirements

```bash
# Generate coverage report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Coverage thresholds
# - Critical modules: 95%+ required
# - Service layer: 90%+ required  
# - Utilities: 85%+ acceptable

# Check specific module coverage
uv run pytest --cov=src/models --cov-report=term-missing
```

### Testing Best Practices

```python
# Good test structure
def test_search_request_validation():
    """Test SearchRequest field validation."""
    # Arrange
    valid_data = {"query": "test", "limit": 10}
    
    # Act
    request = SearchRequest(**valid_data)
    
    # Assert
    assert request.query == "test"
    assert request.limit == 10

def test_search_request_invalid_limit():
    """Test SearchRequest with invalid limit."""
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="test", limit=0)
    assert "greater than 0" in str(exc_info.value)

# Use fixtures for common setup
@pytest.fixture
def mock_config():
    """Create mock configuration for testing."""
    return UnifiedConfig(
        environment="testing",
        embedding_provider="openai",
        openai={"api_key": "test-key"}
    )

# Test async code properly
@pytest.mark.asyncio
async def test_async_service_method():
    """Test async service method."""
    async with ServiceClass(config) as service:
        result = await service.async_method()
        assert result.success is True
```

## Code Review Process

### Before Creating PR

```bash
# Pre-submission checklist
□ All tests pass locally
□ Code follows linting standards
□ New tests added for new functionality
□ Documentation updated if needed
□ No secrets or credentials committed
□ Commit messages follow conventional format

# Run complete validation
ruff check . --fix && ruff format .
uv run pytest tests/unit/ -v
uv run pytest --cov=src
```

### PR Requirements

1. **Clear Description**: Explain what changes and why
2. **Test Coverage**: Include tests for new functionality
3. **Documentation**: Update relevant docs
4. **Breaking Changes**: Clearly marked and explained
5. **Small Scope**: Focus on single feature/fix

### Review Checklist

**Code Quality:**
- [ ] Follows coding standards
- [ ] Proper error handling
- [ ] Appropriate logging
- [ ] Performance considerations

**Testing:**
- [ ] Adequate test coverage
- [ ] Tests pass reliably
- [ ] Edge cases covered
- [ ] Mock usage appropriate

**Security:**
- [ ] No hardcoded secrets
- [ ] Input validation present
- [ ] SQL injection prevention
- [ ] XSS prevention

**Documentation:**
- [ ] Code comments where needed
- [ ] API docs updated
- [ ] README updated if needed
- [ ] Migration notes if needed

## Release Process

### Version Management

```bash
# Semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes
# MINOR: New features (backward compatible)
# PATCH: Bug fixes

# Examples:
v1.0.0 -> v1.0.1  # Bug fix
v1.0.1 -> v1.1.0  # New feature
v1.1.0 -> v2.0.0  # Breaking change
```

### Release Steps

```bash
# 1. Update version
# Edit pyproject.toml
version = "1.1.0"

# 2. Update CHANGELOG.md
# Add new version section with changes

# 3. Run full test suite
uv run pytest tests/ -v
uv run pytest --cov=src

# 4. Tag release
git tag -a v1.1.0 -m "Release v1.1.0: Add hybrid search"
git push origin v1.1.0

# 5. Create GitHub release
# Use GitHub interface to create release from tag
```

### Release Notes Template

```markdown
## v1.1.0 - 2024-01-15

### 🎉 New Features
- Add hybrid vector search with reranking
- Implement comprehensive security validation
- Add 500+ unit tests with 90%+ coverage

### 🐛 Bug Fixes
- Fix memory leak in embedding cache
- Resolve configuration validation edge cases

### 📚 Documentation
- Update API reference with new models
- Add comprehensive testing documentation

### ⚠️ Breaking Changes
- Configuration structure changed (migration guide available)

### 🏗️ Internal
- Refactor service layer architecture
- Improve error handling patterns
```

## Quality Gates

### Continuous Integration

```yaml
# GitHub Actions workflow
name: Quality Gates

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
      
      - name: Install dependencies
        run: uv sync
        
      - name: Lint code
        run: |
          uv run ruff check .
          uv run ruff format --check .
          
      - name: Type check
        run: uv run mypy src/
        
      - name: Run tests
        run: uv run pytest --cov=src --cov-report=xml
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff-check
        name: ruff-check
        entry: uv run ruff check --fix
        language: system
        
      - id: ruff-format
        name: ruff-format
        entry: uv run ruff format
        language: system
        
      - id: pytest-fast
        name: pytest-fast
        entry: uv run pytest tests/unit/ -x --tb=short -q
        language: system
        pass_filenames: false
```

### Local Quality Checks

```bash
# Quick pre-commit check
make check

# Full quality validation
make validate

# Performance benchmarking
make benchmark

# Security scanning
make security-scan
```

### Makefile Targets

```makefile
# Common development commands
.PHONY: setup test lint format type-check coverage clean

setup:
	uv sync
	pre-commit install

test:
	uv run pytest tests/unit/ -v

test-fast:
	uv run pytest tests/unit/ -x --tb=short -q

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy src/

coverage:
	uv run pytest --cov=src --cov-report=html
	open htmlcov/index.html

check: lint format type-check test-fast

validate: lint format type-check test coverage

clean:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	find . -type d -name __pycache__ -delete
```

## Best Practices Summary

### Development
1. **Start with tests** - Write tests before implementation
2. **Small commits** - Make atomic, focused commits
3. **Clear documentation** - Document complex logic and APIs
4. **Error handling** - Always handle errors gracefully
5. **Type safety** - Use type hints throughout

### Testing
1. **High coverage** - Aim for 90%+ on critical modules
2. **Edge cases** - Test boundary conditions
3. **Mock appropriately** - Mock external dependencies
4. **Fast execution** - Keep unit tests under 100ms each
5. **Clear assertions** - Make test failures informative

### Code Review
1. **Be constructive** - Focus on improving code quality
2. **Check tests** - Ensure adequate test coverage
3. **Consider maintainability** - Will this be easy to modify?
4. **Security mindset** - Look for potential vulnerabilities
5. **Performance impact** - Consider performance implications

This development workflow ensures high code quality, maintainability, and reliability across the entire system.