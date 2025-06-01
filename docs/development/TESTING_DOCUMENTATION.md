# Testing Documentation

This document provides comprehensive information about the testing infrastructure, test coverage, and quality assurance practices implemented in the vector knowledge base system.

## Table of Contents

- [Test Overview](#test-overview)
- [Test Structure](#test-structure)
- [Test Categories](#test-categories)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Quality Standards](#quality-standards)
- [Writing Tests](#writing-tests)
- [Continuous Integration](#continuous-integration)

## Test Overview

The project maintains high code quality with comprehensive testing:

- **Total Tests**: 500+ unit tests across 25 test files
- **Coverage**: 90%+ across all critical modules
- **Framework**: pytest with Pydantic v2 validation testing
- **Standards**: Following modern Python testing best practices

### Key Testing Principles

1. **Comprehensive Coverage**: Every model, service, and utility has dedicated tests
2. **Pydantic v2 Validation**: Extensive ValidationError testing for all models
3. **Edge Case Testing**: Boundary conditions, error scenarios, and edge cases
4. **Isolated Testing**: Proper mocking and dependency injection
5. **Fast Execution**: Unit tests optimized for quick feedback cycles

## Test Structure

```
tests/
├── unit/                               # Unit tests (500+ tests)
│   ├── config/                         # Configuration tests
│   │   ├── test_enums.py              # Enum validation (45 tests)
│   │   ├── test_models.py             # Config model tests
│   │   ├── test_security_config.py    # Security config tests
│   │   └── test_unified_config.py     # Unified config tests
│   ├── models/                         # Pydantic model tests
│   │   ├── test_api_contracts.py      # API models (67 tests)
│   │   ├── test_document_processing.py # Document models (33 tests)
│   │   ├── test_validators.py         # Validation functions (57 tests)
│   │   └── test_vector_search.py      # Vector search models (51 tests)
│   ├── mcp/                           # MCP protocol tests
│   │   ├── test_requests.py           # MCP request models
│   │   └── test_responses.py          # MCP response models
│   ├── test_security.py               # Security validation (33 tests)
│   ├── test_chunking.py               # Content chunking (45 tests)
│   ├── test_manage_vector_db.py       # Database operations (76 tests)
│   └── test_crawl4ai_bulk_embedder.py # Scraping pipeline (92 tests)
├── integration/                        # Integration tests (planned)
├── conftest.py                        # pytest configuration
└── __init__.py
```

## Test Categories

### 1. Configuration Tests (45+ tests)

Tests for configuration management and validation:

```python
# Example: Enum validation tests
def test_enum_values():
    """Test enum values match expected strings."""
    assert Environment.DEVELOPMENT.value == "development"
    assert Environment.TESTING.value == "testing"
    assert Environment.PRODUCTION.value == "production"

def test_string_enum_inheritance():
    """Test that enums inherit from str."""
    assert isinstance(Environment.DEVELOPMENT, str)
    assert Environment.DEVELOPMENT == "development"
```

**Coverage includes:**
- All enum types with string inheritance validation
- Unified configuration model validation
- Environment variable loading and validation
- Security configuration parameters

### 2. API Contract Tests (67 tests)

Comprehensive Pydantic v2 model validation for all API contracts:

```python
# Example: Request validation tests
def test_search_request_limit_constraints():
    """Test limit field constraints."""
    # Valid limits
    SearchRequest(query="test", limit=1)
    SearchRequest(query="test", limit=100)
    
    # Invalid limits
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=0)
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=101)
```

**Coverage includes:**
- Request/response model validation
- Field constraints and validation rules
- Default value assignment
- Error response formatting
- Nested model validation

### 3. Document Processing Tests (33 tests)

Tests for document processing and chunking:

```python
# Example: Document metadata tests
def test_document_metadata_defaults():
    """Test default field values."""
    before = datetime.now()
    metadata = DocumentMetadata(url="https://example.com")
    after = datetime.now()
    assert before <= metadata.crawled_at <= after
```

**Coverage includes:**
- Document metadata generation
- Chunk creation and validation
- Content type detection
- Processing statistics

### 4. Vector Search Tests (51 tests)

Tests for vector search functionality:

```python
# Example: Search parameters validation
def test_fusion_config_weight_constraints():
    """Test fusion weight constraints."""
    # Valid weights (sum to 1.0)
    FusionConfig(dense_weight=0.7, sparse_weight=0.3)
    
    # Invalid weights
    with pytest.raises(ValidationError):
        FusionConfig(dense_weight=1.5, sparse_weight=0.3)
```

**Coverage includes:**
- Search parameter validation
- Fusion algorithm configuration
- Prefetch optimization settings
- Search result formatting

### 5. Security Tests (33 tests)

Comprehensive security validation testing:

```python
# Example: URL validation tests
def test_validate_url_dangerous_patterns():
    """Test validation of URLs with dangerous patterns."""
    dangerous_urls = [
        "http://localhost",
        "http://127.0.0.1",
        "http://192.168.1.1"
    ]
    
    for url in dangerous_urls:
        with pytest.raises(SecurityError):
            validator.validate_url(url)
```

**Coverage includes:**
- URL validation and sanitization
- Collection name validation
- Query string sanitization
- API key validation
- Domain filtering

### 6. Service Integration Tests (200+ tests)

Tests for service layer components:

```python
# Example: Service lifecycle tests
@pytest.mark.asyncio
async def test_embedding_manager_lifecycle():
    """Test embedding manager initialization and cleanup."""
    manager = EmbeddingManager(config)
    await manager.initialize()
    
    # Test functionality
    embeddings = await manager.generate_embeddings(["test"])
    assert len(embeddings) == 1
    
    await manager.cleanup()
```

**Coverage includes:**
- Service initialization and cleanup
- Database operations (CRUD)
- Embedding generation
- Caching functionality
- Error handling and recovery

### 7. MCP Protocol Tests (30+ tests)

Tests for Model Context Protocol communication:

```python
# Example: MCP request validation
def test_mcp_request_forbids_extra_fields():
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError):
        MCPRequest(extra_field="not allowed")
```

**Coverage includes:**
- Request/response protocol validation
- Tool communication models
- Error response formatting
- Protocol compliance

## Running Tests

### Basic Test Execution

```bash
# Run all tests with coverage
uv run pytest --cov=src

# Run specific test categories
uv run pytest tests/unit/models/          # Model tests
uv run pytest tests/unit/config/          # Config tests
uv run pytest tests/unit/test_security.py # Security tests

# Run with verbose output
uv run pytest tests/unit/ -v

# Run with short traceback for quick feedback
uv run pytest tests/unit/ --tb=short
```

### Coverage Reports

```bash
# Generate HTML coverage report
uv run pytest --cov=src --cov-report=html
open htmlcov/index.html

# Generate terminal coverage report
uv run pytest --cov=src --cov-report=term-missing

# Coverage with specific modules
uv run pytest --cov=src/models --cov=src/config
```

### Test Selection

```bash
# Run tests by pattern
uv run pytest -k "test_validation"
uv run pytest -k "test_api_contracts"

# Run tests by marker
uv run pytest -m "integration"
uv run pytest -m "slow"

# Stop on first failure
uv run pytest -x

# Run failed tests from last session
uv run pytest --lf
```

### Performance Testing

```bash
# Run with timing information
uv run pytest --durations=10

# Parallel test execution
uv run pytest -n auto  # Requires pytest-xdist

# Memory profiling
uv run pytest --memray
```

## Test Coverage

### Current Coverage Metrics

| Module | Tests | Lines | Coverage |
|--------|-------|--------|----------|
| **src/models/** | 208 tests | 1,200+ lines | 95%+ |
| **src/config/** | 45+ tests | 800+ lines | 90%+ |
| **src/services/** | 200+ tests | 2,000+ lines | 88%+ |
| **src/security.py** | 33 tests | 300+ lines | 95%+ |
| **src/mcp/** | 30+ tests | 500+ lines | 90%+ |

### Coverage Goals

- **Critical modules**: 95%+ coverage required
- **Service layer**: 90%+ coverage required
- **Utilities**: 85%+ coverage acceptable
- **Integration**: 80%+ coverage target

### Coverage Exclusions

```python
# .coveragerc exclusions
[run]
omit = 
    tests/*
    */migrations/*
    */venv/*
    setup.py
    
[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    if __name__ == .__main__.:
```

## Quality Standards

### Code Quality Checks

```bash
# Linting and formatting (required before commit)
ruff check . --fix && ruff format .

# Type checking
mypy src/

# Security scanning
bandit -r src/

# Import sorting
ruff check --select I --fix .
```

### Test Quality Requirements

1. **Test Naming**: Descriptive test names using `test_<functionality>_<scenario>`
2. **Documentation**: Docstrings for complex test scenarios
3. **Isolation**: No test dependencies or shared state
4. **Assertions**: Clear, specific assertions with helpful error messages
5. **Edge Cases**: Comprehensive boundary and error testing

### Example Test Quality Standards

```python
def test_search_request_score_threshold_constraints():
    """Test score_threshold field validation with boundary values."""
    # Valid thresholds (boundary testing)
    SearchRequest(query="test", score_threshold=0.0)  # Lower bound
    SearchRequest(query="test", score_threshold=1.0)  # Upper bound
    SearchRequest(query="test", score_threshold=0.5)  # Middle value
    
    # Invalid thresholds (error scenarios)
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="test", score_threshold=-0.1)
    assert "greater than or equal to 0" in str(exc_info.value)
    
    with pytest.raises(ValidationError) as exc_info:
        SearchRequest(query="test", score_threshold=1.1)
    assert "less than or equal to 1" in str(exc_info.value)
```

## Writing Tests

### Test File Organization

```python
"""Unit tests for <module_name>."""

import pytest
from pydantic import ValidationError
from unittest.mock import Mock, AsyncMock, patch

from src.module import ClassUnderTest


class TestClassName:
    """Test class for ClassName."""
    
    @pytest.fixture
    def mock_dependency(self):
        """Create mock dependency for testing."""
        return Mock()
    
    def test_method_success_scenario(self):
        """Test successful method execution."""
        # Arrange
        instance = ClassUnderTest()
        
        # Act
        result = instance.method()
        
        # Assert
        assert result == expected_value
    
    def test_method_error_scenario(self):
        """Test method with error conditions."""
        with pytest.raises(SpecificError) as exc_info:
            instance.method_with_invalid_input()
        assert "expected error message" in str(exc_info.value)
```

### Pydantic Model Testing Pattern

```python
def test_model_required_fields():
    """Test that required fields are enforced."""
    # Valid model creation
    model = ModelClass(required_field="value")
    assert model.required_field == "value"
    
    # Missing required field
    with pytest.raises(ValidationError) as exc_info:
        ModelClass()
    assert "required_field" in str(exc_info.value)

def test_model_field_validation():
    """Test field validation rules."""
    # Valid values
    ModelClass(constrained_field=5)  # Within range
    
    # Invalid values
    with pytest.raises(ValidationError):
        ModelClass(constrained_field=-1)  # Below range

def test_model_default_values():
    """Test default field values."""
    model = ModelClass(required_field="value")
    assert model.optional_field == "default_value"
```

### Async Service Testing Pattern

```python
@pytest.mark.asyncio
async def test_service_method():
    """Test async service method."""
    # Setup
    service = ServiceClass(config)
    await service.initialize()
    
    try:
        # Test
        result = await service.async_method()
        assert result.success is True
    finally:
        # Cleanup
        await service.cleanup()

# Alternative with async context manager
@pytest.mark.asyncio
async def test_service_with_context_manager():
    """Test service with async context manager."""
    async with ServiceClass(config) as service:
        result = await service.async_method()
        assert result.success is True
```

### Mocking Best Practices

```python
# Mock external dependencies
@patch('src.module.external_service')
def test_with_mocked_service(mock_service):
    """Test with mocked external service."""
    mock_service.return_value.method.return_value = "mocked_result"
    
    result = function_under_test()
    assert result == "expected_result"
    mock_service.return_value.method.assert_called_once()

# Mock async methods
@pytest.mark.asyncio
async def test_with_async_mock():
    """Test with async mock."""
    mock_service = AsyncMock()
    mock_service.async_method.return_value = "async_result"
    
    result = await function_under_test(mock_service)
    assert result == "expected_result"
```

## Continuous Integration

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: astral-sh/setup-uv@v1
      
      - name: Install dependencies
        run: uv sync
        
      - name: Run linting
        run: |
          uv run ruff check .
          uv run ruff format --check .
          
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
        
      - id: pytest
        name: pytest
        entry: uv run pytest tests/unit/ --tb=short
        language: system
        pass_filenames: false
```

### Test Performance Monitoring

```bash
# Monitor test execution time
uv run pytest --durations=0 > test_timings.txt

# Profile memory usage
uv run pytest --memray tests/unit/

# Generate performance reports
uv run pytest --benchmark-only --benchmark-json=benchmark.json
```

## Best Practices Summary

1. **Write tests first**: TDD approach for new features
2. **Keep tests simple**: One assertion per test when possible
3. **Use descriptive names**: Test names should explain what is being tested
4. **Test edge cases**: Boundary conditions and error scenarios
5. **Mock appropriately**: Mock external dependencies, not internal logic
6. **Maintain fast tests**: Unit tests should run in milliseconds
7. **Regular coverage review**: Monitor and improve coverage over time
8. **Clean test data**: Use fixtures and proper setup/teardown

This testing documentation provides comprehensive guidance for maintaining high code quality and ensuring reliable system behavior through thorough testing practices.