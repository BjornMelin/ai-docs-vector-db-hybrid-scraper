# Unit Testing Framework

This directory contains comprehensive unit testing for the AI Documentation Vector DB Hybrid Scraper, providing thorough coverage of individual components, functions, and classes with fast execution and reliable validation.

## Framework Overview

The unit testing framework provides:

- **Comprehensive component coverage** for all system modules
- **Fast execution** with isolated test scenarios and minimal dependencies
- **Modern async testing patterns** with proper async/await support
- **Type-safe testing** with comprehensive type annotations
- **Mock-driven testing** for external dependencies and services

## Directory Structure

- **cli/**: Command-line interface components and commands
- **config/**: Configuration management and validation
- **core/**: Core system components and constants
- **infrastructure/**: Infrastructure and client management
- **mcp_tools/**: MCP (Model Context Protocol) tools and models
- **models/**: Data models and validation
- **security/**: Security components and ML security
- **services/**: Service layer components and business logic
- **utils/**: Utility functions and helpers

## Core Testing Categories

### CLI Testing (`cli/`)

Command-line interface and user interaction testing:

```python
@pytest.mark.unit
@pytest.mark.cli
class TestCLICommands:
    """Test CLI command implementations."""
    
    def test_batch_command(self):
        """Test batch processing command."""
        pass
    
    def test_config_command(self):
        """Test configuration management command."""
        pass
    
    def test_database_command(self):
        """Test database management command."""
        pass
    
    def test_setup_command(self):
        """Test setup and initialization command."""
        pass
```

**Subdirectories:**
- **commands/**: Individual CLI command testing
- **wizard/**: Setup wizard and profile manager testing

### Configuration Testing (`config/`)

Configuration management and validation testing:

```python
@pytest.mark.unit
@pytest.mark.config
class TestConfiguration:
    """Test configuration management components."""
    
    async def test_config_validation(self):
        """Test configuration validation logic."""
        pass
    
    async def test_config_integration(self):
        """Test configuration integration scenarios."""
        pass
    
    def test_config_models(self):
        """Test configuration data models."""
        pass
```

### Service Layer Testing (`services/`)

Business logic and service component testing:

```python
@pytest.mark.unit
@pytest.mark.service
class TestServices:
    """Test service layer components."""
    
    async def test_browser_services(self):
        """Test browser automation services."""
        pass
    
    async def test_cache_services(self):
        """Test caching service implementations."""
        pass
    
    async def test_embedding_services(self):
        """Test embedding generation services."""
        pass
    
    async def test_vector_db_services(self):
        """Test vector database services."""
        pass
```

**Major Service Categories:**
- **browser/**: Browser automation and web scraping
- **cache/**: Caching strategies and implementations
- **content_intelligence/**: Content analysis and classification
- **core/**: Core service infrastructure
- **crawling/**: Web crawling and content extraction
- **embeddings/**: Text embedding generation
- **functional/**: Functional service components
- **hyde/**: HyDE (Hypothetical Document Embeddings) implementation
- **monitoring/**: System monitoring and health checks
- **observability/**: Observability and tracking
- **query_processing/**: Query processing and orchestration
- **task_queue/**: Asynchronous task processing
- **utilities/**: Service utility functions
- **vector_db/**: Vector database operations and management

### MCP Tools Testing (`mcp_tools/`)

Model Context Protocol tools and models:

```python
@pytest.mark.unit
@pytest.mark.mcp
class TestMCPTools:
    """Test MCP tool implementations."""
    
    def test_tool_registry(self):
        """Test MCP tool registration and management."""
        pass
    
    def test_request_models(self):
        """Test MCP request data models."""
        pass
    
    def test_response_models(self):
        """Test MCP response data models."""
        pass
```

### Models Testing (`models/`)

Data models and validation testing:

```python
@pytest.mark.unit
@pytest.mark.models
class TestDataModels:
    """Test data model implementations."""
    
    def test_api_contracts(self):
        """Test API contract models."""
        pass
    
    def test_document_processing(self):
        """Test document processing models."""
        pass
    
    def test_vector_search(self):
        """Test vector search models."""
        pass
    
    def test_model_validators(self):
        """Test model validation logic."""
        pass
```

## Usage Commands

### Quick Start

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific unit test category
uv run pytest tests/unit/services/ -v
uv run pytest tests/unit/config/ -v
uv run pytest tests/unit/mcp_tools/ -v

# Run with unit test markers
uv run pytest -m "unit" -v
```

### Fast Development Testing

```bash
# Run fast unit tests only
uv run pytest tests/unit/ -m "unit and fast" -v

# Run unit tests without slow operations
uv run pytest tests/unit/ -m "unit and not slow" -v

# Run specific service unit tests
uv run pytest tests/unit/services/browser/ -v
uv run pytest tests/unit/services/embeddings/ -v
uv run pytest tests/unit/services/vector_db/ -v
```

### Coverage Analysis

```bash
# Run unit tests with coverage (CI profile scoped to unit folder)
python scripts/dev.py test --profile ci -- tests/unit/

# Run coverage for specific service tests
python scripts/dev.py test --profile ci -- tests/unit/services/

# Generate XML/terminal coverage (produced automatically by the CI profile)
python scripts/dev.py test --profile ci
```

### Parallel Execution

```bash
# Run unit tests in parallel
uv run pytest tests/unit/ -n auto

# Run with specific worker count
uv run pytest tests/unit/ -n 4

# Run with work-stealing for better load balancing
uv run pytest tests/unit/ -n auto --dist=worksteal
```

### CI/CD Integration

```bash
# Fast unit tests for CI
uv run pytest tests/unit/ -m "unit and fast and not slow" --maxfail=10

# Unit tests with JUnit XML output
uv run pytest tests/unit/ --junitxml=unit_test_results.xml

# Unit tests with performance tracking
uv run pytest tests/unit/ --durations=20 --tb=short
```

## Unit Testing Patterns

### Async Testing Pattern

```python
@pytest.mark.asyncio
async def test_async_operation(mock_service: AsyncMock) -> None:
    """Test async operation with proper mocking."""
    # Arrange
    expected_result = {"status": "success", "data": "test"}
    mock_service.process.return_value = expected_result
    
    # Act
    result = await service_function(test_input)
    
    # Assert
    assert result == expected_result
    mock_service.process.assert_called_once_with(test_input)
```

### Parametrized Testing Pattern

```python
@pytest.mark.parametrize("input_data,expected_output", [
    ({"query": "test"}, {"results": []}),
    ({"query": "python"}, {"results": ["doc1", "doc2"]}),
    ({"query": ""}, {"error": "Empty query"}),
])
def test_search_variations(
    input_data: Dict[str, Any],
    expected_output: Dict[str, Any]
) -> None:
    """Test search with various input scenarios."""
    result = search_function(**input_data)
    assert result == expected_output
```

### Mock Service Pattern

```python
@pytest.fixture
def mock_vector_db_service() -> AsyncMock:
    """Mock vector database service."""
    mock = AsyncMock(spec=VectorDBService)
    mock.search.return_value = []
    mock.upsert.return_value = {"status": "success"}
    mock.delete.return_value = {"deleted": 1}
    return mock
```

### Error Handling Testing Pattern

```python
async def test_service_error_handling(mock_service: AsyncMock) -> None:
    """Test proper error handling in service operations."""
    # Setup mock to raise exception
    mock_service.process.side_effect = ConnectionError("Connection failed")
    
    # Test exception is properly handled
    with pytest.raises(ServiceError) as exc_info:
        await service_function()
    
    assert "Connection failed" in str(exc_info.value)
    assert exc_info.value.error_code == "CONNECTION_ERROR"
```

## Test Organization Principles

### Test Structure

Each test file follows a consistent structure:

```python
"""Test module for [component_name].

This module provides comprehensive unit testing for [component_description],
including [specific_areas_covered].
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List

# Test class organization
class TestComponentName:
    """Test suite for ComponentName class."""
    
    def test_basic_functionality(self) -> None:
        """Test basic component functionality."""
        pass
    
    async def test_async_operations(self) -> None:
        """Test async component operations."""
        pass
    
    def test_error_scenarios(self) -> None:
        """Test error handling scenarios."""
        pass
    
    @pytest.mark.parametrize("input,expected", test_cases)
    def test_edge_cases(self, input: Any, expected: Any) -> None:
        """Test edge cases and boundary conditions."""
        pass
```

### Fixture Organization

Fixtures are organized by scope and reusability:

```python
# Session-scoped fixtures (expensive setup)
@pytest.fixture(scope="session")
def test_database_config() -> DatabaseConfig:
    """Test database configuration."""
    return DatabaseConfig(url="sqlite+aiosqlite:///:memory:")

# Module-scoped fixtures (shared across test module)
@pytest.fixture(scope="module")
def mock_embedding_service() -> EmbeddingService:
    """Mock embedding service for module tests."""
    return create_mock_embedding_service()

# Function-scoped fixtures (test isolation)
@pytest.fixture
def sample_document() -> Dict[str, Any]:
    """Sample document for testing."""
    return {
        "url": "https://example.com/doc",
        "title": "Test Document",
        "content": "Test content for document processing"
    }
```

### Assertion Patterns

Use descriptive assertions with clear error messages:

```python
# Good assertion patterns
assert result is not None, "Function should return a value"
assert isinstance(result, dict), "Result should be a dictionary"
assert len(result["items"]) == expected_count, f"Expected {expected_count} items"
assert result["status"] == "success", f"Operation failed: {result.get('error')}"

# Custom assertion helpers
def assert_valid_document(doc: Dict[str, Any]) -> None:
    """Assert document has required fields."""
    required_fields = ["url", "title", "content"]
    for field in required_fields:
        assert field in doc, f"Missing required field: {field}"
    assert isinstance(doc["url"], str), "URL must be a string"
    assert len(doc["content"]) > 0, "Content cannot be empty"
```

## Performance and Quality Targets

### Execution Speed Targets

| Test Category | Target Time | Description |
|---------------|-------------|-------------|
| Individual Unit Test | < 0.1s | Single test execution |
| Test Class | < 1s | Complete test class execution |
| Test Module | < 5s | Complete test module execution |
| Full Unit Suite | < 60s | Complete unit test suite |

### Coverage Targets

| Component | Target Coverage | Description |
|-----------|----------------|-------------|
| Core Logic | 95%+ | Business logic and algorithms |
| Service Layer | 90%+ | Service implementations |
| Models/Validators | 95%+ | Data models and validation |
| Utilities | 85%+ | Utility functions |
| CLI Commands | 80%+ | Command-line interfaces |

### Quality Metrics

- **Test Reliability**: 99%+ consistent pass rate
- **Test Maintainability**: Clear, readable test code
- **Mock Accuracy**: Realistic mock behavior
- **Error Coverage**: Comprehensive error scenario testing
- **Documentation**: Complete test documentation

## Tools and Frameworks

### Core Testing
- **pytest**: Primary testing framework
- **pytest-asyncio**: Async test support
- **pytest-mock**: Mocking and patching
- **pytest-cov**: Coverage analysis

### Mock and Stub Libraries
- **unittest.mock**: Standard library mocking
- **pytest-mock**: Enhanced mocking capabilities
- **responses**: HTTP request mocking
- **freezegun**: Time/date mocking

### Performance and Analysis
- **pytest-benchmark**: Performance benchmarking
- **pytest-profiling**: Test execution profiling
- **memory_profiler**: Memory usage analysis
- **pytest-xdist**: Parallel test execution

### Quality Assurance
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **bandit**: Security analysis
- **coverage**: Code coverage analysis

This unit testing framework ensures comprehensive, fast, and reliable testing of all system components, supporting rapid development and high code quality in the AI Documentation Vector DB Hybrid Scraper.
