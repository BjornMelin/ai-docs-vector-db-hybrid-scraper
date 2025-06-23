# Test Pattern Style Guide 2025

## Overview

This document defines standardized patterns for all tests in the AI Documentation Vector DB Hybrid Scraper project. These patterns ensure consistency, maintainability, and adherence to 2025 best practices.

## Core Principles

1. **Consistency**: All tests follow the same patterns
2. **Type Safety**: Comprehensive type annotations
3. **Async-First**: Proper async/await patterns throughout
4. **Resource Management**: Clean setup/teardown patterns
5. **Readability**: Clear, self-documenting test code

## Async Test Patterns

### Standard Async Test Function

```python
from typing import Any, Dict
import pytest
from unittest.mock import AsyncMock

@pytest.mark.asyncio
async def test_async_operation(
    mock_service: AsyncMock,
    test_data: Dict[str, Any]
) -> None:
    """Test async operation with proper error handling.
    
    Args:
        mock_service: Mocked async service
        test_data: Test data fixture
    """
    # Arrange
    expected_result = {"status": "success"}
    mock_service.process.return_value = expected_result
    
    # Act
    result = await some_async_function(test_data["input"])
    
    # Assert
    assert result == expected_result
    mock_service.process.assert_called_once_with(test_data["input"])
```

### Async Context Manager Pattern

```python
@pytest.mark.asyncio
async def test_async_context_manager(
    async_resource_manager: AsyncContextManager[Any]
) -> None:
    """Test async context manager usage."""
    async with async_resource_manager as resource:
        result = await resource.process()
        assert result is not None
```

## Fixture Standards

### Naming Convention

- Use descriptive snake_case names
- Prefix with mock_ for mock objects
- Suffix with _config for configuration objects
- Suffix with _data for test data

```python
@pytest.fixture(scope="function")
async def mock_vector_db_client() -> AsyncMock:
    """Mock Qdrant vector database client."""
    client = AsyncMock(spec=QdrantClient)
    client.create_collection.return_value = None
    client.search.return_value = []
    return client

@pytest.fixture(scope="session")
def database_config() -> DatabaseConfig:
    """Test database configuration."""
    return DatabaseConfig(
        url="sqlite+aiosqlite:///:memory:",
        pool_size=5,
        echo=False
    )

@pytest.fixture
def sample_document_data() -> Dict[str, Any]:
    """Sample document data for testing."""
    return {
        "url": "https://example.com/doc",
        "title": "Test Document",
        "content": "This is test content",
        "metadata": {"language": "en"}
    }
```

### Scope Guidelines

- `session`: Expensive setup (database schemas, browser instances)
- `module`: Shared state across module tests
- `class`: Class-level test dependencies
- `function`: Default for isolation (most cases)

### Async Fixture Pattern

```python
@pytest.fixture(scope="function")
async def async_service_client() -> AsyncGenerator[ServiceClient, None]:
    """Async service client with proper cleanup."""
    client = ServiceClient()
    await client.connect()
    try:
        yield client
    finally:
        await client.disconnect()
```

## Type Annotations

### Test Function Signatures

```python
# Standard test function
def test_sync_operation(fixture: MockType) -> None:
    """Test description."""
    pass

# Async test function
async def test_async_operation(fixture: MockType) -> None:
    """Test description."""
    pass

# Parametrized test
@pytest.mark.parametrize("input_value,expected", [
    ("input1", "output1"),
    ("input2", "output2"),
])
def test_parametrized(input_value: str, expected: str) -> None:
    """Test description."""
    pass
```

### Mock Type Annotations

```python
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List

@pytest.fixture
def mock_http_client() -> AsyncMock:
    """Mock HTTP client with proper typing."""
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = AsyncMock(
        status_code=200,
        json=AsyncMock(return_value={"data": "test"})
    )
    return client
```

## Assertion Patterns

### Standard Assertions

```python
# Boolean assertions
assert condition, "Descriptive error message"
assert not condition, "Descriptive error message"

# Equality assertions with type checking
assert isinstance(result, ExpectedType)
assert result == expected_value

# Collection assertions
assert len(collection) == expected_length
assert item in collection
assert all(condition for item in collection)

# Exception assertions
with pytest.raises(ExpectedError) as exc_info:
    function_that_raises()
assert "expected message" in str(exc_info.value)
```

### Custom Assertion Helpers

```python
def assert_successful_response(
    response: Dict[str, Any],
    expected_data: Any = None
) -> None:
    """Assert response indicates success."""
    assert response.get("success") is True
    assert "error" not in response
    if expected_data is not None:
        assert response.get("data") == expected_data

def assert_valid_document_chunk(chunk: Dict[str, Any]) -> None:
    """Assert document chunk has required fields."""
    required_fields = ["content", "title", "url", "chunk_index"]
    for field in required_fields:
        assert field in chunk, f"Missing required field: {field}"
    assert isinstance(chunk["chunk_index"], int)
    assert chunk["chunk_index"] >= 0
```

## Parametrization Patterns

### Basic Parametrization

```python
@pytest.mark.parametrize("input_data,expected_output", [
    ({"query": "test"}, {"results": []}),
    ({"query": "python"}, {"results": ["doc1", "doc2"]}),
])
async def test_search_variations(
    input_data: Dict[str, Any],
    expected_output: Dict[str, Any],
    search_service: SearchService
) -> None:
    """Test search with various inputs."""
    result = await search_service.search(**input_data)
    assert result == expected_output
```

### Complex Parametrization with IDs

```python
@pytest.mark.parametrize("test_case", [
    pytest.param(
        {"input": "valid", "expected": True},
        id="valid_input"
    ),
    pytest.param(
        {"input": "invalid", "expected": False},
        id="invalid_input"
    ),
])
def test_validation_cases(test_case: Dict[str, Any]) -> None:
    """Test validation with various cases."""
    result = validate_input(test_case["input"])
    assert result == test_case["expected"]
```

## Error Handling Patterns

### Exception Testing

```python
async def test_service_error_handling(
    mock_service: AsyncMock
) -> None:
    """Test proper error handling."""
    # Setup mock to raise exception
    mock_service.process.side_effect = ConnectionError("Connection failed")
    
    # Test exception is properly handled
    with pytest.raises(ServiceError) as exc_info:
        await service_function()
    
    assert "Connection failed" in str(exc_info.value)
    assert exc_info.value.error_code == "CONNECTION_ERROR"
```

### Error Validation

```python
def assert_error_response(
    response: Dict[str, Any],
    expected_error_code: str,
    expected_message_fragment: str = None
) -> None:
    """Assert response indicates specific error."""
    assert response.get("success") is False
    assert "error" in response
    assert response["error"]["code"] == expected_error_code
    if expected_message_fragment:
        assert expected_message_fragment in response["error"]["message"]
```

## Resource Management

### Temporary Resources

```python
@pytest.fixture
def temp_file_path() -> Generator[Path, None, None]:
    """Temporary file with automatic cleanup."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()
```

### Async Resource Management

```python
@pytest.fixture
async def async_database_session() -> AsyncGenerator[Session, None]:
    """Async database session with transaction rollback."""
    async with async_session_factory() as session:
        transaction = await session.begin()
        try:
            yield session
        finally:
            await transaction.rollback()
```

## Test Data Patterns

### Factories

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentFactory:
    """Factory for creating test documents."""
    
    @staticmethod
    def create_document(
        url: str = "https://example.com/doc",
        title: str = "Test Document",
        content: str = "Test content",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create test document with defaults."""
        return {
            "url": url,
            "title": title,
            "content": content,
            "metadata": metadata or {},
            "timestamp": "2024-01-01T00:00:00Z"
        }
```

### Builder Pattern

```python
class TestDataBuilder:
    """Builder for complex test data structures."""
    
    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
    
    def with_url(self, url: str) -> "TestDataBuilder":
        """Add URL to test data."""
        self._data["url"] = url
        return self
    
    def with_content(self, content: str) -> "TestDataBuilder":
        """Add content to test data."""
        self._data["content"] = content
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build final test data."""
        return self._data.copy()

# Usage
test_data = (TestDataBuilder()
    .with_url("https://example.com")
    .with_content("Test content")
    .build())
```

## Marker Usage Standards

### Performance Markers

```python
@pytest.mark.slow
@pytest.mark.performance
async def test_large_data_processing() -> None:
    """Test processing large datasets."""
    pass

@pytest.mark.benchmark
def test_function_performance(benchmark) -> None:
    """Benchmark function performance."""
    result = benchmark(expensive_function)
    assert result is not None
```

### Integration Markers

```python
@pytest.mark.integration
@pytest.mark.database
async def test_database_integration() -> None:
    """Test database integration."""
    pass

@pytest.mark.network
@pytest.mark.security
async def test_api_security() -> None:
    """Test API security features."""
    pass
```

## Documentation Standards

### Test Docstrings

```python
async def test_complex_operation(
    mock_service: AsyncMock,
    test_data: Dict[str, Any]
) -> None:
    """Test complex operation with multiple steps.
    
    This test verifies that the complex operation correctly:
    1. Validates input data
    2. Processes data through multiple stages
    3. Returns expected results
    4. Handles errors appropriately
    
    Args:
        mock_service: Mocked external service
        test_data: Test input data
    """
    # Test implementation
    pass
```

### Class Documentation

```python
class TestDocumentProcessor:
    """Test suite for DocumentProcessor class.
    
    This test suite covers:
    - Document validation
    - Content extraction
    - Metadata processing
    - Error handling scenarios
    """
    
    def test_document_validation(self) -> None:
        """Test document validation logic."""
        pass
```

## Common Anti-Patterns to Avoid

### ❌ Bad Patterns

```python
# Bad: No type hints
def test_function(fixture):
    pass

# Bad: Generic assertions
assert result
assert result != None

# Bad: Complex setup in test
def test_complex_operation():
    service = Service()
    service.config = Config()
    service.database = Database()
    # ... complex setup
    result = service.process()
    assert result

# Bad: No cleanup
def test_file_operation():
    with open("/tmp/test", "w") as f:
        f.write("test")
    # File left behind
```

### ✅ Good Patterns

```python
# Good: Proper type hints
def test_function(fixture: MockType) -> None:
    pass

# Good: Descriptive assertions
assert result is not None, "Function should return a value"
assert isinstance(result, dict), "Result should be a dictionary"

# Good: Setup in fixtures
def test_complex_operation(configured_service: Service) -> None:
    result = configured_service.process()
    assert result is not None

# Good: Automatic cleanup
@pytest.fixture
def temp_file() -> Generator[Path, None, None]:
    file_path = Path("/tmp/test")
    file_path.touch()
    try:
        yield file_path
    finally:
        file_path.unlink(missing_ok=True)
```

## Migration Checklist

When updating existing tests to follow these patterns:

- [ ] Add type annotations to all functions and fixtures
- [ ] Update async patterns to use proper async/await
- [ ] Standardize fixture naming and scoping
- [ ] Add descriptive docstrings
- [ ] Use standardized assertion helpers
- [ ] Apply consistent parametrization patterns
- [ ] Add proper error handling tests
- [ ] Ensure resource cleanup
- [ ] Apply appropriate markers
- [ ] Update imports to follow standards

## Tools Integration

### Ruff Configuration

The project's ruff configuration supports these patterns:
- Type checking with mypy-compatible rules
- Import sorting with isort
- Code formatting alignment

### Coverage Requirements

All new test patterns should maintain or improve coverage:
- Function coverage: 100% for new test code
- Branch coverage: 95% minimum
- Integration test coverage: 80% minimum

This style guide ensures all tests in the project follow consistent, modern patterns that enhance maintainability and reliability.