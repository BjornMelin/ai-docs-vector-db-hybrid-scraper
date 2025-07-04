# Testing Best Practices

> **Purpose**: Comprehensive testing guidelines and best practices for the AI Docs Vector DB Hybrid Scraper project
> **Audience**: Developers and QA Engineers
> **Last Updated**: 2025-01-04

This guide outlines testing best practices, patterns, and standards for maintaining high code quality and reliability.

## Core Testing Principles

### Test-Driven Development (TDD)
- Write tests before implementation
- Follow Red-Green-Refactor cycle
- Maintain test coverage above 80%

### Test Structure (AAA Pattern)
```python
def test_vector_search_functionality():
    # Arrange - Set up test data and dependencies
    search_query = "AI documentation"
    mock_embeddings = generate_test_embeddings()
    
    # Act - Execute the functionality being tested
    results = vector_search_service.search(search_query, mock_embeddings)
    
    # Assert - Verify expected behavior
    assert len(results) > 0
    assert all(result.score > 0.5 for result in results)
```

## Testing Categories

### Unit Tests
- Test individual functions and classes in isolation
- Mock external dependencies
- Fast execution (< 100ms per test)
- High coverage of business logic

### Integration Tests
- Test component interactions
- Use real dependencies where appropriate
- Test API endpoints end-to-end
- Validate data flow between services

### End-to-End Tests
- Test complete user workflows
- Use production-like environment
- Validate system behavior from user perspective
- Test critical business paths

## Test Data Management

### Test Fixtures
```python
@pytest.fixture
def sample_documents():
    """Provide consistent test documents."""
    return [
        Document(id="1", content="AI testing documentation", embedding=[0.1, 0.2, 0.3]),
        Document(id="2", content="Vector database guide", embedding=[0.4, 0.5, 0.6])
    ]
```

### Factory Pattern
```python
class DocumentFactory:
    @staticmethod
    def create_document(content: str = "Test content") -> Document:
        return Document(
            id=str(uuid.uuid4()),
            content=content,
            embedding=generate_random_embedding()
        )
```

## Mocking and Test Doubles

### External Service Mocking
```python
@pytest.fixture
def mock_qdrant_client():
    with patch('qdrant_client.QdrantClient') as mock:
        mock.return_value.search.return_value = create_mock_search_results()
        yield mock
```

### HTTP Mocking with respx
```python
@pytest.mark.asyncio
async def test_api_endpoint(respx_mock):
    respx_mock.get("https://api.example.com/data").mock(
        return_value=httpx.Response(200, json={"status": "success"})
    )
    
    result = await api_client.fetch_data()
    assert result["status"] == "success"
```

## Async Testing Patterns

### Async Test Setup
```python
@pytest.mark.asyncio
async def test_async_vector_search():
    async with AsyncVectorService() as service:
        results = await service.search_async("test query")
        assert len(results) > 0
```

### Testing Timeouts and Cancellation
```python
@pytest.mark.asyncio
async def test_search_timeout():
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            slow_search_operation(),
            timeout=1.0
        )
```

## Property-Based Testing

### Using Hypothesis
```python
from hypothesis import given, strategies as st

@given(st.text(min_size=1, max_size=1000))
def test_text_processing_with_any_input(text):
    """Test text processing with generated inputs."""
    result = process_text(text)
    assert isinstance(result, str)
    assert len(result) <= len(text)
```

## Performance Testing

### Benchmarking with pytest-benchmark
```python
def test_vector_search_performance(benchmark):
    """Benchmark vector search performance."""
    result = benchmark(vector_service.search, "test query")
    assert len(result) > 0
```

### Memory Testing
```python
def test_memory_usage():
    """Test memory usage remains within bounds."""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    large_operation()
    final_memory = process.memory_info().rss
    
    memory_increase = final_memory - initial_memory
    assert memory_increase < 100 * 1024 * 1024  # 100MB limit
```

## Error Handling Tests

### Exception Testing
```python
def test_invalid_input_handling():
    """Test proper error handling for invalid inputs."""
    with pytest.raises(ValidationError) as exc_info:
        process_invalid_document(None)
    
    assert "document cannot be None" in str(exc_info.value)
```

### Graceful Degradation
```python
def test_fallback_behavior():
    """Test system behavior when external services fail."""
    with patch('external_service.call') as mock_call:
        mock_call.side_effect = ConnectionError("Service unavailable")
        
        result = service_with_fallback.perform_operation()
        assert result.status == "fallback_used"
```

## AI/ML Testing Considerations

### Model Output Validation
```python
def test_embedding_generation():
    """Test embedding generation produces valid outputs."""
    text = "Sample documentation text"
    embedding = embedding_service.generate(text)
    
    # Test embedding properties
    assert len(embedding) == EXPECTED_DIMENSION
    assert all(isinstance(x, float) for x in embedding)
    assert not all(x == 0 for x in embedding)  # Non-zero embedding
```

### Similarity Testing
```python
def test_semantic_similarity():
    """Test that similar texts produce similar embeddings."""
    text1 = "AI documentation"
    text2 = "artificial intelligence docs"
    
    emb1 = embedding_service.generate(text1)
    emb2 = embedding_service.generate(text2)
    
    similarity = cosine_similarity(emb1, emb2)
    assert similarity > 0.7  # Should be semantically similar
```

## Test Organization

### Directory Structure
```
tests/
├── unit/
│   ├── services/
│   ├── models/
│   └── utils/
├── integration/
│   ├── api/
│   ├── database/
│   └── external/
├── e2e/
│   ├── workflows/
│   └── scenarios/
├── fixtures/
│   ├── data/
│   └── mocks/
└── conftest.py
```

### Test Naming Conventions
- `test_<functionality>_<scenario>_<expected_result>`
- Example: `test_vector_search_with_filters_returns_filtered_results`

## Continuous Integration

### Test Execution Strategy
```yaml
# CI pipeline test stages
stages:
  - lint          # Code quality checks
  - unit-tests    # Fast unit tests
  - integration   # Component integration tests
  - e2e          # End-to-end scenarios
  - performance  # Performance benchmarks
```

### Coverage Requirements
- Minimum 80% overall coverage
- 95% coverage for critical business logic
- 100% coverage for security-related code

## Common Testing Anti-Patterns to Avoid

### ❌ What NOT to Do
- Testing implementation details instead of behavior
- Over-mocking internal components
- Writing tests just to hit coverage metrics
- Sharing mutable state between tests
- Using real external services in unit tests
- Magic numbers without explanation

### ✅ What TO Do
- Test observable behavior and contracts
- Mock at system boundaries
- Write meaningful tests that verify business value
- Ensure test isolation and independence
- Use test doubles for external dependencies
- Document test intentions clearly

## Debugging Failed Tests

### Useful pytest Options
```bash
# Run with detailed output
pytest -v --tb=short

# Run specific test with debugging
pytest -v -s tests/test_specific.py::test_function

# Run with coverage report
pytest --cov=src --cov-report=html

# Run only failed tests from last run
pytest --lf
```

### Test Debugging Tips
1. Use `pytest.set_trace()` for interactive debugging
2. Add detailed assertion messages
3. Log intermediate values for complex operations
4. Use parametrized tests to isolate edge cases

## Resources and Further Reading

- [pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Hypothesis Property-Based Testing](https://hypothesis.readthedocs.io/)
- [respx HTTP Mocking](https://lundberg.github.io/respx/)