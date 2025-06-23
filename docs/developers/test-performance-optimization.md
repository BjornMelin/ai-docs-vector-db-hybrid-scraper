# Test Performance Optimization Guide

This guide provides comprehensive strategies for optimizing test execution speed and parallel efficiency in the AI Documentation Vector DB Hybrid Scraper project.

## Performance Targets

| Test Category | Target Time | Description |
|---------------|-------------|-------------|
| Unit Tests | < 0.1s | Individual unit tests should complete in under 100ms |
| Integration Tests | < 2s | Integration tests should complete in under 2 seconds |
| E2E Tests | < 10s | End-to-end tests should complete in under 10 seconds |
| Full Suite (Parallel) | < 5 minutes | Complete test suite with parallel execution |
| CI/CD Pipeline | < 10 minutes | Total test time in CI/CD including setup |

## Quick Start

### Fast Test Execution

```bash
# Run fastest unit tests only (< 1 minute)
python scripts/run_fast_tests.py --profile unit

# Run fast and medium tests (< 3 minutes)  
python scripts/run_fast_tests.py --profile fast

# Run with coverage for CI
python scripts/run_fast_tests.py --profile fast --coverage
```

### Performance Analysis

```bash
# Profile test performance
python scripts/test_performance_profiler.py

# Generate performance dashboard
python scripts/test_performance_dashboard.py --dashboard --report
```

## Optimization Strategies

### 1. Test Categorization and Selection

#### Speed-Based Markers

Use pytest markers to categorize tests by execution speed:

```python
import pytest

@pytest.mark.fast
def test_quick_operation():
    """Tests that complete in < 0.1s"""
    pass

@pytest.mark.slow  
def test_expensive_operation():
    """Tests that take > 2s"""
    pass
```

#### Intelligent Test Selection

```bash
# Run only fast tests for quick feedback
pytest -m "fast and not slow"

# Run medium-priority tests
pytest -m "(unit or integration) and not slow"

# Skip expensive tests in CI
pytest -m "not (slow or browser or load)"
```

### 2. Parallel Execution Optimization

#### Optimal Worker Configuration

```bash
# Auto-detect optimal workers
pytest -n auto

# Specific worker count
pytest -n 4

# Work-stealing distribution for better load balancing
pytest -n auto --dist=worksteal
```

#### Test Isolation Best Practices

```python
# Use session-scoped fixtures for expensive setup
@pytest.fixture(scope="session")
def expensive_resource():
    # Setup once per test session
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()

# Ensure tests are stateless
def test_stateless_operation(isolated_data):
    # Use fresh data for each test
    result = process(isolated_data)
    assert result.success
```

### 3. Fixture Optimization

#### Cached Fixtures

```python
from tests.utils.performance_fixtures import FixtureCache

@pytest.fixture(scope="session")
def cached_database_pool():
    """Reuse expensive database connections."""
    return FixtureCache.get(
        "db_pool",
        create_database_pool,
        ttl=300  # 5 minute cache
    )
```

#### Fast Mock Objects

```python
from tests.utils.performance_fixtures import fast_mock_factory

def test_with_optimized_mocks(fast_mock_factory):
    # Pre-configured async mock
    service = fast_mock_factory["async"](
        fetch_data=AsyncMock(return_value={"result": "success"})
    )
    
    # Test executes quickly with minimal overhead
    result = await service.fetch_data()
    assert result["result"] == "success"
```

### 4. Test Data Optimization

#### Minimal Test Data

```python
# Use minimal data sets for faster tests
MINIMAL_DOCUMENT = {
    "id": 1,
    "content": "Short test content",  # Not long paragraphs
    "metadata": {"title": "Test"}     # Essential fields only
}

# Pre-generated test data to avoid computation
SAMPLE_EMBEDDING = [0.1] * 1536  # Fixed embedding
```

#### Data Factories

```python
import factory

class DocumentFactory(factory.Factory):
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: n)
    content = "Test content"  # Fixed, not random
    title = factory.LazyAttribute(lambda obj: f"Document {obj.id}")
```

### 5. Mock Optimization

#### Replace Slow External Calls

```python
# Instead of real API calls
@pytest.fixture
def mock_embedding_service():
    with patch('src.services.embeddings.OpenAIProvider') as mock:
        # Pre-computed response to avoid API overhead
        mock.embed_text.return_value = [0.1] * 1536
        mock.embed_batch.return_value = [[0.1] * 1536] * 10
        yield mock

# Instead of real database operations
@pytest.fixture
def mock_vector_db():
    with patch('src.services.vector_db.QdrantClient') as mock:
        # Fast, deterministic responses
        mock.search.return_value = []
        mock.upsert.return_value = {"status": "success"}
        yield mock
```

#### Efficient Async Mocks

```python
# Optimized async mock setup
mock_service = AsyncMock()
mock_service.__aenter__ = AsyncMock(return_value=mock_service)
mock_service.__aexit__ = AsyncMock(return_value=None)

# Pre-configure common methods to avoid repeated setup
mock_service.process = AsyncMock(return_value={"result": "success"})
```

### 6. Memory Optimization

#### Garbage Collection Management

```python
import gc

@pytest.fixture(autouse=True)
def memory_cleanup():
    """Clean memory between tests."""
    gc.collect()
    yield
    gc.collect()
```

#### Resource Management

```python
# Use context managers for automatic cleanup
async def test_with_resources():
    async with create_managed_resource() as resource:
        result = await resource.operation()
        assert result.success
    # Resource automatically cleaned up
```

### 7. Configuration Optimization

#### Optimized pytest.ini

```ini
# pytest-optimized.ini
[pytest]
addopts = 
    -n auto                    # Parallel execution
    --dist=worksteal          # Optimal work distribution
    --tb=short                # Minimal traceback
    --disable-warnings        # Skip warnings for speed
    --maxfail=3               # Stop early on failures
    
timeout = 60                  # Faster timeout
asyncio_mode = auto          # Optimized async handling
```

#### Fast Test Runner

```python
# scripts/run_fast_tests.py
def run_unit_tests():
    """Run optimized unit test suite."""
    cmd = [
        "uv", "run", "pytest",
        "-c", "pytest-optimized.ini",
        "-m", "unit and fast and not slow",
        "--maxfail=1",
        "--tb=line"
    ]
    subprocess.run(cmd)
```

## CI/CD Optimization

### GitHub Actions Workflow

```yaml
name: Fast Test Suite

jobs:
  fast-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Setup Python and uv
      uses: astral-sh/setup-uv@v3
      with:
        enable-cache: true
    
    - name: Install dependencies
      run: uv sync --dev --frozen
    
    - name: Run fast tests
      run: |
        python scripts/run_fast_tests.py --profile unit --timeout 60
    
    - name: Cache test results
      uses: actions/cache@v4
      with:
        path: .pytest_cache
        key: test-cache-${{ runner.os }}-${{ github.sha }}
```

### Test Matrix Optimization

```yaml
strategy:
  matrix:
    test-profile: [unit, fast, integration]
    include:
      - test-profile: unit
        timeout: 2
        marker: "unit and fast"
      - test-profile: fast  
        timeout: 5
        marker: "(unit or fast) and not slow"
      - test-profile: integration
        timeout: 10
        marker: "integration and not slow"
```

## Performance Monitoring

### Automated Performance Tracking

```python
# Add to conftest.py
@pytest.fixture(autouse=True)
def track_test_performance(request):
    """Track individual test performance."""
    start_time = time.perf_counter()
    yield
    duration = time.perf_counter() - start_time
    
    # Log slow tests
    if duration > 1.0:
        print(f"SLOW TEST: {request.node.nodeid} took {duration:.2f}s")
```

### Performance Regression Detection

```bash
# Run performance profiler to detect regressions
python scripts/test_performance_profiler.py --pattern "unit"

# Generate performance dashboard
python scripts/test_performance_dashboard.py --dashboard
```

### Performance Alerts

```python
# Set up performance thresholds
PERFORMANCE_THRESHOLDS = {
    "unit_test_max": 0.1,      # 100ms max for unit tests
    "integration_test_max": 2.0, # 2s max for integration tests
    "total_suite_max": 300,    # 5min max for full suite
}

def check_performance_regression(test_results):
    """Check if current run exceeds thresholds."""
    alerts = []
    
    if test_results["average_time"] > PERFORMANCE_THRESHOLDS["unit_test_max"]:
        alerts.append("Unit test average time exceeded threshold")
    
    return alerts
```

## Troubleshooting Performance Issues

### Common Performance Problems

1. **Slow Fixtures**
   - Move expensive setup to session scope
   - Cache computed values
   - Use minimal test data

2. **Blocking I/O Operations**
   - Mock external API calls
   - Use in-memory databases for tests
   - Replace file operations with StringIO

3. **Resource Leaks**
   - Ensure proper cleanup in fixtures
   - Use context managers
   - Monitor memory usage

4. **Poor Parallelization**
   - Check for shared state between tests
   - Use isolated test data
   - Avoid global variables

### Performance Debugging

```python
# Add timing to individual test steps
def test_with_timing():
    with performance_monitor() as timer:
        timer.start()
        
        # Step 1
        setup_data()
        timer.checkpoint("setup")
        
        # Step 2  
        result = process_data()
        timer.checkpoint("processing")
        
        # Step 3
        validate_result(result)
        timer.checkpoint("validation")
        
        # Check which step is slowest
        timer.assert_under(0.1, "Test exceeded 100ms target")
```

### Memory Profiling

```python
# Profile memory usage in tests
import tracemalloc

def test_memory_efficient():
    tracemalloc.start()
    
    # Test code here
    result = expensive_operation()
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Assert memory usage is reasonable
    assert peak < 50 * 1024 * 1024  # 50MB limit
```

## Best Practices Summary

### Do's

✅ **Use test markers** for categorization and selection  
✅ **Cache expensive fixtures** at session scope  
✅ **Mock external dependencies** consistently  
✅ **Use minimal test data** for faster execution  
✅ **Monitor performance** with automated tracking  
✅ **Run tests in parallel** with proper isolation  
✅ **Set performance targets** and enforce them  

### Don'ts

❌ **Don't use real external APIs** in unit tests  
❌ **Don't create large test datasets** unnecessarily  
❌ **Don't ignore slow tests** - optimize or mark appropriately  
❌ **Don't share state** between parallel tests  
❌ **Don't skip performance monitoring** in CI  
❌ **Don't use blocking I/O** without mocking  
❌ **Don't create expensive fixtures** in function scope  

## Integration with Development Workflow

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: fast-tests
        name: Run fast test suite
        entry: python scripts/run_fast_tests.py --profile unit
        language: system
        pass_filenames: false
```

### IDE Integration

```json
// VS Code settings.json
{
    "python.testing.pytestArgs": [
        "-c", "pytest-optimized.ini",
        "-m", "fast and not slow",
        "--tb=short"
    ]
}
```

### Development Scripts

```bash
#!/bin/bash
# scripts/dev-test.sh - Quick development testing

# Fast unit tests only
echo "🚀 Running fast unit tests..."
python scripts/run_fast_tests.py --profile unit

# Show performance summary
echo "📊 Performance summary:"
python scripts/test_performance_dashboard.py
```

This optimization guide ensures fast, efficient test execution that enables rapid development feedback loops while maintaining comprehensive test coverage.