# Testing Guide

Modern test practices for fast, reliable, and maintainable tests.

## Core Principles

### Test Independence
Every test must run independently and in any order.

```python
# ✅ GOOD: Independent test
async def test_user_creation(db_fixture):
    user = await create_user(db_fixture, email="test@example.com")
    assert user.id is not None
    assert user.email == "test@example.com"

# ❌ BAD: Depends on global state
user_id = None
async def test_create_user(db):
    global user_id
    user = await create_user(db, email="test@example.com")
    user_id = user.id  # Sets global state
```

### Test Categorization
Always use appropriate markers:

```python
@pytest.mark.unit
async def test_calculate_price():
    """Pure logic test - no I/O operations."""
    price = calculate_price(100, 0.1)
    assert price == 90

@pytest.mark.service
async def test_payment_processing(payment_gateway_mock):
    """Tests integration with payment service."""
    result = await process_payment(payment_gateway_mock, 100)
    assert result.status == "success"

@pytest.mark.e2e
async def test_complete_purchase_flow(test_client):
    """Full user journey test."""
    response = await test_client.post("/cart/add", json={"item_id": 1})
    assert response.status_code == 200
```

### AAA Pattern
Follow Arrange-Act-Assert:

```python
async def test_search_functionality(search_service):
    # Arrange
    test_documents = [
        Document(id=1, content="Python testing guide"),
        Document(id=2, content="JavaScript tutorial")
    ]
    await search_service.index_documents(test_documents)
    
    # Act
    results = await search_service.search("Python")
    
    # Assert
    assert len(results) == 1
    assert results[0].id == 1
```

## Test Organization

### Directory Structure
```
tests/
├── conftest.py                 # Root fixtures
├── unit/                       # Fast, isolated tests
│   ├── config/test_config_models_complete.py
│   └── services/test_logging_config.py
├── services/                  # Cross-service integration tests with mocked external dependencies
│   └── cache/test_cache_manager_behavior.py
├── data_quality/             # Dataset validation and semantic evaluation harnesses
├── load/                      # Locust-based load and stress testing scenarios
├── scripts/                   # CLI tool and developer workflow regression tests
└── utils/                    # Shared helper tests
```

### Naming Conventions
```python
# Test function naming
def test_<action>_<expected_result>():
    pass

# Examples
def test_create_user_returns_user_with_id():
    pass

def test_search_with_empty_query_returns_empty_list():
    pass

def test_payment_with_invalid_card_raises_validation_error():
    pass
```

## Fixture Best Practices

### Fixture Scoping
```python
# Session scope for expensive, read-only resources
@pytest.fixture(scope="session")
async def ml_model():
    """Load ML model once per test session."""
    model = await load_model("model.pkl")
    yield model

# Function scope for mutable resources
@pytest.fixture
async def db_session():
    """Create isolated database session per test."""
    async with create_session() as session:
        yield session
        await session.rollback()
```

### Fixture Factories
```python
@pytest.fixture
def user_factory(db_session):
    """Factory for creating test users with custom attributes."""
    created_users = []
    
    async def _create_user(**kwargs):
        defaults = {
            "email": f"user_{len(created_users)}@example.com",
            "name": f"User {len(created_users)}",
            "is_active": True
        }
        defaults.update(kwargs)
        
        user = User(**defaults)
        db_session.add(user)
        await db_session.commit()
        created_users.append(user)
        return user
    
    yield _create_user
    
    # Cleanup all created users
    for user in created_users:
        await db_session.delete(user)
    await db_session.commit()

# Usage
async def test_user_permissions(user_factory):
    admin = await user_factory(role="admin")
    regular = await user_factory(role="user")
    
    assert admin.can_delete_users()
    assert not regular.can_delete_users()
```

## Async Testing

### Proper Async Structure
```python
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result == expected_value

# Use async fixtures
@pytest.fixture
async def async_client():
    async with AsyncClient() as client:
        yield client
```

### Concurrent Testing
```python
import asyncio

async def test_concurrent_operations(db_session):
    """Test handling of concurrent operations."""
    tasks = [
        create_user(db_session, f"user{i}@example.com")
        for i in range(10)
    ]
    
    users = await asyncio.gather(*tasks)
    
    assert len(users) == 10
    assert all(user.id is not None for user in users)
```

## Mocking Strategies

### Mock at Boundaries
```python
# ✅ GOOD: Mock external service
@pytest.fixture
def payment_gateway_mock():
    with patch("services.payment.PaymentGateway") as mock:
        mock.process.return_value = {"status": "success", "id": "123"}
        yield mock

async def test_checkout_process(payment_gateway_mock):
    result = await checkout(cart_id="cart123", payment_method="card")
    assert result.payment_id == "123"
```

### HTTP Mocking with respx
```python
import respx
from httpx import Response

@pytest.fixture
def mock_api():
    with respx.mock() as mock:
        mock.get("https://api.example.com/users/1").mock(
            return_value=Response(200, json={"id": 1, "name": "Test User"})
        )
        yield mock

async def test_external_api_integration(mock_api):
    user = await fetch_user(1)
    assert user.name == "Test User"
    assert mock_api["users"].called
```

### Async Mock Patterns
```python
@pytest.fixture
def async_cache_mock():
    mock = AsyncMock()
    mock.get.return_value = None  # Default cache miss
    mock.set.return_value = True
    return mock

async def test_cached_operation(async_cache_mock):
    async_cache_mock.get.return_value = {"cached": "data"}
    
    result = await get_with_cache("key", async_cache_mock)
    
    async_cache_mock.get.assert_awaited_once_with("key")
    assert result == {"cached": "data"}
```

## Performance Testing

### Benchmark Tests
```python
@pytest.mark.benchmark
def test_search_performance(benchmark):
    """Benchmark search operation."""
    documents = generate_test_documents(1000)
    index = SearchIndex(documents)
    
    result = benchmark(index.search, "python programming")
    
    assert len(result) > 0
    assert benchmark.stats["mean"] < 0.1  # 100ms budget
```

### Load Testing with Locust
Run scenario-based load tests against a running FastAPI instance using Locust:

```bash
python scripts/dev.py load --host http://localhost:8000 --users 50 --spawn-rate 5 --run-time 10m
```

The Locust scenarios live in `tests/load/locustfile.py` and replay RAG search
queries from the golden dataset. Disable headless mode with
`python scripts/dev.py load --no-headless` to launch the Locust web UI.

### Memory Profiling
```python
import tracemalloc

@pytest.mark.memory
async def test_memory_usage():
    """Test memory efficiency."""
    tracemalloc.start()
    
    results = await process_large_dataset(size=10000)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    assert peak < 100 * 1024 * 1024  # 100MB limit
    assert len(results) == 10000
```

## Common Patterns

### Parameterized Tests
```python
@pytest.mark.parametrize("input_value,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("", ""),
    (None, ""),
])
def test_uppercase_conversion(input_value, expected):
    assert to_uppercase(input_value) == expected

@pytest.mark.parametrize("status_code,should_retry", [
    (500, True),   # Server error - retry
    (502, True),   # Bad gateway - retry
    (400, False),  # Bad request - don't retry
    (401, False),  # Unauthorized - don't retry
])
async def test_retry_logic(status_code, should_retry):
    response = Mock(status_code=status_code)
    assert should_retry_request(response) == should_retry
```

### Property-Based Testing
```python
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=1, max_size=1000),
    max_length=st.integers(min_value=1, max_value=100)
)
def test_truncate_text_property(text, max_length):
    result = truncate_text(text, max_length)
    
    assert len(result) <= max_length
    assert result == text[:max_length]
    if len(text) > max_length:
        assert result.endswith("...")
```

## Anti-Patterns to Avoid

### ❌ Test Interdependence
```python
# BAD: Tests depend on execution order
class TestUserFlow:
    user_id = None
    
    def test_create_user(self):
        user = create_user("test@example.com")
        TestUserFlow.user_id = user.id
    
    def test_update_user(self):
        # Fails if test_create_user didn't run first
        update_user(TestUserFlow.user_id, name="New Name")
```

### ❌ Over-Mocking
```python
# BAD: Mocking everything makes test meaningless
@patch("models.User")
@patch("services.db.get_session")
@patch("services.validators.validate_email")
@patch("utils.generate_id")
async def test_create_user(mock_id, mock_validate, mock_session, mock_user):
    # This tests nothing but mocks
    pass
```

### ❌ Time-Dependent Tests
```python
# BAD: Depends on real time
async def test_cache_expiry():
    cache.set("key", "value", ttl=1)
    time.sleep(2)  # Bad: Real sleep in tests
    assert cache.get("key") is None

# GOOD: Use time mocking
async def test_cache_expiry(frozen_time):
    cache.set("key", "value", ttl=60)
    frozen_time.shift(61)
    assert cache.get("key") is None
```

## Test Commands

### Basic Commands
```bash
# Run all tests
uv run pytest

# Run specific categories
uv run pytest -m unit
uv run pytest -m service
uv run pytest -m e2e

# Run in parallel
uv run pytest -n auto

# Run with coverage
uv run python scripts/dev.py test --profile ci

# Debug failing tests
uv run pytest tests/unit/test_module.py::test_specific -xvs
```

### Advanced Commands
```bash
# Run tests matching pattern
uv run pytest -k "test_user and not test_delete"

# Show slowest tests
uv run pytest --durations=10

# Run only failed tests from last run
uv run pytest --lf

# Generate test report
uv run pytest --html=report.html --self-contained-html
```

## Troubleshooting

### Flaky Tests
- Check for race conditions
- Fix timing issues with deterministic data
- Use proper synchronization

### Slow Tests
- Use appropriate fixture scoping
- Mock slow operations
- Use lightweight test data

### Parallel Execution Issues
- Ensure unique resource names
- Use proper isolation
- Mark serial tests with `@pytest.mark.serial`

## Test Configuration

### Pytest (pyproject.toml)
- Centralized under `[tool.pytest.ini_options]` in `pyproject.toml` to keep the root clean.
- Defaults: strict config/markers, deterministic `pytest-randomly` seed + preserved seed, doctests enabled, 10 slowest durations reported, and quiet output.
- Discovery: `testpaths = ["tests"]` with `python_files = ["test_*.py", "*_test.py"]` and class/function patterns matching both `Test*` and `*Tests`.
- Async: `asyncio_mode = "auto"` and a function-scoped default loop.
- Markers: see the consolidated list in `pyproject.toml` for unit, service, e2e, performance, resiliency, and browser/RAG-specialized suites.

### Coverage Configuration
- Defined in `[tool.coverage.run]` and `[tool.coverage.report]` within `pyproject.toml` to keep configuration centralized.
- Key defaults:
  - `source = ["src"]`, branch coverage enabled, and targeted omit rules for CLI entrypoints and optional/development modules.
  - `fail_under = 80.0` with `show_missing = true` to keep gaps visible.
  - Common exclusions (`pragma: no cover`, `__repr__`, and `NotImplementedError` guards) handled via `exclude_lines`.
