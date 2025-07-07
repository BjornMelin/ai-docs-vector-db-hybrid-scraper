# Modern Test Practices Developer Guide

> **Version**: 1.0  
> **Last Updated**: 2025-07-07  
> **Audience**: Developers, QA Engineers, DevOps Teams

## Introduction

This guide provides comprehensive guidelines for writing and maintaining tests using our modern test infrastructure. Follow these practices to ensure fast, reliable, and maintainable tests that leverage our parallel execution capabilities.

## Table of Contents

1. [Core Principles](#core-principles)
2. [Test Organization](#test-organization)
3. [Writing Effective Tests](#writing-effective-tests)
4. [Fixture Best Practices](#fixture-best-practices)
5. [Async Testing](#async-testing)
6. [Mocking Strategies](#mocking-strategies)
7. [Performance Testing](#performance-testing)
8. [Common Patterns](#common-patterns)
9. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
10. [Troubleshooting](#troubleshooting)

## Core Principles

### 1. Test Independence

Every test must be completely independent and able to run in any order or in parallel.

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

async def test_get_user(db):
    user = await get_user(db, user_id)  # Depends on previous test
    assert user is not None
```

### 2. Proper Test Categorization

Always use appropriate markers to categorize tests:

```python
@pytest.mark.unit
async def test_calculate_price():
    """Pure logic test - no I/O operations."""
    price = calculate_price(100, 0.1)
    assert price == 90

@pytest.mark.integration
async def test_payment_processing(payment_gateway_mock):
    """Tests integration with payment service."""
    result = await process_payment(payment_gateway_mock, 100)
    assert result.status == "success"

@pytest.mark.e2e
async def test_complete_purchase_flow(test_client):
    """Full user journey test."""
    # Add to cart -> Checkout -> Payment -> Confirmation
    response = await test_client.post("/cart/add", json={"item_id": 1})
    assert response.status_code == 200
```

### 3. AAA Pattern

Follow the Arrange-Act-Assert pattern for clarity:

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
├── conftest.py                 # Root fixtures and configuration
├── unit/                       # Fast, isolated tests
│   ├── conftest.py            # Unit test fixtures
│   ├── models/                # Model tests
│   │   ├── test_user.py
│   │   └── test_document.py
│   ├── services/              # Service logic tests
│   │   ├── test_auth.py
│   │   └── test_search.py
│   └── utils/                 # Utility tests
│       └── test_helpers.py
├── integration/               # Cross-boundary tests
│   ├── conftest.py           # Integration fixtures
│   ├── api/                  # API integration tests
│   │   └── test_endpoints.py
│   └── services/             # Service integration
│       └── test_workflows.py
├── e2e/                      # End-to-end tests
│   ├── conftest.py          # E2E fixtures
│   └── journeys/            # User journey tests
│       └── test_user_flows.py
└── performance/             # Performance tests
    ├── conftest.py         # Benchmark fixtures
    └── benchmarks/         # Performance benchmarks
        └── test_search_performance.py
```

### Naming Conventions

```python
# Test file naming
test_<module_name>.py

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

## Writing Effective Tests

### 1. Clear Test Names

```python
# ✅ GOOD: Descriptive test names
async def test_user_registration_with_valid_email_creates_active_account():
    """Test that registering with a valid email creates an active user account."""
    pass

# ❌ BAD: Vague test names
async def test_user():
    """Test user."""
    pass
```

### 2. Single Responsibility

Each test should verify one specific behavior:

```python
# ✅ GOOD: Tests one thing
async def test_password_reset_sends_email(email_service_mock):
    user = await create_user(email="user@example.com")
    
    await request_password_reset(user.email)
    
    email_service_mock.send.assert_called_once_with(
        to=user.email,
        subject="Password Reset Request"
    )

# ❌ BAD: Tests multiple things
async def test_password_reset(email_service_mock, db):
    # Testing user creation, password reset, email, and token in one test
    user = await create_user(email="user@example.com")
    assert user.is_active
    
    token = await request_password_reset(user.email)
    assert token is not None
    
    email_service_mock.send.assert_called_once()
    
    new_password = "newpass123"
    await reset_password(token, new_password)
    
    assert await authenticate(user.email, new_password)
```

### 3. Meaningful Assertions

```python
# ✅ GOOD: Specific assertions with context
async def test_search_results_ordering():
    results = await search_service.search("python", order_by="relevance")
    
    # Verify specific ordering
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert results[0].score > results[1].score, "Results not ordered by score"
    assert all(r.score >= 0.5 for r in results), "Low relevance results included"

# ❌ BAD: Generic assertions
async def test_search():
    results = await search_service.search("python")
    assert results  # Too vague
```

## Fixture Best Practices

### 1. Fixture Scoping

Choose appropriate fixture scopes for performance:

```python
# Session scope for expensive, read-only resources
@pytest.fixture(scope="session")
async def ml_model():
    """Load ML model once per test session."""
    model = await load_model("model.pkl")
    yield model
    # No cleanup needed for read-only resource

# Function scope for mutable resources
@pytest.fixture
async def db_session():
    """Create isolated database session per test."""
    async with create_session() as session:
        yield session
        await session.rollback()  # Ensure cleanup

# Module scope for shared test data
@pytest.fixture(scope="module")
async def test_dataset():
    """Load test dataset once per module."""
    data = await load_test_data("test_data.json")
    return data
```

### 2. Fixture Composition

Build complex fixtures from simpler ones:

```python
@pytest.fixture
async def test_user(db_session):
    """Create a test user."""
    user = User(email="test@example.com", name="Test User")
    db_session.add(user)
    await db_session.commit()
    return user

@pytest.fixture
async def authenticated_client(test_client, test_user):
    """Create authenticated test client."""
    token = generate_token(test_user)
    test_client.headers["Authorization"] = f"Bearer {token}"
    return test_client

@pytest.fixture
async def test_document(db_session, test_user):
    """Create test document owned by test user."""
    doc = Document(
        title="Test Document",
        content="Test content",
        owner_id=test_user.id
    )
    db_session.add(doc)
    await db_session.commit()
    return doc
```

### 3. Fixture Factories

Use factories for flexible test data creation:

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

### 1. Proper Async Test Structure

```python
# Always use pytest-asyncio
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

### 2. Concurrent Testing

```python
import asyncio

async def test_concurrent_operations(db_session):
    """Test handling of concurrent operations."""
    # Create multiple concurrent tasks
    tasks = [
        create_user(db_session, f"user{i}@example.com")
        for i in range(10)
    ]
    
    # Execute concurrently
    users = await asyncio.gather(*tasks)
    
    # Verify all succeeded
    assert len(users) == 10
    assert all(user.id is not None for user in users)
```

### 3. Async Context Managers

```python
@pytest.fixture
async def websocket_client():
    """Async context manager for WebSocket testing."""
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        yield ws

async def test_websocket_communication(websocket_client):
    # Send message
    await websocket_client.send("Hello")
    
    # Receive response
    response = await websocket_client.recv()
    assert response == "Hello back"
```

## Mocking Strategies

### 1. Mock at Boundaries

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

# ❌ BAD: Mock internal implementation
async def test_checkout_process():
    with patch("services.checkout._calculate_tax") as mock:
        mock.return_value = 10  # Don't mock internal methods
        result = await checkout(cart_id="cart123")
```

### 2. HTTP Mocking with respx

```python
import respx
from httpx import Response

@pytest.fixture
def mock_api():
    with respx.mock() as mock:
        # Mock specific endpoints
        mock.get("https://api.example.com/users/1").mock(
            return_value=Response(200, json={"id": 1, "name": "Test User"})
        )
        
        # Mock with pattern
        mock.post(url__regex=r"https://api\.example\.com/.*").mock(
            return_value=Response(201, json={"created": True})
        )
        
        yield mock

async def test_external_api_integration(mock_api):
    user = await fetch_user(1)
    assert user.name == "Test User"
    assert mock_api["users"].called
```

### 3. Async Mock Patterns

```python
# Async mock fixture
@pytest.fixture
def async_cache_mock():
    mock = AsyncMock()
    mock.get.return_value = None  # Default cache miss
    mock.set.return_value = True
    return mock

async def test_cached_operation(async_cache_mock):
    # Configure specific behavior
    async_cache_mock.get.return_value = {"cached": "data"}
    
    result = await get_with_cache("key", async_cache_mock)
    
    async_cache_mock.get.assert_awaited_once_with("key")
    assert result == {"cached": "data"}
```

## Performance Testing

### 1. Benchmark Tests

```python
import pytest

@pytest.mark.benchmark
def test_search_performance(benchmark):
    """Benchmark search operation."""
    documents = generate_test_documents(1000)
    index = SearchIndex(documents)
    
    # Benchmark the search operation
    result = benchmark(index.search, "python programming")
    
    assert len(result) > 0
    assert benchmark.stats["mean"] < 0.1  # 100ms budget

@pytest.mark.benchmark
async def test_async_performance(benchmark):
    """Benchmark async operation."""
    async def async_operation():
        await asyncio.sleep(0.01)
        return "result"
    
    result = await benchmark(async_operation)
    assert result == "result"
```

### 2. Load Testing

```python
@pytest.mark.load
async def test_concurrent_load():
    """Test system under concurrent load."""
    async def make_request(session, i):
        async with session.get(f"/api/item/{i}") as response:
            return response.status
    
    async with aiohttp.ClientSession() as session:
        # Create 100 concurrent requests
        tasks = [make_request(session, i) for i in range(100)]
        results = await asyncio.gather(*tasks)
    
    # Verify all requests succeeded
    assert all(status == 200 for status in results)
    
    # Check performance metrics
    assert max(response_times) < 1.0  # 1 second max
    assert statistics.mean(response_times) < 0.1  # 100ms average
```

### 3. Memory Profiling

```python
import tracemalloc

@pytest.mark.memory
async def test_memory_usage():
    """Test memory efficiency."""
    tracemalloc.start()
    
    # Perform memory-intensive operation
    results = await process_large_dataset(size=10000)
    
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Verify memory usage is within bounds
    assert peak < 100 * 1024 * 1024  # 100MB limit
    assert len(results) == 10000
```

## Common Patterns

### 1. Parameterized Tests

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
    (503, True),   # Service unavailable - retry
    (400, False),  # Bad request - don't retry
    (401, False),  # Unauthorized - don't retry
    (404, False),  # Not found - don't retry
])
async def test_retry_logic(status_code, should_retry):
    response = Mock(status_code=status_code)
    assert should_retry_request(response) == should_retry
```

### 2. Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(
    text=st.text(min_size=1, max_size=1000),
    max_length=st.integers(min_value=1, max_value=100)
)
def test_truncate_text_property(text, max_length):
    result = truncate_text(text, max_length)
    
    # Properties that should always hold
    assert len(result) <= max_length
    assert result == text[:max_length]
    if len(text) > max_length:
        assert result.endswith("...")

@given(st.lists(st.integers()))
def test_sort_property(items):
    sorted_items = custom_sort(items)
    
    # Properties of sorted list
    assert len(sorted_items) == len(items)
    assert all(a <= b for a, b in zip(sorted_items, sorted_items[1:]))
    assert set(sorted_items) == set(items)
```

### 3. Test Helpers

```python
# tests/helpers.py
async def create_test_data(db_session, **kwargs):
    """Helper to create consistent test data."""
    defaults = {
        "users": 3,
        "documents": 10,
        "tags": 5
    }
    defaults.update(kwargs)
    
    data = TestData()
    
    # Create users
    for i in range(defaults["users"]):
        user = await create_user(
            db_session,
            email=f"user{i}@test.com",
            name=f"Test User {i}"
        )
        data.users.append(user)
    
    # Create documents
    for i in range(defaults["documents"]):
        doc = await create_document(
            db_session,
            title=f"Document {i}",
            owner=data.users[i % len(data.users)]
        )
        data.documents.append(doc)
    
    return data

# Usage
async def test_search_with_permissions(db_session):
    data = await create_test_data(db_session, users=2, documents=5)
    
    # Test user can only see their documents
    user_docs = await search_documents(
        db_session,
        user_id=data.users[0].id
    )
    
    assert all(doc.owner_id == data.users[0].id for doc in user_docs)
```

## Anti-Patterns to Avoid

### 1. ❌ Test Interdependence

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

### 2. ❌ Over-Mocking

```python
# BAD: Mocking everything makes test meaningless
@patch("models.User")
@patch("services.db.get_session")
@patch("services.validators.validate_email")
@patch("utils.generate_id")
async def test_create_user(mock_id, mock_validate, mock_session, mock_user):
    mock_id.return_value = "123"
    mock_validate.return_value = True
    mock_user.return_value = Mock(id="123")
    
    # This tests nothing but mocks
    result = await create_user("test@example.com")
    assert result.id == "123"
```

### 3. ❌ Shared Mutable State

```python
# BAD: Shared state causes flaky tests
TEST_CACHE = {}

async def test_cache_set():
    TEST_CACHE["key"] = "value"
    assert TEST_CACHE["key"] == "value"

async def test_cache_get():
    # Fails if tests run in different order
    value = TEST_CACHE.get("key")
    assert value == "value"
```

### 4. ❌ Time-Dependent Tests

```python
# BAD: Depends on real time
import time

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

## Troubleshooting

### Common Issues and Solutions

#### 1. Flaky Tests

**Symptoms**: Tests pass sometimes, fail other times

**Solutions**:
```python
# Check for race conditions
async def test_concurrent_updates():
    # Add explicit synchronization
    async with asyncio.Lock():
        await update_resource()

# Fix timing issues
@pytest.mark.flaky(reruns=3, reruns_delay=1)
async def test_eventually_consistent():
    # Retry for eventually consistent systems
    pass

# Use deterministic data
def test_with_fixed_seed():
    random.seed(42)  # Fixed seed for reproducibility
    data = generate_random_data()
```

#### 2. Slow Tests

**Symptoms**: Tests take too long to run

**Solutions**:
```python
# Use appropriate fixtures
@pytest.fixture(scope="session")  # Reuse expensive resources
async def expensive_resource():
    return await load_large_model()

# Mock slow operations
@pytest.fixture
def fast_external_api():
    with respx.mock() as mock:
        mock.get(url__regex=r".*").mock(
            return_value=Response(200, json={})
        )
        yield mock

# Use test data factories
@pytest.fixture
def lightweight_test_data():
    # Create minimal data needed for test
    return {"id": 1, "name": "Test"}
```

#### 3. Parallel Execution Issues

**Symptoms**: Tests fail when run in parallel but pass serially

**Solutions**:
```python
# Ensure unique resource names
@pytest.fixture
def unique_db_name():
    return f"test_db_{uuid.uuid4().hex[:8]}"

# Use proper isolation
@pytest.fixture
async def isolated_test_dir(tmp_path_factory):
    # Each worker gets unique temp directory
    path = tmp_path_factory.mktemp("test")
    yield path
    shutil.rmtree(path)

# Mark tests that can't run in parallel
@pytest.mark.serial
async def test_requires_exclusive_resource():
    # This test needs exclusive access
    pass
```

### Debugging Tips

1. **Run tests in isolation**:
   ```bash
   # Run single test
   uv run pytest tests/unit/test_module.py::test_specific -xvs
   
   # Run with debug output
   uv run pytest tests/unit/test_module.py -xvs --log-cli-level=DEBUG
   ```

2. **Check for worker issues**:
   ```bash
   # Run without parallelization
   uv run pytest -n 0
   
   # Run with specific worker count
   uv run pytest -n 2
   ```

3. **Profile test execution**:
   ```bash
   # Show slowest tests
   uv run pytest --durations=10
   
   # Generate detailed timing report
   uv run pytest --benchmark-only
   ```

## Conclusion

Following these modern test practices ensures our test suite remains fast, reliable, and maintainable. Remember:

- **Independence**: Every test stands alone
- **Clarity**: Tests document behavior
- **Performance**: Leverage parallel execution
- **Maintenance**: Keep tests simple and focused

For additional help, consult the test modernization documentation or reach out to the testing team.