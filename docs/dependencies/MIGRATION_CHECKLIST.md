# Migration Checklist for Dependency Updates

> **For Developers** - Use this checklist when pulling latest changes

## Pre-Migration Steps

- [ ] **Backup Current Environment**
  ```bash
  uv pip freeze > requirements.backup.txt
  cp .env .env.backup
  ```

- [ ] **Check Python Version**
  ```bash
  python --version  # Should be 3.11, 3.12, or 3.13
  ```

## Migration Steps

### 1. Environment Update

- [ ] **Pull Latest Changes**
  ```bash
  git pull origin main
  ```

- [ ] **Clean Python Cache**
  ```bash
  task clean  # or: find . -type d -name __pycache__ -exec rm -rf {} +
  ```

- [ ] **Update Dependencies**
  ```bash
  # Full reinstall (recommended)
  uv pip sync
  
  # Or selective update
  uv pip install -e ".[dev,contract,accessibility]"
  ```

### 2. Configuration Updates

- [ ] **Update .env File**
  ```bash
  # Add new environment variables
  echo "SLOWAPI_RATELIMIT_STORAGE_URL=redis://localhost:6379" >> .env
  echo "CIRCUIT_BREAKER_REDIS_URL=redis://localhost:6379/1" >> .env
  echo "AIOCACHE_REDIS_ENDPOINT=redis://localhost:6379/2" >> .env
  ```

- [ ] **Update FastAPI Imports**
  ```python
  # Old
  from fastapi import FastAPI
  # app = FastAPI()  # with [standard] extra
  
  # New
  from fastapi import FastAPI
  import httpx  # Now required separately
  app = FastAPI()
  ```

### 3. Code Updates

- [ ] **Add Rate Limiting**
  ```python
  # src/api/main.py
  from slowapi import Limiter, _rate_limit_exceeded_handler
  from slowapi.util import get_remote_address
  from slowapi.errors import RateLimitExceeded
  
  limiter = Limiter(key_func=get_remote_address)
  app.state.limiter = limiter
  app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
  ```

- [ ] **Add Circuit Breakers**
  ```python
  # src/services/external_api.py
  from purgatory import CircuitBreaker
  
  api_breaker = CircuitBreaker(
      name="external-api",
      failure_threshold=5,
      recovery_timeout=60,
      expected_exception=httpx.HTTPError
  )
  
  @api_breaker
  async def call_external_api():
      async with httpx.AsyncClient() as client:
          return await client.get("https://api.example.com")
  ```

- [ ] **Update Caching**
  ```python
  # Old synchronous caching
  from cachetools import TTLCache
  cache = TTLCache(maxsize=100, ttl=300)
  
  # New async caching
  from aiocache import cached, Cache
  
  @cached(ttl=300, cache=Cache.REDIS)
  async def get_cached_data(key: str):
      return await fetch_data(key)
  ```

### 4. Test Updates

- [ ] **Update HTTP Mocks**
  ```python
  # tests/conftest.py
  import pytest
  import respx
  import httpx
  
  @pytest.fixture
  def mock_http():
      with respx.mock(assert_all_called=False) as respx_mock:
          yield respx_mock
  ```

- [ ] **Add Performance Benchmarks**
  ```python
  # tests/benchmarks/test_performance.py
  import pytest
  
  @pytest.mark.benchmark(group="embeddings")
  def test_embedding_performance(benchmark):
      result = benchmark(generate_embeddings, test_documents)
      assert result.mean < 0.1  # 100ms average
  ```

### 5. Quality Checks

- [ ] **Run Linting**
  ```bash
  task lint  # or: ruff check . --fix
  ```

- [ ] **Run Formatting**
  ```bash
  task format  # or: ruff format .
  ```

- [ ] **Run Type Checking**
  ```bash
  task typecheck  # or: mypy src/
  ```

- [ ] **Run Security Scan**
  ```bash
  task security-check  # or: bandit -r src/
  ```

### 6. Testing

- [ ] **Run Unit Tests**
  ```bash
  task test-unit
  ```

- [ ] **Run Integration Tests**
  ```bash
  task test-integration
  ```

- [ ] **Run Full Test Suite**
  ```bash
  task test-full
  ```

- [ ] **Check Test Coverage**
  ```bash
  task coverage  # Should be ≥80%
  ```

### 7. Performance Validation

- [ ] **Run Benchmarks**
  ```bash
  task benchmark
  ```

- [ ] **Compare Performance**
  ```bash
  python scripts/compare_benchmarks.py --baseline=requirements.backup.txt
  ```

## Post-Migration Validation

### API Endpoints

- [ ] **Test Rate Limiting**
  ```bash
  # Should get 429 after 5 requests/minute
  for i in {1..6}; do curl http://localhost:8000/api/test; done
  ```

- [ ] **Test Circuit Breaker**
  ```python
  # Monitor logs for circuit breaker state changes
  tail -f logs/app.log | grep "CircuitBreaker"
  ```

### Monitoring

- [ ] **Check Prometheus Metrics**
  ```bash
  curl http://localhost:8000/metrics | grep -E "(rate_limit|circuit_breaker)"
  ```

- [ ] **Verify Redis Connections**
  ```bash
  redis-cli ping  # Should return PONG
  redis-cli info clients  # Check connection count
  ```

## Rollback Plan

If issues occur:

1. **Restore Environment**
   ```bash
   uv pip install -r requirements.backup.txt
   cp .env.backup .env
   ```

2. **Revert Code Changes**
   ```bash
   git checkout HEAD~1
   ```

3. **Clear Caches**
   ```bash
   redis-cli FLUSHALL
   rm -rf .ruff_cache .mypy_cache .pytest_cache
   ```

## Common Issues & Solutions

### Issue: Import Errors
```python
# Error: ModuleNotFoundError: No module named 'httpx'
# Solution: Install httpx separately
uv pip install httpx>=0.28.1
```

### Issue: Redis Connection Failed
```bash
# Error: redis.exceptions.ConnectionError
# Solution: Start Redis service
docker-compose up -d redis
# or
redis-server --daemonize yes
```

### Issue: Type Checking Failures
```python
# Error: Mypy errors after update
# Solution: Clear mypy cache
rm -rf .mypy_cache
mypy src/ --no-incremental
```

### Issue: Test Failures
```bash
# Error: Tests failing with new dependencies
# Solution: Update test fixtures
pytest --fixtures  # List all fixtures
pytest -k "test_name" -vv  # Debug specific test
```

## Final Verification

- [ ] All tests pass (`task test-full`)
- [ ] Coverage ≥80% (`task coverage`)
- [ ] No security issues (`task security-check`)
- [ ] Performance acceptable (`task benchmark`)
- [ ] API endpoints functional
- [ ] Monitoring operational
- [ ] Documentation updated

## Sign-off

- [ ] Developer: _________________ Date: _______
- [ ] Reviewer: _________________ Date: _______

---

**Need Help?** 
- Check [DEPENDENCY_UPGRADE_GUIDE.md](./DEPENDENCY_UPGRADE_GUIDE.md)
- Run `python scripts/check_breaking_changes.py`
- Ask in #dev-support channel