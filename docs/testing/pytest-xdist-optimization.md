# pytest-xdist Optimization Guide

This guide documents the pytest-xdist configuration and optimizations implemented for parallel test execution in CI/CD environments.

## Overview

Our test infrastructure is optimized for parallel execution using pytest-xdist with:
- **Automatic worker detection** based on CI environment
- **Platform-specific optimizations** for GitHub Actions, GitLab CI, Jenkins, etc.
- **Test isolation** to prevent race conditions
- **Performance monitoring** to track and optimize execution
- **Resource management** to prevent exhaustion

## Key Components

### 1. Automatic Configuration (`tests/ci/pytest_xdist_config.py`)

The `XDistOptimizer` class automatically detects your environment and configures optimal settings:

```python
from tests.ci.pytest_xdist_config import XDistOptimizer

optimizer = XDistOptimizer()
config = optimizer.get_optimal_config()
```

**Features:**
- Detects CI platform (GitHub Actions, GitLab CI, Jenkins, etc.)
- Determines optimal worker count based on available resources
- Sets appropriate timeouts and distribution strategies
- Applies platform-specific optimizations

### 2. CI Environment Detection (`tests/ci/test_environments.py`)

Provides platform-specific test configurations:

```python
from tests.ci.test_environments import detect_environment

env = detect_environment()
pytest_args = env.get_pytest_args()
env_vars = env.get_env_vars()
```

**Supported Platforms:**
- GitHub Actions (2-4 workers)
- GitLab CI (2 workers)
- Jenkins (adaptive)
- Azure DevOps (4 workers)
- CircleCI (resource-class based)
- Local development (2 workers)

### 3. Test Isolation (`tests/ci/test_isolation.py`)

Ensures proper isolation for parallel execution:

```python
from tests.ci.test_isolation import IsolatedTestResources

resources = IsolatedTestResources()
temp_dir = resources.get_isolated_temp_dir()
port = resources.get_isolated_port()
db_name = resources.get_isolated_database_name()
```

**Isolation Features:**
- Worker-specific temporary directories
- Non-conflicting port allocation
- Unique database/collection names
- Environment variable isolation

### 4. Performance Reporting (`tests/ci/performance_reporter.py`)

Tracks test execution performance:

```bash
pytest --performance-report=test-performance.json
```

**Metrics Tracked:**
- Test duration and distribution
- Worker efficiency
- Memory usage
- Load balance score
- Slow test identification

## Configuration Files

### pytest.ini (Base Configuration)

```ini
[pytest]
addopts = 
    --dist=loadscope
    --numprocesses=auto
    --maxprocesses=4
    --tx=popen//python=python
    --import-mode=importlib
```

### pytest-ci.ini (CI-Specific)

```ini
[pytest]
addopts = 
    # Inherits from pytest.ini
    --timeout=180
    --max-worker-restart=2
    --performance-report=test-performance.json
```

## Usage

### Running Tests with Optimization

Use the provided script for automatic optimization:

```bash
# Automatic environment detection and optimization
python scripts/run_ci_tests.py --test-type all --coverage

# Override worker count
python scripts/run_ci_tests.py --workers 2

# Generate performance report
python scripts/run_ci_tests.py --performance-report
```

### Manual Configuration

```bash
# Basic parallel execution
pytest -n auto

# With specific worker count
pytest -n 4

# With distribution strategy
pytest -n 4 --dist=loadscope

# With performance reporting
pytest --performance-report=report.json
```

## CI/CD Integration

### GitHub Actions

```yaml
- name: Run optimized tests
  run: |
    python scripts/run_ci_tests.py \
      --test-type ${{ matrix.test-type }} \
      --coverage \
      --performance-report
```

### GitLab CI

```yaml
test:
  script:
    - python scripts/run_ci_tests.py --test-type unit
```

### Jenkins

```groovy
sh 'python scripts/run_ci_tests.py --test-type all --coverage'
```

## Best Practices

### 1. Test Organization

- Group related tests in the same module for `loadscope` distribution
- Use markers for test categorization
- Keep test files reasonably sized

### 2. Resource Management

```python
@pytest.fixture
def isolated_resources(request):
    """Use isolated resources for parallel safety."""
    resources = IsolatedTestResources()
    yield resources
    resources.cleanup()
```

### 3. Avoid Shared State

```python
# Bad - shared state
class TestExample:
    shared_data = []
    
# Good - isolated state
class TestExample:
    @pytest.fixture
    def test_data(self):
        return []
```

### 4. Use Worker-Safe Fixtures

```python
@pytest.fixture(scope="session")
def app_config(worker_config):
    """Configuration that's safe for parallel execution."""
    config = load_base_config()
    if worker_config["is_master"]:
        return config
    
    # Worker-specific overrides
    config["port"] = worker_config["ports"]["http"]
    return config
```

## Performance Optimization

### 1. Test Distribution Strategies

- **loadscope** (default): Distributes by module/class
- **loadfile**: Distributes by file
- **loadgroup**: Custom grouping
- **no**: No distribution (sequential)

### 2. Worker Count Guidelines

| Environment | Recommended Workers | Max Workers |
|-------------|-------------------|-------------|
| GitHub Actions | 4 | 4 |
| GitLab CI | 2 | 2 |
| Jenkins | CPU/2 | 8 |
| Local Dev | 2 | 2 |

### 3. Timeout Configuration

```ini
# Per-test timeout
--timeout=300

# Timeout method
--timeout-method=thread  # Better for I/O bound tests
--timeout-method=signal  # Better for CPU bound tests
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Use `isolated_port` fixture
2. **File conflicts**: Use `isolated_temp_dir` fixture
3. **Database conflicts**: Use unique collection names
4. **Memory exhaustion**: Reduce worker count
5. **Flaky tests**: Add isolation or use `@pytest.mark.flaky`

### Debugging Parallel Execution

```bash
# Disable parallel execution for debugging
pytest -n 0

# Run with single worker
pytest -n 1

# Verbose output
pytest -n auto -v -s

# Show worker output
pytest -n auto --capture=no
```

### Performance Analysis

Check the generated performance report:

```json
{
  "metadata": {
    "total_duration": 45.23,
    "total_tests": 250,
    "workers_used": 4
  },
  "worker_metrics": {
    "gw0": {
      "total_tests": 65,
      "efficiency": 92.5
    }
  },
  "recommendations": [
    "Consider using --dist=loadfile for better distribution"
  ]
}
```

## Advanced Configuration

### Custom Worker Allocation

```python
def get_custom_worker_count():
    """Custom logic for worker count."""
    if is_memory_intensive_suite():
        return min(2, os.cpu_count())
    elif is_io_bound_suite():
        return os.cpu_count() * 2
    else:
        return os.cpu_count()
```

### Test Grouping

```python
# pytest_plugins.py
def pytest_collection_modifyitems(items):
    """Group tests for optimal distribution."""
    for item in items:
        if "database" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("db"))
        elif "api" in item.nodeid:
            item.add_marker(pytest.mark.xdist_group("api"))
```

### Resource Limits

```python
# Limit memory per worker
config.max_memory_per_worker_mb = 1024

# CPU affinity (Linux only)
config.cpu_affinity = True

# Process priority
config.platform_optimizations["process_priority"] = "below_normal"
```

## Monitoring and Metrics

### Key Metrics to Track

1. **Total execution time**: Should decrease with more workers
2. **Worker efficiency**: Should be > 80%
3. **Load balance score**: Should be > 80%
4. **Memory usage**: Should not exceed limits
5. **Test failures**: Should not increase with parallelization

### Continuous Improvement

1. Review performance reports regularly
2. Identify and optimize slow tests
3. Adjust worker counts based on metrics
4. Update distribution strategy as needed
5. Monitor for flaky tests

## Future Enhancements

- [ ] Dynamic worker scaling based on test load
- [ ] Predictive test distribution using ML
- [ ] Real-time performance dashboard
- [ ] Automatic flaky test detection and isolation
- [ ] Cross-CI platform performance comparison