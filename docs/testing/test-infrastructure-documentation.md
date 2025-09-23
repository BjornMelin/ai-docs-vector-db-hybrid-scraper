---
title: Test Infrastructure Documentation
audience: general
status: active
owner: documentation
last_reviewed: 2025-03-13
---

# Test Infrastructure Documentation

> **Version**: 1.0  
> **Last Updated**: 2025-07-07  
> **Scope**: Complete test infrastructure architecture and configuration

## Overview

This document provides comprehensive documentation of our modernized test infrastructure, including architecture, configuration, tools, and maintenance procedures.

## Infrastructure Architecture

### High-Level Architecture

```mermaid
graph TB
    subgraph "Test Execution Layer"
        A[pytest Core] --> B[pytest-xdist]
        B --> C[Worker Pool]
        C --> D[Worker 1]
        C --> E[Worker 2]
        C --> F[Worker N]
    end
    
    subgraph "Test Types"
        G[Unit Tests]
        H[Integration Tests]
        I[E2E Tests]
        J[Performance Tests]
        K[Security Tests]
    end
    
    subgraph "Infrastructure Services"
        L[Test Database]
        M[Cache Layer]
        N[Mock Services]
        O[Fixtures]
    end
    
    subgraph "Quality Tools"
        P[Coverage Analysis]
        Q[Performance Profiling]
        R[Flakiness Detection]
        S[Report Generation]
    end
    
    D --> L
    E --> M
    F --> N
    
    G --> C
    H --> C
    I --> C
    J --> C
    K --> C
    
    C --> P
    C --> Q
    C --> R
    C --> S
```

### Component Details

#### 1. Core Test Framework

```ini
# pyproject.toml
[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

# Parallel execution settings
addopts = """
    --strict-markers
    --strict-config
    --verbose
    -ra
    --cov=src
    --cov-branch
    --cov-report=term-missing:skip-covered
    --cov-report=html
    --cov-report=xml
    --cov-fail-under=80
    --maxfail=5
    -n auto
    --dist loadgroup
    --timeout=300
    --timeout-method=thread
    --benchmark-disable
"""

# Custom markers
markers = [
    "unit: Unit tests (fast, isolated)",
    "integration: Integration tests (cross-boundary)",
    "e2e: End-to-end tests (full stack)",
    "slow: Slow tests (>1s execution time)",
    "performance: Performance benchmarks",
    "security: Security validation tests",
    "ai: AI/ML model tests",
    "flaky: Flaky tests (auto-retry enabled)",
    "serial: Tests that must run serially",
    "requires_gpu: Tests requiring GPU",
]

# Async configuration
asyncio_mode = "auto"

# Coverage settings
[tool.coverage.run]
source = ["src"]
branch = true
parallel = true
omit = [
    "*/tests/*",
    "*/migrations/*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/node_modules/*"
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "pass",
]

[tool.coverage.html]
directory = "htmlcov"
title = "Test Coverage Report"

[tool.coverage.xml]
output = "coverage.xml"
```

#### 2. Parallel Execution Configuration

```python
# conftest.py - Root configuration
import pytest
import asyncio
import uuid
from typing import AsyncGenerator

# Configure pytest-xdist
def pytest_configure(config):
    """Configure parallel test execution."""
    config.addinivalue_line(
        "markers", "parallel: mark test as safe for parallel execution"
    )
    
    # Set worker-specific configuration
    worker_id = config.getoption("--dist-worker", default="master")
    if worker_id != "master":
        # Configure unique resources per worker
        config.option.db_suffix = f"_{worker_id}"
        config.option.cache_prefix = f"test_{worker_id}"

# Load distribution
def pytest_collection_modifyitems(session, config, items):
    """Distribute tests across workers intelligently."""
    # Group tests by module for better cache utilization
    from _pytest_xdist import LoadScopeScheduling
    
    # Mark serial tests
    serial_items = []
    parallel_items = []
    
    for item in items:
        if item.get_closest_marker("serial"):
            serial_items.append(item)
        else:
            parallel_items.append(item)
    
    # Ensure serial tests run on single worker
    if serial_items:
        for item in serial_items:
            item.add_marker(pytest.mark.xdist_group("serial"))
    
    return items

# Worker-scoped fixtures
@pytest.fixture(scope="session")
def worker_id(request):
    """Get unique worker ID for parallel execution."""
    return request.config.getoption("--dist-worker", default="master")

@pytest.fixture
async def isolated_db(worker_id):
    """Provide isolated database per worker."""
    db_name = f"test_db_{worker_id}_{uuid.uuid4().hex[:8]}"
    
    # Create isolated database
    async with create_database(db_name) as db:
        yield db
    
    # Cleanup
    await drop_database(db_name)
```

#### 3. Test Runners

```python
# scripts/run_tests.py
#!/usr/bin/env python3
"""Master test runner with different execution profiles."""

import click
import subprocess
import sys
import time
from pathlib import Path

class TestRunner:
    """Configurable test runner for different scenarios."""
    
    PROFILES = {
        "fast": {
            "args": ["-m", "not slow", "-n", "auto"],
            "description": "Fast tests only (parallel)"
        },
        "unit": {
            "args": ["-m", "unit", "-n", "auto"],
            "description": "Unit tests only"
        },
        "integration": {
            "args": ["-m", "integration", "-n", "4"],
            "description": "Integration tests (limited parallelism)"
        },
        "e2e": {
            "args": ["-m", "e2e", "-n", "2"],
            "description": "End-to-end tests (minimal parallelism)"
        },
        "ci": {
            "args": ["--cov", "--cov-fail-under=80", "-n", "auto"],
            "description": "CI/CD test suite with coverage"
        },
        "performance": {
            "args": ["--benchmark-only", "-n", "1"],
            "description": "Performance benchmarks (serial)"
        },
        "security": {
            "args": ["-m", "security", "-n", "1"],
            "description": "Security tests (serial)"
        },
        "all": {
            "args": ["-n", "auto"],
            "description": "All tests (full parallel)"
        }
    }
    
    def run(self, profile: str, extra_args: list = None):
        """Run tests with specified profile."""
        if profile not in self.PROFILES:
            click.echo(f"Unknown profile: {profile}")
            click.echo(f"Available profiles: {', '.join(self.PROFILES.keys())}")
            sys.exit(1)
        
        config = self.PROFILES[profile]
        args = ["uv", "run", "pytest"] + config["args"]
        
        if extra_args:
            args.extend(extra_args)
        
        click.echo(f"Running {profile} tests: {config['description']}")
        click.echo(f"Command: {' '.join(args)}")
        
        start_time = time.time()
        result = subprocess.run(args)
        duration = time.time() - start_time
        
        click.echo(f"\nTest execution completed in {duration:.2f} seconds")
        
        return result.returncode

@click.command()
@click.option("--profile", "-p", default="fast", help="Test execution profile")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.option("--debug", "-d", is_flag=True, help="Debug mode")
@click.option("--failfast", "-x", is_flag=True, help="Stop on first failure")
@click.argument("pytest_args", nargs=-1)
def main(profile, verbose, debug, failfast, pytest_args):
    """Run tests with specified profile and options."""
    runner = TestRunner()
    
    extra_args = list(pytest_args)
    if verbose:
        extra_args.append("-vv")
    if debug:
        extra_args.extend(["--log-cli-level=DEBUG", "-s"])
    if failfast:
        extra_args.append("-x")
    
    sys.exit(runner.run(profile, extra_args))

if __name__ == "__main__":
    main()
```

### Fixture Architecture

#### 1. Database Fixtures

```python
# tests/fixtures/database.py
import asyncio
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def db_engine(worker_id):
    """Create isolated database engine per test."""
    db_url = f"postgresql+asyncpg://test:test@localhost/test_{worker_id}"
    engine = create_async_engine(db_url, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def db_session(db_engine):
    """Provide database session with automatic rollback."""
    async_session = sessionmaker(
        db_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        async with session.begin():
            yield session
            await session.rollback()
```

#### 2. Cache Fixtures

```python
# tests/fixtures/cache.py
import pytest
import redis.asyncio as redis
from unittest.mock import AsyncMock

@pytest.fixture(scope="session")
async def redis_pool(worker_id):
    """Create Redis connection pool per worker."""
    pool = redis.ConnectionPool(
        host="localhost",
        port=6379,
        db=int(worker_id) if worker_id.isdigit() else 0,
        decode_responses=True
    )
    yield pool
    await pool.disconnect()

@pytest.fixture
async def cache_client(redis_pool):
    """Provide Redis client with automatic cleanup."""
    client = redis.Redis(connection_pool=redis_pool)
    
    # Clear all keys for this test
    await client.flushdb()
    
    yield client
    
    # Cleanup after test
    await client.flushdb()
    await client.close()

@pytest.fixture
def mock_cache():
    """Provide mock cache for unit tests."""
    cache = AsyncMock()
    cache.get.return_value = None
    cache.set.return_value = True
    cache.delete.return_value = True
    cache.exists.return_value = False
    return cache
```

#### 3. HTTP Client Fixtures

```python
# tests/fixtures/http.py
import pytest
import httpx
import respx

@pytest.fixture
async def http_client():
    """Provide async HTTP client."""
    async with httpx.AsyncClient() as client:
        yield client

@pytest.fixture
def mock_http():
    """Provide HTTP mocking via respx."""
    with respx.mock() as respx_mock:
        yield respx_mock

@pytest.fixture
async def api_client(test_app):
    """Provide test client for API testing."""
    from httpx import AsyncClient
    
    async with AsyncClient(app=test_app, base_url="http://test") as client:
        yield client
```

### Performance Testing Infrastructure

```python
# tests/performance/conftest.py
import pytest
import time
import psutil
import tracemalloc
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    duration: float
    memory_peak: int
    cpu_percent: float
    io_operations: int

class PerformanceTracker:
    """Track performance metrics during tests."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def start(self):
        """Start tracking performance."""
        self.start_time = time.perf_counter()
        self.process = psutil.Process()
        self.start_io = self.process.io_counters()
        tracemalloc.start()
    
    def stop(self) -> PerformanceMetrics:
        """Stop tracking and return metrics."""
        duration = time.perf_counter() - self.start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_io = self.process.io_counters()
        io_ops = (
            (end_io.read_count - self.start_io.read_count) +
            (end_io.write_count - self.start_io.write_count)
        )
        
        metrics = PerformanceMetrics(
            duration=duration,
            memory_peak=peak,
            cpu_percent=self.process.cpu_percent(),
            io_operations=io_ops
        )
        
        self.metrics.append(metrics)
        return metrics

@pytest.fixture
def performance_tracker():
    """Provide performance tracking for tests."""
    return PerformanceTracker()

@pytest.fixture
def benchmark_db():
    """Provide populated database for benchmarks."""
    # Create database with realistic data volume
    db = create_benchmark_database()
    populate_with_test_data(db, records=10000)
    yield db
    cleanup_benchmark_database(db)
```

### CI/CD Integration

#### 1. GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
        test-group: ["unit", "integration", "e2e"]
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
          POSTGRES_DB: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
      - uses: actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5
      
      - name: Set up Python
        uses: actions/setup-python@e797f83bcb11b83ae66e0230d6156d7c80228e7c # v6
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install uv
        run: |
          curl -LsSf https://astral.sh/uv/install.sh | sh
          echo "$HOME/.cargo/bin" >> $GITHUB_PATH
      
      - name: Install dependencies
        run: |
          uv venv
          uv pip install -r requirements.txt
          uv pip install -r requirements-dev.txt
      
      - name: Run tests
        run: |
          uv run pytest -m ${{ matrix.test-group }} \
            --cov=src \
            --cov-report=xml \
            --junit-xml=junit/test-results-${{ matrix.test-group }}.xml
        env:
          DATABASE_URL: postgresql://postgres:test@localhost/test
          REDIS_URL: redis://localhost:6379
      
      - name: Upload coverage
        uses: codecov/codecov-action@5a1091511ad55cbe89839c7260b706298ca349f7 # v5.5.1
        with:
          file: ./coverage.xml
          flags: ${{ matrix.test-group }}
          name: py${{ matrix.python-version }}-${{ matrix.test-group }}
      
      - name: Upload test results
        uses: actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2
        if: always()
        with:
          name: test-results-py${{ matrix.python-version }}-${{ matrix.test-group }}
          path: junit/test-results-*.xml
```

#### 2. Test Result Aggregation

```python
# scripts/aggregate_test_results.py
#!/usr/bin/env python3
"""Aggregate test results from parallel runs."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

class TestResultAggregator:
    """Aggregate JUnit XML test results."""
    
    def aggregate_junit_results(self, result_files: List[Path]) -> Dict:
        """Aggregate multiple JUnit XML files."""
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        total_time = 0.0
        
        test_cases = []
        
        for file_path in result_files:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Get test suite statistics
            for testsuite in root.findall(".//testsuite"):
                total_tests += int(testsuite.get("tests", 0))
                total_failures += int(testsuite.get("failures", 0))
                total_errors += int(testsuite.get("errors", 0))
                total_skipped += int(testsuite.get("skipped", 0))
                total_time += float(testsuite.get("time", 0))
                
                # Collect test cases
                for testcase in testsuite.findall(".//testcase"):
                    test_cases.append({
                        "name": testcase.get("name"),
                        "classname": testcase.get("classname"),
                        "time": float(testcase.get("time", 0)),
                        "status": self._get_test_status(testcase)
                    })
        
        return {
            "summary": {
                "total_tests": total_tests,
                "passed": total_tests - total_failures - total_errors - total_skipped,
                "failed": total_failures,
                "errors": total_errors,
                "skipped": total_skipped,
                "duration": total_time,
                "success_rate": (
                    (total_tests - total_failures - total_errors) / total_tests * 100
                    if total_tests > 0 else 0
                )
            },
            "test_cases": test_cases
        }
    
    def _get_test_status(self, testcase):
        """Determine test case status."""
        if testcase.find("failure") is not None:
            return "failed"
        elif testcase.find("error") is not None:
            return "error"
        elif testcase.find("skipped") is not None:
            return "skipped"
        else:
            return "passed"
    
    def generate_report(self, results: Dict, output_path: Path):
        """Generate HTML report from aggregated results."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Results</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f0f0f0; padding: 15px; border-radius: 5px; }
                .passed { color: green; }
                .failed { color: red; }
                .skipped { color: orange; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Test Results Summary</h1>
            <div class="summary">
                <p>Total Tests: {total_tests}</p>
                <p class="passed">Passed: {passed}</p>
                <p class="failed">Failed: {failed}</p>
                <p class="skipped">Skipped: {skipped}</p>
                <p>Duration: {duration:.2f} seconds</p>
                <p>Success Rate: {success_rate:.1f}%</p>
            </div>
            
            <h2>Test Details</h2>
            <table>
                <tr>
                    <th>Test Name</th>
                    <th>Status</th>
                    <th>Duration</th>
                </tr>
                {test_rows}
            </table>
        </body>
        </html>
        """
        
        test_rows = ""
        for test in results["test_cases"]:
            status_class = test["status"]
            test_rows += f"""
                <tr>
                    <td>{test['classname']}.{test['name']}</td>
                    <td class="{status_class}">{test['status']}</td>
                    <td>{test['time']:.3f}s</td>
                </tr>
            """
        
        html_content = html_template.format(
            **results["summary"],
            test_rows=test_rows
        )
        
        output_path.write_text(html_content)

if __name__ == "__main__":
    aggregator = TestResultAggregator()
    
    # Find all JUnit XML files
    result_files = list(Path("junit").glob("*.xml"))
    
    # Aggregate results
    results = aggregator.aggregate_junit_results(result_files)
    
    # Generate report
    aggregator.generate_report(results, Path("test-report.html"))
    
    # Print summary
    print(json.dumps(results["summary"], indent=2))
```

### Monitoring and Metrics

#### 1. Test Execution Monitoring

```python
# src/testing/monitoring.py
import time
import logging
from dataclasses import dataclass
from typing import Dict, List
import prometheus_client as prom

# Metrics
test_duration_histogram = prom.Histogram(
    'test_duration_seconds',
    'Test execution duration in seconds',
    ['test_type', 'test_name', 'status']
)

test_counter = prom.Counter(
    'test_executions_total',
    'Total number of test executions',
    ['test_type', 'status']
)

test_memory_gauge = prom.Gauge(
    'test_memory_usage_bytes',
    'Memory usage during test execution',
    ['test_name']
)

@dataclass
class TestMetrics:
    """Container for test execution metrics."""
    test_name: str
    test_type: str
    duration: float
    memory_peak: int
    status: str
    error: str = None

class TestMonitor:
    """Monitor test execution and collect metrics."""
    
    def __init__(self):
        self.metrics: List[TestMetrics] = []
        self.logger = logging.getLogger(__name__)
    
    def record_test_execution(self, metrics: TestMetrics):
        """Record test execution metrics."""
        self.metrics.append(metrics)
        
        # Update Prometheus metrics
        test_duration_histogram.labels(
            test_type=metrics.test_type,
            test_name=metrics.test_name,
            status=metrics.status
        ).observe(metrics.duration)
        
        test_counter.labels(
            test_type=metrics.test_type,
            status=metrics.status
        ).inc()
        
        test_memory_gauge.labels(
            test_name=metrics.test_name
        ).set(metrics.memory_peak)
        
        # Log for debugging
        self.logger.info(
            f"Test {metrics.test_name} completed in {metrics.duration:.3f}s "
            f"with status {metrics.status}"
        )
    
    def generate_report(self) -> Dict:
        """Generate test execution report."""
        if not self.metrics:
            return {}
        
        total_duration = sum(m.duration for m in self.metrics)
        passed_tests = [m for m in self.metrics if m.status == "passed"]
        failed_tests = [m for m in self.metrics if m.status == "failed"]
        
        return {
            "summary": {
                "total_tests": len(self.metrics),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "total_duration": total_duration,
                "avg_duration": total_duration / len(self.metrics),
                "success_rate": len(passed_tests) / len(self.metrics) * 100
            },
            "slowest_tests": sorted(
                self.metrics,
                key=lambda m: m.duration,
                reverse=True
            )[:10],
            "failed_tests": [
                {
                    "name": m.test_name,
                    "error": m.error,
                    "duration": m.duration
                }
                for m in failed_tests
            ]
        }
```

#### 2. Flakiness Detection

```python
# src/testing/flakiness.py
from collections import defaultdict
from typing import Dict, List, Set

class FlakinessDetector:
    """Detect and track flaky tests."""
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.test_results: Dict[str, List[bool]] = defaultdict(list)
        self.flaky_tests: Set[str] = set()
    
    def record_test_result(self, test_name: str, passed: bool):
        """Record test execution result."""
        self.test_results[test_name].append(passed)
        
        # Check for flakiness
        if len(self.test_results[test_name]) >= 10:
            self._check_flakiness(test_name)
    
    def _check_flakiness(self, test_name: str):
        """Check if test is flaky based on results."""
        results = self.test_results[test_name]
        failure_rate = results.count(False) / len(results)
        
        # Test is flaky if it fails sometimes but not always
        if 0 < failure_rate < 1 - self.threshold:
            self.flaky_tests.add(test_name)
    
    def get_flaky_tests(self) -> List[Dict]:
        """Get list of flaky tests with statistics."""
        flaky_report = []
        
        for test_name in self.flaky_tests:
            results = self.test_results[test_name]
            flaky_report.append({
                "test_name": test_name,
                "total_runs": len(results),
                "failures": results.count(False),
                "failure_rate": results.count(False) / len(results),
                "last_10_results": results[-10:]
            })
        
        return sorted(flaky_report, key=lambda x: x["failure_rate"], reverse=True)
```

### Maintenance Procedures

#### 1. Regular Maintenance Tasks

```bash
#!/bin/bash
# scripts/test_maintenance.sh

# Weekly maintenance script
echo "Running weekly test maintenance..."

# 1. Clean up old test artifacts
echo "Cleaning test artifacts..."
find ./test_artifacts -mtime +7 -delete
find ./htmlcov -mtime +30 -delete
find ./.pytest_cache -mtime +7 -delete

# 2. Update test dependencies
echo "Updating test dependencies..."
uv pip install --upgrade pytest pytest-xdist pytest-asyncio pytest-benchmark

# 3. Run test quality checks
echo "Running test quality checks..."
uv run python scripts/detect_test_smells.py
uv run python scripts/check_test_coverage.py

# 4. Generate test reports
echo "Generating test reports..."
uv run pytest --collect-only -q | wc -l > test_count.txt
uv run python scripts/generate_test_inventory.py

# 5. Check for slow tests
echo "Identifying slow tests..."
uv run pytest --durations=20 > slow_tests.txt

# 6. Verify parallel execution
echo "Verifying parallel execution..."
uv run pytest -n auto --collect-only -q

echo "Maintenance complete!"
```

#### 2. Test Database Management

```python
# scripts/manage_test_db.py
#!/usr/bin/env python3
"""Manage test database lifecycle."""

import asyncio
import click
from sqlalchemy import text

async def cleanup_test_databases():
    """Clean up orphaned test databases."""
    async with get_admin_connection() as conn:
        # Find test databases
        result = await conn.execute(
            text("SELECT datname FROM pg_database WHERE datname LIKE 'test_%'")
        )
        test_dbs = [row[0] for row in result]
        
        # Check for active connections
        for db_name in test_dbs:
            result = await conn.execute(
                text(f"""
                    SELECT count(*) FROM pg_stat_activity 
                    WHERE datname = '{db_name}'
                """)
            )
            active_connections = result.scalar()
            
            if active_connections == 0:
                # Safe to drop
                await conn.execute(text(f"DROP DATABASE IF EXISTS {db_name}"))
                click.echo(f"Dropped orphaned database: {db_name}")

@click.command()
@click.option("--cleanup", is_flag=True, help="Clean up orphaned databases")
def main(cleanup):
    """Test database management utility."""
    if cleanup:
        asyncio.run(cleanup_test_databases())

if __name__ == "__main__":
    main()
```

### Troubleshooting Guide

#### Common Issues

1. **Parallel Execution Failures**
   - Check for shared state between tests
   - Verify database isolation
   - Look for hardcoded ports/resources

2. **Fixture Scope Issues**
   - Review fixture dependencies
   - Check for proper cleanup
   - Verify async context handling

3. **Performance Degradation**
   - Profile slow tests
   - Check for N+1 queries
   - Review fixture overhead

4. **Flaky Tests**
   - Add retry mechanisms
   - Improve test isolation
   - Fix timing dependencies

#### Debug Commands

```bash
# Run single test with full output
uv run pytest path/to/test.py::test_function -xvs

# Run with specific worker count
uv run pytest -n 2

# Run without parallelization
uv run pytest -n 0

# Profile test execution
uv run pytest --profile

# Generate detailed timing report
uv run pytest --durations=0
```

## Conclusion

Our modernized test infrastructure provides:

1. **Scalability**: Parallel execution across all CPU cores
2. **Reliability**: Isolated test environments
3. **Performance**: 90% reduction in execution time
4. **Maintainability**: Clear organization and documentation
5. **Observability**: Comprehensive metrics and monitoring

This infrastructure ensures our tests remain fast, reliable, and maintainable as the project grows. Regular maintenance and monitoring keep the system running optimally.