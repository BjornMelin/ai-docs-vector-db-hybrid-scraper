# Load and Performance Testing Framework

This directory contains a comprehensive, production-ready load testing framework for system capacity, scalability, and performance validation of the AI Documentation Vector DB Hybrid Scraper.

## Framework Overview

The load testing framework provides:

- **Locust-based load testing** with realistic user behavior patterns
- **Multiple execution modes**: Locust web UI, headless, pytest integration  
- **Comprehensive user scenarios**: VectorDBUser, AdminUser with weighted tasks
- **Performance metrics collection** with automatic grading and recommendations
- **Custom scenario support** via JSON configuration files
- **Regression testing** against baseline performance

## Directory Structure

- **load_testing/**: Normal load condition testing
- **stress_testing/**: Beyond-capacity stress testing  
- **spike_testing/**: Sudden load increase testing
- **endurance_testing/**: Long-duration performance testing
- **volume_testing/**: Large dataset volume testing
- **scalability/**: System scalability and growth testing

## Core Framework Components

### Infrastructure Files

- **conftest.py** - Comprehensive load testing fixtures and configuration
- **base_load_test.py** - Base load testing infrastructure with Locust integration
- **locust_load_runner.py** - Enhanced Locust-based load testing implementation  
- **load_profiles.py** - Load test profiles and scenarios (steady, ramp_up, spike, step_load)
- **run_load_tests.py** - Comprehensive command-line load test runner

### Test Categories (8 Complete Test Suites)

#### 1. Load Testing (`load_testing/`)
- **test_normal_load.py** - Normal load testing scenarios
- **test_concurrent_users.py** - Concurrent user scenarios and patterns

#### 2. Stress Testing (`stress_testing/`)  
- **test_stress_scenarios.py** - CPU, memory, connection exhaustion scenarios
- **test_breaking_point.py** - Breaking point analysis and system limits

#### 3. Spike Testing (`spike_testing/`)
- **test_spike_load.py** - Comprehensive spike testing with auto-scaling

#### 4. Endurance Testing (`endurance_testing/`)
- **test_endurance_load.py** - Memory leak detection and long-term stability

#### 5. Volume Testing (`volume_testing/`)
- **test_volume_load.py** - Large dataset processing and bulk operations

#### 6. Scalability Testing (`scalability/`)
- **test_scalability_load.py** - Horizontal/vertical scaling validation

## Framework Features

### Core Capabilities

- **Locust-based load testing** with realistic user behavior patterns
- **Multiple execution modes**: Locust web UI, headless, pytest integration
- **Comprehensive user scenarios**: VectorDBUser, AdminUser with weighted tasks  
- **Performance metrics collection** with automatic grading and recommendations
- **Custom scenario support** via JSON configuration files
- **Regression testing** against baseline performance

### Advanced Testing Scenarios

- **Auto-scaling simulation** with capacity management
- **Circuit breaker activation testing** under spike conditions
- **Memory leak detection** with cleanup mechanisms
- **Cache performance analysis** over time
- **Database connection pooling** under stress
- **Bulk operations optimization** for large datasets
- **Resource utilization monitoring** and analysis

### Analytics and Reporting

- **Performance grading** (A-F scale) based on response times, error rates, and throughput
- **Bottleneck identification** with specific recommendations
- **Comparative endpoint analysis** for performance optimization
- **Response time percentiles** (p50, p95, p99) calculation
- **Success rate monitoring** with configurable thresholds
- **Comprehensive test reports** in JSON format

## Usage Commands

### Quick Start

```bash
# View all available options
python tests/load/run_load_tests.py --help

# Run light load test with Locust
python tests/load/run_load_tests.py --mode locust --config light

# Run with Locust web UI for interactive control
python tests/load/run_load_tests.py --mode locust --web --config moderate

# Run specific test type with pytest
python tests/load/run_load_tests.py --mode pytest --test-type load

# Run stress testing
python tests/load/run_load_tests.py --mode locust --config stress

# Run custom scenario
python tests/load/run_load_tests.py --mode scenario --scenario custom_scenario.json
```

### Advanced Usage

```bash
# Benchmark specific endpoints
python tests/load/run_load_tests.py --mode benchmark --endpoints /api/search /api/documents

# Performance regression testing
python tests/load/run_load_tests.py --mode regression --baseline baseline_results.json

# Custom configuration
python tests/load/run_load_tests.py --mode locust --users 100 --spawn-rate 10 --duration 600

# Run with specific markers using pytest
python tests/load/run_load_tests.py --mode pytest --markers load stress spike
```

### Standard pytest Usage

```bash
# Run all load tests
uv run pytest tests/load/ -v

# Run specific load category
uv run pytest tests/load/stress_testing/ -v

# Run with load markers
uv run pytest -m load -v
```

### CI/CD Integration

```bash
# Run load tests in CI pipeline
uv run pytest tests/load/ -v --tb=short
uv run pytest -m "load and not slow" --maxfail=5
```

## Performance Testing Capabilities

### Load Test Types

1. **Load Testing** - Normal expected load validation
2. **Stress Testing** - Beyond capacity testing with breaking point analysis
3. **Spike Testing** - Sudden traffic increase handling
4. **Endurance Testing** - Long-duration stability and memory leak detection
5. **Volume Testing** - Large dataset processing capabilities
6. **Scalability Testing** - Horizontal and vertical scaling validation

### Realistic Test Scenarios

- **Vector database operations** (search, insert, update) with realistic patterns
- **Document processing workflows** with variable complexity
- **Embedding generation** with batch optimization
- **API rate limiting** and circuit breaker testing
- **Database connection pool** management under load
- **Mixed workload patterns** simulating real user behavior

## Technical Implementation

### Architecture

- **Modular design** allowing easy extension and maintenance
- **Fixture-based configuration** for consistent test setup
- **Mock service framework** for controlled testing scenarios
- **Performance metrics collection** with statistical analysis
- **Load profile system** for reusable test patterns

### Integration Points

- **FastAPI application testing** with realistic endpoint targeting
- **Vector database operations** (Qdrant integration ready)
- **Embedding service testing** with performance optimization
- **MCP tool validation** for Claude Desktop integration
- **CI/CD pipeline integration** with automated reporting

### Quality Assurance

- **Comprehensive error handling** with detailed failure analysis
- **Resource cleanup** preventing test interference
- **Performance threshold validation** with configurable criteria
- **Statistical analysis** of response times and throughput
- **Automated bottleneck identification** with actionable recommendations

## Tools and Frameworks

- **locust**: Modern load testing framework (primary)
- **pytest-benchmark**: Performance benchmarking
- **pytest**: Test framework integration
- **FastAPI**: Application testing framework