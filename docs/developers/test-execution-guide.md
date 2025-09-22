# Test Execution and Maintenance Guide
## Enterprise-Grade Testing Workflow

**Version:** 1.0  
**Last Updated:** 2025-06-29  
**Target Audience:** Development Team, CI/CD Engineers  

---

## Overview

This guide provides comprehensive instructions for executing and maintaining the test suite to achieve and maintain 90%+ enterprise-grade test coverage with meaningful business logic validation.

---

## Quick Start Commands

### Essential Daily Commands
```bash
# Fast unit tests for immediate feedback
uv run pytest tests/unit/ -x --tb=short

# Coverage check for current work
uv run pytest tests/unit/ --cov=src --cov-report=term-missing

# Rerun only failed tests
uv run pytest --lf
```

### Pre-Commit Validation
```bash
# Complete validation before commit
uv run pytest --cov=src --cov-fail-under=80 --maxfail=5

# Lint and format
ruff check . --fix && ruff format .

# Type checking
mypy src/ --config-file pyproject.toml
```

---

## Test Suite Organization

### Test Categories

| Category | Location | Purpose | Execution Time |
|----------|----------|---------|----------------|
| **Unit Tests** | `tests/unit/` | Fast, isolated component testing | < 30 seconds |
| **Integration Tests** | `tests/integration/` | Service interaction validation | < 5 minutes |
| **End-to-End Tests** | `tests/e2e/` | Complete workflow validation | < 15 minutes |
| **Performance Tests** | `tests/performance/` | Benchmarking and optimization | < 10 minutes |
| **Contract Tests** | `tests/contract/` | API contract validation | < 2 minutes |
| **Security Tests** | `tests/security/` | Security vulnerability testing | < 5 minutes |

### Test Markers

Execute specific test categories using markers:

```bash
# Fast unit tests only
uv run pytest -m "fast"

# AI/ML specific tests
uv run pytest -m "ai or embedding or vector_db"

# Performance and benchmarking
uv run pytest -m "performance or benchmark" --benchmark-only

# Security testing
uv run pytest -m "security"

# Integration testing
uv run pytest -m "integration"

# Property-based testing
uv run pytest -m "property"
```

---

## Coverage Analysis Commands

### Component-Specific Coverage

#### Core Business Logic Components
```bash
# AgenticOrchestrator (Target: 93%+)
uv run pytest tests/unit/services/agents/test_agentic_orchestrator*.py \
  --cov=src/services/agents/agentic_orchestrator \
  --cov-report=term-missing

# QueryOrchestrator (Target: 85%+)
uv run pytest tests/unit/services/agents/test_query_orchestrator*.py \
  --cov=src/services/agents/query_orchestrator \
  --cov-report=term-missing

# DynamicToolDiscovery (Target: 80%+)
uv run pytest tests/unit/services/agents/test_dynamic_tool_discovery*.py \
  --cov=src/services/agents/dynamic_tool_discovery \
  --cov-report=term-missing

# MCP Services (Target: 90%+)
uv run pytest tests/unit/mcp_services/ \
  --cov=src/mcp_services \
  --cov-report=term-missing
```

#### System-Wide Coverage
```bash
# Complete coverage analysis
uv run pytest --cov=src --cov-report=html:htmlcov --cov-report=term-missing

# Coverage with branch analysis
uv run pytest --cov=src --cov-branch --cov-report=html:htmlcov

# Fail if coverage below threshold
uv run pytest --cov=src --cov-fail-under=90
```

### Coverage Report Generation

#### HTML Reports (Detailed Analysis)
```bash
# Generate comprehensive HTML coverage report
uv run pytest --cov=src --cov-report=html:htmlcov

# Open report in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

#### Terminal Reports (Quick Check)
```bash
# Brief coverage summary
uv run pytest --cov=src --cov-report=term

# Detailed missing lines
uv run pytest --cov=src --cov-report=term-missing

# Show covered lines
uv run pytest --cov=src --cov-report=term-missing --cov-report=term:skip-covered
```

#### XML Reports (CI/CD Integration)
```bash
# Generate XML for CI/CD tools
uv run pytest --cov=src --cov-report=xml:coverage.xml
```

---

## Test Execution Profiles

### Development Profiles

#### Fast Feedback Loop
```bash
# Ultra-fast tests for TDD
uv run pytest tests/unit/ -m "fast" --maxfail=1 --tb=line

# Modified files only (with coverage tracking)
uv run pytest tests/unit/ --lf --cov=src --cov-report=term
```

#### Component Focus
```bash
# Focus on specific component during development
uv run pytest tests/unit/services/agents/ -v --cov=src/services/agents

# Test single module with detailed output
uv run pytest tests/unit/services/agents/test_agentic_orchestrator.py -v -s
```

### Quality Assurance Profiles

#### Pre-Merge Validation
```bash
# Complete quality check
uv run pytest tests/unit/ tests/integration/ \
  --cov=src --cov-fail-under=85 \
  --maxfail=10 --tb=short
```

#### Release Validation
```bash
# Full test suite with all categories
uv run pytest tests/ \
  --cov=src --cov-fail-under=90 \
  --cov-report=html:htmlcov \
  --maxfail=20 --tb=short \
  --durations=10
```

### Performance Profiles

#### Performance Regression Testing
```bash
# Run only performance benchmarks
uv run pytest tests/performance/ --benchmark-only --benchmark-sort=mean

# Performance tests with coverage
uv run pytest tests/performance/ \
  --cov=src/services --cov-report=term-missing \
  --benchmark-skip
```

#### Load Testing
```bash
# Load testing suite
uv run pytest tests/load/ -v --tb=short

# Stress testing
uv run pytest tests/load/stress_testing/ -v
```

---

## CI/CD Integration

### GitHub Actions Configuration

#### Basic Test Pipeline
```yaml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6.7.0
      
      - name: Install dependencies
        run: uv sync --all-extras
        
      - name: Run linting
        run: |
          uv run ruff check .
          uv run ruff format --check .
          
      - name: Run type checking
        run: uv run mypy src/
        
      - name: Run tests with coverage
        run: |
          uv run pytest tests/unit/ tests/integration/ \
            --cov=src --cov-report=xml \
            --cov-fail-under=85
            
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

#### Advanced Pipeline with Multiple Test Categories
```yaml
name: Comprehensive Testing
on: [push, pull_request]

jobs:
  test-matrix:
    strategy:
      matrix:
        test-category: [unit, integration, security, performance]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: astral-sh/setup-uv@v6.7.0
      
      - name: Install dependencies
        run: uv sync --all-extras
        
      - name: Run ${{ matrix.test-category }} tests
        run: |
          case "${{ matrix.test-category }}" in
            unit)
              uv run pytest tests/unit/ --cov=src --cov-report=xml
              ;;
            integration)
              uv run pytest tests/integration/ --maxfail=5
              ;;
            security)
              uv run pytest tests/security/ -v
              ;;
            performance)
              uv run pytest tests/performance/ --benchmark-only
              ;;
          esac
```

### Coverage Monitoring

#### Coverage Thresholds
```bash
# Set different thresholds for different components
uv run pytest tests/unit/services/agents/ \
  --cov=src/services/agents \
  --cov-fail-under=90

uv run pytest tests/unit/mcp_services/ \
  --cov=src/mcp_services \
  --cov-fail-under=85
```

#### Coverage Trending
```bash
# Generate coverage badge
uv run coverage-badge -o coverage.svg

# Export coverage data for trending
uv run pytest --cov=src --cov-report=json:coverage.json
```

---

## Debugging and Troubleshooting

### Common Test Failures

#### Import Errors
```bash
# Check Python path and module availability
uv run python -c "import src; print(src.__file__)"

# Verify test environment
uv run pytest --collect-only tests/unit/
```

#### Async Test Issues
```bash
# Run with asyncio debug mode
uv run pytest tests/integration/ -v --tb=long --asyncio-mode=auto

# Check for hanging async operations
uv run pytest tests/integration/ --timeout=30
```

#### Coverage Collection Issues
```bash
# Debug coverage collection
uv run coverage debug sys

# Check coverage configuration
uv run coverage debug config
```

### Performance Debugging

#### Slow Tests Identification
```bash
# Show slowest 20 tests
uv run pytest tests/ --durations=20

# Profile test execution
uv run pytest tests/unit/ --profile --profile-svg
```

#### Memory Usage Monitoring
```bash
# Monitor memory usage during tests
uv run pytest tests/integration/ --tb=short --memray
```

---

## Test Maintenance Procedures

### Daily Maintenance

#### Test Health Check
```bash
# Quick health check of test suite
uv run pytest tests/unit/ --collect-only -q

# Check for outdated test dependencies
uv run pytest --version
```

#### Coverage Monitoring
```bash
# Daily coverage report
uv run pytest tests/unit/ --cov=src --cov-report=term-missing | head -20
```

### Weekly Maintenance

#### Test Performance Review
```bash
# Identify performance regressions
uv run pytest tests/ --durations=10 --tb=no

# Benchmark comparison
uv run pytest tests/performance/ --benchmark-only --benchmark-compare
```

#### Dependency Updates
```bash
# Update test dependencies
uv lock --upgrade

# Verify compatibility after updates
uv run pytest tests/unit/ --maxfail=5
```

### Monthly Maintenance

#### Test Quality Audit
```bash
# Dead test detection
uv run pytest tests/ --collect-only --quiet | grep -c "collected"

# Test coverage quality review
uv run pytest tests/ --cov=src --cov-report=html
# Review htmlcov/index.html for quality assessment
```

#### Performance Baseline Update
```bash
# Update performance baselines
uv run pytest tests/performance/ --benchmark-only --benchmark-save=baseline

# Compare with previous baseline
uv run pytest tests/performance/ --benchmark-only --benchmark-compare=baseline
```

---

## Test Data Management

### Test Fixtures

#### Loading Test Data
```bash
# Regenerate test fixtures
uv run python scripts/generate_test_fixtures.py

# Validate test data integrity
uv run pytest tests/fixtures/ -v
```

#### Database Test Data
```bash
# Reset test database
uv run python scripts/reset_test_db.py

# Seed test data
uv run python scripts/seed_test_data.py
```

### Mock Data Management

#### AI/ML Test Data
```bash
# Generate embedding test data
uv run python tests/utils/generate_embedding_fixtures.py

# Validate AI model mock responses
uv run pytest tests/unit/ai/ -m "property" -v
```

---

## Reporting and Analytics

### Coverage Reports

#### Executive Summary Report
```bash
# Generate executive coverage summary
uv run coverage report --format=markdown > coverage_summary.md
```

#### Detailed Component Analysis
```bash
# Per-component coverage analysis
uv run coverage report --show-missing --include="src/services/agents/*"
uv run coverage report --show-missing --include="src/mcp_services/*"
```

### Test Metrics Dashboard

#### Test Execution Metrics
```bash
# Test execution time trends
uv run pytest tests/ --durations=0 --tb=no > test_durations.log

# Test success rate analysis
uv run pytest tests/ --tb=no --quiet | grep -E "(passed|failed|error)"
```

#### Quality Metrics
```bash
# Test quality scoring
uv run pytest tests/ --cov=src --cov-report=json:coverage.json
# Process coverage.json for quality metrics
```

---

## Security and Compliance

### Security Test Execution
```bash
# OWASP compliance testing
uv run pytest tests/security/compliance/ -v

# Vulnerability scanning
uv run pytest tests/security/vulnerability/ -v

# Authentication testing
uv run pytest tests/security/authentication/ -v
```

### Compliance Reporting
```bash
# Generate compliance report
uv run pytest tests/security/ --html=security_report.html --self-contained-html
```

---

## Best Practices Summary

### Test Development
1. **Write tests first** for new features (TDD approach)
2. **Focus on behavior** over line coverage metrics
3. **Use property-based testing** for AI/ML components
4. **Mock at service boundaries** not internal logic
5. **Test error scenarios** comprehensively

### Test Execution
1. **Run fast tests frequently** during development
2. **Use appropriate test markers** for targeted execution
3. **Monitor coverage trends** not just absolute numbers
4. **Validate performance** with every test run
5. **Maintain test data quality** through regular updates

### Test Maintenance
1. **Review test quality** monthly through audits
2. **Update test dependencies** regularly
3. **Refactor slow tests** to maintain productivity
4. **Archive obsolete tests** to reduce maintenance burden
5. **Document test patterns** for team consistency

---

## Support and Resources

### Documentation Links
- [Testing Best Practices](./testing-best-practices.md)
- [AI/ML Testing Patterns](./ai-ml-testing-guide.md)
- [Performance Testing Guide](./performance-testing-guide.md)

### Tool Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [coverage.py Documentation](https://coverage.readthedocs.io/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)

### Team Resources
- **Test Lead:** [Team Lead Contact]
- **CI/CD Support:** [DevOps Team Contact]
- **Test Infrastructure:** [Infrastructure Team Contact]

---

**Last Updated:** 2025-06-29  
**Next Review:** 2025-07-29