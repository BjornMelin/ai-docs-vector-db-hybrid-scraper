# Test Cleanup Action Plan

## Phase 1: Stop the Bleeding (Week 1)

### 1.1 Update Testing Guidelines
- [ ] Update CLAUDE.md with explicit anti-patterns to avoid
- [ ] Add pre-commit hook to reject new tests with problematic patterns
- [ ] Create test review checklist for PRs

### 1.2 Remove Coverage-Driven Tests
**Files to fix immediately:**
- [ ] `tests/unit/processing/test_chunking_performance.py`
- [ ] `tests/unit/services/agents/test_dynamic_tool_discovery.py` 
- [ ] `tests/unit/services/query_processing/test_preprocessor.py`
- [ ] `tests/unit/infrastructure/test_client_manager.py`

**Action**: Delete coverage-specific tests, keep only behavior tests

## Phase 2: Naming Cleanup (Week 2)

### 2.1 Bulk Rename Operations
```bash
# Script to identify files needing rename
grep -r "enhanced\|modern\|optimized\|advanced" tests/ --include="*.py" | cut -d: -f1 | sort -u
```

### 2.2 Priority Renames
1. **Browser Router Tests**
   - `EnhancedAutomationRouter` → `AutomationRouter`
   - `test_enhanced_*` → `test_router_*`

2. **Query Processing Tests**
   - `AdvancedSearchOrchestrator` → `SearchOrchestrator`
   - `test_advanced_*` → `test_orchestrator_*`

3. **MCP Tools Tests**
   - Remove "Modern" from all pipeline factory tests
   - Remove "Advanced" from response converter tests

## Phase 3: Reduce Mocking Complexity (Week 3)

### 3.1 Identify Over-Mocked Tests
```python
# Tests with 5+ mocks to refactor
tests/unit/mcp_tools/test_tool_registry.py
tests/unit/services/observability/test_observability_integration.py
tests/unit/services/monitoring/test_initialization.py
tests/unit/mcp_services/test_unified_mcp_server.py
```

### 3.2 Refactoring Strategy
1. Create test doubles at service boundaries only
2. Use real objects for internal components
3. Replace mock chains with simple test implementations

## Phase 4: Flatten Directory Structure (Week 4)

### 4.1 New Structure
```
tests/
├── unit/
│   ├── api/           # API endpoint tests
│   ├── services/      # Service layer tests
│   ├── models/        # Model validation tests
│   └── utils/         # Utility function tests
├── integration/       # Service integration tests
├── e2e/              # End-to-end workflow tests
├── benchmarks/       # Performance benchmarks
└── fixtures/         # Shared test data
```

### 4.2 Migration Plan
1. Move nested integration tests to flat structure
2. Consolidate security tests into relevant unit/integration tests
3. Move load tests to benchmarks/
4. Remove empty __init__.py files

## Phase 5: Fix Test Infrastructure (Week 5)

### 5.1 pytest.ini Cleanup
```ini
[pytest]
testpaths = tests
python_files = test_*.py

# Keep only essential markers
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Tests taking >5 seconds
    asyncio: Async tests
    benchmark: Performance benchmarks
```

### 5.2 conftest.py Simplification
- Remove global path manipulation
- Create focused fixtures without complex inheritance
- Move platform-specific logic to separate module

## Phase 6: Split Large Test Files (Week 6)

### 6.1 Files to Split
1. **test_dynamic_tool_discovery.py** (2,164 lines)
   - Split into: discovery, scoring, execution, integration
   
2. **test_orchestrator.py** (1,000+ lines)
   - Split into: initialization, search_modes, pipeline, results

3. **test_pipeline.py** (1,400+ lines)
   - Split by pipeline stage tests

### 6.2 Splitting Strategy
- Max 500 lines per test file
- Group by functionality, not coverage
- One test class per major feature

## Success Metrics

### Quantitative
- [ ] 0 tests with "coverage" in name
- [ ] 0 files with "enhanced/modern/advanced" in test names
- [ ] Average test file <500 lines
- [ ] <30 pytest markers
- [ ] <3 directory levels in tests/

### Qualitative
- [ ] Tests describe behavior, not implementation
- [ ] Mocks only at service boundaries
- [ ] Clear test organization
- [ ] Fast test execution (<30s for unit tests)

## Tools & Scripts

### Automated Checks
```python
# pre-commit hook to check for anti-patterns
#!/usr/bin/env python3
import sys
import re

FORBIDDEN_PATTERNS = [
    r'test.*coverage',
    r'test.*enhanced',
    r'test.*modern',
    r'test.*advanced',
    r'test.*optimized',
    r'Mock.*Mock.*Mock',
    r'test.*_private',
    r'test.*_internal'
]

# Add to .pre-commit-config.yaml
```

### Refactoring Scripts
1. Bulk rename script for test functions
2. Mock complexity analyzer
3. Test file size checker
4. Directory structure flattener

## Timeline

- **Week 1**: Stop new anti-patterns, remove coverage tests
- **Week 2**: Bulk naming cleanup
- **Week 3**: Reduce mock complexity
- **Week 4**: Flatten directory structure
- **Week 5**: Fix test infrastructure
- **Week 6**: Split large files

Total effort: ~150 hours across 6 weeks