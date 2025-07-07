# Subagent Task Assignments

## Agent 1: Test Infrastructure Foundation

### Primary Objectives
- Modernize fixture architecture
- Implement parallel execution support
- Create reusable mock factories

### Specific Tasks

1. **Update Core Fixtures** (Priority: HIGH)
   ```python
   # In tests/conftest.py
   - Implement session-scoped fixtures for expensive resources
   - Add async fixture cleanup with proper error handling
   - Create fixture factories for common patterns
   - Add performance monitoring fixtures
   ```

2. **External Service Mocks** (Priority: HIGH)
   ```python
   # In tests/fixtures/external_services.py
   - Complete mock_qdrant_cloud fixture
   - Add mock_redis_cluster for HA testing
   - Implement mock_openai_with_rate_limits
   - Create mock_webhook_server
   ```

3. **CI Configuration** (Priority: HIGH)
   ```ini
   # In pytest.ini
   - Add pytest-xdist configuration
   - Configure coverage thresholds per module
   - Set up proper timeout handling
   - Add CI-specific markers
   ```

4. **Async Test Support** (Priority: MEDIUM)
   ```python
   # In tests/fixtures/async_fixtures.py
   - Implement async_test_client with retry logic
   - Add async_rate_limiter for API testing
   - Create async_task_manager for concurrent ops
   - Build async_event_emitter for event testing
   ```

### Deliverables
- [ ] Updated conftest.py with modern patterns
- [ ] Complete fixture library in tests/fixtures/
- [ ] CI-optimized pytest.ini
- [ ] Fixture documentation with examples

---

## Agent 2: Naming and Structure Cleanup

### Primary Objectives
- Remove all "enhanced/modern/advanced" naming
- Flatten directory structure
- Consolidate test organization

### Specific Tasks

1. **Bulk Rename Script** (Priority: HIGH)
   ```bash
   # Create and execute rename script
   - Map old names to new descriptive names
   - Update all imports automatically
   - Preserve git history with --follow
   - Generate rename report
   ```

2. **Directory Flattening** (Priority: HIGH)
   ```
   # Target structure:
   tests/
   ├── unit/
   ├── integration/
   ├── e2e/
   └── benchmarks/
   ```

3. **File Consolidation** (Priority: MEDIUM)
   - Merge related small test files
   - Remove empty __init__.py files
   - Consolidate fixture imports
   - Update test discovery paths

### Deliverables
- [ ] Rename mapping document
- [ ] Flattened directory structure
- [ ] Updated import statements
- [ ] Migration verification report

---

## Agent 3: Coverage-Driven Test Removal

### Primary Objectives
- Eliminate coverage-focused tests
- Remove implementation detail tests
- Replace with behavior-driven tests

### Specific Tasks

1. **Identify and Remove** (Priority: HIGH)
   ```python
   # Files to clean:
   - test_chunking_performance.py: Remove all *_coverage tests
   - test_dynamic_tool_discovery.py: Remove coverage class
   - test_preprocessor.py: Remove private method tests
   ```

2. **Behavior Test Creation** (Priority: HIGH)
   ```python
   # Replace with:
   - Test public API contracts
   - Test error handling behavior
   - Test integration points
   - Test user-facing functionality
   ```

3. **Documentation** (Priority: MEDIUM)
   - Document why tests were removed
   - Create behavior test guidelines
   - Update test strategy docs

### Deliverables
- [ ] List of removed tests with rationale
- [ ] New behavior-driven test suite
- [ ] Updated test coverage report
- [ ] Testing best practices guide

---

## Agent 4: Mock Complexity Reduction

### Primary Objectives
- Implement boundary-only mocking
- Replace mock chains with test doubles
- Use real objects internally

### Specific Tasks

1. **Mock Analysis** (Priority: HIGH)
   ```python
   # Identify problematic patterns:
   - Count mock decorators per test
   - Find Mock().mock().mock() chains
   - Locate internal component mocks
   ```

2. **Boundary Mock Implementation** (Priority: HIGH)
   ```python
   # Use respx for HTTP:
   @respx.mock
   async def test_api_call(respx_mock):
       respx_mock.get("https://api.example.com").mock(
           return_value=Response(200, json={"data": "test"})
       )
   ```

3. **Test Double Creation** (Priority: MEDIUM)
   ```python
   # Create simple test implementations:
   class TestRepository:
       def __init__(self):
           self.data = {}
       
       async def save(self, item):
           self.data[item.id] = item
   ```

### Deliverables
- [ ] Mock complexity report
- [ ] Refactored tests with boundary mocking
- [ ] Test double library
- [ ] Mock usage guidelines

---

## Agent 5: Large File Splitting

### Primary Objectives
- Split files >500 lines
- Organize by functionality
- Maintain test isolation

### Specific Tasks

1. **File Analysis** (Priority: HIGH)
   ```python
   # Target files:
   - test_dynamic_tool_discovery.py → 5 files
   - test_orchestrator.py → 3 files
   - test_pipeline.py → 4 files
   ```

2. **Split Strategy** (Priority: HIGH)
   ```python
   # Example split for test_dynamic_tool_discovery.py:
   - test_tool_discovery_init.py
   - test_tool_scoring.py
   - test_tool_execution.py
   - test_tool_registry.py
   - test_tool_integration.py
   ```

3. **Dependency Management** (Priority: MEDIUM)
   - Update imports in split files
   - Ensure fixture availability
   - Maintain test independence
   - Update test collection

### Deliverables
- [ ] File splitting plan
- [ ] Refactored test modules
- [ ] Import dependency graph
- [ ] Test execution verification

---

## Agent 6: Async Test Modernization

### Primary Objectives
- Implement proper async patterns
- Add async cleanup
- Use modern async testing tools

### Specific Tasks

1. **Async Pattern Updates** (Priority: HIGH)
   ```python
   # Modern async test pattern:
   @pytest.mark.asyncio
   async def test_async_operation():
       async with AsyncClient() as client:
           response = await client.get("/api/data")
           assert response.status_code == 200
   ```

2. **Cleanup Implementation** (Priority: HIGH)
   ```python
   # Proper async cleanup:
   @pytest_asyncio.fixture
   async def async_resource():
       resource = await create_resource()
       yield resource
       await resource.cleanup()
   ```

3. **Async Mock Patterns** (Priority: MEDIUM)
   ```python
   # Using respx for async HTTP:
   async with respx.mock:
       respx.get("https://api.example.com").mock(
           return_value=Response(200)
       )
   ```

### Deliverables
- [ ] Async test pattern guide
- [ ] Refactored async tests
- [ ] Async fixture library
- [ ] Performance comparison report

---

## Agent 7: CI/CD Optimization

### Primary Objectives
- Configure parallel execution
- Optimize for CI environments
- Implement test sharding

### Specific Tasks

1. **Parallel Configuration** (Priority: HIGH)
   ```yaml
   # GitHub Actions example:
   strategy:
     matrix:
       test-group: [1, 2, 3, 4]
   steps:
     - run: pytest --shard=${{ matrix.test-group }}/4
   ```

2. **Performance Optimization** (Priority: HIGH)
   ```ini
   # pytest.ini CI settings:
   [pytest]
   addopts = 
       -n auto
       --dist loadscope
       --maxprocesses 4
       --timeout 300
   ```

3. **Test Isolation** (Priority: MEDIUM)
   - Ensure no shared state between tests
   - Implement proper database isolation
   - Add fixture scope verification
   - Create isolation test suite

### Deliverables
- [ ] CI configuration files
- [ ] Parallel execution benchmarks
- [ ] Test isolation verification
- [ ] CI optimization guide

---

## Agent 8: Documentation and Validation

### Primary Objectives
- Update all documentation
- Create validation tools
- Ensure consistency

### Specific Tasks

1. **Documentation Updates** (Priority: HIGH)
   ```markdown
   # Update CLAUDE.md:
   - New fixture patterns
   - Testing best practices
   - Anti-pattern examples
   - Migration guide
   ```

2. **Validation Tools** (Priority: HIGH)
   ```python
   # Pre-commit hooks:
   - Check for forbidden patterns
   - Validate fixture usage
   - Ensure proper mocking
   - Verify file sizes
   ```

3. **Migration Support** (Priority: MEDIUM)
   - Create migration checklist
   - Build validation scripts
   - Document rollback procedures
   - Provide troubleshooting guide

### Deliverables
- [ ] Updated CLAUDE.md
- [ ] Pre-commit configuration
- [ ] Validation script suite
- [ ] Migration documentation

---

## Coordination Protocol

### Daily Standup Format
1. Progress update (2 min per agent)
2. Blockers and dependencies
3. Integration points for the day
4. Performance metrics review

### Shared Resources
- Fixture registry: `tests/fixtures/registry.py`
- Mock definitions: `tests/fixtures/mocks/`
- Performance baselines: `tests/benchmarks/baselines.json`
- Migration tracking: `tests/MIGRATION_STATUS.md`

### Integration Checkpoints
- Day 1 EOD: Foundation complete
- Day 2 EOD: Cleanup verified
- Day 3 EOD: Refactoring tested
- Day 4: Full integration testing

### Success Criteria
- All tests passing in parallel
- Execution time <30s for unit tests
- No flaky tests in CI
- 80%+ code coverage maintained
- Clean validation reports