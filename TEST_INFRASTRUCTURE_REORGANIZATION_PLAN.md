# Test Infrastructure Reorganization Plan

## Current Issues Identified

1. **Multiple conftest.py files** with overlapping functionality
2. **Duplicate pytest configuration files** (pytest.ini, pytest-modern.ini, pytest-fast.ini, pytest-platform.ini)
3. **Inconsistent test naming conventions** across different directories
4. **Excessive marker duplication** in pytest.ini
5. **Test utilities scattered** across different locations
6. **Incomplete directory structures** (empty directories like e2e/, fastmcp/)

## Reorganization Strategy

### 1. Consolidate Configuration Files
- Keep only `pytest.ini` as the main configuration
- Remove redundant configuration files
- Streamline marker definitions (remove duplicates)

### 2. Consolidate conftest.py Files
- Keep main `tests/conftest.py` as the primary fixture source
- Merge specialized conftest files into the main one with proper scoping
- Remove redundant conftest files

### 3. Fix Test Naming Conventions
- Ensure all test files follow `test_*.py` pattern
- Verify all test functions follow `test_*` pattern
- Fix any non-compliant naming

### 4. Clean Up Directory Structure
- Remove empty directories that serve no purpose
- Organize test utilities in a single location
- Ensure proper `__init__.py` files in test packages

### 5. Optimize Test Discovery
- Fix pytest collection issues
- Ensure proper module imports
- Clean up norecursedirs settings

## Implementation Steps

1. Audit and consolidate pytest configuration
2. Merge and clean up conftest.py files
3. Fix test naming conventions
4. Clean up empty/redundant directories
5. Organize test utilities
6. Validate pytest collection works properly
7. Update documentation

## Expected Benefits

- Faster test discovery and collection
- Consistent test organization
- Easier maintenance and debugging
- Better test isolation
- Clearer test structure for new developers