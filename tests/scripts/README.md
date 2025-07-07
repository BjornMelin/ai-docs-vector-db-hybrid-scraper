# Test Modernization Scripts

This directory contains automation scripts to support the parallel test modernization effort. These tools help 8 parallel subagents work efficiently without conflicts.

## Available Scripts

### 1. `validate_test_quality.py`
Validates test files for anti-patterns and quality issues.

```bash
# Check all tests
python tests/scripts/validate_test_quality.py tests/

# Check specific directory
python tests/scripts/validate_test_quality.py tests/unit/services/

# Output as JSON
python tests/scripts/validate_test_quality.py tests/ --json

# Exit with error if issues found (for CI)
python tests/scripts/validate_test_quality.py tests/ --strict
```

**Detects:**
- Forbidden naming patterns (enhanced, modern, advanced)
- Coverage-driven testing anti-patterns
- Excessive mocking complexity
- File size violations (>500 lines)
- Directory depth violations (>4 levels)
- Test function size violations (>50 lines)

### 2. `rename_antipattern_files.py`
Automatically renames files containing anti-pattern names.

```bash
# Dry run (see what would be renamed)
python tests/scripts/rename_antipattern_files.py tests/

# Actually rename files
python tests/scripts/rename_antipattern_files.py tests/ --execute

# Target specific pattern
python tests/scripts/rename_antipattern_files.py tests/ --pattern enhanced --execute
```

**Features:**
- Smart renaming with meaningful alternatives
- Updates imports in other files automatically
- Handles naming conflicts
- Preserves file functionality

### 3. `parallel_coordinator.py`
Coordinates work between parallel subagents to avoid conflicts.

```bash
# Register an agent
python tests/scripts/parallel_coordinator.py register agent1 naming-cleanup

# Claim files for exclusive processing
python tests/scripts/parallel_coordinator.py claim agent1 tests/unit/test_foo.py tests/unit/test_bar.py

# Release file locks
python tests/scripts/parallel_coordinator.py release agent1 tests/unit/test_foo.py

# Report progress
python tests/scripts/parallel_coordinator.py progress agent1 --processed 5 --fixed naming:10 structure:3

# Get next task for agent
python tests/scripts/parallel_coordinator.py task agent1 naming-cleanup

# Get overall status
python tests/scripts/parallel_coordinator.py status
```

**Agent Roles:**
- `naming-cleanup`: Fix enhanced/modern/advanced naming
- `fixture-migration`: Migrate to modern fixtures
- `structure-flattening`: Reduce directory nesting
- `mock-boundary`: Implement boundary-only mocking
- `behavior-testing`: Convert to behavior-driven tests
- `async-patterns`: Modernize async test patterns
- `performance-fixtures`: Optimize fixture performance
- `ci-optimization`: Optimize for CI execution

### 4. `convert_to_behavior_tests.py`
Converts coverage-driven tests to behavior-driven tests.

```bash
# Analyze directory for conversion opportunities
python tests/scripts/convert_to_behavior_tests.py tests/unit/

# Analyze specific file
python tests/scripts/convert_to_behavior_tests.py --file tests/unit/services/test_user.py

# Actually convert files
python tests/scripts/convert_to_behavior_tests.py tests/unit/ --convert
```

**Conversions:**
- `test_init` → `test_creates_instance_with_valid_data`
- `test_private_method` → Test through public API
- `test_get_property` → `test_provides_expected_value`
- Adds conversion notes as comments

## Parallel Execution Workflow

### For Subagent Implementers:

1. **Register your agent:**
   ```bash
   AGENT_ID="agent-$(date +%s)"
   python tests/scripts/parallel_coordinator.py register $AGENT_ID naming-cleanup
   ```

2. **Get your next task:**
   ```bash
   python tests/scripts/parallel_coordinator.py task $AGENT_ID naming-cleanup
   ```

3. **Claim files before working:**
   ```bash
   python tests/scripts/parallel_coordinator.py claim $AGENT_ID tests/unit/test_example.py
   ```

4. **Do the work:**
   ```bash
   # Example: Fix naming issues
   python tests/scripts/rename_antipattern_files.py tests/unit/test_example.py --execute
   ```

5. **Report progress:**
   ```bash
   python tests/scripts/parallel_coordinator.py progress $AGENT_ID --processed 1 --fixed naming:3
   ```

6. **Release files:**
   ```bash
   python tests/scripts/parallel_coordinator.py release $AGENT_ID tests/unit/test_example.py
   ```

## Quality Gates

Before marking any task complete, run validation:

```bash
# Validate specific file
python tests/scripts/validate_test_quality.py tests/unit/test_example.py --strict

# Should show:
# ✅ No quality issues found!
```

## Progress Tracking

Monitor overall progress:

```bash
python tests/scripts/parallel_coordinator.py status
```

Output shows:
- Active agents and their roles
- Files processed vs total
- Issues fixed by category
- Overall completion percentage

## Tips for Parallel Execution

1. **Always claim files** before modifying to avoid conflicts
2. **Release files promptly** after completion
3. **Report progress frequently** for accurate tracking
4. **Validate changes** before marking complete
5. **Check coordinator status** if unsure about next task

## Integration with CI

These scripts can be integrated into CI pipelines:

```yaml
# Example GitHub Actions step
- name: Validate Test Quality
  run: |
    python tests/scripts/validate_test_quality.py tests/ --strict
```

## Troubleshooting

### File Lock Issues
If files remain locked after agent failure:
```bash
# Manually check locks
cat tests/.coordinator/state.json | jq .file_locks

# Force release if needed (use carefully)
# Edit tests/.coordinator/state.json and remove the lock
```

### Validation Failures
If validation keeps failing:
1. Run validation on specific file for detailed output
2. Check for patterns the script might miss
3. Ensure you're not introducing new anti-patterns

### Coordinator Issues
If coordinator seems stuck:
1. Check `tests/.coordinator/state.json` for corruption
2. Verify agent registration
3. Check for completed tasks that weren't reported