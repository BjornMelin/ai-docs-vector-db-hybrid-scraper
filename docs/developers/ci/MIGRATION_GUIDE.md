# Configuration Deployment Workflow Migration Guide

This guide explains how to migrate from the monolithic 761-line workflow to the new modular approach using composite actions.

## Overview

The refactoring breaks down the original workflow into 4 reusable composite actions:

1. **detect-changes**: Analyzes changes and determines deployment parameters
2. **validate-config**: Validates configuration files (JSON, YAML, Docker Compose, security)
3. **deploy-config**: Handles deployment with snapshot, backup, and rollback capabilities
4. **smoke-tests**: Runs post-deployment verification tests

## Benefits

### Before (Monolithic Workflow)
- 761 lines in a single file
- Difficult to maintain and test
- Lots of duplication if similar logic needed elsewhere
- Hard to understand the full flow
- Changes require editing the entire workflow

### After (Modular Approach)
- Main workflow reduced to ~300 lines
- Each action is ~150-200 lines and focused on one task
- Reusable across multiple workflows
- Easier to test individual components
- Better separation of concerns

## Migration Steps

### 1. Deploy Composite Actions
```bash
# The composite actions are already created in:
.github/actions/detect-changes/
.github/actions/validate-config/
.github/actions/deploy-config/
.github/actions/smoke-tests/
```

### 2. Test Composite Actions
```bash
# Run the test workflow to ensure all actions work
gh workflow run "Test Composite Actions" --ref your-branch
```

### 3. Gradual Migration

You can run both workflows in parallel during the transition:

```yaml
# In config-deployment.yml, add a condition to skip on certain branches
jobs:
  detect-config-changes:
    if: github.ref != 'refs/heads/migrate-to-composite'
    # ... rest of job
```

### 4. Update References

If you have other workflows that depend on the config deployment, update them to use the new workflow name:

```yaml
# Old
uses: ./.github/workflows/config-deployment.yml

# New
uses: ./.github/workflows/config-deployment-refactored.yml
```

### 5. Switch Over

Once validated:
1. Rename `config-deployment.yml` to `config-deployment-legacy.yml`
2. Rename `config-deployment-refactored.yml` to `config-deployment.yml`
3. Update any workflow references

## Customization

### Adding New Validation Steps

Add to `.github/actions/validate-config/action.yml`:

```yaml
- name: My custom validation
  shell: bash
  run: |
    echo "Running custom validation..."
    # Your validation logic here
```

### Modifying Deployment Strategies

Edit `.github/actions/deploy-config/action.yml` to add new strategies:

```yaml
case $STRATEGY in
  "canary")
    echo "  1. Deploy to 10% of instances"
    echo "  2. Monitor metrics for 30 minutes"
    echo "  3. Gradually increase to 100%"
    ;;
```

### Extending Smoke Tests

Add new test types to `.github/actions/smoke-tests/action.yml`:

```yaml
- name: Performance smoke test
  shell: bash
  run: |
    echo "Running performance smoke test..."
    # Your performance test logic
```

## Rollback Plan

If issues arise, you can quickly rollback:

1. Revert the workflow file changes
2. The composite actions can remain (they won't be used)
3. No changes to external systems are required

## Monitoring

Monitor the following after migration:

1. **Workflow execution time**: Should be similar or faster
2. **Success rate**: Should remain the same
3. **Action logs**: Check for any unexpected behavior
4. **Artifact generation**: Ensure all artifacts are still created

## FAQ

**Q: Can I use these actions in other workflows?**
A: Yes! That's one of the main benefits. Any workflow in this repo can use these actions.

**Q: How do I version these actions?**
A: Since they're in the same repo, they use the same ref as the workflow. For external use, you'd need to publish them separately.

**Q: What if I need to debug an action?**
A: Add debug steps within the action or use workflow debugging with `ACTIONS_RUNNER_DEBUG=true`.

**Q: Can I override action behavior?**
A: Yes, through inputs. If you need more flexibility, add new optional inputs to the actions.

## Support

For questions or issues:
1. Check the action logs for detailed error messages
2. Review the test workflow for usage examples
3. Open an issue with the `workflow` label