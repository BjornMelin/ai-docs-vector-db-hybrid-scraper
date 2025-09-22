# GitHub Composite Actions

This directory contains reusable composite actions for configuration deployment workflows.

## Available Actions

### 1. detect-changes
Detects configuration changes and determines deployment strategy.

**Inputs:**
- `github-event-name`: GitHub event name (required)
- `github-ref`: GitHub reference (required)
- `workflow-dispatch-environment`: Environment from workflow dispatch (optional)

**Outputs:**
- `profiles-changed`: Whether configuration profiles changed
- `templates-changed`: Whether configuration templates changed
- `monitoring-changed`: Whether monitoring configuration changed
- `security-changed`: Whether security configuration changed
- `infrastructure-changed`: Whether infrastructure configuration changed
- `should-deploy`: Whether deployment should proceed
- `target-environment`: Target deployment environment
- `deployment-strategy`: Deployment strategy to use

### 2. validate-config
Validates all configuration files including JSON, YAML, Docker Compose, and security checks.

**Inputs:**
- `target-environment`: Target environment to validate (required)
- `python-version`: Python version to use (default: '3.12')

**Outputs:**
- `validation-passed`: Whether all validation checks passed

### 3. deploy-config
Deploys configuration using the specified strategy with backup and rollback support.

**Inputs:**
- `target-environment`: Target environment for deployment (required)
- `deployment-strategy`: Deployment strategy (blue-green, rolling, direct) (required)
- `rollback-on-failure`: Automatically rollback on deployment failure (default: 'true')
- `python-version`: Python version to use (default: '3.12')

**Outputs:**
- `deployment-url`: URL of the deployed service
- `snapshot-id`: ID of the deployment snapshot
- `deployment-status`: Status of the deployment

### 4. smoke-tests
Runs smoke tests to verify configuration deployment.

**Inputs:**
- `target-environment`: Target environment to test (required)
- `deployment-url`: URL of the deployed service (optional)
- `python-version`: Python version to use (default: '3.12')

**Outputs:**
- `test-status`: Status of the smoke tests
- `test-report`: Path to test report

## Usage Example

```yaml
jobs:
  detect-changes:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: ./.github/actions/detect-changes
        with:
          github-event-name: ${{ github.event_name }}
          github-ref: ${{ github.ref }}

  validate:
    needs: detect-changes
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: ./.github/actions/validate-config
        with:
          target-environment: ${{ needs.detect-changes.outputs.target-environment }}

  deploy:
    needs: [detect-changes, validate]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: ./.github/actions/deploy-config
        with:
          target-environment: ${{ needs.detect-changes.outputs.target-environment }}
          deployment-strategy: ${{ needs.detect-changes.outputs.deployment-strategy }}

  smoke-tests:
    needs: [detect-changes, deploy]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: ./.github/actions/smoke-tests
        with:
          target-environment: ${{ needs.detect-changes.outputs.target-environment }}
          deployment-url: ${{ needs.deploy.outputs.deployment-url }}
```

## Benefits

1. **Modularity**: Each action focuses on a specific responsibility
2. **Reusability**: Actions can be used in multiple workflows
3. **Maintainability**: Easier to update and test individual components
4. **Reduced Duplication**: Common logic is centralized
5. **Better Testing**: Each action can be tested independently

## Contributing

When creating new composite actions:

1. Keep actions focused on a single responsibility
2. Document all inputs and outputs clearly
3. Use semantic versioning for breaking changes
4. Include error handling and validation
5. Test actions thoroughly before merging