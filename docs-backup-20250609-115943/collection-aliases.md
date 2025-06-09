# Collection Aliases and Zero-Downtime Deployments

> **Status**: Current  
> **Last Updated**: 2025-06-09  
> **Purpose**: Collection Aliases documentation  
> **Audience**: Developers

This guide covers the collection alias system that enables zero-downtime deployments, A/B testing, and canary rollouts for your vector database.

## Overview

Collection aliases provide:

- **Zero-downtime deployments**: Instantly switch between collection versions
- **Safe rollbacks**: Keep the old version until confident
- **A/B testing**: Test changes on real traffic
- **Canary deployments**: Gradual rollout with health monitoring

## Available MCP Tools

### Alias Management

#### search_with_alias

Search using a collection alias instead of a direct collection name.

```bash
# In Claude Desktop/Code
"Search the documentation alias for vector database optimization"
```

#### list_aliases

View all configured aliases and their target collections.

```bash
"Show me all collection aliases"
```

#### create_alias

Create or update an alias to point to a collection.

```bash
"Create an alias 'production' pointing to 'docs_v2' collection"
```

### Deployment Patterns

#### deploy_new_index

Deploy a new collection version with zero downtime using blue-green deployment.

```bash
"Deploy a new version of the documentation index from the latest crawl"
```

The deployment process:

1. Creates a new collection with timestamp suffix
2. Populates it from the specified source
3. Runs validation queries
4. Atomically switches the alias
5. Monitors for issues
6. Schedules old collection cleanup

#### start_ab_test

Start an A/B test between two collections.

```bash
"Start an A/B test between 'docs_v1' and 'docs_v2' with 20% traffic to v2"
```

Features:

- Deterministic user routing
- Automatic metric collection (latency, relevance, clicks)
- Statistical analysis with p-values
- Effect size calculation

#### analyze_ab_test

Get statistical analysis of an A/B test.

```bash
"Analyze the results of experiment exp_embeddings_test_1234567890"
```

#### start_canary_deployment

Start a gradual rollout with health monitoring.

```bash
"Start a canary deployment for the 'documentation' alias to 'docs_v3' collection"
```

Default stages:

- 5% traffic for 30 minutes
- 25% traffic for 60 minutes
- 50% traffic for 120 minutes
- 100% traffic

#### get_canary_status

Check the progress of a canary deployment.

```bash
"What's the status of canary deployment canary_1234567890?"
```

#### pause_canary / resume_canary

Control canary deployment progression.

```bash
"Pause the canary deployment canary_1234567890"
"Resume the canary deployment canary_1234567890"
```

## Usage Examples

### Blue-Green Deployment

```python
# Deploy new documentation version
result = await deploy_new_index(
    alias="documentation",
    source="collection:docs_v1",  # Copy from existing
    validation_queries=[
        "python asyncio",
        "react hooks",
        "fastapi authentication"
    ],
    rollback_on_failure=True
)
```

### A/B Testing Workflow

```python
# 1. Start experiment
experiment_id = await start_ab_test(
    experiment_name="new_embeddings",
    control_collection="docs_openai",
    treatment_collection="docs_bge",
    traffic_split=0.2,  # 20% to treatment
    metrics=["latency", "relevance", "clicks"]
)

# 2. Route queries through experiment
for query in user_queries:
    variant, results = await ab_testing.route_query(
        experiment_id=experiment_id,
        query_vector=query_embedding,
        user_id=user_id
    )
    
    # Track user feedback
    await ab_testing.track_feedback(
        experiment_id=experiment_id,
        variant=variant,
        metric="clicks",
        value=1.0 if clicked else 0.0
    )

# 3. Analyze results
analysis = await analyze_ab_test(experiment_id)
# Returns:
# {
#     "metrics": {
#         "latency": {
#             "control_mean": 45.2,
#             "treatment_mean": 38.7,
#             "improvement": -14.4%,
#             "p_value": 0.023,
#             "significant": true
#         }
#     }
# }
```

### Canary Deployment Example

```python
# Custom canary stages for faster rollout
stages = [
    {"percentage": 10, "duration_minutes": 15},
    {"percentage": 50, "duration_minutes": 30},
    {"percentage": 100, "duration_minutes": 0}
]

deployment_id = await start_canary_deployment(
    alias="production",
    new_collection="docs_2024_01_15",
    stages=stages,
    auto_rollback=True
)

# Monitor progress
status = await get_canary_status(deployment_id)
# {
#     "current_stage": 1,
#     "current_percentage": 50,
#     "avg_latency": 42.3,
#     "avg_error_rate": 0.001
# }
```

## Best Practices

### 1. Validation Queries

Always include representative validation queries that cover:

- Common search patterns
- Edge cases
- Different content types
- Various query lengths

### 2. Monitoring Thresholds

Set appropriate thresholds for canary deployments:

- Error rate: < 5% (default)
- Latency: < 200ms (default)
- Adjust based on your SLAs

### 3. Rollback Strategy

- Always enable `rollback_on_failure` for production
- Keep old collections for at least 24 hours
- Test rollback procedures regularly

### 4. A/B Test Design

- Run tests for statistical significance (minimum 100 samples per variant)
- Use consistent user assignment for better results
- Track multiple metrics to avoid optimization bias

### 5. Collection Naming

Use descriptive naming conventions:

- `docs_v1`, `docs_v2` for versions
- `docs_2024_01_15` for dated releases
- `docs_experimental_bge` for feature branches

## Architecture

### Alias Resolution Flow

```plaintext
User Query → Alias → Current Collection → Search Results
             ↓
        Alias Manager
             ↓
        Collection Mapping
```

### Blue-Green Pattern

```plaintext
Alias: "production"
  ├─ Blue (Current): docs_v1 ← Live Traffic
  └─ Green (New): docs_v2 ← Building & Testing
                    ↓
              Validation Pass
                    ↓
              Atomic Switch
                    ↓
  ├─ Blue (Old): docs_v1 ← Scheduled Deletion
  └─ Green (Current): docs_v2 ← Live Traffic
```

### Canary Stages

```plaintext
Stage 1: 5% → Monitor → Health Check → Continue/Rollback
Stage 2: 25% → Monitor → Health Check → Continue/Rollback
Stage 3: 50% → Monitor → Health Check → Continue/Rollback
Stage 4: 100% → Complete
```

## Troubleshooting

### Deployment Failures

If a deployment fails:

1. Check validation query results
2. Verify collection schema compatibility
3. Review error logs
4. Manual rollback: `switch_alias(alias, old_collection)`

### A/B Test Issues

- Ensure both collections have the same schema
- Verify traffic split adds up to 1.0
- Check metric tracking is working

### Canary Stuck

If canary deployment is stuck:

1. Check current status
2. Review health metrics
3. Manually pause/resume
4. Force completion if needed

## Performance Considerations

- Alias resolution adds < 1ms overhead
- Collection switching is atomic (instant)
- No downtime during deployments
- Parallel collection building supported
