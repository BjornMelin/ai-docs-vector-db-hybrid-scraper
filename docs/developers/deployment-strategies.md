# Enterprise Deployment Strategies

> **Status**: Active  
> **Last Updated**: 2025-01-16  
> **Purpose**: Advanced deployment patterns for enterprise environments  
> **Audience**: DevOps engineers and system architects

This guide covers enterprise-grade deployment strategies available in the AI Documentation Vector DB system, 
including A/B testing, blue-green deployments, canary releases, and feature flag management.

## 🚀 Overview

The deployment module provides sophisticated deployment patterns controlled by feature flags, enabling:

- **Zero-downtime deployments**
- **Progressive traffic routing**
- **A/B testing with statistical analysis**
- **Feature flag-based tier control**
- **Automated rollback capabilities**

## 📊 A/B Testing

### Overview

A/B testing allows you to compare two versions of your deployment simultaneously with traffic splitting 
and statistical analysis.

```python
from src.services.deployment import ABTestingManager
from src.config.enums import ABTestVariant

# Initialize A/B test manager
ab_manager = ABTestingManager(config)

# Create a new A/B test
test_id = await ab_manager.create_test(
    name="embedding-model-comparison",
    description="Compare ada-002 vs text-embedding-3-small",
    variant_a_config={
        "embedding_model": "text-embedding-ada-002",
        "chunk_size": 512
    },
    variant_b_config={
        "embedding_model": "text-embedding-3-small", 
        "chunk_size": 1024
    },
    traffic_split=0.5,  # 50/50 split
    minimum_sample_size=1000
)
```

### Tracking Metrics

```python
# Record conversion for a variant
await ab_manager.record_conversion(
    test_id=test_id,
    variant=ABTestVariant.VARIANT_B,
    user_id="user123",
    conversion_value=1.0
)

# Get current test results
results = await ab_manager.get_test_results(test_id)
print(f"Variant A: {results.variant_a_conversions} conversions")
print(f"Variant B: {results.variant_b_conversions} conversions")
print(f"Statistical significance: {results.significance_level}")
```

## 🔄 Blue-Green Deployments

### Overview

Blue-green deployments maintain two identical production environments, allowing instant switching 
between versions with zero downtime.

```python
from src.services.deployment import BlueGreenDeployment

# Initialize blue-green deployment
bg_deployment = BlueGreenDeployment(config)

# Create deployment
deployment_id = await bg_deployment.create_deployment(
    blue_config={
        "version": "v1.0.0",
        "collection": "docs_v1"
    },
    green_config={
        "version": "v1.1.0",
        "collection": "docs_v2"
    },
    health_check_interval=30,  # seconds
    rollback_threshold=0.95    # 95% health required
)

# Switch to green environment
await bg_deployment.switch_to_green(
    deployment_id=deployment_id,
    validate_health=True
)
```

### Health Monitoring

```python
# Monitor deployment health
health = await bg_deployment.get_health_status(deployment_id)
print(f"Blue health: {health.blue_health}%")
print(f"Green health: {health.green_health}%")
print(f"Active environment: {health.active_environment}")

# Automatic rollback on failure
if health.green_health < 0.95:
    await bg_deployment.rollback(deployment_id)
```

## 🕯️ Canary Deployments

### Overview

Canary deployments gradually roll out changes to a small subset of users before full deployment.

```python
from src.services.deployment import CanaryDeployment

# Initialize canary deployment
canary = CanaryDeployment(config)

# Create canary deployment
deployment = await canary.create_deployment(
    name="vector-search-optimization",
    stages=[
        {"traffic": 0.05, "duration_minutes": 30},   # 5% for 30 min
        {"traffic": 0.25, "duration_minutes": 60},   # 25% for 1 hour
        {"traffic": 0.50, "duration_minutes": 120},  # 50% for 2 hours
        {"traffic": 1.00, "duration_minutes": None}  # 100% final
    ],
    success_criteria={
        "error_rate": {"max": 0.01},      # < 1% errors
        "latency_p99": {"max": 500},      # < 500ms p99
        "success_rate": {"min": 0.99}     # > 99% success
    }
)

# Monitor canary progress
status = await canary.get_deployment_status(deployment.id)
print(f"Current stage: {status.current_stage}")
print(f"Traffic percentage: {status.traffic_percentage}%")
print(f"Health score: {status.health_score}")
```

### Automated Progression

```python
# Enable auto-progression based on metrics
await canary.enable_auto_progression(
    deployment_id=deployment.id,
    monitoring_interval=60,  # Check every minute
    require_manual_approval=True  # For final stage
)

# Manual promotion/rollback
if status.health_score > 0.95:
    await canary.promote_to_next_stage(deployment.id)
else:
    await canary.rollback(deployment.id)
```

## 🏴 Feature Flags

### Overview

Feature flags provide fine-grained control over feature availability based on deployment tiers.

```python
from src.services.deployment import FeatureFlagManager
from src.config.deployment_tiers import DeploymentTier

# Initialize feature flag manager
flags = FeatureFlagManager(config)

# Define feature flags
await flags.create_flag(
    name="advanced-hybrid-search",
    description="Enable advanced hybrid search with SPLADE",
    enabled_tiers=[
        DeploymentTier.ENTERPRISE,
        DeploymentTier.PROFESSIONAL
    ],
    rollout_percentage=1.0,
    metadata={
        "requires": ["splade-model", "gpu-acceleration"],
        "impact": "high"
    }
)

# Check feature availability
if await flags.is_enabled(
    "advanced-hybrid-search", 
    tier=DeploymentTier.PROFESSIONAL
):
    # Use advanced search
    search_service = AdvancedHybridSearchService(config)
else:
    # Use standard search
    search_service = StandardSearchService(config)
```

### Dynamic Feature Control

```python
# Update feature rollout
await flags.update_rollout(
    flag_name="advanced-hybrid-search",
    rollout_percentage=0.5,  # 50% of eligible users
    sticky_sessions=True     # Consistent per user
)

# Emergency kill switch
await flags.disable_flag(
    "advanced-hybrid-search",
    reason="Performance degradation detected"
)

# Get all active features for a tier
active_features = await flags.get_active_features(
    tier=DeploymentTier.ENTERPRISE
)
```

## 🎯 Integration Examples

### Combined Deployment Strategy

```python
# Use canary deployment with feature flags
async def deploy_with_canary_and_flags():
    # Create canary deployment
    canary_id = await canary.create_deployment(
        name="ml-security-features",
        stages=[
            {"traffic": 0.1, "duration_minutes": 60},
            {"traffic": 0.5, "duration_minutes": 120},
            {"traffic": 1.0, "duration_minutes": None}
        ]
    )
    
    # Enable feature flag for canary users
    await flags.create_flag(
        name="ml-security-validator",
        enabled_tiers=[DeploymentTier.ENTERPRISE],
        rollout_percentage=0.0  # Start at 0%
    )
    
    # Gradually increase feature flag as canary progresses
    async def update_feature_rollout(stage: int, traffic: float):
        await flags.update_rollout(
            flag_name="ml-security-validator",
            rollout_percentage=traffic
        )
    
    # Monitor and progress
    while True:
        status = await canary.get_deployment_status(canary_id)
        
        if status.current_stage > previous_stage:
            await update_feature_rollout(
                status.current_stage,
                status.traffic_percentage / 100
            )
        
        if status.is_complete:
            break
            
        await asyncio.sleep(60)
```

### A/B Test with Blue-Green Fallback

```python
async def ab_test_with_fallback():
    # Run A/B test in green environment
    bg_deployment = await blue_green.create_deployment(
        blue_config={"version": "stable"},
        green_config={"version": "experimental"}
    )
    
    # Create A/B test in green
    ab_test = await ab_manager.create_test(
        name="embedding-optimization",
        variant_a_config={"model": "current"},
        variant_b_config={"model": "optimized"},
        environment="green"
    )
    
    # Monitor test results
    while not ab_test.is_complete:
        results = await ab_manager.get_test_results(ab_test.id)
        
        # Check for degradation
        if results.variant_b_error_rate > 0.05:
            # Rollback to blue
            await blue_green.switch_to_blue(bg_deployment.id)
            await ab_manager.stop_test(ab_test.id)
            break
        
        await asyncio.sleep(300)  # Check every 5 minutes
```

## 📈 Monitoring and Observability

### Deployment Metrics

```python
from src.services.deployment.models import DeploymentMetrics

# Get comprehensive deployment metrics
metrics = await deployment_monitor.get_metrics(
    deployment_id=deployment.id,
    time_range="1h"
)

print(f"Request rate: {metrics.request_rate}/s")
print(f"Error rate: {metrics.error_rate}%")
print(f"P99 latency: {metrics.latency_p99}ms")
print(f"Active users: {metrics.active_users}")
```

### Health Checks

```python
# Configure health checks
health_config = {
    "endpoints": [
        "/health",
        "/api/v1/search",
        "/api/v1/collections"
    ],
    "interval_seconds": 30,
    "timeout_seconds": 5,
    "success_threshold": 0.95
}

# Monitor health across deployments
health_status = await deployment_monitor.check_health(
    deployment_ids=[canary_id, bg_deployment_id],
    config=health_config
)
```

## 🛡️ Safety and Rollback

### Automated Rollback Policies

```python
# Define rollback policies
rollback_policy = {
    "error_rate_threshold": 0.05,      # 5% errors
    "latency_increase": 1.5,           # 50% latency increase  
    "success_rate_minimum": 0.95,      # 95% success rate
    "evaluation_window": 300,          # 5 minutes
    "consecutive_failures": 3          # 3 failed checks
}

# Apply to deployment
await deployment_manager.set_rollback_policy(
    deployment_id=deployment.id,
    policy=rollback_policy,
    auto_rollback=True
)
```

### Manual Controls

```python
# Emergency stop
await deployment_manager.emergency_stop(
    deployment_id=deployment.id,
    reason="Critical bug discovered"
)

# Pause deployment
await deployment_manager.pause(
    deployment_id=deployment.id,
    duration_minutes=30
)

# Resume with modified parameters
await deployment_manager.resume(
    deployment_id=deployment.id,
    modified_config={
        "traffic_percentage": 0.1,  # Reduce to 10%
        "monitoring_interval": 60   # More frequent checks
    }
)
```

## 🔧 Configuration

### Environment Variables

```bash
# Deployment configuration
DEPLOYMENT_STRATEGY=canary
DEPLOYMENT_TIER=enterprise
FEATURE_FLAGS_ENABLED=true

# A/B Testing
AB_TEST_MINIMUM_SAMPLE_SIZE=1000
AB_TEST_CONFIDENCE_LEVEL=0.95

# Blue-Green
BLUE_GREEN_HEALTH_CHECK_INTERVAL=30
BLUE_GREEN_ROLLBACK_THRESHOLD=0.95

# Canary
CANARY_INITIAL_TRAFFIC=0.05
CANARY_MONITORING_INTERVAL=60
CANARY_AUTO_PROGRESSION=true
```

### Tier-Based Configuration

```python
# Configure deployment features per tier
deployment_config = {
    DeploymentTier.PERSONAL: {
        "strategies": [],  # No advanced deployments
        "feature_flags": False,
        "monitoring": "basic"
    },
    DeploymentTier.PROFESSIONAL: {
        "strategies": ["blue-green"],
        "feature_flags": True,
        "monitoring": "enhanced"
    },
    DeploymentTier.ENTERPRISE: {
        "strategies": ["blue-green", "canary", "ab-testing"],
        "feature_flags": True,
        "monitoring": "comprehensive"
    }
}
```

## 📚 Best Practices

1. **Start Small**: Begin with canary deployments at 5% traffic
2. **Monitor Actively**: Set up comprehensive monitoring before deployment
3. **Define Clear Criteria**: Establish success/failure metrics upfront
4. **Automate Rollbacks**: Configure automatic rollback policies
5. **Test in Staging**: Always test deployment strategies in non-production
6. **Document Changes**: Maintain deployment runbooks and changelogs
7. **Gradual Rollout**: Use progressive traffic increases
8. **Feature Flag First**: Test new features behind flags before full deployment

## 🔗 Related Documentation

- [Configuration Management](./configuration.md)
- [Performance Benchmarking](./benchmarking-and-performance.md)
- [Security Architecture](../operators/security.md)
- [Monitoring Guide](../operators/monitoring.md)