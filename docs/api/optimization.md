# Performance Optimization Agent (POA) API

The Performance Optimization Agent provides real-time performance monitoring, optimization recommendations, and automatic tuning capabilities.

## Endpoints

### Get POA Status

```http
GET /api/v1/poa/status
```

Retrieves the current status of the Performance Optimization Agent.

**Response:**

```json
{
  "status": "running",
  "mode": "monitoring",
  "uptime_seconds": 3600,
  "optimizations_applied": 42,
  "current_metrics": {
    "cpu_usage": 45.2,
    "memory_usage": 62.8,
    "response_time_p95": 98.5
  }
}
```

### Start POA

```http
POST /api/v1/poa/start
```

Starts the Performance Optimization Agent with specified configuration.

**Request Body:**

```json
{
  "mode": "aggressive",
  "targets": {
    "response_time_p95": 100,
    "throughput_rps": 1000
  },
  "auto_tune": true
}
```

**Response:**

```json
{
  "message": "Performance Optimization Agent started",
  "agent_id": "poa_123456",
  "mode": "aggressive"
}
```

### Stop POA

```http
POST /api/v1/poa/stop
```

Stops the Performance Optimization Agent.

**Response:**

```json
{
  "message": "Performance Optimization Agent stopped",
  "optimizations_retained": true
}
```

### Get Performance Metrics

```http
GET /api/v1/poa/metrics
```

Retrieves detailed performance metrics.

**Query Parameters:**

- `time_range`: Time range for metrics (e.g., "1h", "24h", "7d")
- `metrics`: Comma-separated list of metrics to retrieve

**Response:**

```json
{
  "time_range": "1h",
  "metrics": {
    "response_times": {
      "p50": 45.2,
      "p95": 98.5,
      "p99": 145.3,
      "avg": 52.1
    },
    "throughput": {
      "current_rps": 850,
      "peak_rps": 1250,
      "avg_rps": 720
    },
    "error_rates": {
      "rate": 0.02,
      "count": 15
    },
    "resource_usage": {
      "cpu": 45.2,
      "memory": 62.8,
      "connections": 120
    }
  }
}
```

### Get Optimization Recommendations

```http
GET /api/v1/poa/recommendations
```

Retrieves AI-generated optimization recommendations.

**Response:**

```json
{
  "recommendations": [
    {
      "id": "rec_001",
      "priority": "high",
      "category": "caching",
      "title": "Enable Redis caching for vector search results",
      "description": "Implement caching layer for frequently accessed vectors",
      "expected_improvement": {
        "metric": "response_time_p95",
        "current": 98.5,
        "projected": 45.0,
        "improvement_percent": 54.3
      },
      "implementation": {
        "difficulty": "medium",
        "estimated_hours": 4,
        "code_changes": true
      }
    },
    {
      "id": "rec_002",
      "priority": "medium",
      "category": "connection_pooling",
      "title": "Increase Qdrant connection pool size",
      "description": "Current pool size is causing connection bottlenecks",
      "expected_improvement": {
        "metric": "throughput_rps",
        "current": 850,
        "projected": 1100,
        "improvement_percent": 29.4
      },
      "implementation": {
        "difficulty": "easy",
        "estimated_hours": 1,
        "code_changes": false
      }
    }
  ],
  "total_recommendations": 5,
  "estimated_total_improvement": 72.5
}
```

### Apply Optimization

```http
POST /api/v1/poa/optimize/{recommendation_id}
```

Applies a specific optimization recommendation.

**Path Parameters:**

- `recommendation_id`: ID of the recommendation to apply

**Request Body:**

```json
{
  "apply_immediately": true,
  "rollback_on_failure": true,
  "monitoring_duration": 300
}
```

**Response:**

```json
{
  "status": "applied",
  "recommendation_id": "rec_001",
  "metrics_before": {
    "response_time_p95": 98.5
  },
  "monitoring": {
    "duration_seconds": 300,
    "status": "active"
  }
}
```

### Get Optimization History

```http
GET /api/v1/poa/history
```

Retrieves the history of applied optimizations.

**Query Parameters:**

- `limit`: Number of records to return (default: 50)
- `offset`: Offset for pagination

**Response:**

```json
{
  "optimizations": [
    {
      "id": "opt_123",
      "recommendation_id": "rec_001",
      "applied_at": "2025-01-09T10:30:00Z",
      "status": "successful",
      "metrics_improvement": {
        "response_time_p95": {
          "before": 98.5,
          "after": 47.2,
          "improvement_percent": 52.1
        }
      }
    }
  ],
  "total": 42,
  "limit": 50,
  "offset": 0
}
```

## Configuration

### Performance Targets

Configure performance targets for automatic optimization:

```json
{
  "targets": {
    "response_time_p50": 50,
    "response_time_p95": 100,
    "response_time_p99": 200,
    "throughput_rps": 1000,
    "error_rate": 0.01,
    "cpu_usage": 70,
    "memory_usage": 80
  }
}
```

### Optimization Modes

- **monitoring**: Collect metrics only, no automatic changes
- **conservative**: Apply only low-risk optimizations
- **balanced**: Apply medium-risk optimizations with monitoring
- **aggressive**: Apply all optimizations to meet targets

## Webhooks

Configure webhooks for optimization events:

```json
{
  "webhook_url": "https://your-app.com/poa-webhook",
  "events": [
    "optimization_applied",
    "target_breached",
    "recommendation_available"
  ]
}
```

## Best Practices

1. **Start Conservative**: Begin with monitoring mode to establish baselines
2. **Set Realistic Targets**: Use historical data to set achievable targets
3. **Monitor After Changes**: Keep POA active for at least 24h after optimizations
4. **Review Recommendations**: Not all recommendations suit every use case
5. **Test in Staging**: Apply optimizations to staging environment first

## Error Codes

- `POA_NOT_RUNNING`: POA is not currently active
- `INVALID_RECOMMENDATION`: Recommendation ID not found
- `OPTIMIZATION_FAILED`: Failed to apply optimization
- `ROLLBACK_INITIATED`: Optimization caused issues, rolling back