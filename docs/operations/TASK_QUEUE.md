# Task Queue Operations Guide

This guide covers the persistent task queue system implemented using ARQ for background job processing.

## Overview

The system uses ARQ (Async Redis Queue) for persistent background task execution. This ensures critical operations like collection deletion, cache persistence, and deployment orchestration continue even if the application restarts.

## Architecture

### Components

1. **Task Queue Manager** (`src/services/task_queue/manager.py`)
   - Manages Redis connection pool
   - Enqueues jobs with optional delays
   - Provides job status and monitoring

2. **Task Functions** (`src/services/task_queue/tasks.py`)
   - `delete_collection`: Delayed deletion of Qdrant collections
   - `persist_cache`: Write-behind cache persistence
   - `run_canary_deployment`: Canary deployment orchestration

3. **Worker Process** (`src/services/task_queue/worker.py`)
   - Processes queued tasks
   - Configurable concurrency and retry settings

## Configuration

Task queue settings in `UnifiedConfig`:

```yaml
task_queue:
  redis_url: "redis://localhost:6379"  # DragonflyDB URL
  redis_database: 1                     # Dedicated DB for tasks
  max_jobs: 10                          # Concurrent jobs per worker
  job_timeout: 3600                     # Default timeout (seconds)
  job_ttl: 86400                        # Result TTL (24 hours)
  max_tries: 3                          # Retry attempts
  retry_delay: 60                       # Delay between retries
  queue_name: "default"                 # Queue name
  worker_pool_size: 4                   # Number of workers
```

## Running the Worker

### Development

```bash
# Using the CLI script
./scripts/start-worker.sh

# Or using the Python command
uv run task-worker

# With custom settings
uv run task-worker --workers 2 --max-jobs 20 --queue high_priority
```

### Production (Docker)

```bash
# Run with Docker Compose
docker-compose --profile worker up task-worker

# Or use the helper script
./scripts/run-worker-docker.sh
```

### Production (Systemd)

Create `/etc/systemd/system/ai-docs-task-worker.service`:

```ini
[Unit]
Description=AI Docs Task Queue Worker
After=network.target redis.service

[Service]
Type=simple
User=app
WorkingDirectory=/app
Environment="PYTHONPATH=/app/src"
ExecStart=/app/.venv/bin/arq src.services.task_queue.worker.WorkerSettings
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Task Usage Examples

### Collection Deletion

```python
# In QdrantAliasManager
await self.safe_delete_collection("old_collection", grace_period_minutes=60)
```

This schedules deletion after 60 minutes, surviving server restarts.

### Cache Write-Behind

```python
# In CachePatterns
await cache_patterns.write_behind(
    key="user:123",
    value={"name": "John", "email": "john@example.com"},
    persist_func=save_to_database,
    delay=5.0  # Persist after 5 seconds
)
```

### Canary Deployment

```python
# In CanaryDeployment
deployment_id = await canary.start_canary(
    alias_name="production",
    new_collection="v2_collection",
    stages=[
        {"percentage": 10, "duration_minutes": 30},
        {"percentage": 50, "duration_minutes": 60},
        {"percentage": 100, "duration_minutes": 0}
    ]
)
```

## Monitoring

### Check Worker Health

```bash
# Check if worker is processing
redis-cli -n 1 ping

# View worker info
redis-cli -n 1 info
```

### Job Status

```python
# Get job status
manager = TaskQueueManager(config)
await manager.initialize()

status = await manager.get_job_status("job_id_123")
print(f"Job status: {status['status']}")
print(f"Result: {status.get('result')}")
```

### Queue Statistics

```python
stats = await manager.get_queue_stats()
print(f"Pending jobs: {stats['pending']}")
print(f"Running jobs: {stats['running']}")
```

## Error Handling

### Automatic Retries

Failed jobs are automatically retried based on configuration:
- Max retries: 3 (configurable)
- Retry delay: 60 seconds (exponential backoff)

### Dead Letter Queue

Failed jobs after max retries are stored for investigation:

```bash
# View failed jobs
redis-cli -n 1 ZRANGE arq:dead 0 -1
```

### Manual Retry

```python
# Retry a failed job
await manager.retry_job("failed_job_id")
```

## Best Practices

1. **Idempotency**: Design tasks to be safely retryable
2. **Timeouts**: Set appropriate timeouts for long-running tasks
3. **Monitoring**: Use health checks and alerts
4. **Graceful Shutdown**: Workers complete current jobs before stopping
5. **Resource Limits**: Configure memory and CPU limits in production

## Troubleshooting

### Worker Not Starting

1. Check Redis/DragonflyDB connectivity:
   ```bash
   redis-cli -h localhost -p 6379 ping
   ```

2. Verify configuration:
   ```bash
   python -c "from src.config.loader import load_config; print(load_config().task_queue)"
   ```

### Jobs Not Processing

1. Check worker logs:
   ```bash
   docker logs task-worker
   ```

2. Verify queue has jobs:
   ```bash
   redis-cli -n 1 LLEN arq:queue:default
   ```

### Memory Issues

1. Monitor worker memory:
   ```bash
   docker stats task-worker
   ```

2. Adjust worker pool size and max jobs:
   ```yaml
   task_queue:
     worker_pool_size: 2  # Reduce workers
     max_jobs: 5          # Reduce concurrent jobs
   ```

## Migration from asyncio.create_task

The system has been migrated from `asyncio.create_task` to provide:

1. **Persistence**: Jobs survive server restarts
2. **Reliability**: Automatic retries and error handling
3. **Scalability**: Multiple workers can process jobs
4. **Monitoring**: Job status and queue statistics
5. **Flexibility**: Delayed execution and priority queues

All critical background operations now use the task queue:
- Collection deletion (60-minute grace period)
- Cache persistence (write-behind pattern)
- Deployment orchestration (canary/blue-green)

## Performance Tuning

### Redis/DragonflyDB

```yaml
# docker-compose.yml
dragonfly:
  environment:
    - DRAGONFLY_THREADS=8
    - DRAGONFLY_MEMORY_LIMIT=4gb
```

### Worker Concurrency

```yaml
task_queue:
  worker_pool_size: 4   # Number of worker processes
  max_jobs: 10          # Jobs per worker
  # Total concurrent jobs = 4 * 10 = 40
```

### Job Priorities

Use different queues for priority:

```python
# High priority
await manager.enqueue("task", _queue_name="high_priority")

# Low priority
await manager.enqueue("task", _queue_name="low_priority")
```

Run dedicated workers per queue:

```bash
# High priority worker
uv run task-worker --queue high_priority --workers 4

# Low priority worker  
uv run task-worker --queue low_priority --workers 1
```