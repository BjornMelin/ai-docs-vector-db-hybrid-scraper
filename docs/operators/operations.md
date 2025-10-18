
# Operations Guide

## Agentic Workflow Runbooks

### Browser Automation Tiers
- The unified manager (`src/services/browser/unified_manager.py`) orders tiers as: HTTP fetcher → lightweight headless Chromium → Crawl4AI → Browser-use → Playwright with supervised escalation.
- Configure limits under `config/browser.yml` (`max_parallel_sessions`, `retry_budget`).
- Use `python scripts/dev.py services --action status --stack browser` before releases to confirm tier readiness.
- When a tier flaps, disable it via `BROWSER__DISABLE_<TIER>` environment flags and watch the `*_browser_tier_health_status` gauge until stable.

### Crawling Strategy
- Tier routing settings live in `config/crawling.yml`; adjust thresholds instead of editing code.
- Each tier emits `crawler_tier_health` and `crawler_request_latency_seconds` metrics. Alert when health <0.9 for 10 minutes.
- Preload frontier queues with `python scripts/dev.py crawling prime --profile standard` after cache flushes.

### Retrieval and RAG Self-Healing
- Retry budgets derive from `agentic.max_retries` in `config/agentic.yml`; they apply separately per LangGraph stage (`discover`, `retrieve`, `execute`).
- Persist checkpoints by selecting a durable saver in the dependency wiring; checkpoints land under `storage/langgraph-checkpoints/`.
- Run `python scripts/dev.py benchmark --suite performance` before changing retrieval configuration.

### Vector Database Stewardship
- Nightly optimisation: `python scripts/dev.py vector optimize` merges Qdrant segments using thresholds in `config/vector_db.yml`.
- Monitor `qdrant_collection_optimizer_in_progress` during compaction; expect temporary retrieval latency increases.
- Adjust `vector_db.max_write_qps` ahead of bulk loads to keep `*_rag_stage_latency_seconds{stage="retrieve"}` under SLO limits.

## Daily Operations

### Daily Health Check
```bash
docker-compose ps
curl -s http://localhost:6333/health | jq '.'
redis-cli ping
df -h | grep -E "(/$|/var|/tmp)"
docker logs --since 24h qdrant | grep -i error
docker logs --since 24h dragonfly | grep -i error
redis-cli -n 1 info | grep -E "(connected_clients|used_memory_human)"
free -h && top -bn1 | head -5
```

### Log Management
```bash
find /var/log/ai-docs -name "*.log" -mtime +30 -delete
docker system prune -f --volumes --filter "until=24h"
tar -czf /backup/logs/daily-logs-$(date +%Y%m%d).tar.gz /var/log/ai-docs/
```

## Service Management

### Core Service Commands
```bash
# Start services
python scripts/dev.py services start
docker-compose up -d

# Stop services
docker-compose down --timeout 30

# Restart specific services
docker-compose restart qdrant
docker-compose restart dragonfly

# View logs
docker-compose logs -f --tail=100 qdrant
docker-compose logs -f --tail=100 dragonfly

# Resource monitoring
docker stats --no-stream
```

## Backup & Recovery

### Daily Backup
```bash
BACKUP_DATE=$(date +%Y%m%d)
BACKUP_DIR="/backup/daily/${BACKUP_DATE}"
mkdir -p ${BACKUP_DIR}

# Vector database backup
curl -X POST "http://localhost:6333/snapshots" \
  -H "Content-Type: application/json" \
  -d '{"collection_name": "documents"}'

# Configuration backup
cp -r /app/config ${BACKUP_DIR}/
cp docker-compose.yml ${BACKUP_DIR}/
cp .env ${BACKUP_DIR}/env.backup

# Cache backup
redis-cli save
cp /var/lib/redis/dump.rdb ${BACKUP_DIR}/redis-backup.rdb
```

### Recovery
```bash
RECOVERY_DATE=$1
BACKUP_DIR="/backup/daily/${RECOVERY_DATE}"

# Stop services
docker-compose down

# Restore vector database
docker run --rm -v ai-docs_qdrant_data:/data -v ${BACKUP_DIR}:/backup \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/qdrant-data.tar.gz -C /data"

# Restore cache
cp ${BACKUP_DIR}/redis-backup.rdb /var/lib/redis/dump.rdb

# Restart services
docker-compose up -d
```

## User Management

### User Operations
```bash
# Create user
USERNAME="newuser"
API_KEY=$(openssl rand -hex 32)
ROLE="viewer"
redis-cli hset "user:${USERNAME}" \
  "api_key" "${API_KEY}" \
  "role" "${ROLE}"

# List users
redis-cli keys "user:*" | sed 's/user://' | sort

# Disable user
redis-cli hset "user:username" "status" "disabled"

# Rotate API keys
NEW_KEY=$(openssl rand -hex 32)
redis-cli hset "user:username" "api_key" "${NEW_KEY}"
```

## Incident Response

### Common Incidents
```bash
# Database down
docker-compose restart qdrant

# Memory exhausted
docker-compose restart dragonfly
redis-cli flushall

# High load
docker-compose scale worker=3

# Service unresponsive
timeout 60 docker-compose down
docker kill $(docker ps -q)
docker system prune -f
docker-compose up -d
```

## Performance Management

### Resource Monitoring
```bash
# System metrics
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Vector database metrics
curl -s "http://localhost:6333/metrics" | grep -E "(qdrant_collections_total|qdrant_points_total)"

# Cache metrics
redis-cli info memory | grep -E "(used_memory_human|used_memory_peak_human)"
```

### Performance Optimization
```bash
# Scale workers
docker-compose scale worker=3

# Optimize cache
redis-cli config set maxmemory-policy allkeys-lru

# Optimize vector database
curl -X POST "http://localhost:6333/collections/documents/optimize"
```

## Troubleshooting

### Health Verification
```bash
# Service connectivity
docker exec api ping qdrant
docker exec api ping dragonfly

# API endpoints
curl -v http://localhost:8000/health
curl -v http://localhost:6333/health

# Port checks
netstat -tlnp | grep -E "(8000|6333|6379)"
```

### Common Fixes
```bash
# Clear cache
redis-cli flushall

# Restart services
docker-compose restart

# Clean resources
docker system prune -af

# Check logs
docker-compose logs --tail=50
```
