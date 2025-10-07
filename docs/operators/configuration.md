# Configuration Guide

## Environment Variables

### Required Configuration
```bash
# Core API Keys
export AI_DOCS__OPENAI__API_KEY=${OPENAI_API_KEY}
export AI_DOCS__FIRECRAWL__API_KEY=${FIRECRAWL_API_KEY}

# Database URLs
export AI_DOCS__CACHE__DRAGONFLY_URL="redis://dragonfly:6379"
export AI_DOCS__QDRANT__URL="http://qdrant:6333"
export AI_DOCS__QDRANT__COLLECTION_NAME="documents"

# Environment
export AI_DOCS__ENVIRONMENT=production
export AI_DOCS_CONFIG_PATH="config/production.json"
```

### Production Settings
```bash
export AI_DOCS__DEBUG=false
export AI_DOCS__LOG_LEVEL=INFO
export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=20
export AI_DOCS__PERFORMANCE__REQUEST_TIMEOUT=30
export AI_DOCS__PERFORMANCE__BATCH_SIZE=100
```

### Development Settings
```bash
export AI_DOCS__ENVIRONMENT=development
export AI_DOCS__DEBUG=true
export AI_DOCS__LOG_LEVEL=DEBUG
export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=5
```

## Docker Compose Configuration

### Production docker-compose.yml
```yaml
services:
  qdrant:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
        reservations:
          memory: 2G
          cpus: "1.0"
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
      QDRANT__LOG_LEVEL: INFO

  dragonfly:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"
    command: ["dragonfly", "--maxmemory=1gb", "--save_schedule=*/10"]
```

## Configuration Files

### config/production.json
```json
{
  "environment": "production",
  "debug": false,
  "log_level": "INFO",
  "performance": {
    "max_concurrent_requests": 20,
    "request_timeout": 30,
    "batch_size": 100,
    "cache_ttl": 3600
  },
  "qdrant": {
    "collection_name": "documents",
    "collection_config": {
      "vector_size": 1536,
      "distance": "Cosine",
      "hnsw_config": {
        "m": 16,
        "ef_construct": 100
      }
    }
  }
}
```

### config/development.json
```json
{
  "environment": "development",
  "debug": true,
  "log_level": "DEBUG",
  "performance": {
    "max_concurrent_requests": 5,
    "request_timeout": 60,
    "batch_size": 50
  }
}
```

## Security Configuration

### API Authentication
```bash
# Generate secure API keys
API_KEY=$(openssl rand -hex 32)
export AI_DOCS__API__SECRET_KEY="${API_KEY}"

# Rate limiting
export AI_DOCS__RATE_LIMIT__REQUESTS_PER_MINUTE=60
export AI_DOCS__RATE_LIMIT__BURST_SIZE=10
```

### TLS Configuration
```bash
# Enable TLS
export AI_DOCS__TLS__ENABLED=true
export AI_DOCS__TLS__CERT_PATH="/etc/ssl/certs/ai-docs.crt"
export AI_DOCS__TLS__KEY_PATH="/etc/ssl/private/ai-docs.key"
```

## Performance Tuning

### Vector Database Optimization
```bash
# Qdrant configuration
export QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32
export QDRANT__SERVICE__MAX_WORKERS=4
export QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=2
```

### Cache Optimization
```bash
# DragonflyDB configuration
export DRAGONFLY__MAXMEMORY=2gb
export DRAGONFLY__MAXMEMORY_POLICY=allkeys-lru
export DRAGONFLY__SAVE_SCHEDULE="*/10"
```

## Monitoring Configuration

### Metrics Collection
```bash
export AI_DOCS__METRICS__ENABLED=true
export AI_DOCS__METRICS__PORT=9090
export AI_DOCS__METRICS__PATH="/metrics"
```

### Logging Configuration
```bash
export AI_DOCS__LOGGING__FORMAT=json
export AI_DOCS__LOGGING__LEVEL=INFO
export AI_DOCS__LOGGING__FILE_PATH="/var/log/ai-docs/app.log"
export AI_DOCS__LOGGING__MAX_SIZE=100MB
export AI_DOCS__LOGGING__BACKUP_COUNT=5
```

## Configuration Validation

### Verify Configuration
```bash
# Check environment variables
env | grep AI_DOCS__

# Validate configuration file
python -c "import json; json.load(open('config/production.json'))"

# Test database connections
curl -s http://localhost:6333/health
redis-cli ping
```

### Configuration Reload
```bash
# Reload configuration without restart
docker-compose kill -s SIGHUP api

# Full restart with new configuration
docker-compose down
docker-compose up -d
```
