# Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the AI Documentation Vector DB Hybrid Scraper to production environments. The system is designed for production-ready deployment with enterprise-grade features.

## System Architecture

### Core Components

```mermaid
architecture-beta
    group api(cloud)[API Layer]
    group services(cloud)[Service Layer]
    group storage(database)[Storage Layer]
    group monitoring(shield)[Monitoring Layer]
    
    service fastapi(server)[FastAPI Server] in api
    service workers(server)[Background Workers] in api
    service mcp(server)[MCP Server] in api
    
    service embeddings(internet)[Embedding Service] in services
    service crawling(server)[Crawling Service] in services
    service search(database)[Search Service] in services
    service cache(disk)[Cache Service] in services
    
    service qdrant(database)[Qdrant Vector DB] in storage
    service dragonfly(disk)[DragonflyDB] in storage
    service postgres(database)[PostgreSQL] in storage
    
    service prometheus(shield)[Prometheus] in monitoring
    service grafana(shield)[Grafana] in monitoring
    service jaeger(shield)[Jaeger Tracing] in monitoring
    
    fastapi:B --> embeddings:T
    fastapi:B --> crawling:T
    fastapi:B --> search:T
    workers:B --> cache:T
    
    embeddings:R --> qdrant:L
    search:R --> qdrant:L
    cache:R --> dragonfly:L
    
    fastapi:B --> prometheus:T
    search:B --> prometheus:T
    embeddings:B --> prometheus:T
```

### Deployment Tiers

| Tier | Description | Use Case | Resources |
|------|-------------|----------|-----------|
| **Personal** | Single-node deployment | Development, prototyping | 2 CPU, 4GB RAM |
| **Professional** | Multi-service setup | Small teams, staging | 4 CPU, 8GB RAM |
| **Enterprise** | Full feature stack | Production, enterprise | 8+ CPU, 16GB+ RAM |

## Prerequisites

### System Requirements

#### Minimum Requirements (Personal Tier)
- **CPU**: 2 cores, 2.4GHz
- **RAM**: 4GB (8GB recommended)
- **Storage**: 20GB SSD
- **Network**: 100Mbps

#### Recommended Requirements (Professional Tier)
- **CPU**: 4 cores, 3.0GHz
- **RAM**: 8GB (16GB recommended)
- **Storage**: 50GB SSD
- **Network**: 1Gbps

#### Production Requirements (Enterprise Tier)
- **CPU**: 8+ cores, 3.5GHz
- **RAM**: 16GB+ (32GB recommended)
- **Storage**: 100GB+ NVMe SSD
- **Network**: 10Gbps

### Software Dependencies

#### Required Software
- **Docker**: 20.10.0 or later
- **Docker Compose**: 2.0.0 or later
- **Python**: 3.11-3.13
- **uv**: 0.7.16 or later

#### Optional Software
- **kubectl**: For Kubernetes deployment
- **Helm**: For Kubernetes package management
- **Terraform**: For infrastructure as code

## Environment Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Copy environment template
cp .env.example .env

# Edit with your specific values
nano .env
```

### Required Environment Variables

#### Core API Keys
```env
# OpenAI API Key (required for embeddings)
OPENAI_API_KEY=sk-your-openai-api-key-here

# Firecrawl API Key (optional, for premium features)
FIRECRAWL_API_KEY=fc-your-firecrawl-api-key-here

# Qdrant API Key (for cloud deployment)
QDRANT_API_KEY=your-qdrant-api-key
```

#### Service Configuration
```env
# Application Mode
AI_DOCS__MODE=enterprise
AI_DOCS__ENVIRONMENT=production
AI_DOCS__DEBUG=false

# Database URLs
AI_DOCS__QDRANT__URL=http://qdrant:6333
AI_DOCS__CACHE__REDIS_URL=redis://redis:6379

# Performance Settings
AI_DOCS__PERFORMANCE__MAX_CONCURRENT_CRAWLS=20
AI_DOCS__PERFORMANCE__MAX_CONCURRENT_EMBEDDINGS=50
AI_DOCS__PERFORMANCE__MAX_MEMORY_USAGE_MB=2000
```

#### Security Configuration
```env
# Security Settings
AI_DOCS__SECURITY__REQUIRE_API_KEYS=true
AI_DOCS__SECURITY__ENABLE_RATE_LIMITING=true
AI_DOCS__SECURITY__RATE_LIMIT_REQUESTS_PER_MINUTE=1000
AI_DOCS__SECURITY__ALLOWED_DOMAINS=["your-domain.com"]
```

#### Monitoring Configuration
```env
# Observability
AI_DOCS__OBSERVABILITY__ENABLED=true
AI_DOCS__OBSERVABILITY__OTLP_ENDPOINT=http://jaeger:4317
AI_DOCS__MONITORING__ENABLE_METRICS=true
AI_DOCS__MONITORING__METRICS_PORT=8001
```

## Deployment Methods

### Method 1: Docker Compose (Recommended)

#### Simple Deployment
```bash
# Clone repository
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Configure environment
cp .env.example .env
# Edit .env with your values

# Start services
docker-compose up -d

# Verify deployment
docker-compose ps
```

#### Enterprise Deployment
```bash
# Use enterprise configuration
docker-compose -f docker-compose.enterprise.yml up -d

# Check health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### Method 2: Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.20+)
- Helm 3.0+
- kubectl configured

#### Deployment Steps
```bash
# Create namespace
kubectl create namespace ai-docs

# Deploy with Helm
helm install ai-docs ./helm-chart \
  --namespace ai-docs \
  --set image.tag=latest \
  --set environment=production \
  --set replicaCount=3

# Verify deployment
kubectl get pods -n ai-docs
kubectl get services -n ai-docs
```

### Method 3: Cloud Platform Deployment

#### Railway (Recommended for simplicity)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway deploy

# Configure environment variables in Railway dashboard
railway variables set OPENAI_API_KEY=your-key
railway variables set AI_DOCS__MODE=enterprise
```

#### AWS ECS/Fargate
```bash
# Build and push image
docker build -t ai-docs:latest .
docker tag ai-docs:latest your-repo/ai-docs:latest
docker push your-repo/ai-docs:latest

# Deploy with Terraform
cd infrastructure/aws
terraform init
terraform plan
terraform apply
```

## Database Setup

### Qdrant Configuration

#### Local Deployment
```yaml
# docker-compose.yml
qdrant:
  image: qdrant/qdrant:latest
  ports:
    - "6333:6333"
    - "6334:6334"
  volumes:
    - qdrant_data:/qdrant/storage
  environment:
    - QDRANT__LOG_LEVEL=INFO
    - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=128
    - QDRANT__STORAGE__QUANTIZATION__ALWAYS_RAM=true
    - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8
```

#### Cloud Deployment
```env
# Use Qdrant Cloud
AI_DOCS__QDRANT__URL=https://your-cluster.qdrant.io
AI_DOCS__QDRANT__API_KEY=your-api-key
AI_DOCS__QDRANT__USE_GRPC=true
```

### Cache Configuration

#### DragonflyDB Setup
```yaml
# docker-compose.yml
redis:
  image: docker.dragonflydb.io/dragonflydb/dragonfly
  ports:
    - "6379:6379"
  volumes:
    - redis_data:/data
  command: ["dragonfly", "--logtostderr", "--cache_mode"]
  environment:
    - DRAGONFLY_THREADS=8
    - DRAGONFLY_MEMORY_LIMIT=4gb
```

### PostgreSQL (Enterprise)
```yaml
# docker-compose.yml
postgres:
  image: postgres:15
  environment:
    - POSTGRES_DB=ai_docs_enterprise
    - POSTGRES_USER=ai_docs_user
    - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
  volumes:
    - postgres_data:/var/lib/postgresql/data
  ports:
    - "5432:5432"
```

## Monitoring and Observability

### Prometheus Configuration

#### prometheus.yml
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-docs-api'
    static_configs:
      - targets: ['api:8001']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant:6333']
    metrics_path: '/metrics'

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
```

### Grafana Dashboards

#### Key Metrics to Monitor
- **API Performance**: Request rate, response time, error rate
- **Vector Search**: Query latency, result accuracy, index size
- **Embedding Generation**: Throughput, queue depth, error rate
- **Cache Performance**: Hit ratio, memory usage, eviction rate
- **System Resources**: CPU, memory, disk, network

#### Pre-built Dashboards
- `grafana/dashboards/api-performance.json`
- `grafana/dashboards/vector-search.json`
- `grafana/dashboards/system-resources.json`

### Jaeger Tracing

#### Configuration
```yaml
# docker-compose.yml
jaeger:
  image: jaegertracing/all-in-one:latest
  ports:
    - "16686:16686"  # Web UI
    - "4317:4317"    # OTLP gRPC
    - "4318:4318"    # OTLP HTTP
  environment:
    - COLLECTOR_OTLP_ENABLED=true
```

## Security Configuration

### SSL/TLS Setup

#### Certificate Management
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Or use Let's Encrypt (production)
certbot certonly --webroot -w /var/www/html -d your-domain.com
```

#### Nginx Configuration
```nginx
server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/ssl/certs/cert.pem;
    ssl_certificate_key /etc/ssl/private/key.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Authentication & Authorization

#### API Key Management
```env
# Configure API keys
AI_DOCS__SECURITY__REQUIRE_API_KEYS=true
AI_DOCS__SECURITY__API_KEY_HEADER=X-API-Key

# Rate limiting
AI_DOCS__SECURITY__ENABLE_RATE_LIMITING=true
AI_DOCS__SECURITY__RATE_LIMIT_REQUESTS_PER_MINUTE=1000
```

#### Role-Based Access Control
```yaml
# Configure RBAC in deployment
security:
  rbac:
    enabled: true
    roles:
      - name: admin
        permissions: ["read", "write", "delete"]
      - name: user
        permissions: ["read", "write"]
      - name: readonly
        permissions: ["read"]
```

## Performance Optimization

### Resource Allocation

#### CPU Optimization
```yaml
# docker-compose.yml resource limits
api:
  deploy:
    resources:
      limits:
        cpus: "2.0"
        memory: 4G
      reservations:
        cpus: "1.0"
        memory: 2G
```

#### Memory Optimization
```env
# Configure memory limits
AI_DOCS__PERFORMANCE__MAX_MEMORY_USAGE_MB=2000
AI_DOCS__CACHE__LOCAL_MAX_MEMORY_MB=500
AI_DOCS__PERFORMANCE__BATCH_EMBEDDING_SIZE=100
```

### Database Optimization

#### Qdrant Tuning
```env
# Performance settings
QDRANT__STORAGE__QUANTIZATION__ALWAYS_RAM=true
QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8
QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=128
```

#### Cache Optimization
```env
# DragonflyDB optimization
DRAGONFLY_THREADS=8
DRAGONFLY_MEMORY_LIMIT=4gb
DRAGONFLY_SNAPSHOT_INTERVAL=3600
```

## Health Checks and Monitoring

### Health Check Endpoints

#### API Health
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check
curl http://localhost:8000/api/v1/config/status

# Metrics endpoint
curl http://localhost:8000/metrics
```

#### Service Health
```bash
# Qdrant health
curl http://localhost:6333/health

# Redis health
redis-cli -p 6379 ping

# Database health
curl http://localhost:8000/api/v1/health/database
```

### Automated Monitoring

#### Health Check Script
```bash
#!/bin/bash
# health-check.sh

set -e

echo "Checking API health..."
curl -f http://localhost:8000/health || exit 1

echo "Checking Qdrant health..."
curl -f http://localhost:6333/health || exit 1

echo "Checking Redis health..."
redis-cli -p 6379 ping || exit 1

echo "All services healthy!"
```

#### Monitoring Alerts
```yaml
# alertmanager.yml
groups:
  - name: ai-docs-alerts
    rules:
      - alert: APIDown
        expr: up{job="ai-docs-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "AI Docs API is down"
          
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
```

## Backup and Recovery

### Database Backup

#### Qdrant Backup
```bash
# Create backup
docker exec qdrant_container qdrant --backup /backup/qdrant-backup.tar.gz

# Restore backup
docker exec qdrant_container qdrant --restore /backup/qdrant-backup.tar.gz
```

#### PostgreSQL Backup
```bash
# Create backup
pg_dump -h localhost -U ai_docs_user ai_docs_enterprise > backup.sql

# Restore backup
psql -h localhost -U ai_docs_user ai_docs_enterprise < backup.sql
```

### Automated Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# Backup Qdrant
docker exec qdrant_container qdrant --backup "$BACKUP_DIR/qdrant-backup.tar.gz"

# Backup PostgreSQL
pg_dump -h localhost -U ai_docs_user ai_docs_enterprise > "$BACKUP_DIR/postgres-backup.sql"

# Backup configuration
cp .env "$BACKUP_DIR/env-backup"
cp docker-compose.yml "$BACKUP_DIR/docker-compose-backup.yml"

echo "Backup completed: $BACKUP_DIR"
```

## Scaling and Load Balancing

### Horizontal Scaling

#### Load Balancer Configuration
```nginx
upstream ai_docs_backend {
    least_conn;
    server api-1:8000;
    server api-2:8000;
    server api-3:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://ai_docs_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### Auto-scaling Configuration
```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-docs-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-docs-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Check memory usage
docker stats

# Reduce memory usage
export AI_DOCS__PERFORMANCE__MAX_MEMORY_USAGE_MB=1000
export AI_DOCS__CACHE__LOCAL_MAX_MEMORY_MB=200
```

#### Network Issues
```bash
# Check network connectivity
curl -v http://localhost:8000/health

# Check DNS resolution
nslookup qdrant
nslookup redis
```

#### Database Connection Issues
```bash
# Check Qdrant connection
curl http://localhost:6333/health

# Check Redis connection
redis-cli -p 6379 ping

# Check database logs
docker logs qdrant_container
docker logs redis_container
```

### Performance Debugging

#### Slow Response Times
1. Check CPU and memory usage
2. Verify database performance
3. Check cache hit rates
4. Monitor network latency

#### High Error Rates
1. Check application logs
2. Verify API key validity
3. Monitor rate limits
4. Check external service status

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
- Monitor system health
- Check error logs
- Verify backup completion
- Monitor resource usage

#### Weekly Tasks
- Update security patches
- Optimize database indexes
- Clean up old logs
- Review performance metrics

#### Monthly Tasks
- Update dependencies
- Review and rotate API keys
- Optimize resource allocation
- Update documentation

### Update Procedures

#### Application Updates
```bash
# Pull latest code
git pull origin main

# Update dependencies
uv sync

# Rebuild containers
docker-compose build

# Rolling update
docker-compose up -d --no-deps api
```

#### Database Updates
```bash
# Backup before update
./backup.sh

# Update database
docker-compose stop qdrant
docker-compose pull qdrant
docker-compose up -d qdrant
```

## Support and Documentation

### Getting Help

#### Documentation
- [API Reference](docs/api/index.md)
- [Configuration Guide](docs/configuration/index.md)
- [Troubleshooting Guide](docs/troubleshooting/index.md)

#### Support Channels
- GitHub Issues: [Issues](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues)
- Documentation: [Docs](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/tree/main/docs)

### Monitoring Resources

#### Key Metrics
- API response time < 100ms (P95)
- Search accuracy > 95%
- System uptime > 99.9%
- Cache hit ratio > 80%

#### Dashboard URLs
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Jaeger: http://localhost:16686

This deployment guide provides comprehensive instructions for production deployment. For specific deployment scenarios or advanced configurations, refer to the platform-specific documentation in the `docs/deployment/` directory.