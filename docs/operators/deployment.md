# Deployment Guide

## Quick Start

### Docker Compose Deployment
```bash
# Clone repository
git clone <repository-url>
cd ai-docs-vector-db-hybrid-scraper

# Copy environment configuration
cp .env.example .env

# Edit required variables
export AI_DOCS__OPENAI__API_KEY="your-openai-key"
export AI_DOCS__FIRECRAWL__API_KEY="your-firecrawl-key"

# Start services
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
curl http://localhost:6333/health
```

### Environment Modes

#### Simple Mode (Development)
```bash
# Start basic services
docker-compose up -d

# Services included:
# - API server (port 8000)
# - Qdrant vector database (port 6333)
# - DragonflyDB cache (port 6379)
```

#### Enterprise Mode (Production)
```bash
# Start with enterprise configuration
docker-compose -f docker-compose.enterprise.yml up -d

# Additional services:
# - Monitoring (Prometheus, Grafana)
# - Load balancer
# - Worker nodes
# - Backup services
```

## Production Deployment

### Prerequisites
```bash
# System requirements
# - 16GB+ RAM
# - 8+ CPU cores
# - 100GB+ storage
# - Docker 20.10+
# - Docker Compose 2.0+

# Install dependencies
sudo apt update
sudo apt install -y docker.io docker-compose-plugin
sudo systemctl enable docker
sudo systemctl start docker
```

### Production Configuration
```bash
# Set production environment
export AI_DOCS__ENVIRONMENT=production
export CONFIG_FILE=config/production.json

# Security settings
export AI_DOCS__API__SECRET_KEY=$(openssl rand -hex 32)
export AI_DOCS__TLS__ENABLED=true

# Performance settings
export AI_DOCS__PERFORMANCE__MAX_CONCURRENT_REQUESTS=20
export AI_DOCS__PERFORMANCE__BATCH_SIZE=100
```

### SSL/TLS Setup
```bash
# Generate SSL certificates
sudo certbot certonly --standalone -d your-domain.com

# Configure nginx proxy
# /etc/nginx/sites-available/ai-docs
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Kubernetes Deployment

### Basic Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-docs-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-docs-api
  template:
    metadata:
      labels:
        app: ai-docs-api
    spec:
      containers:
      - name: api
        image: ai-docs:latest
        ports:
        - containerPort: 8000
        env:
        - name: AI_DOCS__ENVIRONMENT
          value: "production"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### Service Configuration
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ai-docs-service
spec:
  selector:
    app: ai-docs-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deploy to Kubernetes
```bash
# Apply configurations
kubectl apply -f k8s/

# Verify deployment
kubectl get pods
kubectl get services

# Check logs
kubectl logs -f deployment/ai-docs-api
```

## Cloud Deployment

### AWS ECS
```bash
# Create ECS cluster
aws ecs create-cluster --cluster-name ai-docs-cluster

# Create task definition
aws ecs register-task-definition --cli-input-json file://ecs-task-definition.json

# Create service
aws ecs create-service \
  --cluster ai-docs-cluster \
  --service-name ai-docs-service \
  --task-definition ai-docs-task \
  --desired-count 2
```

### Google Cloud Run
```bash
# Build and deploy
gcloud run deploy ai-docs \
  --image gcr.io/PROJECT-ID/ai-docs \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

## Database Setup

### Qdrant Cloud
```bash
# Configuration for Qdrant Cloud
export AI_DOCS__QDRANT__URL="https://your-cluster.qdrant.io"
export AI_DOCS__QDRANT__API_KEY="your-qdrant-api-key"
```

### Redis Cloud
```bash
# Configuration for Redis Cloud
export AI_DOCS__CACHE__DRAGONFLY_URL="redis://user:password@redis-host:port"
```

## Monitoring Setup

### Prometheus Configuration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ai-docs'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Start Monitoring Stack
```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

## Scaling

### Horizontal Scaling
```bash
# Scale API services
docker-compose scale api=3

# Scale workers
docker-compose scale worker=5

# Kubernetes scaling
kubectl scale deployment ai-docs-api --replicas=5
```

### Vertical Scaling
```bash
# Update resource limits in docker-compose.yml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
```

## Health Checks

### Deployment Verification
```bash
# Health check script
#!/bin/bash
echo "Verifying deployment..."

# Check API health
curl -f http://localhost:8000/health || exit 1

# Check vector database
curl -f http://localhost:6333/health || exit 1

# Check cache
redis-cli ping || exit 1

echo "Deployment verification complete"
```

## Troubleshooting

### Common Issues
```bash
# Port conflicts
netstat -tlnp | grep -E "(8000|6333|6379)"

# Container logs
docker-compose logs api
docker-compose logs qdrant
docker-compose logs dragonfly

# Resource usage
docker stats --no-stream

# Restart services
docker-compose restart
```

### Recovery Procedures
```bash
# Emergency restart
docker-compose down --timeout 30
docker system prune -f
docker-compose up -d

# Rollback deployment
git checkout previous-stable-commit
docker-compose up -d
```