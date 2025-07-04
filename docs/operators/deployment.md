# Deployment Guide

> **Purpose**: Comprehensive deployment guide for production and development environments
> **Audience**: DevOps engineers, system administrators, and deployment teams
> **Last Updated**: 2025-01-04

This guide covers deployment strategies, environment setup, and configuration for the AI Docs Vector DB Hybrid Scraper system across different platforms and environments.

## Deployment Modes

### Simple Mode (Development/Small Scale)
- **Resources**: Minimal requirements (4GB RAM, 2 CPU cores)
- **Features**: Basic MCP server, standard vector search, 3-tier browser automation
- **Use Case**: Development, testing, small-scale deployments

### Enterprise Mode (Production/Large Scale)
- **Resources**: Recommended (16GB+ RAM, 8+ CPU cores)
- **Features**: Full FastMCP 2.0, hybrid search, 5-tier AI browser system, ML optimization
- **Use Case**: Production environments, high-volume processing

## Quick Start Deployment

### Using Docker Compose
```bash
# Clone the repository
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Start services (Simple Mode)
docker-compose up -d

# Start services (Enterprise Mode)
docker-compose -f docker-compose.enterprise.yml up -d
```

### Using Docker
```bash
# Build the image
docker build -t ai-docs-scraper .

# Run Simple Mode
docker run -d \
  --name ai-docs-scraper \
  -p 8000:8000 \
  -e MODE=simple \
  ai-docs-scraper

# Run Enterprise Mode
docker run -d \
  --name ai-docs-scraper-enterprise \
  -p 8000:8000 \
  -e MODE=enterprise \
  ai-docs-scraper
```

### Using uv (Local Development)
```bash
# Install dependencies
uv install

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run Simple Mode
uv run python -m src.main --mode simple

# Run Enterprise Mode
uv run python -m src.main --mode enterprise
```

## Environment Configuration

### Required Environment Variables
```bash
# Core Configuration
MODE=simple|enterprise
LOG_LEVEL=INFO
PORT=8000

# Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
REDIS_URL=redis://localhost:6379

# AI Services (Optional)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key

# Security
SECRET_KEY=your_secret_key
CORS_ORIGINS=http://localhost:3000,http://localhost:8080
```

### Optional Environment Variables
```bash
# Performance Tuning
MAX_WORKERS=4
EMBEDDING_BATCH_SIZE=100
CACHE_TTL=3600

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
ENABLE_METRICS=true

# Browser Automation
BROWSER_TYPE=chromium
HEADLESS=true
BROWSER_TIMEOUT=30000
```

## Infrastructure Requirements

### Minimum Requirements (Simple Mode)
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 20GB SSD
- **Network**: 100 Mbps

### Recommended Requirements (Enterprise Mode)
- **CPU**: 8 cores
- **RAM**: 16GB
- **Storage**: 100GB SSD
- **Network**: 1 Gbps

### Storage Requirements
- **Vector Database**: 1-10GB depending on document volume
- **Cache**: 1-5GB for Redis cache
- **Logs**: 1-5GB for application logs
- **Temp Files**: 1-2GB for document processing

## Deployment Platforms

### Docker Compose (Recommended)
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODE=simple
    depends_on:
      - qdrant
      - redis

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

volumes:
  qdrant_data:
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-docs-scraper
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-docs-scraper
  template:
    metadata:
      labels:
        app: ai-docs-scraper
    spec:
      containers:
      - name: app
        image: ai-docs-scraper:latest
        ports:
        - containerPort: 8000
        env:
        - name: MODE
          value: "enterprise"
        - name: QDRANT_HOST
          value: "qdrant-service"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
```

### Cloud Deployment

#### AWS ECS
- Use the provided ECS task definition
- Configure ALB for load balancing
- Use EFS for persistent storage
- Set up CloudWatch for monitoring

#### Google Cloud Run
- Deploy using Cloud Build
- Configure custom domains
- Use Cloud SQL for metadata
- Enable Cloud Monitoring

#### Azure Container Instances
- Use Azure Container Registry
- Configure Application Gateway
- Set up Azure Monitor
- Use Azure Files for storage

## Database Setup

### Qdrant Vector Database
```bash
# Using Docker
docker run -p 6333:6333 qdrant/qdrant:latest

# Using Qdrant Cloud (Recommended for production)
# 1. Sign up at https://cloud.qdrant.io/
# 2. Create a cluster
# 3. Configure QDRANT_URL and QDRANT_API_KEY
```

### Redis Cache
```bash
# Using Docker
docker run -p 6379:6379 redis:alpine

# Using Redis Cloud (Recommended for production)
# 1. Sign up at https://redis.com/
# 2. Create a database
# 3. Configure REDIS_URL
```

## Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# Use Let's Encrypt (production)
certbot certonly --standalone -d your-domain.com
```

### API Security
```bash
# Set strong secret key
export SECRET_KEY=$(openssl rand -hex 32)

# Configure CORS origins
export CORS_ORIGINS="https://your-domain.com,https://app.your-domain.com"

# Enable rate limiting
export ENABLE_RATE_LIMITING=true
export RATE_LIMIT_REQUESTS=100
export RATE_LIMIT_WINDOW=60
```

## Monitoring and Observability

### Prometheus Metrics
The application exposes metrics at `/metrics` endpoint:
- Request count and latency
- Vector search performance
- Cache hit/miss rates
- Database connection pool status

### Health Checks
- **Health**: `/health` - Basic application health
- **Ready**: `/ready` - Application readiness
- **Live**: `/live` - Kubernetes liveness probe

### Logging Configuration
```python
# Configure structured logging
LOGGING_CONFIG = {
    "version": 1,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["console"]
    }
}
```

## Production Checklist

### Pre-Deployment
- [ ] Environment variables configured
- [ ] SSL certificates installed
- [ ] Database connections tested
- [ ] Security settings verified
- [ ] Performance benchmarks run
- [ ] Backup procedures tested

### Deployment
- [ ] Application deployed successfully
- [ ] Health checks passing
- [ ] Database migrations applied
- [ ] SSL/TLS working correctly
- [ ] Monitoring dashboards operational
- [ ] Load balancer configured

### Post-Deployment
- [ ] End-to-end tests passing
- [ ] Performance metrics normal
- [ ] Error rates within acceptable limits
- [ ] Backup jobs scheduled
- [ ] Alerting rules configured
- [ ] Documentation updated

## Troubleshooting

### Common Issues

#### Application Won't Start
1. Check environment variables
2. Verify database connections
3. Check port availability
4. Review application logs

#### High Memory Usage
1. Monitor vector database size
2. Check cache configuration
3. Review batch processing settings
4. Consider scaling up resources

#### Slow Response Times
1. Check database performance
2. Review cache hit rates
3. Monitor network latency
4. Analyze query complexity

### Log Analysis
```bash
# View application logs
docker logs ai-docs-scraper

# Filter error logs
docker logs ai-docs-scraper 2>&1 | grep ERROR

# Monitor real-time logs
docker logs -f ai-docs-scraper
```

## Scaling and Performance

### Horizontal Scaling
- Use load balancer (nginx, ALB, etc.)
- Deploy multiple application instances
- Implement sticky sessions if needed
- Monitor resource utilization

### Vertical Scaling
- Increase CPU cores for processing
- Add RAM for larger datasets
- Use faster storage (NVMe SSD)
- Optimize database parameters

### Performance Optimization
- Enable caching at multiple levels
- Use connection pooling
- Implement batch processing
- Optimize vector search parameters

## Backup and Recovery

### Database Backup
```bash
# Qdrant backup
curl -X POST "http://localhost:6333/collections/{collection_name}/snapshots"

# Redis backup
redis-cli BGSAVE
```

### Application Backup
- Configuration files
- Custom models and embeddings
- User data and preferences
- System logs and metrics

## Maintenance

### Regular Tasks
- Monitor system resources
- Update dependencies
- Review security patches
- Clean up old logs
- Optimize database performance

### Scheduled Maintenance
- Monthly: Full system backup
- Weekly: Database optimization
- Daily: Log rotation
- Hourly: Health check validation

## Support and Resources

### Documentation Links
- [Operations Guide](./operations.md) - Daily operational procedures
- [Monitoring Guide](./monitoring.md) - System monitoring and alerting
- [Security Guide](./security.md) - Security configuration and best practices
- [Configuration Guide](./configuration.md) - Advanced configuration options

### External Resources
- [Docker Documentation](https://docs.docker.com/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Redis Documentation](https://redis.io/documentation)

### Getting Help
- Check the troubleshooting section
- Review application logs
- Consult the operations guide
- Contact the development team

---

This deployment guide provides comprehensive instructions for setting up and maintaining the AI Docs Vector DB Hybrid Scraper system in various environments. Follow the security best practices and monitoring guidelines for optimal performance and reliability.