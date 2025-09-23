# Production Deployment Guide

## 1. Docker Deployment Basics

# Build optimized image
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:8000"]

# Run container with restart policy
docker run -d --restart=unless-stopped --name myapp myapp:latest

# Use docker-compose for multi-container setup
version: '3.8'
services:
  web:
    image: myapp:latest
    restart: unless-stopped
    ports:
      - "8000:8000"
    depends_on:
      - db
      - redis
  db:
    image: postgres:13
    restart: unless-stopped
    environment:
      POSTGRES_DB: myapp
  redis:
    image: redis:6-alpine
    restart: unless-stopped

## 2. Environment Configuration

# Environment variables file (.env.prod)
DATABASE_URL=postgresql://user:pass@db:5432/myapp
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key-here
DEBUG=False
ALLOWED_HOSTS=yourdomain.com,api.yourdomain.com

# Load environment in application
import os
from dotenv import load_dotenv

load_dotenv('.env.prod')
DATABASE_URL = os.getenv('DATABASE_URL')
SECRET_KEY = os.getenv('SECRET_KEY')

## 3. Essential Service Dependencies

# Database (PostgreSQL)
- Connection pooling
- Backup strategy
- Read replicas for scaling

# Cache (Redis)
- Session storage
- Rate limiting
- Background task queue

# Reverse proxy (Nginx)
- SSL termination
- Static file serving
- Load balancing

# Logging service
- Centralized log aggregation
- Log rotation
- Error tracking integration

## 4. Health Checks and Monitoring

# Application health endpoint
@app.route('/health')
def health_check():
    return {'status': 'healthy'}, 200

# Docker health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Container resource limits
docker run --memory=512m --cpus=1.0 --name myapp myapp:latest

# Monitoring setup
- CPU and memory usage tracking
- HTTP response time monitoring
- Database connection count
- Error rate metrics

## 5. Basic Scaling Considerations

# Horizontal scaling
docker-compose up --scale web=3

# Vertical scaling limits
- Memory: 512MB-2GB per container
- CPU: 1-4 cores per container
- Database connections: 100-200 max

# Load balancer configuration
upstream app_servers {
    server web1:8000;
    server web2:8000;
    server web3:8000;
}

# Session persistence
- Use Redis for shared sessions
- Enable sticky sessions if needed

## 6. Security Configuration

# Container security
- Run as non-root user
- Disable unnecessary capabilities
- Use read-only filesystem where possible

# Network security
- Internal network segmentation
- Firewall rules for service ports
- SSL/TLS for all external traffic

# Application security
- Environment variable secrets
- CORS policy configuration
- Rate limiting implementation
- Input validation and sanitization

# Image scanning
docker scan myapp:latest

# Runtime security
- Enable Docker content trust
- Regular security updates
- Log security events