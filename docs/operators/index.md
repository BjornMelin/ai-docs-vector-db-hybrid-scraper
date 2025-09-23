---
title: Operator Documentation
audience: operators
status: active
owner: operations-engineering
last_reviewed: 2025-09-22
---

## Operator Documentation

Essential operational guides for running the AI Docs Vector DB platform in production.

## Core Operational Guides

### [Operations](./operations.md)

Daily operational procedures, service management, backup/recovery, and incident response commands.

### [Deployment](./deployment.md)

Production deployment procedures for Docker, Kubernetes, and cloud platforms.

### [Configuration](./configuration.md)

Essential environment variables, performance tuning, and configuration management.

### [Monitoring](./monitoring.md)

Health checks, metrics collection, alerting, and performance monitoring.

### [Security](./security.md)

Authentication, network security, container hardening, and incident response.

## Quick Reference

### Essential Commands

```bash
# Daily health check
docker-compose ps && curl http://localhost:6333/health && redis-cli ping

# Service management
docker-compose up -d      # Start services
docker-compose restart    # Restart all services
docker-compose logs -f    # View logs

# Emergency procedures
docker-compose down --timeout 30  # Graceful shutdown
docker kill $(docker ps -q)       # Force stop all containers
```

### Key Ports

- **8000**: API server
- **6333**: Qdrant vector database
- **6379**: DragonflyDB cache
- **9090**: Prometheus metrics
- **3000**: Grafana dashboard

### Critical Directories

- `/backup/`: System backups
- `/var/log/ai-docs/`: Application logs
- `/app/config/`: Configuration files
- `/app/data/`: Application data

## Getting Help

- **Operations Issues**: Check troubleshooting section in each guide
- **Service Health**: Use health check commands in operations guide
- **Performance Problems**: See monitoring and configuration guides
- **Security Incidents**: Follow security incident response procedures

---

For comprehensive system information, see [Developer Architecture](../developers/architecture.md).
For user-facing documentation, see [User Documentation](../users/index.md).
