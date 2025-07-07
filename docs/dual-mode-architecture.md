# Dual-Mode Architecture Guide

The AI Docs Vector DB implements a sophisticated dual-mode architecture that resolves the "Enterprise Paradox" - the conflict between building impressive portfolio projects and maintaining daily usability.

## Architecture Overview

### The Enterprise Paradox Solution

**Problem**: Enterprise-grade projects showcase advanced capabilities but become too complex for daily use.

**Solution**: Dual-mode architecture that provides:

- **Simple Mode**: 25K lines optimized for solo developers (64% complexity reduction)
- **Enterprise Mode**: 70K lines with full enterprise capabilities

### Core Architecture Components

```text
src/architecture/
├── modes.py              # Application mode definitions and configs
├── features.py           # Feature flag system for mode-aware functionality
├── service_factory.py    # Mode-aware service instantiation
└── __init__.py           # Architecture module exports

src/api/
├── app_factory.py        # Mode-aware FastAPI application factory
├── main.py               # Application entry point with mode detection
└── routers/
    ├── simple/           # Simplified API endpoints
    └── enterprise/       # Full-featured API endpoints

src/services/
├── simple/               # Simple mode service implementations
└── enterprise/           # Enterprise mode service implementations
```

## Mode Comparison

| Feature                 | Simple Mode           | Enterprise Mode                      |
| ----------------------- | --------------------- | ------------------------------------ |
| **Target Users**        | Solo developers       | Enterprise teams                     |
| **Lines of Code**       | ~25K                  | ~70K                                 |
| **Memory Usage**        | 500MB                 | 4GB                                  |
| **Concurrent Requests** | 5                     | 100                                  |
| **Cache Size**          | 50MB                  | 1GB                                  |
| **Monitoring**          | Basic health checks   | Full observability stack             |
| **Search**              | Vector search only    | Hybrid search + reranking            |
| **Deployment**          | Single container      | Blue-green, canary, A/B testing      |
| **Dependencies**        | Minimal (Qdrant only) | Full stack (Redis, PostgreSQL, etc.) |

## Getting Started

### Simple Mode (Recommended for Development)

```bash
# Using convenience script
./scripts/start-simple-mode.sh

# Or with Docker
./scripts/start-simple-mode.sh --docker

# Or manually
export AI_DOCS_MODE=simple
cp .env.simple .env
uvicorn src.api.main:app --reload
```

**Simple Mode Features:**

- ✅ Basic vector search
- ✅ Document management
- ✅ Simple caching
- ✅ Health monitoring
- ❌ Advanced analytics
- ❌ Deployment features
- ❌ A/B testing

### Enterprise Mode (Portfolio Demonstrations)

```bash
# Using convenience script
./scripts/start-enterprise-mode.sh

# Or with Docker (includes full stack)
./scripts/start-enterprise-mode.sh --docker

# Or manually
export AI_DOCS_MODE=enterprise
cp .env.enterprise .env
uvicorn src.api.main:app --reload
```

**Enterprise Mode Features:**

- ✅ Hybrid search with reranking
- ✅ Advanced analytics
- ✅ Distributed caching
- ✅ Comprehensive monitoring
- ✅ Deployment services
- ✅ A/B testing
- ✅ Circuit breakers
- ✅ Observability stack

## Configuration

### Environment Variables

The application automatically detects the mode from the `AI_DOCS_MODE` environment variable:

```bash
# Simple mode
export AI_DOCS_MODE=simple

# Enterprise mode
export AI_DOCS_MODE=enterprise
```

### Configuration Files

Use mode-specific configuration files:

- `.env.simple` - Optimized for solo developers
- `.env.enterprise` - Full enterprise configuration

### Mode Detection Precedence

1. `AI_DOCS_MODE` environment variable
2. `AI_DOCS_DEPLOYMENT__TIER` (legacy support)
3. Defaults to `simple` mode

## Service Architecture

### Service Factory Pattern

The `ModeAwareServiceFactory` creates different service implementations based on the current mode:

```python
from src.architecture.service_factory import get_service

# Automatically gets appropriate implementation for current mode
search_service = await get_service("search_service")
cache_service = await get_service("cache_service")
```

### Service Registration

Services are registered with mode-specific implementations:

```python
from src.architecture.service_factory import register_service
from src.services.simple.search import SimpleSearchService
from src.services.enterprise.search import EnterpriseSearchService

register_service(
    "search_service",
    simple_impl=SimpleSearchService,
    enterprise_impl=EnterpriseSearchService
)
```

### Feature Flags

Use decorators to enable/disable features based on mode:

```python
from src.architecture.features import enterprise_only, conditional_feature

@enterprise_only(fallback_value=None)
async def advanced_analytics():
    return await generate_advanced_metrics()

@conditional_feature("enable_hybrid_search", fallback_value=[])
async def hybrid_search(query):
    return await perform_hybrid_search(query)
```

## API Differences

### Simple Mode API

- **Endpoints**: Basic CRUD operations
- **Documentation**: `/docs` only
- **Rate Limiting**: Basic (50 req/min)
- **Authentication**: Optional

Example endpoints:

```
GET  /search?q=query&limit=10
POST /documents
GET  /health
```

### Enterprise Mode API

- **Endpoints**: Full feature set with advanced capabilities
- **Documentation**: `/docs` and `/redoc`
- **Rate Limiting**: Adaptive (1000 req/min)
- **Authentication**: Required

Additional endpoints:

```
POST /search/hybrid
GET  /analytics/dashboard
POST /deployment/canary
GET  /metrics
```

## Development Workflow

### 1. Daily Development (Simple Mode)

```bash
# Start in simple mode for fast iteration
./scripts/start-simple-mode.sh

# Develop with minimal complexity
# Test basic functionality
# Deploy to development environment
```

### 2. Portfolio/Demo Preparation (Enterprise Mode)

```bash
# Switch to enterprise mode for demonstrations
./scripts/start-enterprise-mode.sh --docker

# Full stack with monitoring
# Advanced features enabled
# Production-ready configuration
```

### 3. Testing Both Modes

```bash
# Run mode-specific tests
pytest tests/unit/architecture/test_dual_mode.py

# Test simple mode
AI_DOCS_MODE=simple pytest tests/integration/

# Test enterprise mode
AI_DOCS_MODE=enterprise pytest tests/integration/
```

## Docker Deployment

### Simple Mode Docker

```yaml
# docker-compose.simple.yml
services:
  api:
    environment:
      - AI_DOCS_MODE=simple
    depends_on:
      - qdrant # Minimal dependencies
```

### Enterprise Mode Docker

```yaml
# docker-compose.enterprise.yml
services:
  api:
    environment:
      - AI_DOCS_MODE=enterprise
    depends_on:
      - qdrant
      - redis # Caching
      - postgres # Data persistence
      - jaeger # Tracing
      - prometheus # Metrics
```

## Performance Characteristics

### Simple Mode Performance

- **Startup Time**: < 5 seconds
- **Memory Footprint**: 100-500MB
- **Response Time**: < 100ms (95th percentile)
- **Throughput**: 100 RPS
- **Dependencies**: 1 (Qdrant)

### Enterprise Mode Performance

- **Startup Time**: 15-30 seconds (full stack)
- **Memory Footprint**: 1-4GB
- **Response Time**: < 200ms (95th percentile)
- **Throughput**: 1000+ RPS
- **Dependencies**: 6+ (full observability stack)

## Monitoring and Observability

### Simple Mode

- Basic health checks
- Simple logging
- Essential metrics only
- No distributed tracing

Access:

- Health: `http://localhost:8000/health`
- Mode info: `http://localhost:8000/mode`

### Enterprise Mode

- Comprehensive monitoring
- Structured logging
- Full metrics collection
- Distributed tracing
- Performance profiling

Access:

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Jaeger: `http://localhost:16686`
- Health: `http://localhost:8000/health`

## Migration Between Modes

### Simple to Enterprise

1. Update configuration:

   ```bash
   cp .env.enterprise .env
   export AI_DOCS_MODE=enterprise
   ```

2. Start additional services:

   ```bash
   docker-compose -f docker-compose.enterprise.yml up -d
   ```

3. Restart application:
   ```bash
   uvicorn src.api.main:app --reload
   ```

### Enterprise to Simple

1. Update configuration:

   ```bash
   cp .env.simple .env
   export AI_DOCS_MODE=simple
   ```

2. Stop additional services:

   ```bash
   docker-compose -f docker-compose.enterprise.yml down
   ```

3. Restart application:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## Best Practices

### For Solo Developers

1. **Start Simple**: Use simple mode for daily development
2. **Fast Iteration**: Leverage reduced complexity for quick testing
3. **Resource Efficiency**: Benefit from lower resource usage
4. **Easy Debugging**: Minimal dependencies make debugging easier

### For Enterprise/Portfolio

1. **Full Stack**: Use enterprise mode for demonstrations
2. **Production Ready**: Leverage enterprise-grade features
3. **Comprehensive Monitoring**: Showcase observability capabilities
4. **Scalability**: Demonstrate production scalability

### Development Guidelines

1. **Mode-Aware Development**: Consider both modes when adding features
2. **Feature Flags**: Use decorators for mode-specific functionality
3. **Service Abstraction**: Implement services for both modes
4. **Configuration**: Maintain separate config files
5. **Testing**: Test both modes in CI/CD pipeline

## Troubleshooting

### Mode Detection Issues

```bash
# Check current mode
curl http://localhost:8000/mode

# Verify environment
echo $AI_DOCS_MODE

# Check configuration loading
curl http://localhost:8000/info
```

### Service Availability

```bash
# Check service status
curl http://localhost:8000/health

# Check available services
curl http://localhost:8000/mode | jq '.enabled_services'
```

### Common Issues

1. **Wrong Mode**: Ensure `AI_DOCS_MODE` is set correctly
2. **Missing Services**: Check Docker containers are running
3. **Configuration**: Verify `.env` file is loaded
4. **Dependencies**: Ensure required services are available

## Architecture Benefits

### Complexity Management

- ✅ 64% reduction in daily-use complexity
- ✅ Maintains full enterprise capabilities
- ✅ Clean separation of concerns
- ✅ Gradual feature adoption

### Developer Experience

- ✅ Fast startup for development
- ✅ Minimal resource usage
- ✅ Easy debugging and testing
- ✅ Portfolio demonstration ready

### Operational Benefits

- ✅ Production-ready enterprise mode
- ✅ Development-optimized simple mode
- ✅ Flexible deployment options
- ✅ Mode-aware monitoring

## Future Enhancements

1. **Automatic Mode Switching**: Based on resource availability
2. **Hybrid Mode**: Custom feature combinations
3. **Performance Profiling**: Automatic optimization recommendations
4. **Dynamic Scaling**: Resource-based mode adaptation

---

This dual-mode architecture successfully resolves the Enterprise Paradox by providing an elegant solution that maintains both usability and capability through intelligent architectural choices rather than feature removal.
