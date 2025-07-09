# API Documentation

Welcome to the AI Documentation Vector Database Hybrid Scraper API documentation. This comprehensive guide covers all endpoints, authentication methods, and usage patterns for both Simple and Enterprise modes.

## Overview

The API is built with FastAPI and provides automatic OpenAPI/Swagger documentation. The system operates in two modes:

- **Simple Mode**: Optimized for solo developers with minimal complexity
- **Enterprise Mode**: Full enterprise feature set with advanced capabilities

## Base URL

```
http://localhost:8000/api/v1
```

## Authentication

The API supports multiple authentication methods:

- **API Key**: Pass via `X-API-Key` header
- **JWT Bearer Token**: Pass via `Authorization: Bearer <token>` header

## Interactive Documentation

Access interactive API documentation at:

- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc) (Enterprise mode only)

## API Endpoints

### Core Endpoints

- [Search API](./search.md) - Vector and hybrid search capabilities
- [Documents API](./documents.md) - Document management and processing
- [Embeddings API](./embeddings.md) - Text embedding generation
- [Crawling API](./crawling.md) - Web scraping and content extraction

### Enterprise Endpoints

- [Analytics API](./analytics.md) - Advanced analytics and insights
- [Deployment API](./deployment.md) - Canary deployment and A/B testing
- [Monitoring API](./monitoring.md) - Performance and health monitoring
- [Security API](./security.md) - Authentication and authorization

### Performance Optimization

- [POA API](./optimization.md) - Performance Optimization Agent endpoints

## Rate Limiting

| Mode | Rate Limit | Burst |
|------|------------|-------|
| Simple | 100 req/min | 10 |
| Enterprise | 1000 req/min | 100 |

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": {
    "message": "Error description",
    "error_code": "ERROR_CODE",
    "context": {}
  }
}
```

Common error codes:

- `VALIDATION_ERROR` - Invalid request parameters
- `AUTHENTICATION_ERROR` - Missing or invalid credentials
- `RATE_LIMIT_ERROR` - Rate limit exceeded
- `INTERNAL_ERROR` - Server error

## Pagination

List endpoints support pagination:

```
GET /api/v1/documents?page=1&page_size=20
```

Response includes pagination metadata:

```json
{
  "items": [...],
  "total": 100,
  "page": 1,
  "page_size": 20,
  "pages": 5
}
```

## Versioning

The API uses URL versioning. Current version: `v1`

Future versions will maintain backward compatibility where possible.

## OpenAPI Schema

Download the OpenAPI schema:

```
GET /openapi.json
```

## SDK Support

Official SDKs are available for:

- Python: `pip install ai-docs-sdk`
- TypeScript: `npm install @ai-docs/sdk`
- Go: `go get github.com/ai-docs/sdk-go`

## Need Help?

- Check the [FAQ](../user-guide/faq.md)
- Review [Examples](../user-guide/examples.md)
- Contact support: api-support@ai-docs.dev