# API Reference

> **Purpose**: Complete API specifications and endpoint documentation  
> **Audience**: Developers integrating with the system

## Available APIs

### Core APIs
- [**REST API**](../reference/api/rest-api.md) - Vector operations, search, and document management
- [**Browser API**](../reference/api/browser-api.md) - Web scraping and automation endpoints

## API Overview

### REST API Features
- **Vector Operations**: Store, search, and manage embeddings
- **Document Management**: Upload, process, and retrieve documents
- **Collection Management**: Create and manage vector collections
- **Search Capabilities**: Hybrid search, semantic search, and filtering

### Browser API Features
- **5-Tier Scraping**: Lightweight to heavy automation
- **Rate Limiting**: Intelligent request throttling
- **Caching**: Browser state and content caching
- **Monitoring**: Real-time scraping metrics

## Authentication

All APIs use:
- **API Key**: Include in `Authorization: Bearer <token>` header
- **Rate Limiting**: 1000 requests/hour default
- **CORS**: Configurable cross-origin policies

## Quick Examples

### Vector Search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "limit": 10}'
```

### Browser Scraping
```bash
curl -X POST "http://localhost:8000/scrape" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com", "tier": "lightweight"}'
```

## Related Documentation

- 🛠️ [How-to Guides](../../how-to-guides/) - Implementation examples
- ⚙️ [Configuration](../configuration/) - API configuration options
- 🧠 [Concepts](../../concepts/) - Understanding API design