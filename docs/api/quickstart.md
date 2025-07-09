# API Quick Start Guide

## 🚀 Getting Started in 2 Minutes

### Prerequisites
- Python 3.11+ (3.13 recommended)
- uv package manager
- Docker (optional, for Qdrant)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper

# Install with uv (recommended)
uv pip install -e .

# Or use pip
pip install -e .
```

### Quick Start

```python
from src.api.app_factory import create_app
from src.models.api_contracts import SearchRequest

# Create the FastAPI app
app = create_app()

# Example: Search for documents
async def search_example():
    request = SearchRequest(
        query="How to implement vector search?",
        top_k=5,
        filters={"category": "documentation"}
    )
    
    # Use the API client or direct endpoint
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/search",
            json=request.model_dump()
        )
        results = response.json()
        return results
```

## 🔥 Core API Endpoints

### Search Documents
```http
POST /api/v1/search
Content-Type: application/json

{
  "query": "vector databases",
  "top_k": 10,
  "filters": {
    "source": "technical"
  }
}
```

### Embed Documents
```http
POST /api/v1/embed
Content-Type: application/json

{
  "documents": [
    {
      "content": "Document text to embed",
      "metadata": {"source": "api"},
      "source": "manual"
    }
  ]
}
```

### Health Check
```http
GET /api/v1/health
```

## 🛡️ Authentication

The API supports multiple authentication methods:

### API Key Authentication
```python
headers = {
    "X-API-Key": "your-api-key-here"
}

response = await client.get(
    "http://localhost:8000/api/v1/search",
    headers=headers
)
```

### Bearer Token
```python
headers = {
    "Authorization": "Bearer your-jwt-token"
}
```

## ⚡ Performance Tips

1. **Batch Operations**: Use batch endpoints for multiple documents
2. **Caching**: Results are cached for 5 minutes by default
3. **Rate Limiting**: 100 requests/minute per API key
4. **Connection Pooling**: Reuse HTTP clients for better performance

## 🔧 Configuration

Environment variables for quick configuration:

```bash
# Required
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333

# Optional
REDIS_URL=redis://localhost:6379
API_RATE_LIMIT=100
CACHE_TTL=300
```

## 📚 Next Steps

- [Full API Reference](/api/reference)
- [Advanced Configuration](/configuration)
- [Performance Tuning](/performance)
- [Security Best Practices](/security)