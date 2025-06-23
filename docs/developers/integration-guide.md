# Integration Guide

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Complete integration patterns and implementation examples  
> **Audience**: Developers integrating the system into their applications

This guide shows you how to integrate the AI Documentation Vector DB system into
your applications, whether you're building web apps, CLI tools, or enterprise
systems.

## 🚀 Quick Integration Start

### Choose Your Integration Method

1. **Python SDK** - Direct programmatic access (recommended)
2. **REST API** - HTTP endpoints for any language
3. **MCP Tools** - Claude Desktop/Code integration
4. **Docker Container** - Containerized deployment

### 5-Minute Integration Example

```python
# Install and basic usage
uv add ai-docs-vector-db

from ai_docs_vector_db import DocumentDB

# Initialize
db = DocumentDB(openai_api_key="sk-...")
await db.initialize()

# Add documents
await db.add_url("https://docs.example.com")

# Search
results = await db.search("vector database")
print(f"Found {len(results)} results")
```

## 🐍 Python SDK Integration

### Installation and Setup

#### Install Package

```bash
# Production installation
uv add ai-docs-vector-db

# Development installation
git clone https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper.git
cd ai-docs-vector-db-hybrid-scraper
uv sync --dev
```

#### Environment Configuration

```python
# config.py
import os
from ai_docs_vector_db import UnifiedConfig

config = UnifiedConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    enable_cache=True,
    cache_ttl=3600
)
```

### Core SDK Usage Patterns

#### Basic Search Integration

```python
from ai_docs_vector_db import DocumentDB, SearchRequest

class MyApplication:
    def __init__(self):
        self.db = DocumentDB(config)

    async def initialize(self):
        """Initialize the document database."""
        await self.db.initialize()

    async def search_documents(self, query: str, limit: int = 10):
        """Search documents with automatic relevance ranking."""
        request = SearchRequest(
            query=query,
            limit=limit,
            score_threshold=0.7,
            enable_hyde=True  # Enhanced query expansion
        )

        results = await self.db.search(request)

        return [
            {
                "title": result.title,
                "content": result.content[:500] + "...",
                "url": result.url,
                "score": result.score,
                "metadata": result.metadata
            }
            for result in results.results
        ]

    async def add_documentation_site(self, base_url: str):
        """Add entire documentation site to index."""
        return await self.db.crawl_site(
            base_url=base_url,
            max_pages=100,
            max_depth=3
        )

    async def cleanup(self):
        """Cleanup resources."""
        await self.db.cleanup()

# Usage
app = MyApplication()
await app.initialize()

# Search
results = await app.search_documents("authentication guide")

# Add documentation
await app.add_documentation_site("https://docs.myproject.com")

await app.cleanup()
```

#### Advanced Search with Reranking

```python
from ai_docs_vector_db import AdvancedSearchRequest, SearchStrategy

async def advanced_search_integration(query: str):
    """Advanced search with multiple strategies and reranking."""

    # Configure advanced search
    request = AdvancedSearchRequest(
        query=query,
        search_strategy=SearchStrategy.HYBRID,
        accuracy_level="balanced",
        enable_reranking=True,
        hyde_config={
            "temperature": 0.7,
            "expand_queries": 3
        },
        limit=20,
        filters={
            "doc_type": ["tutorial", "guide", "reference"],
            "language": "en"
        }
    )

    # Execute search
    response = await db.advanced_search(request)

    # Process results with confidence scoring
    high_confidence = [r for r in response.results if r.score > 0.85]
    medium_confidence = [r for r in response.results if 0.7 <= r.score <= 0.85]

    return {
        "high_confidence": high_confidence,
        "medium_confidence": medium_confidence,
        "total_results": len(response.results),
        "search_time_ms": response.query_time_ms,
        "strategy_used": response.search_strategy
    }
```

#### Document Management Integration

```python
from ai_docs_vector_db import DocumentRequest, BulkDocumentRequest

class DocumentManager:
    def __init__(self, db: DocumentDB):
        self.db = db

    async def add_single_document(self, url: str, metadata: dict = None):
        """Add single document with metadata."""
        request = DocumentRequest(
            url=url,
            collection_name="my_docs",
            doc_type="documentation",
            metadata=metadata or {},
            force_recrawl=False
        )

        response = await self.db.add_document(request)

        if response.success:
            return {
                "document_id": response.document_id,
                "chunks_created": response.chunks_created,
                "processing_time": response.processing_time_ms,
                "status": "indexed"
            }
        else:
            raise Exception(f"Failed to add document: {response.error}")

    async def bulk_add_documents(self, urls: list[str]):
        """Efficiently add multiple documents."""
        request = BulkDocumentRequest(
            urls=urls,
            max_concurrent=5,  # Adjust based on rate limits
            collection_name="my_docs",
            force_recrawl=False
        )

        response = await self.db.bulk_add_documents(request)

        return {
            "successful": len([r for r in response.results if r.success]),
            "failed": len([r for r in response.results if not r.success]),
            "total_processing_time": sum(r.processing_time_ms for r in response.results),
            "details": response.results
        }

    async def update_document_metadata(self, document_id: str, metadata: dict):
        """Update document metadata without re-processing content."""
        return await self.db.update_document(
            document_id=document_id,
            metadata=metadata
        )

    async def remove_document(self, document_id: str):
        """Remove document from index."""
        return await self.db.delete_document(document_id)
```

### Web Scraping Integration

#### Basic Web Scraping

```python
from ai_docs_vector_db import UnifiedBrowserManager, UnifiedScrapingRequest

class WebScrapingIntegration:
    def __init__(self, config):
        self.browser_manager = UnifiedBrowserManager(config)

    async def initialize(self):
        await self.browser_manager.initialize()

    async def scrape_with_smart_routing(self, url: str):
        """Scrape with automatic tier selection for optimal performance."""

        # Analyze URL first for insights
        analysis = await self.browser_manager.analyze_url(url)

        # Scrape with automatic optimization
        response = await self.browser_manager.scrape(
            url=url,
            tier="auto",  # Let system choose optimal tier
            extract_metadata=True
        )

        return {
            "success": response.success,
            "content": response.content,
            "tier_used": response.tier_used,
            "execution_time": response.execution_time_ms,
            "quality_score": response.quality_score,
            "recommended_tier": analysis["recommended_tier"],
            "metadata": response.metadata
        }

    async def scrape_complex_spa(self, url: str, interactions: list):
        """Scrape complex single-page applications with custom interactions."""

        request = UnifiedScrapingRequest(
            url=url,
            tier="browser_use",  # AI-powered interactions
            interaction_required=True,
            custom_actions=interactions,
            timeout=60000,  # Longer timeout for complex interactions
            wait_for_selector=".content-loaded",
            extract_metadata=True
        )

        return await self.browser_manager.scrape(request)

    async def batch_scrape_with_concurrency(self, urls: list[str]):
        """Efficiently scrape multiple URLs with concurrency control."""
        import asyncio

        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

        async def scrape_single(url):
            async with semaphore:
                return await self.scrape_with_smart_routing(url)

        tasks = [scrape_single(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if not (isinstance(r, dict) and r.get("success"))]

        return {
            "successful_count": len(successful),
            "failed_count": len(failed),
            "results": successful,
            "errors": failed
        }

    async def cleanup(self):
        await self.browser_manager.cleanup()

# Usage examples
scraper = WebScrapingIntegration(config)
await scraper.initialize()

# Simple scraping
result = await scraper.scrape_with_smart_routing("https://docs.example.com")

# Complex SPA scraping
spa_result = await scraper.scrape_complex_spa(
    "https://app.example.com/docs",
    interactions=[
        {"type": "click", "selector": "#docs-section"},
        {"type": "wait", "duration": 2000},
        {"type": "extract", "target": "documentation"}
    ]
)

# Batch scraping
batch_results = await scraper.batch_scrape_with_concurrency([
    "https://docs.example1.com",
    "https://docs.example2.com",
    "https://docs.example3.com"
])

await scraper.cleanup()
```

## 🌐 REST API Integration

### HTTP Client Integration

#### Python Requests Integration

```python
import requests
import asyncio
import aiohttp
from typing import Dict, List, Any

class APIClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: str = None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}" if api_key else None
        }

    def search_documents(self, query: str, **kwargs) -> Dict[str, Any]:
        """Synchronous document search."""
        payload = {
            "query": query,
            "limit": kwargs.get("limit", 10),
            "score_threshold": kwargs.get("score_threshold", 0.7),
            "collection_name": kwargs.get("collection", "documents"),
            "enable_hyde": kwargs.get("enable_hyde", False)
        }

        response = requests.post(
            f"{self.base_url}/search",
            json=payload,
            headers=self.headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()

    def add_document(self, url: str, **kwargs) -> Dict[str, Any]:
        """Add single document."""
        payload = {
            "url": url,
            "collection_name": kwargs.get("collection", "documents"),
            "doc_type": kwargs.get("doc_type", "documentation"),
            "metadata": kwargs.get("metadata", {}),
            "force_recrawl": kwargs.get("force_recrawl", False)
        }

        response = requests.post(
            f"{self.base_url}/documents",
            json=payload,
            headers=self.headers,
            timeout=120  # Longer timeout for document processing
        )
        response.raise_for_status()
        return response.json()

    def list_collections(self) -> Dict[str, Any]:
        """List all collections."""
        response = requests.get(
            f"{self.base_url}/collections",
            headers=self.headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()

# Usage
client = APIClient(api_key="your-api-key")

# Search
results = client.search_documents(
    query="authentication tutorial",
    limit=5,
    enable_hyde=True
)

# Add document
doc_result = client.add_document(
    url="https://docs.example.com/auth",
    metadata={"category": "security", "version": "v2.0"}
)

# List collections
collections = client.list_collections()
```

#### Async HTTP Client Integration

```python
import aiohttp
import asyncio
from typing import Dict, List, Any, Optional

class AsyncAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000/api/v1", api_key: Optional[str] = None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json"
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def search_documents(self, query: str, **kwargs) -> Dict[str, Any]:
        """Async document search."""
        payload = {
            "query": query,
            "limit": kwargs.get("limit", 10),
            "score_threshold": kwargs.get("score_threshold", 0.7),
            "collection_name": kwargs.get("collection", "documents"),
            "enable_hyde": kwargs.get("enable_hyde", False)
        }

        async with self.session.post(
            f"{self.base_url}/search",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def bulk_add_documents(self, urls: List[str], **kwargs) -> Dict[str, Any]:
        """Async bulk document addition."""
        payload = {
            "urls": urls,
            "max_concurrent": kwargs.get("max_concurrent", 5),
            "collection_name": kwargs.get("collection", "documents"),
            "force_recrawl": kwargs.get("force_recrawl", False)
        }

        async with self.session.post(
            f"{self.base_url}/documents/bulk",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes for bulk processing
        ) as response:
            response.raise_for_status()
            return await response.json()

    async def stream_search_results(self, query: str, batch_size: int = 10):
        """Stream search results in batches."""
        offset = 0

        while True:
            results = await self.search_documents(
                query=query,
                limit=batch_size,
                offset=offset
            )

            if not results["results"]:
                break

            yield results["results"]
            offset += batch_size

            if len(results["results"]) < batch_size:
                break

# Usage
async def main():
    async with AsyncAPIClient(api_key="your-api-key") as client:
        # Search
        results = await client.search_documents(
            query="vector database tutorial",
            limit=10,
            enable_hyde=True
        )

        # Bulk add
        bulk_result = await client.bulk_add_documents([
            "https://docs.example1.com",
            "https://docs.example2.com",
            "https://docs.example3.com"
        ])

        # Stream results
        async for batch in client.stream_search_results("machine learning", batch_size=5):
            for result in batch:
                print(f"- {result['title']}: {result['score']:.3f}")

asyncio.run(main())
```

### JavaScript/Node.js Integration

#### Basic Node.js Client

```javascript
// api-client.js
const axios = require("axios");

class DocumentDBClient {
  constructor(baseUrl = "http://localhost:8000/api/v1", apiKey = null) {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      headers: {
        "Content-Type": "application/json",
        ...(apiKey && { Authorization: `Bearer ${apiKey}` }),
      },
      timeout: 30000,
    });
  }

  async searchDocuments(query, options = {}) {
    const payload = {
      query,
      limit: options.limit || 10,
      score_threshold: options.scoreThreshold || 0.7,
      collection_name: options.collection || "documents",
      enable_hyde: options.enableHyde || false,
    };

    try {
      const response = await this.client.post("/search", payload);
      return response.data;
    } catch (error) {
      throw new Error(
        `Search failed: ${error.response?.data?.error || error.message}`
      );
    }
  }

  async addDocument(url, options = {}) {
    const payload = {
      url,
      collection_name: options.collection || "documents",
      doc_type: options.docType || "documentation",
      metadata: options.metadata || {},
      force_recrawl: options.forceRecrawl || false,
    };

    try {
      const response = await this.client.post("/documents", payload, {
        timeout: 120000, // 2 minutes for document processing
      });
      return response.data;
    } catch (error) {
      throw new Error(
        `Add document failed: ${error.response?.data?.error || error.message}`
      );
    }
  }

  async listCollections() {
    try {
      const response = await this.client.get("/collections");
      return response.data;
    } catch (error) {
      throw new Error(
        `List collections failed: ${
          error.response?.data?.error || error.message
        }`
      );
    }
  }
}

module.exports = DocumentDBClient;

// Usage example
const client = new DocumentDBClient(
  "http://localhost:8000/api/v1",
  "your-api-key"
);

async function example() {
  try {
    // Search documents
    const searchResults = await client.searchDocuments("authentication guide", {
      limit: 5,
      enableHyde: true,
    });

    console.log(`Found ${searchResults.results.length} results:`);
    searchResults.results.forEach((result) => {
      console.log(`- ${result.title} (${result.score.toFixed(3)})`);
    });

    // Add document
    const addResult = await client.addDocument(
      "https://docs.example.com/auth",
      {
        metadata: { category: "security", version: "v2.0" },
      }
    );

    console.log(`Document added: ${addResult.document_id}`);

    // List collections
    const collections = await client.listCollections();
    console.log(
      "Collections:",
      collections.collections.map((c) => c.name)
    );
  } catch (error) {
    console.error("Error:", error.message);
  }
}

example();
```

#### React Integration

```jsx
// hooks/useDocumentDB.js
import { useState, useCallback } from "react";
import DocumentDBClient from "../api/client";

export function useDocumentDB(apiKey) {
  const [client] = useState(
    () => new DocumentDBClient(process.env.REACT_APP_API_URL, apiKey)
  );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const searchDocuments = useCallback(
    async (query, options = {}) => {
      setLoading(true);
      setError(null);

      try {
        const results = await client.searchDocuments(query, options);
        return results;
      } catch (err) {
        setError(err.message);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  const addDocument = useCallback(
    async (url, options = {}) => {
      setLoading(true);
      setError(null);

      try {
        const result = await client.addDocument(url, options);
        return result;
      } catch (err) {
        setError(err.message);
        throw err;
      } finally {
        setLoading(false);
      }
    },
    [client]
  );

  return {
    searchDocuments,
    addDocument,
    loading,
    error,
  };
}

// components/DocumentSearch.jsx
import React, { useState } from "react";
import { useDocumentDB } from "../hooks/useDocumentDB";

function DocumentSearch({ apiKey }) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const { searchDocuments, loading, error } = useDocumentDB(apiKey);

  const handleSearch = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    try {
      const searchResults = await searchDocuments(query, {
        limit: 10,
        enableHyde: true,
      });
      setResults(searchResults.results);
    } catch (err) {
      console.error("Search failed:", err);
    }
  };

  return (
    <div className="document-search">
      <form onSubmit={handleSearch}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search documentation..."
          disabled={loading}
        />
        <button type="submit" disabled={loading || !query.trim()}>
          {loading ? "Searching..." : "Search"}
        </button>
      </form>

      {error && <div className="error">Error: {error}</div>}

      <div className="results">
        {results.map((result, index) => (
          <div key={index} className="result-item">
            <h3>
              <a href={result.url} target="_blank" rel="noopener noreferrer">
                {result.title}
              </a>
            </h3>
            <p>{result.content.substring(0, 200)}...</p>
            <div className="metadata">
              <span className="score">Score: {result.score.toFixed(3)}</span>
              <span className="type">{result.doc_type}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default DocumentSearch;
```

## 🐳 Docker Integration

### Docker Compose Setup

#### Complete Docker Compose Configuration

```yaml
# docker-compose.yml
version: "3.8"

services:
  ai-docs-vector-db:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - LOG_LEVEL=INFO
    depends_on:
      - qdrant
      - redis
    volumes:
      - ./config:/app/config
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: dragonflydb/dragonfly:latest
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: ["dragonfly", "--logtostderr"]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  qdrant_data:
  redis_data:

networks:
  default:
    name: ai-docs-network
```

#### Application Integration with Docker

```python
# docker_integration.py
import os
import asyncio
from contextlib import asynccontextmanager

class DockerizedDocumentDB:
    def __init__(self):
        self.base_url = os.getenv("API_BASE_URL", "http://localhost:8000")
        self.api_key = os.getenv("API_KEY")
        self.client = None

    @asynccontextmanager
    async def get_client(self):
        """Context manager for HTTP client with proper cleanup."""
        import aiohttp

        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )

        timeout = aiohttp.ClientTimeout(
            total=30,
            connect=10,
            sock_read=10
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        ) as session:
            yield session

    async def health_check(self):
        """Check if the dockerized service is healthy."""
        async with self.get_client() as client:
            try:
                async with client.get(f"{self.base_url}/health") as response:
                    return response.status == 200
            except Exception:
                return False

    async def wait_for_service(self, max_attempts=30, delay=2):
        """Wait for the dockerized service to become available."""
        for attempt in range(max_attempts):
            if await self.health_check():
                return True
            await asyncio.sleep(delay)
        return False

    async def search_with_retry(self, query: str, max_retries=3):
        """Search with automatic retry for transient failures."""
        for attempt in range(max_retries):
            try:
                async with self.get_client() as client:
                    payload = {"query": query, "limit": 10}
                    async with client.post(f"{self.base_url}/api/v1/search", json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status >= 500:
                            # Retry on server errors
                            if attempt < max_retries - 1:
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                        response.raise_for_status()
            except aiohttp.ClientError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        raise Exception(f"Failed to search after {max_retries} attempts")

# Usage
async def main():
    db = DockerizedDocumentDB()

    # Wait for service to be ready
    if not await db.wait_for_service():
        raise Exception("Service failed to start")

    # Use the service
    results = await db.search_with_retry("vector database")
    print(f"Found {len(results['results'])} results")

if __name__ == "__main__":
    asyncio.run(main())
```

### Kubernetes Deployment

#### Kubernetes Manifests

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-docs-vector-db
  labels:
    app: ai-docs-vector-db
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-docs-vector-db
  template:
    metadata:
      labels:
        app: ai-docs-vector-db
    spec:
      containers:
        - name: ai-docs-vector-db
          image: ai-docs-vector-db:latest
          ports:
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: api-secrets
                  key: openai-api-key
            - name: QDRANT_URL
              value: "http://qdrant:6333"
            - name: REDIS_URL
              value: "redis://redis:6379"
          resources:
            requests:
              memory: "512Mi"
              cpu: "250m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ai-docs-vector-db
spec:
  selector:
    app: ai-docs-vector-db
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: Secret
metadata:
  name: api-secrets
type: Opaque
data:
  openai-api-key: # base64 encoded API key
```

## 🔌 MCP Integration

### Claude Desktop Integration

#### Complete MCP Setup

```json
{
  "mcpServers": {
    "ai-docs-vector-db": {
      "command": "uv",
      "args": ["run", "python", "src/unified_mcp_server.py"],
      "cwd": "/absolute/path/to/ai-docs-vector-db-hybrid-scraper",
      "env": {
        "OPENAI_API_KEY": "sk-...",
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "FIRECRAWL_API_KEY": "fc-...",
        "QDRANT_URL": "http://localhost:6333",
        "REDIS_URL": "redis://localhost:6379",
        "LOG_LEVEL": "INFO",
        "ENABLE_CACHE": "true",
        "CACHE_TTL": "3600",
        "ENABLE_MONITORING": "true",
        "MAX_SEARCH_RESULTS": "20"
      }
    }
  }
}
```

#### Custom MCP Tool Development

```python
# custom_mcp_tool.py
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server

from src.models.api_contracts import SearchRequest
from src.services import QdrantService, EmbeddingManager

server = Server("custom-docs-tools")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available custom tools."""
    return [
        types.Tool(
            name="semantic_search",
            description="Perform semantic search across documentation with advanced filters",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "collection": {
                        "type": "string",
                        "description": "Collection to search in",
                        "default": "documents"
                    },
                    "semantic_threshold": {
                        "type": "number",
                        "description": "Semantic similarity threshold (0.0-1.0)",
                        "default": 0.7
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "Include document metadata",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="analyze_documentation_gaps",
            description="Analyze documentation for gaps and coverage",
            inputSchema={
                "type": "object",
                "properties": {
                    "collection": {
                        "type": "string",
                        "description": "Collection to analyze",
                        "default": "documents"
                    },
                    "topic_areas": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Topic areas to check coverage for"
                    }
                },
                "required": ["topic_areas"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None
) -> list[types.TextContent]:
    """Handle tool calls."""

    if name == "semantic_search":
        return await handle_semantic_search(arguments or {})
    elif name == "analyze_documentation_gaps":
        return await handle_gap_analysis(arguments or {})
    else:
        raise ValueError(f"Unknown tool: {name}")

async def handle_semantic_search(args: dict) -> list[types.TextContent]:
    """Handle semantic search with advanced filtering."""
    query = args["query"]
    collection = args.get("collection", "documents")
    threshold = args.get("semantic_threshold", 0.7)
    include_metadata = args.get("include_metadata", True)

    # Use the services
    config = get_config()
    async with QdrantService(config) as qdrant:
        # Perform search
        request = SearchRequest(
            query=query,
            collection_name=collection,
            score_threshold=threshold,
            limit=10
        )

        results = await qdrant.search_vectors(
            collection_name=request.collection_name,
            query=request.query,
            limit=request.limit,
            score_threshold=request.score_threshold
        )

        # Format results
        formatted_results = []
        for result in results:
            result_text = f"**{result.title}** (Score: {result.score:.3f})\n"
            result_text += f"URL: {result.url}\n"
            result_text += f"Content: {result.content[:300]}...\n"

            if include_metadata and result.metadata:
                result_text += f"Metadata: {result.metadata}\n"

            formatted_results.append(result_text)

        response_text = f"Found {len(results)} results for '{query}':\n\n"
        response_text += "\n---\n".join(formatted_results)

        return [types.TextContent(type="text", text=response_text)]

async def handle_gap_analysis(args: dict) -> list[types.TextContent]:
    """Analyze documentation gaps for specified topic areas."""
    collection = args.get("collection", "documents")
    topic_areas = args["topic_areas"]

    config = get_config()
    async with QdrantService(config) as qdrant:
        gap_analysis = []

        for topic in topic_areas:
            # Search for content in this topic area
            results = await qdrant.search_vectors(
                collection_name=collection,
                query=f"{topic} documentation guide tutorial",
                limit=5,
                score_threshold=0.5
            )

            coverage_score = len(results) / 5.0  # Normalize to 0-1
            quality_score = sum(r.score for r in results) / len(results) if results else 0

            gap_analysis.append({
                "topic": topic,
                "coverage_score": coverage_score,
                "quality_score": quality_score,
                "documents_found": len(results),
                "top_documents": [r.title for r in results[:3]]
            })

        # Generate report
        report = "# Documentation Gap Analysis\n\n"

        for analysis in gap_analysis:
            status = "✅ Good" if analysis["coverage_score"] > 0.6 else "⚠️ Partial" if analysis["coverage_score"] > 0.2 else "❌ Poor"

            report += f"## {analysis['topic']} - {status}\n"
            report += f"- Coverage: {analysis['coverage_score']:.1%}\n"
            report += f"- Quality: {analysis['quality_score']:.3f}\n"
            report += f"- Documents: {analysis['documents_found']}\n"

            if analysis["top_documents"]:
                report += f"- Top docs: {', '.join(analysis['top_documents'])}\n"

            report += "\n"

        return [types.TextContent(type="text", text=report)]

# Run the server
if __name__ == "__main__":
    import asyncio
    asyncio.run(server.run())
```

## 🔍 Testing Integration

### Integration Test Examples

#### API Integration Tests

```python
# test_integration.py
import pytest
import asyncio
import aiohttp
from ai_docs_vector_db import DocumentDB

class TestAPIIntegration:
    @pytest.fixture
    async def api_client(self):
        """Create API client for testing."""
        async with aiohttp.ClientSession() as session:
            yield session

    @pytest.fixture
    async def document_db(self):
        """Create document DB instance for testing."""
        config = get_test_config()
        db = DocumentDB(config)
        await db.initialize()
        try:
            yield db
        finally:
            await db.cleanup()

    async def test_search_integration(self, api_client):
        """Test search API integration."""
        payload = {
            "query": "test query",
            "limit": 5,
            "score_threshold": 0.7
        }

        async with api_client.post(
            "http://localhost:8000/api/v1/search",
            json=payload
        ) as response:
            assert response.status == 200
            data = await response.json()

            assert data["success"] is True
            assert "results" in data
            assert "query_time_ms" in data
            assert len(data["results"]) <= 5

    async def test_document_lifecycle(self, document_db):
        """Test complete document lifecycle."""
        # Add document
        add_result = await document_db.add_url("https://example.com/test-doc")
        assert add_result.success
        document_id = add_result.document_id

        # Search for document
        search_results = await document_db.search("test document")
        assert any(r.id == document_id for r in search_results.results)

        # Update metadata
        update_result = await document_db.update_document(
            document_id,
            metadata={"updated": True}
        )
        assert update_result.success

        # Delete document
        delete_result = await document_db.delete_document(document_id)
        assert delete_result.success

        # Verify deletion
        search_results = await document_db.search("test document")
        assert not any(r.id == document_id for r in search_results.results)

    async def test_error_handling(self, api_client):
        """Test API error handling."""
        # Invalid request
        async with api_client.post(
            "http://localhost:8000/api/v1/search",
            json={"query": ""}  # Empty query should fail
        ) as response:
            assert response.status == 400
            data = await response.json()
            assert data["success"] is False
            assert "error" in data

    async def test_performance_requirements(self, document_db):
        """Test performance requirements are met."""
        import time

        # Search should complete within 1 second for simple queries
        start_time = time.time()
        results = await document_db.search("quick test query")
        end_time = time.time()

        assert end_time - start_time < 1.0
        assert results.query_time_ms < 1000
```

#### Browser Integration Tests

```python
# test_browser_integration.py
import pytest
from ai_docs_vector_db import UnifiedBrowserManager, UnifiedScrapingRequest

class TestBrowserIntegration:
    @pytest.fixture
    async def browser_manager(self):
        """Create browser manager for testing."""
        config = get_test_config()
        manager = UnifiedBrowserManager(config)
        await manager.initialize()
        try:
            yield manager
        finally:
            await manager.cleanup()

    async def test_tier_selection(self, browser_manager):
        """Test automatic tier selection works correctly."""
        # Static site should use lightweight tier
        static_response = await browser_manager.scrape(
            url="https://httpbin.org/html",
            tier="auto"
        )
        assert static_response.success
        assert static_response.tier_used in ["lightweight", "crawl4ai"]

        # Complex site might use higher tier
        complex_response = await browser_manager.scrape(
            url="https://docs.python.org/3/",
            tier="auto"
        )
        assert complex_response.success
        assert complex_response.tier_used in ["crawl4ai", "crawl4ai_enhanced"]

    async def test_custom_interactions(self, browser_manager):
        """Test custom interactions work correctly."""
        request = UnifiedScrapingRequest(
            url="https://httpbin.org/forms/post",
            tier="playwright",
            interaction_required=True,
            custom_actions=[
                {"type": "fill", "selector": "input[name='email']", "value": "test@example.com"},
                {"type": "click", "selector": "button[type='submit']"}
            ]
        )

        response = await browser_manager.scrape(request)
        assert response.success
        assert "test@example.com" in response.content

    async def test_error_recovery(self, browser_manager):
        """Test error recovery and fallback mechanisms."""
        # Force a tier that might fail, should fallback
        response = await browser_manager.scrape(
            url="https://httpbin.org/status/503",  # Server error
            tier="auto"
        )

        # Should either succeed with fallback or fail gracefully
        if not response.success:
            assert response.error is not None
            assert response.fallback_attempted is True
```

## 📊 Monitoring and Observability

### Application Monitoring Integration

#### Metrics Collection

```python
# monitoring_integration.py
import time
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str]

class MetricsCollector:
    def __init__(self):
        self.metrics: List[MetricPoint] = []

    def record_search_latency(self, latency_ms: float, collection: str):
        """Record search latency metric."""
        self.metrics.append(MetricPoint(
            name="search_latency_ms",
            value=latency_ms,
            timestamp=time.time(),
            tags={"collection": collection}
        ))

    def record_document_processing_time(self, processing_ms: float, doc_type: str):
        """Record document processing time."""
        self.metrics.append(MetricPoint(
            name="document_processing_ms",
            value=processing_ms,
            timestamp=time.time(),
            tags={"doc_type": doc_type}
        ))

    def record_cache_hit_rate(self, hit_rate: float, cache_type: str):
        """Record cache hit rate."""
        self.metrics.append(MetricPoint(
            name="cache_hit_rate",
            value=hit_rate,
            timestamp=time.time(),
            tags={"cache_type": cache_type}
        ))

    def get_metrics(self, since: Optional[float] = None) -> List[MetricPoint]:
        """Get metrics since timestamp."""
        if since is None:
            return self.metrics.copy()
        return [m for m in self.metrics if m.timestamp >= since]

    def export_prometheus_format(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for metric in self.metrics:
            tags_str = ",".join(f'{k}="{v}"' for k, v in metric.tags.items())
            line = f'{metric.name}{{{tags_str}}} {metric.value} {int(metric.timestamp * 1000)}'
            lines.append(line)

        return "\n".join(lines)

# Integration with DocumentDB
class MonitoredDocumentDB:
    def __init__(self, config, metrics_collector: MetricsCollector):
        self.db = DocumentDB(config)
        self.metrics = metrics_collector

    async def initialize(self):
        await self.db.initialize()

    async def search_with_metrics(self, query: str, **kwargs):
        """Search with automatic metrics collection."""
        start_time = time.time()

        try:
            results = await self.db.search(query, **kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Record metrics
            self.metrics.record_search_latency(
                latency_ms=latency_ms,
                collection=kwargs.get("collection", "documents")
            )

            return results

        except Exception as e:
            # Record error metrics
            self.metrics.record_search_latency(
                latency_ms=-1,  # Error indicator
                collection=kwargs.get("collection", "documents")
            )
            raise

    async def cleanup(self):
        await self.db.cleanup()

# Usage
metrics = MetricsCollector()
db = MonitoredDocumentDB(config, metrics)

await db.initialize()
results = await db.search_with_metrics("test query")

# Export metrics
prometheus_metrics = metrics.export_prometheus_format()
print(prometheus_metrics)
```

#### Health Check Integration

```python
# health_checks.py
import asyncio
import aiohttp
from typing import Dict, Any, List
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthChecker:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def check_api_health(self) -> Dict[str, Any]:
        """Check API endpoint health."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        return {
                            "status": HealthStatus.HEALTHY,
                            "response_time_ms": response.headers.get("X-Response-Time", "unknown"),
                            "details": await response.json()
                        }
                    else:
                        return {
                            "status": HealthStatus.UNHEALTHY,
                            "error": f"HTTP {response.status}",
                            "details": {}
                        }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": str(e),
                "details": {}
            }

    async def check_search_functionality(self) -> Dict[str, Any]:
        """Check if search functionality works."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {"query": "health check test", "limit": 1}
                async with session.post(
                    f"{self.base_url}/api/v1/search",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "status": HealthStatus.HEALTHY,
                            "query_time_ms": data.get("query_time_ms", "unknown"),
                            "results_count": len(data.get("results", []))
                        }
                    else:
                        return {
                            "status": HealthStatus.UNHEALTHY,
                            "error": f"Search failed with HTTP {response.status}"
                        }
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "error": f"Search check failed: {str(e)}"
            }

    async def comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health checks."""
        checks = await asyncio.gather(
            self.check_api_health(),
            self.check_search_functionality(),
            return_exceptions=True
        )

        api_health = checks[0] if not isinstance(checks[0], Exception) else {
            "status": HealthStatus.UNHEALTHY,
            "error": str(checks[0])
        }

        search_health = checks[1] if not isinstance(checks[1], Exception) else {
            "status": HealthStatus.UNHEALTHY,
            "error": str(checks[1])
        }

        # Determine overall status
        statuses = [api_health["status"], search_health["status"]]

        if all(s == HealthStatus.HEALTHY for s in statuses):
            overall_status = HealthStatus.HEALTHY
        elif any(s == HealthStatus.UNHEALTHY for s in statuses):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED

        return {
            "overall_status": overall_status,
            "timestamp": time.time(),
            "checks": {
                "api": api_health,
                "search": search_health
            },
            "summary": {
                "healthy_checks": sum(1 for s in statuses if s == HealthStatus.HEALTHY),
                "total_checks": len(statuses)
            }
        }

# Usage
health_checker = HealthChecker("http://localhost:8000")
health_report = await health_checker.comprehensive_health_check()

if health_report["overall_status"] == HealthStatus.UNHEALTHY:
    # Alert or take corrective action
    print("System is unhealthy!")
    for check_name, check_result in health_report["checks"].items():
        if check_result["status"] == HealthStatus.UNHEALTHY:
            print(f"  {check_name}: {check_result.get('error', 'Unknown error')}")
```

## 🔧 Production Deployment Patterns

### Load Balancing Integration

```python
# load_balancer_integration.py
import random
import asyncio
import aiohttp
from typing import List, Dict, Any

class LoadBalancedClient:
    def __init__(self, endpoints: List[str], api_key: str = None):
        self.endpoints = endpoints
        self.api_key = api_key
        self.endpoint_health = {endpoint: True for endpoint in endpoints}

    def get_healthy_endpoint(self) -> str:
        """Get a healthy endpoint using round-robin."""
        healthy_endpoints = [ep for ep, healthy in self.endpoint_health.items() if healthy]

        if not healthy_endpoints:
            # Fallback to any endpoint if all are marked unhealthy
            healthy_endpoints = self.endpoints

        return random.choice(healthy_endpoints)

    async def request_with_failover(self, path: str, method: str = "GET", **kwargs):
        """Make request with automatic failover."""
        last_exception = None

        for attempt in range(len(self.endpoints)):
            endpoint = self.get_healthy_endpoint()

            try:
                headers = kwargs.get("headers", {})
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                kwargs["headers"] = headers

                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=f"{endpoint}{path}",
                        timeout=aiohttp.ClientTimeout(total=30),
                        **kwargs
                    ) as response:
                        if response.status < 500:
                            # Mark endpoint as healthy on success
                            self.endpoint_health[endpoint] = True
                            return await response.json()
                        else:
                            # Server error, try next endpoint
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )

            except Exception as e:
                # Mark endpoint as unhealthy
                self.endpoint_health[endpoint] = False
                last_exception = e
                continue

        # All endpoints failed
        raise Exception(f"All endpoints failed. Last error: {last_exception}")

    async def search(self, query: str, **kwargs) -> Dict[str, Any]:
        """Search with load balancing and failover."""
        payload = {
            "query": query,
            "limit": kwargs.get("limit", 10),
            **kwargs
        }

        return await self.request_with_failover(
            "/api/v1/search",
            method="POST",
            json=payload
        )

# Usage
client = LoadBalancedClient([
    "http://api1.example.com",
    "http://api2.example.com",
    "http://api3.example.com"
], api_key="your-api-key")

results = await client.search("load balanced query")
```

This comprehensive integration guide provides everything you need to integrate
the AI Documentation Vector DB system into your applications. Choose the
integration method that best fits your needs and follow the patterns shown for
robust, production-ready implementations.

---

_🔗 Ready to integrate! Whether you're building web applications, CLI tools, or
enterprise systems, these patterns provide a solid foundation for leveraging the
full power of the AI Documentation Vector DB system._
