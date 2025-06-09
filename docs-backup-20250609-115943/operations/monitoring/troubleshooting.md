# 🔧 Troubleshooting Guide

> **Status**: Current  
> **Last Updated**: 2025-06-09  
> **Purpose**: Troubleshooting operations guide  
> **Audience**: System administrators and DevOps teams

> **V1 Status**: Enhanced with Query API, HyDE, and DragonflyDB troubleshooting  
> **Coverage**: Common issues and V1-specific performance problems

## 📋 Quick Diagnosis

Run our V1 diagnostic scripts first:

```bash
# Health check for entire system
./scripts/health-check.sh

# MCP-specific health check  
./scripts/health-check-mcp.sh

# Performance benchmark
python src/performance_test.py --quick
```

## 🔍 Installation Issues

### Issue: "uv not found" or "uvx command not recognized"

**Symptoms**: Command not found errors when running setup
**Solution**:

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell environment
source ~/.bashrc
# or
exec $SHELL

# Verify installation
uv --version
uvx --version
```

### Issue: "ModuleNotFoundError: No module named 'FlagEmbedding'"

**Symptoms**: Import errors when running reranking features
**Solution**:

```bash
# Install with uv (recommended)
uv add FlagEmbedding>=1.3.0

# Alternative: pip installation
pip install FlagEmbedding>=1.3.0

# If you need PyTorch
uv add torch>=2.0.0

# Verify installation
python -c "from FlagEmbedding import FlagReranker; print('✅ FlagEmbedding installed')"
```

### Issue: "crawl4ai installation fails"

**Symptoms**: Errors during pip/uv install of crawl4ai
**Solution**:

```bash
# Use latest installation method
uv add "crawl4ai[all]>=0.6.0"

# If playwright issues occur
playwright install

# For Ubuntu/Debian systems
sudo apt-get update
sudo apt-get install -y libnss3-dev libxss1 libasound2

# Verify installation
python -c "import crawl4ai; print('✅ Crawl4AI installed')"
```

## 🐳 Docker & Qdrant Issues

### Issue: "Connection refused" to Qdrant

**Symptoms**:

- `curl http://localhost:6333/health` fails
- MCP servers can't connect to Qdrant
- Scraper fails with connection errors

**Diagnosis**:

```bash
# Check if Docker is running
docker ps

# Check if Qdrant container is running
docker ps | grep qdrant

# Check port availability
netstat -tulpn | grep 6333
lsof -i :6333
```

**Solutions**:

```bash
# Solution 1: Start/restart Qdrant
docker-compose down
docker-compose up -d

# Solution 2: Check Docker Desktop (Windows/macOS)
# Ensure Docker Desktop is running and WSL integration is enabled

# Solution 3: Fix port conflicts
# If port 6333 is in use, modify docker-compose.yml:
ports:
  - "6334:6333"  # Use different external port

# Solution 4: Reset Docker networking
docker network prune
docker-compose down && docker-compose up -d

# Solution 5: Check firewall (Linux)
sudo ufw allow 6333
sudo iptables -A INPUT -p tcp --dport 6333 -j ACCEPT
```

### Issue: "Qdrant container keeps restarting"

**Symptoms**:

- Container shows "Restarting" status
- Logs show memory or permission errors

**Diagnosis**:

```bash
# Check container logs
docker logs qdrant-advanced

# Check resource usage
docker stats qdrant-advanced

# Check disk space
df -h ~/.qdrant_data
```

**Solutions**:

```bash
# Solution 1: Increase memory allocation
# Edit docker-compose.yml memory limits:
deploy:
  resources:
    limits:
      memory: 4G  # Increase if needed
    reservations:  
      memory: 2G

# Solution 2: Fix data directory permissions
sudo chown -R $USER:$USER ~/.qdrant_data
chmod -R 755 ~/.qdrant_data

# Solution 3: Clear corrupted data (WARNING: loses data)
docker-compose down
rm -rf ~/.qdrant_data/*
docker-compose up -d

# Solution 4: Use alternative data path
# Edit docker-compose.yml volumes:
volumes:
  - ./qdrant_data:/qdrant/storage:z
```

### Issue: "Slow Qdrant performance"

**Symptoms**:

- Search queries take >5 seconds
- High memory usage
- CPU constantly high

**Solutions**:

```bash
# Solution 1: Enable quantization (if not already)
# Check collection settings:
curl http://localhost:6333/collections/documents

# Solution 2: Optimize Docker resource allocation
# In docker-compose.yml:
environment:
  - QDRANT__STORAGE__PERFORMANCE__MAX_SEARCH_THREADS=8
  - QDRANT__STORAGE__QUANTIZATION__ALWAYS_RAM=true

# Solution 3: Database maintenance
python src/manage_vector_db.py optimize

# Solution 4: Monitor and tune
docker exec qdrant-sota-2025 qdrant-cli collection info documents
```

## 🤖 MCP Server Issues

### Issue: "MCP server not found" in Claude Desktop

**Symptoms**:

- Claude Desktop shows "No MCP servers"
- MCP-related commands don't work
- Server appears disconnected

**Diagnosis**:

```bash
# Check if servers are installed
uvx list | grep qdrant
npx -y firecrawl-mcp --version

# Test servers directly
uvx mcp-server-qdrant --help
npx -y firecrawl-mcp --help
```

**Solutions**:

```bash
# Solution 1: Reinstall MCP servers
uvx uninstall mcp-server-qdrant
uvx install mcp-server-qdrant

npm uninstall -g firecrawl-mcp  # if globally installed
# Test with npx instead

# Solution 2: Check Claude Desktop config file location
# Windows: %APPDATA%\Claude\claude_desktop_config.json
# macOS: ~/Library/Application Support/Claude/claude_desktop_config.json  
# Linux: ~/.config/claude-desktop/config.json

# Verify config file exists and has correct JSON syntax
cat ~/.config/claude-desktop/config.json | jq .

# Solution 3: Use absolute paths in config
{
  "mcpServers": {
    "qdrant": {
      "command": "/home/user/.local/bin/uvx",
      "args": ["mcp-server-qdrant"],
      "env": { ... }
    }
  }
}

# Solution 4: Restart Claude Desktop completely
# Close all instances, wait 10 seconds, restart
```

### Issue: "Invalid API key" errors

**Symptoms**:

- OpenAI embedding requests fail
- Firecrawl scraping fails
- Authentication errors in logs

**Solutions**:

```bash
# Solution 1: Verify API keys
# Test OpenAI key:
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  "https://api.openai.com/v1/models"

# Test Firecrawl key:
curl -H "Authorization: Bearer $FIRECRAWL_API_KEY" \
  "https://api.firecrawl.dev/v0/account"

# Solution 2: Check environment variables in Claude config
# Ensure quotes are properly escaped in JSON:
"OPENAI_API_KEY": "sk-your-key-here"
"FIRECRAWL_API_KEY": "fc-your-key-here"

# Solution 3: Use .env file for consistency
# Create .env file:
echo "OPENAI_API_KEY=your_key_here" >> .env
echo "FIRECRAWL_API_KEY=your_key_here" >> .env

# Update Claude config to reference environment:
"env": {
  "OPENAI_API_KEY": "${OPENAI_API_KEY}",
  "FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"
}
```

### Issue: "MCP timeouts" or slow responses

**Symptoms**:

- Claude operations timeout
- Long waits for MCP responses
- Partial results returned

**Solutions**:

```bash
# Solution 1: Increase timeouts in MCP config
"env": {
  "TIMEOUT": "60000",           # 60 seconds
  "REQUEST_TIMEOUT": "45000",   # 45 seconds
  "SEARCH_TIMEOUT": "30000"     # 30 seconds
}

# Solution 2: Optimize batch sizes
"env": {
  "BATCH_SIZE": "32",           # Reduce if memory issues
  "MAX_CONCURRENT": "4",        # Reduce concurrent operations
  "RATE_LIMIT_RPM": "60"        # API rate limiting
}

# Solution 3: Enable streaming for large operations
"env": {
  "ENABLE_STREAMING": "true",
  "STREAM_CHUNK_SIZE": "1000"
}
```

## ⚡ Performance Issues

### Issue: "Slow embedding generation"

**Symptoms**:

- Scraping takes hours instead of minutes
- High CPU usage during embedding
- Memory exhaustion

**Solutions**:

```bash
# Solution 1: Enable FastEmbed (50% speedup)
# Edit src/crawl4ai_bulk_embedder.py:
EMBEDDING_CONFIG = EmbeddingConfig(
    provider=EmbeddingProvider.FASTEMBED,  # Instead of OpenAI for speed
    dense_model=EmbeddingModel.BGE_SMALL_EN_V15
)

# Solution 2: Use quantization
EMBEDDING_CONFIG = EmbeddingConfig(
    enable_quantization=True,
    matryoshka_dimensions=[1024, 512, 256]  # Smaller dimensions
)

# Solution 3: Optimize batch processing
MAX_CONCURRENT_CRAWLS = 5  # Reduce if system is overwhelmed
CHUNK_SIZE = 1600          # Optimal research-backed size
BATCH_SIZE = 16            # Reduce if memory issues

# Solution 4: Use CPU optimization
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python src/crawl4ai_bulk_embedder.py
```

### Issue: "High memory usage"

**Symptoms**:

- System runs out of RAM
- OOM (Out of Memory) errors
- Swap thrashing

**Solutions**:

```bash
# Solution 1: Enable memory-efficient settings
# In docker-compose.yml:
environment:
  - QDRANT__STORAGE__ON_DISK_PAYLOAD=true
  - QDRANT__STORAGE__QUANTIZATION__ALWAYS_RAM=false

# Solution 2: Reduce batch sizes
# Edit configuration:
BATCH_SIZE = 8             # Smaller batches
MAX_CONCURRENT_CRAWLS = 3  # Fewer concurrent operations

# Solution 3: Use memory monitoring
# Add to scraper:
import psutil
import gc

def monitor_memory():
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        gc.collect()  # Force garbage collection
        
# Solution 4: Process in chunks
# For large documentation sites, process in smaller batches:
python src/crawl_single_site.py "https://docs.qdrant.tech/" 20
python src/crawl_single_site.py "https://docs.qdrant.tech/concepts/" 20
```

### Issue: "Slow search performance"

**Symptoms**:

- Vector searches take >5 seconds
- Timeouts in Claude Desktop
- High CPU during search

**Solutions**:

```bash
# Solution 1: Optimize Qdrant HNSW parameters
curl -X PUT "http://localhost:6333/collections/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "hnsw_config": {
      "m": 16,
      "ef_construct": 128,
      "ef": 64
    }
  }'

# Solution 2: Use quantization
curl -X PUT "http://localhost:6333/collections/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "quantization_config": {
      "scalar": {
        "type": "int8",
        "always_ram": true
      }
    }
  }'

# Solution 3: Index optimization
python src/manage_vector_db.py optimize

# Solution 4: Reduce search scope
# Limit search results:
results = qdrant_client.search(
    collection_name="documents",
    query_vector=embedding,
    limit=10,  # Instead of 50+
    with_payload=False  # If metadata not needed
)
```

## 📊 Data & Content Issues

### Issue: "Poor search accuracy"

**Symptoms**:

- Irrelevant search results
- Missing expected content
- Low similarity scores

**Solutions**:

```bash
# Solution 1: Enable hybrid search + reranking
EMBEDDING_CONFIG = EmbeddingConfig(
    provider=EmbeddingProvider.HYBRID,
    search_strategy=VectorSearchStrategy.HYBRID_RRF,
    enable_reranking=True,
    reranker_model="BAAI/bge-reranker-v2-m3"
)

# Solution 2: Optimize chunk size
CHUNK_SIZE = 1600      # Research-optimal for most content
CHUNK_OVERLAP = 200    # Better context preservation

# Solution 3: Update embeddings with better model
# Migrate to text-embedding-3-small:
python src/manage_vector_db.py migrate-embeddings \
  --new-model text-embedding-3-small

# Solution 4: Re-index with metadata
# Include more metadata for filtering:
metadata = {
    "title": page_title,
    "url": page_url, 
    "section": section_name,
    "last_updated": datetime.now().isoformat()
}
```

### Issue: "Embedding dimension mismatch"

**Symptoms**:

- Errors when adding new vectors
- "Vector dimension doesn't match"
- Collection creation failures

**Solutions**:

```bash
# Solution 1: Check current collection config
curl http://localhost:6333/collections/documents

# Solution 2: Create new collection with correct dimensions
curl -X DELETE http://localhost:6333/collections/documents
curl -X PUT "http://localhost:6333/collections/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "vectors": {
      "size": 1536,  # text-embedding-3-small dimension
      "distance": "Cosine"
    }
  }'

# Solution 3: Migrate data to new collection
python src/manage_vector_db.py create-collection \
  --name documents_new \
  --dimension 1536 \
  --distance cosine

python src/manage_vector_db.py migrate \
  --from documents \
  --to documents_new

# Solution 4: Update configuration consistency
# Ensure all configs use same model:
# - Claude Desktop MCP config
# - Local scraper config  
# - Environment variables
```

### Issue: "Duplicate content in database"

**Symptoms**:

- Same content appears multiple times
- Inflated database size
- Redundant search results

**Solutions**:

```bash
# Solution 1: Enable deduplication
python src/manage_vector_db.py deduplicate \
  --similarity-threshold 0.95

# Solution 2: Use content hashing
# In scraper configuration:
ENABLE_DEDUPLICATION = True
DEDUP_THRESHOLD = 0.95
CONTENT_HASH_ALGORITHM = "sha256"

# Solution 3: Clean database
python src/manage_vector_db.py clean \
  --remove-duplicates \
  --remove-empty \
  --remove-invalid

# Solution 4: Prevent duplicates during scraping
# Add URL tracking:
processed_urls = set()
def should_process_url(url):
    if url in processed_urls:
        return False
    processed_urls.add(url)
    return True
```

## 🔐 Security & API Issues

### Issue: "Rate limiting errors"

**Symptoms**:

- "Rate limit exceeded" errors
- 429 HTTP status codes
- Slow API responses

**Solutions**:

```bash
# Solution 1: Implement rate limiting
# Add to scraper config:
RATE_LIMIT_RPM = 60        # 60 requests per minute
RATE_LIMIT_TPM = 150000    # 150k tokens per minute
BACKOFF_FACTOR = 2         # Exponential backoff

# Solution 2: Use batching
# Process embeddings in batches:
import time
for batch in chunks(texts, batch_size=20):
    embeddings = get_embeddings(batch)
    time.sleep(1)  # Rate limiting pause

# Solution 3: Switch to local models for bulk operations
# Use FastEmbed for bulk processing:
EMBEDDING_CONFIG = EmbeddingConfig(
    provider=EmbeddingProvider.FASTEMBED,  # No API limits
    dense_model=EmbeddingModel.BGE_SMALL_EN_V15
)

# Solution 4: Upgrade API plan
# For high-volume usage, consider:
# - OpenAI Tier 2+ accounts
# - Dedicated Firecrawl plans
```

### Issue: "API key security warnings"

**Symptoms**:

- Keys visible in config files
- Security scanner alerts
- Accidental key exposure

**Solutions**:

```bash
# Solution 1: Use environment variables only
# Remove keys from Claude config, use env vars:
"env": {
  "OPENAI_API_KEY": "${OPENAI_API_KEY}",
  "FIRECRAWL_API_KEY": "${FIRECRAWL_API_KEY}"
}

# Solution 2: Use secure credential storage
# Windows Credential Manager
# macOS Keychain
# Linux secret-service

# Solution 3: Rotate API keys regularly
# Set calendar reminders to rotate keys monthly

# Solution 4: Use restricted API keys
# Create API keys with minimal required permissions
# Restrict by IP address if possible
```

## 🧪 Testing & Validation

### Automated Diagnostic Script

Create `scripts/diagnose-issues.sh`:

```bash
#!/bin/bash
set -e

echo "🔍 SOTA 2025 System Diagnostics"
echo "================================"

# Check Python environment
echo "Checking Python environment..."
python --version
uv --version || echo "❌ uv not installed"

# Check required packages
echo "Checking Python packages..."
python -c "import crawl4ai; print('✅ crawl4ai')" || echo "❌ crawl4ai"
python -c "import qdrant_client; print('✅ qdrant_client')" || echo "❌ qdrant_client"
python -c "import openai; print('✅ openai')" || echo "❌ openai"
python -c "from FlagEmbedding import FlagReranker; print('✅ FlagEmbedding')" || echo "❌ FlagEmbedding"

# Check Docker and Qdrant
echo "Checking Docker services..."
docker --version || echo "❌ Docker not available"
docker ps | grep qdrant || echo "❌ Qdrant container not running"

# Test Qdrant connection
echo "Testing Qdrant connection..."
curl -s http://localhost:6333/health | grep -q "ok" && echo "✅ Qdrant healthy" || echo "❌ Qdrant unhealthy"

# Check MCP servers
echo "Checking MCP servers..."
uvx mcp-server-qdrant --help > /dev/null 2>&1 && echo "✅ Qdrant MCP" || echo "❌ Qdrant MCP"
npx -y firecrawl-mcp --help > /dev/null 2>&1 && echo "✅ Firecrawl MCP" || echo "❌ Firecrawl MCP"

# Check API keys
echo "Checking API configuration..."
[ -n "$OPENAI_API_KEY" ] && echo "✅ OpenAI API key set" || echo "❌ OpenAI API key missing"
[ -n "$FIRECRAWL_API_KEY" ] && echo "✅ Firecrawl API key set" || echo "⚠️  Firecrawl API key missing (optional)"

# Test database
echo "Testing database operations..."
python -c "
from qdrant_client import QdrantClient
client = QdrantClient('localhost', port=6333)
collections = client.get_collections()
print(f'✅ Found {len(collections.collections)} collections')
" || echo "❌ Database test failed"

echo "================================"
echo "Diagnostics complete"
```

### Performance Benchmark

Create `scripts/benchmark-system.sh`:

```bash
#!/bin/bash
set -e

echo "⚡ SOTA 2025 Performance Benchmark"
echo "=================================="

# Test embedding speed
echo "Testing embedding generation speed..."
python -c "
import time
from openai import OpenAI
client = OpenAI()

start = time.time()
response = client.embeddings.create(
    input='This is a test sentence for benchmarking embedding generation speed.',
    model='text-embedding-3-small'
)
end = time.time()
print(f'✅ Embedding generation: {(end-start)*1000:.1f}ms')
" || echo "❌ Embedding test failed"

# Test vector search speed
echo "Testing vector search speed..."
python -c "
import time
from qdrant_client import QdrantClient
import numpy as np

client = QdrantClient('localhost', port=6333)
test_vector = np.random.random(1536).tolist()

start = time.time()
results = client.search(
    collection_name='documents',
    query_vector=test_vector,
    limit=5
)
end = time.time()
print(f'✅ Vector search: {(end-start)*1000:.1f}ms')
" || echo "❌ Search test failed"

# Test Docker performance
echo "Testing Docker performance..."
docker stats --no-stream qdrant-sota-2025 | tail -n 1

echo "=================================="
echo "Benchmark complete"
```

## 🆕 V1-Specific Troubleshooting

### Issue: Query API Returns Fewer Results Than Expected

**Symptoms**: `query_points()` returns fewer results than traditional `search()`  
**Cause**: Aggressive prefetch filtering or fusion settings

**Solution**:

```python
# Increase prefetch limits
query_request = QueryRequest(
    prefetch=[
        PrefetchQuery(
            query=embedding,
            using="dense",
            limit=200,  # Increase from 100
            filter=None  # Remove filters temporarily
        )
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=20  # Get more candidates
)
```

### Issue: HyDE Increasing Latency Too Much

**Symptoms**: Search latency > 500ms with HyDE enabled  
**Cause**: Synchronous LLM generation or no caching

**Solution**:

```python
# 1. Enable aggressive caching
hyde_service = HyDEService(
    llm_client=llm,
    embedding_service=embedder,
    cache=DragonflyCache(ttl=86400)  # 24hr cache
)

# 2. Use faster LLM
hyde_config = {
    "model": "gpt-3.5-turbo",  # Not GPT-4
    "max_tokens": 150,  # Reduce from 200
    "temperature": 0.5  # Lower for consistency
}

# 3. Pre-generate for common queries
common_queries = ["how to", "what is", "tutorial"]
await hyde_service.pre_warm_cache(common_queries)
```

### Issue: DragonflyDB Memory Usage Growing

**Symptoms**: DragonflyDB consuming > 4GB RAM  
**Cause**: No eviction policy or compression disabled

**Solution**:

```yaml
# docker-compose.yml
dragonfly:
  image: docker.dragonflydb.io/dragonflydb/dragonfly:latest
  command: >
    --cache_mode
    --maxmemory_policy=allkeys-lru
    --maxmemory=4gb
    --compression=zstd
    --compression_level=3
  environment:
    - DRAGONFLY_THREADS=8
```

### Issue: Payload Index Queries Still Slow

**Symptoms**: Filtered searches taking > 100ms  
**Cause**: Indexes not created or wrong field types

**Solution**:

```python
# 1. Verify indexes exist
info = await qdrant.get_collection(collection_name)
print(f"Indexed fields: {info.payload_schema}")

# 2. Create missing indexes
await qdrant.create_payload_index(
    collection_name=collection_name,
    field_name="language",
    field_schema=PayloadSchemaType.KEYWORD,  # Not TEXT
    wait=True  # Wait for completion
)

# 3. Monitor index usage
await qdrant.search(
    collection_name=collection_name,
    query_vector=embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="language",
                match=MatchValue(value="python")
            )
        ]
    ),
    with_payload=True,
    score_threshold=0.7
)
```

### Issue: Collection Alias Update Failing

**Symptoms**: Alias operations timeout or fail  
**Cause**: Concurrent operations or invalid collection state

**Solution**:

```python
# 1. Use atomic operations
async def safe_alias_update(old_collection: str, new_collection: str, alias: str):
    try:
        # Single atomic operation
        await qdrant.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(
                    delete_alias=DeleteAlias(
                        alias_name=alias
                    )
                ),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        collection_name=new_collection,
                        alias_name=alias
                    )
                )
            ]
        )
    except Exception as e:
        # Rollback on failure
        logger.error(f"Alias update failed: {e}")
        # Keep old collection active
```

### V1 Performance Degradation Checklist

If V1 performance degrades over time:

1. **Cache Health**:

   ```bash
   # Check DragonflyDB stats
   docker exec dragonfly redis-cli INFO stats
   # Look for evicted_keys, hit_rate
   ```

2. **Query API Efficiency**:

   ```python
   # Monitor prefetch effectiveness
   metrics = await qdrant.get_collection(collection_name)
   print(f"Points: {metrics.points_count}")
   print(f"Segments: {metrics.segments_count}")
   # More segments = slower queries
   ```

3. **HyDE Cache Coverage**:

   ```python
   # Check cache hit rate
   stats = await hyde_service.get_cache_stats()
   if stats["hit_rate"] < 0.6:
       # Increase cache TTL or size
       pass
   ```

4. **Index Utilization**:

   ```python
   # Profile query to ensure index usage
   import time
   
   start = time.time()
   results = await qdrant.search(
       collection_name=collection_name,
       query_vector=embedding,
       query_filter=complex_filter
   )
   
   if time.time() - start > 0.1:  # 100ms
       print("Consider adding more indexes")
   ```

### V1 Debugging Tools

```python
# V1 Debug Helper
class V1DebugHelper:
    """Debug V1 performance issues."""
    
    async def diagnose_search_performance(self, query: str):
        """Complete V1 search diagnosis."""
        
        results = {
            "query": query,
            "timestamps": {},
            "cache_hits": {},
            "bottlenecks": []
        }
        
        # 1. Check cache
        start = time.time()
        cached = await self.cache.get(query)
        results["timestamps"]["cache_check"] = time.time() - start
        results["cache_hits"]["query"] = cached is not None
        
        # 2. HyDE generation
        if not cached:
            start = time.time()
            hyde_embedding = await self.hyde_service.enhance(query)
            hyde_time = time.time() - start
            results["timestamps"]["hyde"] = hyde_time
            
            if hyde_time > 0.2:
                results["bottlenecks"].append("HyDE generation slow")
        
        # 3. Query API search
        start = time.time()
        search_results = await self.search_with_query_api(query)
        search_time = time.time() - start
        results["timestamps"]["search"] = search_time
        
        if search_time > 0.1:
            results["bottlenecks"].append("Query API slow")
        
        # 4. Reranking
        start = time.time()
        reranked = await self.rerank(query, search_results)
        rerank_time = time.time() - start
        results["timestamps"]["rerank"] = rerank_time
        
        if rerank_time > 0.05:
            results["bottlenecks"].append("Reranking slow")
        
        return results
```

## 📞 Getting Help

### Community Resources

- [GitHub Issues](https://github.com/BjornMelin/ai-docs-vector-db-hybrid-scraper/issues)
- [Crawl4AI Community](https://github.com/unclecode/crawl4ai/discussions)
- [Qdrant Discord](https://discord.gg/qdrant)
- [Model Context Protocol Docs](https://modelcontextprotocol.io/)

### Creating Bug Reports

When reporting issues, please include:

1. **System Information**:

   ```bash
   # Run and include output:
   ./scripts/diagnose-issues.sh
   ```

2. **Configuration**:
   - Your `docker-compose.yml`
   - Your Claude Desktop MCP config (with API keys redacted)
   - Your `config/documentation-sites.json`

3. **Logs**:

   ```bash
   # Include relevant logs:
   docker logs qdrant-advanced
   uvx mcp-server-qdrant 2>&1 | tail -50
   npx -y firecrawl-mcp 2>&1 | tail -50
   ```

4. **Minimal Reproduction**:
   - Exact command that fails
   - Expected vs actual behavior
   - Error messages (full stack trace)

### Professional Support

For production deployments or complex issues:

- **Qdrant Cloud**: Professional managed service
- **Firecrawl Enterprise**: Enhanced scraping capabilities
- **Custom Integration**: Professional services available

---

🔧 **Remember**: Most issues can be resolved by ensuring all components are using the SOTA 2025 configuration and are properly connected. When in doubt, restart the entire stack and run diagnostics!
