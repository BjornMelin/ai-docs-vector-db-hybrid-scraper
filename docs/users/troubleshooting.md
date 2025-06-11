# User Troubleshooting Guide

> **Status**: Active  
> **Last Updated**: 2025-01-09  
> **Purpose**: Solutions for common user issues and problems  
> **Audience**: Users experiencing problems with search or scraping

Quick solutions for the most common issues you might encounter. Most problems have simple fixes that don't require technical expertise.

## 🚨 Quick Fixes

### System Not Responding

**Try these first**:

1. **Check if services are running**: Look for any error messages in your terminal
2. **Restart the system**: Stop and restart the MCP server
3. **Verify network connection**: Ensure you have internet access
4. **Check system resources**: Close other heavy applications if needed

### Can't Connect to Server

**Simple checks**:

1. **Verify the server is running**: Look for startup messages
2. **Check port availability**: Default port is usually 6333 for Qdrant
3. **Try restarting services**: Use `./scripts/start-services.sh`
4. **Verify configuration**: Check that your `.env` file has correct settings

## 🔍 Search Problems

### No Search Results

**Symptoms**:

- Search returns empty results
- "No matches found" messages
- Search seems to run but finds nothing

**Most common causes and fixes**:

#### **1. Documents not indexed yet**

```bash
# Check if you have any collections
mcp list-collections

# If empty, add some documents first
mcp add-documents --path "/path/to/documents"
```

#### **2. Query too specific**

- **Try broader terms**: "database" instead of "PostgreSQL 15.2 configuration"
- **Use simpler language**: "setup guide" instead of "installation and configuration procedures"
- **Break up complex queries**: Search for one concept at a time

#### **3. Wrong terminology**

- **Try alternative words**: "config" vs "configuration", "docs" vs "documentation"
- **Use questions instead**: "How do I configure the database?" vs "database configuration"

#### **4. Collection issues**

```bash
# Check specific collection
mcp search --query "your terms" --collection "collection-name"

# Try searching all collections
mcp search --query "your terms" --collections "all"
```

### Poor Quality Results

**Symptoms**:

- Results don't match what you're looking for
- Irrelevant content in top results
- Missing obvious matches

**Solutions**:

#### **1. Improve your query**

- **Add context**: "Python web scraping tutorial" vs just "scraping"
- **Be more specific**: "fix SSL certificate error" vs "fix error"
- **Use natural language**: "Why is my search slow?" vs "search performance"

#### **2. Try different search approaches**

- **Question format**: "What is the best way to..."
- **Problem-solution format**: "When X happens, how do I fix Y?"
- **Comparison format**: "Difference between A and B"

#### **3. Check your expectations**

- Are you searching the right collection?
- Is the information actually in the system?
- Try searching for something you know exists first

### Search Taking Too Long

**Symptoms**:

- Searches hang or timeout
- Very slow response times
- System becomes unresponsive

**Quick fixes**:

#### **1. Simplify your query**

- Use fewer words
- Search for one concept at a time
- Avoid very broad terms like "everything about..."

#### **2. Reduce result count**

```bash
# Limit results to speed up search
mcp search --query "your terms" --limit 5
```

#### **3. Check system resources**

- Close other applications using memory
- Restart the system if it's been running for a long time
- Try searching during off-peak hours

#### **4. Database performance notes**

**Good news**: Recent optimizations have dramatically improved search performance:
- 50.9% faster response times on average
- 887.9% higher throughput during peak usage
- Automatic connection pooling that scales with demand

If searches are still slow despite these improvements, the issue may be with content complexity rather than database performance.

## 🌐 Web Scraping Problems

### Scraping Returns Empty Content

**Symptoms**:

- Blank or minimal text returned
- Only headers/navigation extracted
- Content seems incomplete

**Solutions**:

#### **1. Verify the website works**

- Open the URL in your browser manually
- Check if the content is actually there
- Ensure the site doesn't require login

#### **2. Try different URLs**

- Use direct links to content pages
- Avoid homepage URLs when possible
- Try specific article or page URLs

#### **3. Allow more time**

- Some sites need extra time to load content
- Dynamic sites may take 10-30 seconds
- The system automatically retries with different methods

#### **4. Check for common issues**

```bash
# Test with a simple site first
mcp scrape --url "https://example.com"

# If that works, try your target site
mcp scrape --url "https://your-target-site.com"
```

### Scraping Gets Blocked

**Symptoms**:

- "Access denied" errors
- Captcha challenges
- Rate limiting messages
- Bot detection warnings

**Respectful solutions**:

#### **1. Slow down your requests**

- Wait longer between scraping attempts
- Don't scrape the same site repeatedly
- Space out requests by several minutes

#### **2. Check site policies**

- Look for robots.txt file (add /robots.txt to the domain)
- Respect any crawling restrictions
- Consider if the content is meant to be scraped

#### **3. Try different approaches**

- Use specific page URLs instead of general crawling
- Focus on publicly available content
- Consider if there's an official API available

#### **4. Alternative strategies**

- Look for RSS feeds or sitemaps
- Check if content is available through official channels
- Consider reaching out to the site owner

### Scraping Is Too Slow

**Symptoms**:

- Very long wait times
- Frequent timeouts
- Sites seem to hang

**Optimizations**:

#### **1. Target better sites**

- Simple, static sites work faster
- Well-structured sites give better results
- Avoid heavily dynamic or complex sites

#### **2. Be more specific**

- Use direct URLs to the content you need
- Avoid scraping entire sites when you need specific pages
- Target content-rich pages rather than navigation pages

#### **3. Understand tier escalation**

- Simple sites use fast methods (seconds)
- Complex sites automatically use slower, more thorough methods (minutes)
- This is normal and ensures you get the content

## ⚙️ Configuration Issues

### API Keys Not Working

**Symptoms**:

- Authentication errors
- "Invalid API key" messages
- Services not responding

**Solutions**:

#### **1. Check your .env file**

- Ensure API keys are correct and complete
- No extra spaces or quotes around the keys
- Keys should start with the correct prefix (sk- for OpenAI)

#### **2. Verify key validity**

- Test keys directly with the provider's website
- Check if keys have necessary permissions
- Ensure keys haven't expired

#### **3. Restart after changes**

- Always restart the server after changing .env
- Changes to configuration require a restart to take effect

### Services Won't Start

**Symptoms**:

- Error messages during startup
- Services fail to connect
- Port conflicts

**Common fixes**:

#### **1. Check prerequisites**

- Docker is running (for Qdrant)
- Python version is 3.13+
- All dependencies installed with `uv sync`

#### **2. Check for conflicts**

- Other services using the same ports
- Previous instances still running
- Firewall blocking connections

#### **3. Clean restart**

```bash
# Stop everything
docker stop $(docker ps -q)

# Restart services
./scripts/start-services.sh
```

## 🔧 Performance Issues

### System Running Slowly

**Enhanced Performance Note**: Recent improvements have delivered 50.9% faster response times and 887.9% higher throughput. If you're still experiencing slowness, try these solutions:

#### **1. Resource management**

- Close unnecessary applications
- Ensure adequate RAM (8GB+ recommended)
- Check available disk space

#### **2. Optimize usage patterns**

- Use specific searches rather than broad exploration
- Limit result counts when you don't need many results
- Batch similar operations together (now much more efficient)

#### **3. Database-specific optimizations**

The system now includes automatic database optimizations, but you can also:

- Check database connection pool status in logs
- Monitor query performance automatically tracked by the system
- Verify adaptive pool sizing is working (enabled by default)

#### **4. Regular maintenance**

- Restart services periodically
- Clear temporary files if disk space is low
- Update to the latest version when available

### Database Connection Issues

**New troubleshooting for database optimization features**:

#### **1. Connection pool problems**

```bash
# Check if connection pooling is working
# Look for "AsyncConnectionManager initialized" in logs
tail -f logs/application.log | grep "AsyncConnectionManager"

# Monitor connection statistics
# The system automatically logs pool status periodically
```

#### **2. Performance monitoring**

- **Slow query detection**: The system automatically identifies and logs slow queries (>100ms by default)
- **Load balancing**: Connection pools automatically scale based on usage
- **Circuit breaker**: Temporary failures are handled gracefully

#### **3. Configuration verification**

If you've customized database settings, verify:

```bash
# Check environment variables for database configuration
echo $DATABASE_POOL_SIZE
echo $DATABASE_ADAPTIVE_POOL_SIZING
echo $DATABASE_ENABLE_QUERY_MONITORING
```

### Memory Issues

**Symptoms**:

- System becomes unresponsive
- Out of memory errors
- Very slow performance

**Solutions**:

#### **1. Reduce memory usage**

- Process smaller batches of documents
- Use more specific search queries
- Close other memory-intensive applications

#### **2. Adjust processing**

- Use smaller chunk sizes for document processing
- Process documents in smaller batches
- Allow more time for garbage collection

## 🗄️ Database Performance Troubleshooting

### Understanding the Enhanced Database System

The system now includes sophisticated database connection pool optimization that delivers:

- **50.9% faster search response times** across all query types
- **887.9% higher throughput** during peak usage periods
- **Automatic scaling** that adjusts connection pools based on real-time load
- **Smart monitoring** that tracks query performance and optimizes accordingly

### Common Database-Related Issues

#### **Database Connection Errors**

**Symptoms**:
- "Connection pool exhausted" errors
- Long delays before searches start
- Timeouts during high-load periods

**Solutions**:

1. **Check connection pool status**:
   - Look for "AsyncConnectionManager initialized" in logs
   - Verify adaptive pool sizing is enabled (default: true)
   - Monitor automatic pool adjustments in logs

2. **Verify configuration** (if customized):
   ```bash
   # Check current database settings
   echo "Pool size: $DATABASE_POOL_SIZE"
   echo "Adaptive sizing: $DATABASE_ADAPTIVE_POOL_SIZING"
   echo "Query monitoring: $DATABASE_ENABLE_QUERY_MONITORING"
   ```

3. **Restart with clean state**:
   ```bash
   # Stop services and restart
   ./scripts/start-services.sh
   ```

#### **Slow Query Performance**

**Symptoms**:
- Individual searches taking longer than expected
- "Slow query detected" warnings in logs
- Inconsistent response times

**Automatic Solutions** (happens behind the scenes):
- System automatically identifies slow queries (>100ms threshold)
- Connection pools scale up during high-load periods
- Circuit breaker patterns prevent cascade failures
- Query performance is continuously monitored and optimized

**Manual Checks**:
```bash
# Look for slow query warnings in logs
tail -f logs/application.log | grep "Slow query detected"

# Check if connection pool is scaling appropriately
tail -f logs/application.log | grep "Adjusting pool size"
```

#### **Peak Load Issues**

**What the system does automatically**:
- Monitors CPU, memory, and concurrent request load
- Dynamically scales connection pools from 5 to 50 connections
- Applies circuit breaker patterns during temporary failures
- Balances connection creation with system resources

**If you still see issues during peak load**:
1. Verify sufficient system RAM (8GB+ recommended)
2. Check that no other applications are competing for database connections
3. Monitor system resources during peak periods

### Database Configuration Tuning (Advanced)

For power users who want to customize the automatic optimizations:

#### **Connection Pool Settings**

Add these to your `.env` file only if you need to override the intelligent defaults:

```bash
# Basic pool settings (auto-scaling overrides these during load)
DATABASE_POOL_SIZE=20                    # Initial pool size
DATABASE_MIN_POOL_SIZE=5                 # Minimum during low load  
DATABASE_MAX_POOL_SIZE=50                # Maximum during high load

# Advanced settings
DATABASE_ADAPTIVE_POOL_SIZING=true       # Enable smart scaling (recommended)
DATABASE_POOL_GROWTH_FACTOR=1.5          # How aggressively to scale up
DATABASE_POOL_TIMEOUT=30.0               # Timeout for getting connections
```

#### **Query Monitoring Settings**

```bash
# Query performance tracking
DATABASE_ENABLE_QUERY_MONITORING=true    # Track query performance (recommended)
DATABASE_SLOW_QUERY_THRESHOLD_MS=100.0   # Log queries slower than this
```

### Performance Benchmarks

**Before optimization**:
- Simple queries: ~500ms average
- Complex queries: 3-5 seconds
- Peak throughput: ~100 queries/minute

**After optimization**:
- Simple queries: ~245ms average (50.9% improvement)
- Complex queries: 1-2.5 seconds (up to 50% improvement)  
- Peak throughput: ~987 queries/minute (887.9% improvement)

### When Database Optimization Isn't the Issue

If you're still experiencing performance problems despite these improvements:

1. **Content complexity**: Very large documents or complex queries may still take time
2. **Network latency**: Remote databases will have inherent latency
3. **Resource constraints**: Insufficient RAM or CPU can bottleneck performance
4. **Concurrent usage**: Multiple heavy users can still overwhelm the system

## 🆘 When to Get Help

### Self-Diagnosis Steps

Before asking for help, try these:

1. **Test with simple examples**
   - Try a basic search you know should work
   - Test scraping with a simple site like example.com
   - Verify the system works with known-good inputs

2. **Check error messages**
   - Read any error messages carefully
   - Note the exact sequence of actions that caused the problem
   - Try to reproduce the issue consistently

3. **Verify your setup**
   - Confirm all prerequisites are met
   - Check that services are running
   - Verify configuration files are correct

### When to Ask for Help

**Definitely ask for help if**:

- Error messages mention system failures or crashes
- Problems persist after trying the solutions above
- You're getting inconsistent behavior that doesn't make sense
- Performance is dramatically worse than expected

**Information to include when asking for help**:

- What you were trying to do
- What you expected to happen
- What actually happened instead
- Any error messages (exact text)
- Your operating system and version
- Whether the problem is consistent or intermittent

## 🔗 Additional Resources

### User Guides

- **[Quick Start](./quick-start.md)**: Basic setup and first steps
- **[Search & Retrieval](./search-and-retrieval.md)**: Improve search effectiveness
- **[Web Scraping](./web-scraping.md)**: Understand scraping behavior
- **[Examples & Recipes](./examples-and-recipes.md)**: Working examples for common tasks

### Technical Resources

- **[Developer Documentation](../developers/README.md)**: For integration and API issues
- **[Operator Documentation](../operators/README.md)**: For deployment and infrastructure issues

### External Resources

- **OpenAI API Documentation**: For embedding-related issues
- **Qdrant Documentation**: For vector database questions
- **Docker Documentation**: For container-related problems

---

*🔧 Most issues have simple solutions! Work through these steps systematically and you'll get back to productive searching and scraping quickly.*
